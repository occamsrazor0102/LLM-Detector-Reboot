"""Split-conformal prediction wrapper for binary (LLM vs human) classification.

Produces valid marginal-coverage prediction sets: after calibration on a
held-out set of scores + labels, `predict_set(p)` returns a subset of
{"LLM", "human"} that contains the true label with probability at least
`1 - alpha` over the calibration distribution, as long as the test
point is exchangeable with the calibration set.

This replaces a prior implementation that mapped conformal scores to
four UI severity bands (RED/AMBER/YELLOW/GREEN) via a midpoint
heuristic — that mapping did **not** preserve marginal coverage and
made the documented guarantee false. The severity-band mapping has
moved to `beet/fusion/ebm.py::_p_llm_to_labels` where it's honestly
labelled as a cosmetic display mapping.

Math:
  - Nonconformity score s_i = 1 - p(true_label_i) for calibration point i.
  - Quantile q_hat = ceil((n + 1) * (1 - alpha)) / n quantile of {s_i}.
  - For a test point with score p_llm, include class c in the set iff
    1 - p(c | p_llm) <= q_hat, i.e. the nonconformity score of the
    hypothesis "the label is c" doesn't exceed the calibrated quantile.
  - Uncalibrated: return both classes (maximally conservative, honest).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


_LABELS = ["human", "LLM"]  # index 0 = label 0 = human; index 1 = LLM


class ConformalWrapper:
    """Honest split-conformal for the binary LLM/human decision."""

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._threshold: float | None = None
        self._n_calibration: int = 0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def calibrated(self) -> bool:
        return self._threshold is not None

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the conformal threshold on a held-out calibration set.

        scores: array of fused p(LLM) estimates, shape (n,), values in [0, 1].
        labels: array of ground-truth labels, shape (n,), 0 = human, 1 = LLM.
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if scores.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays")
        n = len(scores)
        if n < 10:
            raise ValueError(
                f"too few calibration points ({n}); split-conformal needs at "
                "least ~10 and usually many more for a useful interval"
            )
        # Nonconformity = 1 - p(true class).  For LLM-labelled points, the
        # true-class probability is p_llm itself; for human-labelled points
        # it is 1 - p_llm.
        nonconformity = np.where(labels == 1, 1.0 - scores, scores)
        # Finite-sample split-conformal quantile (Vovk et al.).
        q_level = math.ceil((n + 1) * (1.0 - self._alpha)) / n
        # For very small n and very small alpha, q_level can exceed 1 —
        # clamp so np.quantile doesn't raise.
        q_level = min(q_level, 1.0)
        self._threshold = float(np.quantile(nonconformity, q_level, method="higher"))
        self._n_calibration = int(n)

    def predict_set(self, p_llm: float) -> list[str]:
        """Return the conformal prediction set for one fused score.

        Returns a list drawn from ["human", "LLM"]. Possible values:
          - ["LLM"]            — confident LLM
          - ["human"]           — confident human
          - ["human", "LLM"]   — abstain; both hypotheses are conformal
        When uncalibrated, always returns ["human", "LLM"] (honest
        maximum-uncertainty set, not a coverage guarantee).
        """
        p_llm = max(0.0, min(1.0, float(p_llm)))
        if self._threshold is None:
            return list(_LABELS)
        out: list[str] = []
        # Nonconformity of each class hypothesis for this test point.
        nc = {"human": p_llm, "LLM": 1.0 - p_llm}
        for cls in _LABELS:
            if nc[cls] <= self._threshold:
                out.append(cls)
        # It's mathematically possible to produce the empty set when
        # alpha is high and both classes look unconformable. Honest thing
        # is to report the empty set — but most downstream code assumes
        # at least one member. Callers should check .calibrated and fall
        # back; returning both labels here would misreport coverage.
        return out

    def save(self, path: Path) -> None:
        if self._threshold is None:
            raise RuntimeError("ConformalWrapper.save: not calibrated")
        Path(path).write_text(json.dumps({
            "alpha": self._alpha,
            "threshold": self._threshold,
            "n_calibration": self._n_calibration,
            "schema": "binary-split-conformal-v1",
        }))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        schema = data.get("schema")
        if schema is not None and schema != "binary-split-conformal-v1":
            raise ValueError(
                f"ConformalWrapper.load: unknown schema {schema!r}; "
                "artifact was produced by a different version"
            )
        self._alpha = float(data["alpha"])
        self._threshold = float(data["threshold"])
        self._n_calibration = int(data.get("n_calibration", 0))
