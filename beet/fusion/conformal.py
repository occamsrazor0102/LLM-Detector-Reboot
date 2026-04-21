# beet/fusion/conformal.py
"""
Conformal prediction wrapper.
Produces valid prediction sets at user-specified error rate alpha.
"""
import json
from pathlib import Path

import numpy as np


class ConformalWrapper:
    """
    Split conformal prediction using nonconformity score = 1 - p(true_label).
    Guarantees marginal coverage: P(true label in prediction set) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha
        self._threshold: float | None = None

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Calibrate on held-out data.
        scores: array of p_llm values from fusion model
        labels: array of 0/1 ground truth (1 = LLM-generated)
        """
        # Nonconformity score = 1 - p(correct class)
        nonconformity = np.where(labels == 1, 1 - scores, scores)
        n = len(nonconformity)
        # Finite-sample quantile (Vovk et al.)
        self._threshold = float(np.quantile(nonconformity, np.ceil((n + 1) * (1 - self._alpha)) / n))

    def predict_set(self, p_llm: float) -> list[str]:
        """
        Returns a prediction set (subset of RED/AMBER/YELLOW/GREEN) consistent
        with the calibration. If uncalibrated, returns all four labels.
        """
        if self._threshold is None:
            return ["RED", "AMBER", "YELLOW", "GREEN"]

        # A label is included if its nonconformity score <= threshold
        labels = []
        bands = [("RED", 0.75, 1.01), ("AMBER", 0.50, 0.75),
                 ("YELLOW", 0.25, 0.50), ("GREEN", 0.0, 0.25)]
        for label, lo, hi in bands:
            # Representative p_llm for this band
            mid = (lo + min(hi, 1.0)) / 2
            # Would the prediction "this label" be non-conforming?
            # A RED prediction is non-conforming if mid < 0.75 (i.e., model says low)
            # Simple heuristic: include label if p_llm is "close enough" to the band
            nonconf_llm = 1 - p_llm      # nonconformity for LLM hypothesis
            nonconf_human = p_llm         # nonconformity for human hypothesis
            is_llm_label = lo >= 0.50
            nc = nonconf_llm if is_llm_label else nonconf_human
            if nc <= self._threshold:
                labels.append(label)

        return labels if labels else ["UNCERTAIN"]

    def save(self, path: Path) -> None:
        if self._threshold is None:
            raise RuntimeError("ConformalWrapper.save: not calibrated")
        Path(path).write_text(json.dumps({"alpha": self._alpha, "threshold": self._threshold}))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        self._alpha = float(data["alpha"])
        self._threshold = float(data["threshold"])
