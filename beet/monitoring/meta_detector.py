"""Meta-detector: monitors per-detector output distributions over time.

Tracks each detector's rolling p_llm distribution. Flags detectors whose
behaviour has materially shifted (mean/variance), or — when confirmed labels
are available — whose accuracy has fallen below baseline.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from beet.contracts import LayerResult


@dataclass
class DetectorHealth:
    detector_id: str
    verdict: str                   # "stable" | "drifting" | "degrading" | "unknown"
    mean_p_llm: float
    n_observations: int
    n_confirmed: int
    accuracy_vs_confirmed: float | None
    reason: str


class MetaDetector:
    def __init__(
        self,
        window_size: int = 500,
        mean_shift_threshold: float = 0.15,
        min_accuracy: float = 0.60,
    ):
        self._window_size = int(window_size)
        self._mean_shift = float(mean_shift_threshold)
        self._min_accuracy = float(min_accuracy)
        self._rolling: dict[str, Deque[tuple[float, int | None]]] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )
        self._baseline_means: dict[str, float] = {}

    def set_baseline(self, baselines: dict[str, float]) -> None:
        self._baseline_means = dict(baselines)

    def record(self, layer_result: LayerResult, confirmed_label: int | None = None) -> None:
        if layer_result.determination == "SKIP":
            return
        self._rolling[layer_result.layer_id].append((float(layer_result.p_llm), confirmed_label))

    def health(self) -> dict[str, DetectorHealth]:
        out: dict[str, DetectorHealth] = {}
        for det_id, obs in self._rolling.items():
            if not obs:
                continue
            p_values = [p for p, _ in obs]
            n = len(p_values)
            mean = sum(p_values) / n
            confirmed = [(p, c) for p, c in obs if c is not None]
            n_conf = len(confirmed)
            acc = None
            verdict = "stable"
            reason = ""

            baseline = self._baseline_means.get(det_id)
            if baseline is not None and abs(mean - baseline) > self._mean_shift:
                verdict = "drifting"
                reason = f"mean {mean:.3f} drifted from baseline {baseline:.3f}"

            if n_conf >= 20:
                # Treat p >= 0.5 as predicting LLM
                correct = sum(1 for p, c in confirmed if (p >= 0.5) == (c == 1))
                acc = correct / n_conf
                if acc < self._min_accuracy:
                    verdict = "degrading"
                    reason = f"accuracy {acc:.3f} below min {self._min_accuracy}"

            out[det_id] = DetectorHealth(
                detector_id=det_id, verdict=verdict,
                mean_p_llm=round(mean, 4), n_observations=n,
                n_confirmed=n_conf,
                accuracy_vs_confirmed=None if acc is None else round(acc, 4),
                reason=reason or "within expected range",
            )
        return out

    def check_degradation(self) -> dict[str, str]:
        return {k: v.verdict for k, v in self.health().items()}
