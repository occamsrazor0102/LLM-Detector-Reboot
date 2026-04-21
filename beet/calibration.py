# beet/calibration.py
"""Detector-level probability calibration via isotonic regression.

Maps each detector's raw_score -> calibrated p_llm using a dataset of
(raw_score, label) pairs. Stored as JSON for portability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression


class DetectorCalibrator:
    """Fits and stores isotonic regression per detector."""

    def __init__(self) -> None:
        self._models: dict[str, IsotonicRegression] = {}

    def fit(self, detector_id: str, raw_scores: Iterable[float], labels: Iterable[int]) -> None:
        x = np.asarray(list(raw_scores), dtype=float)
        y = np.asarray(list(labels), dtype=float)
        if len(x) < 2 or len(set(y.tolist())) < 2:
            # Not enough variation to fit; skip silently.
            return
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x, y)
        self._models[detector_id] = ir

    def transform(self, detector_id: str, raw_score: float) -> float:
        model = self._models.get(detector_id)
        if model is None:
            return float(raw_score)
        return float(model.predict([raw_score])[0])

    def has(self, detector_id: str) -> bool:
        return detector_id in self._models

    def save(self, path: Path) -> None:
        data: dict[str, dict[str, list[float]]] = {}
        for det_id, ir in self._models.items():
            data[det_id] = {
                "X": ir.X_thresholds_.tolist(),
                "y": ir.y_thresholds_.tolist(),
            }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        for det_id, params in data.items():
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            x = np.asarray(params["X"], dtype=float)
            y = np.asarray(params["y"], dtype=float)
            ir.fit(x, y)
            self._models[det_id] = ir
