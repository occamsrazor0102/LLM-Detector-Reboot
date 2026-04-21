"""Drift monitoring: tracks pipeline output and per-feature distributions.

Alerts fire when the current window diverges materially from a stored baseline:
- Population drift: mean P(LLM) drift > threshold
- Feature distribution drift: symmetric KL divergence between baseline and current histogram
- Calibration drift: ECE between predicted p_llm and confirmed labels above threshold
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _kl_symmetric(p: list[float], q: list[float], eps: float = 1e-6) -> float:
    n = min(len(p), len(q))
    if n == 0:
        return 0.0
    sp = sum(p[:n]) or 1.0
    sq = sum(q[:n]) or 1.0
    p_norm = [(x / sp) + eps for x in p[:n]]
    q_norm = [(x / sq) + eps for x in q[:n]]
    kl_pq = sum(a * math.log(a / b) for a, b in zip(p_norm, q_norm))
    kl_qp = sum(b * math.log(b / a) for a, b in zip(p_norm, q_norm))
    return (kl_pq + kl_qp) / 2


def _histogram(values: list[float], n_bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    if not values:
        return [0.0] * n_bins
    bins = [0.0] * n_bins
    width = (hi - lo) / n_bins
    for v in values:
        idx = int((v - lo) / max(width, 1e-9))
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        bins[idx] += 1
    return bins


def _ece(y_true: list[int], y_score: list[float], n_bins: int = 10) -> float:
    if len(y_true) != len(y_score) or not y_true:
        return 0.0
    edges = [i / n_bins for i in range(n_bins + 1)]
    total = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = [(s >= lo and (s < hi or (i == n_bins - 1 and s <= hi))) for s in y_score]
        m = sum(mask)
        if m == 0:
            continue
        bucket_conf = sum(s for s, mk in zip(y_score, mask) if mk) / m
        bucket_acc = sum(t for t, mk in zip(y_true, mask) if mk) / m
        total += (m / n) * abs(bucket_conf - bucket_acc)
    return total


class DriftMonitor:
    def __init__(self, store_path: Path, config: dict):
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)
        drift_cfg = config.get("drift_monitoring", {}) if isinstance(config, dict) else {}
        self._window = drift_cfg.get("window_size", 1000)
        self._kl_threshold = drift_cfg.get("kl_threshold", 0.20)
        self._ece_threshold = drift_cfg.get("ece_threshold", 0.15)
        self._observations: list[dict] = []
        self._baseline_features: dict[str, list[float]] = {}
        self._baseline_hist: dict[str, list[float]] = {}
        self._feature_bounds: dict[str, tuple[float, float]] = {}

    def set_baseline(self, feature_vectors: list[dict]) -> None:
        """Record baseline per-feature distributions for drift comparison.

        Bin bounds are fixed from baseline min/max so that subsequent current-
        window histograms use the same bucketing and are directly comparable.
        """
        if not feature_vectors:
            return
        by_name: dict[str, list[float]] = defaultdict(list)
        for fv in feature_vectors:
            for k, v in fv.items():
                if isinstance(v, (int, float)) and not math.isnan(float(v)):
                    by_name[k].append(float(v))
        self._baseline_features = dict(by_name)
        self._feature_bounds = {}
        self._baseline_hist = {}
        for name, vals in by_name.items():
            lo = min(vals)
            hi = max(vals)
            if hi <= lo:
                hi = lo + 1.0  # degenerate — give bins some room
            self._feature_bounds[name] = (lo, hi)
            self._baseline_hist[name] = _histogram(vals, n_bins=10, lo=lo, hi=hi)

    def record(self, p_llm: float, determination: str, feature_vector: dict,
               confirmed_label: int | None = None) -> list[str]:
        self._observations.append({
            "p_llm": p_llm, "determination": determination,
            "features": feature_vector, "confirmed": confirmed_label,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        })
        alerts: list[str] = []
        if len(self._observations) >= self._window:
            alerts = self.check_drift()
            self._flush_alerts(alerts)
            self._observations = []
        return alerts

    def check_drift(self) -> list[str]:
        alerts: list[str] = []
        n = len(self._observations)
        if n < 10:
            return alerts

        p_values = [o["p_llm"] for o in self._observations]
        mean_p = sum(p_values) / n
        if mean_p > 0.70:
            alerts.append(f"POPULATION_DRIFT: mean P(LLM)={mean_p:.3f} exceeds 0.70")
        elif mean_p < 0.15:
            alerts.append(f"POPULATION_DRIFT: mean P(LLM)={mean_p:.3f} below 0.15")

        if self._baseline_hist:
            current_hists = self._current_feature_hists()
            for name, cur_hist in current_hists.items():
                base_hist = self._baseline_hist.get(name)
                if not base_hist:
                    continue
                kl = _kl_symmetric(base_hist, cur_hist)
                if kl > self._kl_threshold:
                    alerts.append(
                        f"FEATURE_DRIFT: '{name}' KL={kl:.3f} exceeds {self._kl_threshold}"
                    )

        confirmed = [
            (int(o["confirmed"]), float(o["p_llm"]))
            for o in self._observations
            if o.get("confirmed") is not None
        ]
        if len(confirmed) >= 20:
            y_true = [c[0] for c in confirmed]
            y_score = [c[1] for c in confirmed]
            ece_val = _ece(y_true, y_score)
            if ece_val > self._ece_threshold:
                alerts.append(f"CALIBRATION_DRIFT: ECE={ece_val:.3f} exceeds {self._ece_threshold}")

        return alerts

    def _current_feature_hists(self) -> dict[str, list[float]]:
        by_name: dict[str, list[float]] = defaultdict(list)
        for o in self._observations:
            for k, v in (o.get("features") or {}).items():
                if isinstance(v, (int, float)) and not math.isnan(float(v)):
                    by_name[k].append(float(v))
        out: dict[str, list[float]] = {}
        for name, vals in by_name.items():
            bounds = self._feature_bounds.get(name)
            if bounds is not None:
                lo, hi = bounds
            else:
                lo = min(vals) if vals else 0.0
                hi = max(vals) if vals else 1.0
                if hi <= lo:
                    hi = lo + 1.0
            out[name] = _histogram(vals, n_bins=10, lo=lo, hi=hi)
        return out

    def _flush_alerts(self, alerts: list[str]) -> None:
        if not alerts:
            return
        path = self._path / "alerts.jsonl"
        import json
        with open(path, "a", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps({
                    "alert": a,
                    "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                }) + "\n")

    def get_summary(self) -> dict:
        if not self._observations:
            return {"n_observations": 0}
        p_values = [o["p_llm"] for o in self._observations]
        n = len(p_values)
        mean = sum(p_values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in p_values) / n)
        return {
            "n_observations": n,
            "mean_p_llm": round(mean, 4),
            "std_p_llm": round(std, 4),
            "high_confidence_fraction":
                sum(1 for p in p_values if p > 0.75 or p < 0.25) / n,
        }
