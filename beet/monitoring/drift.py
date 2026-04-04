import math
import json
from pathlib import Path
from datetime import datetime

class DriftMonitor:
    def __init__(self, store_path: Path, config: dict):
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)
        drift_cfg = config.get("drift_monitoring", {})
        self._window = drift_cfg.get("window_size", 1000)
        self._observations: list[dict] = []

    def record(self, p_llm: float, determination: str, feature_vector: dict) -> None:
        self._observations.append({"p_llm": p_llm, "determination": determination,
            "features": feature_vector, "timestamp": datetime.utcnow().isoformat()})
        if len(self._observations) >= self._window:
            self._check_drift()
            self._observations = []

    def _check_drift(self) -> list[str]:
        alerts = []
        n = len(self._observations)
        if n < 10: return alerts
        p_values = [o["p_llm"] for o in self._observations]
        mean_p = sum(p_values) / n
        if mean_p > 0.70:
            alerts.append(f"DRIFT_ALERT: Mean P(LLM)={mean_p:.2f} unusually high")
        elif mean_p < 0.15:
            alerts.append(f"DRIFT_ALERT: Mean P(LLM)={mean_p:.2f} unusually low")
        return alerts

    def get_summary(self) -> dict:
        if not self._observations: return {"n_observations": 0}
        p_values = [o["p_llm"] for o in self._observations]
        n = len(p_values)
        mean = sum(p_values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in p_values) / n)
        return {"n_observations": n, "mean_p_llm": round(mean, 4), "std_p_llm": round(std, 4),
            "high_confidence_fraction": sum(1 for p in p_values if p > 0.75 or p < 0.25) / n}
