"""Leave-one-out ablation harness."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

from beet.evaluation.dataset import EvalSample
from beet.evaluation.runner import EvalReport, run_eval


@dataclass(frozen=True)
class AblationReport:
    baseline: EvalReport
    per_detector: dict[str, EvalReport]
    deltas: dict[str, dict]
    ranked: list[tuple[str, float]]


def _default_pipeline_factory(config: dict):
    from beet.pipeline import BeetPipeline
    return BeetPipeline(config)


def _enabled_detectors(config: dict) -> list[str]:
    out = []
    for name, cfg in config.get("detectors", {}).items():
        if cfg.get("enabled", True):
            out.append(name)
    return out


def _disable(config: dict, detector: str) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("detectors", {}).setdefault(detector, {})["enabled"] = False
    return cfg


def _safe_sub(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def run_ablation(
    base_config: dict,
    dataset: list[EvalSample],
    *,
    detectors: list[str] | None = None,
    progress: bool = False,
    pipeline_factory: Callable[[dict], object] = _default_pipeline_factory,
) -> AblationReport:
    baseline_pipeline = pipeline_factory(base_config)
    baseline = run_eval(baseline_pipeline, dataset, progress=progress)

    targets = detectors if detectors is not None else _enabled_detectors(base_config)

    per_detector: dict[str, EvalReport] = {}
    deltas: dict[str, dict] = {}

    for det in targets:
        ablated_cfg = _disable(base_config, det)
        ablated_pipeline = pipeline_factory(ablated_cfg)
        ablated = run_eval(ablated_pipeline, dataset, progress=progress)
        per_detector[det] = ablated
        deltas[det] = {
            "delta_auroc": _safe_sub(baseline.metrics["auroc"], ablated.metrics["auroc"]),
            "delta_ece": _safe_sub(baseline.metrics["ece"], ablated.metrics["ece"]),
            "delta_brier": _safe_sub(baseline.metrics["brier"], ablated.metrics["brier"]),
        }

    def _abs_delta(d):
        v = d.get("delta_auroc", 0.0)
        return 0.0 if math.isnan(v) else abs(v)

    ranked = sorted(
        ((name, _abs_delta(d)) for name, d in deltas.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    return AblationReport(
        baseline=baseline,
        per_detector=per_detector,
        deltas=deltas,
        ranked=ranked,
    )


def verdict_for(delta_auroc: float) -> str:
    if math.isnan(delta_auroc):
        return "unknown"
    if delta_auroc < 0:
        return "hurting"
    if delta_auroc >= 0.05:
        return "load-bearing"
    if delta_auroc >= 0.02:
        return "helpful"
    if delta_auroc >= 0.01:
        return "marginal"
    return "negligible"
