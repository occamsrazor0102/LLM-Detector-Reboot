"""Evaluation runner — applies a Pipeline to an EvalSample list and produces an EvalReport."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.metrics import (
    auroc, ece, brier, tpr_at_fpr, summarize, confusion_at_threshold,
)


class PipelineLike(Protocol):
    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination: ...


@dataclass(frozen=True)
class EvalReport:
    predictions: list[dict]
    metrics: dict
    per_tier: dict[str, dict]
    n_samples: int
    config_hash: str
    failed_samples: list[dict] = field(default_factory=list)
    per_attack: dict[str, dict] = field(default_factory=dict)


def _config_hash(config: dict | None) -> str:
    if config is None:
        config = {}
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _extract_config(pipeline) -> dict:
    for attr in ("config", "_config"):
        if hasattr(pipeline, attr):
            cfg = getattr(pipeline, attr)
            if isinstance(cfg, dict):
                return cfg
    return {}


def run_eval(
    pipeline: PipelineLike,
    dataset: list[EvalSample],
    *,
    progress: bool = False,
) -> EvalReport:
    predictions: list[dict] = []
    failed: list[dict] = []
    kept: list[EvalSample] = []

    iterable = dataset
    if progress:
        try:
            from tqdm import tqdm
            iterable = tqdm(dataset)
        except ImportError:
            pass

    for sample in iterable:
        try:
            det = pipeline.analyze(sample.text)
        except Exception as exc:
            failed.append({"id": sample.id, "error": str(exc)})
            continue
        predictions.append({
            "id": sample.id,
            "label": sample.label,
            "p_llm": float(det.p_llm),
            "determination": det.label,
            "tier": sample.tier,
        })
        kept.append(sample)

    y_true = [p["label"] for p in predictions]
    y_score = [p["p_llm"] for p in predictions]
    metrics = summarize(y_true, y_score)

    tier_groups: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        if p["tier"] is None:
            continue
        tier_groups[p["tier"]].append(p)

    per_tier = {}
    for tier, preds in tier_groups.items():
        yt = [p["label"] for p in preds]
        ys = [p["p_llm"] for p in preds]
        per_tier[tier] = summarize(yt, ys)

    sample_by_id = {s.id: s for s in kept}
    attack_groups: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        s = sample_by_id.get(p["id"])
        if s is None or s.attack_name is None:
            continue
        attack_groups[s.attack_name].append(p)
    per_attack = {}
    for attack, preds in attack_groups.items():
        yt = [p["label"] for p in preds]
        ys = [p["p_llm"] for p in preds]
        per_attack[attack] = summarize(yt, ys)

    return EvalReport(
        predictions=predictions,
        metrics=metrics,
        per_tier=per_tier,
        n_samples=len(predictions),
        config_hash=_config_hash(_extract_config(pipeline)),
        failed_samples=failed,
        per_attack=per_attack,
    )


def _clean_metrics(d: dict) -> dict:
    """NaN -> None for JSON friendliness; other floats rounded."""
    import math
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def eval_report_to_dict(
    report: EvalReport,
    *,
    include_predictions: bool = True,
    threshold: float = 0.5,
) -> dict:
    """Serialize an EvalReport to a JSON-safe dict for the UI."""
    y_true = [p["label"] for p in report.predictions]
    y_score = [p["p_llm"] for p in report.predictions]
    confusion = confusion_at_threshold(y_true, y_score, threshold=threshold)
    tp, fp, tn, fn = confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision is not None and recall is not None and (precision + recall))
        else None
    )
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else None
    return {
        "n_samples": report.n_samples,
        "n_failed": len(report.failed_samples),
        "config_hash": report.config_hash,
        "metrics": _clean_metrics(report.metrics),
        "per_tier": {k: _clean_metrics(v) for k, v in report.per_tier.items()},
        "per_attack": {k: _clean_metrics(v) for k, v in report.per_attack.items()},
        "confusion": {
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
        },
        "failed_samples": list(report.failed_samples),
        "predictions": list(report.predictions) if include_predictions else [],
    }
