"""Evaluation runner — applies a Pipeline to an EvalSample list and produces an EvalReport."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.metrics import auroc, ece, brier, tpr_at_fpr, summarize


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
