"""Fairness evaluation: stratify pipeline errors by a group key.

Given a labeled dataset and a group key (e.g. `tier`, `source`), compute
per-group FPR and ECE, then flag disparities exceeding a parity ratio.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from beet.evaluation.dataset import EvalSample
from beet.evaluation.metrics import confusion_at_threshold, ece
from beet.evaluation.runner import run_eval


@dataclass(frozen=True)
class FairnessReport:
    overall_fpr: float
    per_group_fpr: dict[str, float]
    fpr_parity_ratio: float
    per_group_ece: dict[str, float]
    flagged_disparities: list[str] = field(default_factory=list)
    n_per_group: dict[str, int] = field(default_factory=dict)


def _safe_fpr(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = fp + tn
    return float(fp) / denom if denom else 0.0


def run_fairness_eval(
    pipeline,
    dataset: list[EvalSample],
    *,
    group_key: str = "tier",
    threshold: float = 0.50,
    max_fpr_ratio: float = 2.0,
    progress: bool = False,
) -> FairnessReport:
    report = run_eval(pipeline, dataset, progress=progress)
    sample_by_id = {s.id: s for s in dataset}

    overall = confusion_at_threshold(
        [p["label"] for p in report.predictions],
        [p["p_llm"] for p in report.predictions],
        threshold,
    )
    overall_fpr = _safe_fpr(**overall)

    groups: dict[str, list[dict]] = defaultdict(list)
    for p in report.predictions:
        s = sample_by_id.get(p["id"])
        if s is None:
            continue
        val = getattr(s, group_key, None)
        if val is None:
            continue
        groups[val].append(p)

    per_group_fpr: dict[str, float] = {}
    per_group_ece: dict[str, float] = {}
    n_per_group: dict[str, int] = {}
    for g, preds in groups.items():
        y_true = [p["label"] for p in preds]
        y_score = [p["p_llm"] for p in preds]
        conf = confusion_at_threshold(y_true, y_score, threshold)
        per_group_fpr[g] = _safe_fpr(**conf)
        per_group_ece[g] = ece(y_true, y_score)
        n_per_group[g] = len(preds)

    # FPR parity ratio: max / min across groups. With fewer than two groups,
    # or when every group has FPR=0, parity is trivially 1.0. When max>0 but
    # min=0, ratio is infinite (flagged as maximal disparity).
    if len(per_group_fpr) < 2:
        parity_ratio = 1.0
    else:
        fpr_values = list(per_group_fpr.values())
        max_fpr, min_fpr = max(fpr_values), min(fpr_values)
        if max_fpr == 0.0:
            parity_ratio = 1.0
        elif min_fpr == 0.0:
            parity_ratio = float("inf")
        else:
            parity_ratio = max_fpr / min_fpr

    flagged: list[str] = []
    if parity_ratio > max_fpr_ratio:
        hi = max(per_group_fpr.items(), key=lambda kv: kv[1])
        lo = min(per_group_fpr.items(), key=lambda kv: kv[1])
        flagged.append(
            f"FPR parity ratio {parity_ratio:.2f} exceeds max {max_fpr_ratio}: "
            f"highest {hi[0]}={hi[1]:.3f}, lowest {lo[0]}={lo[1]:.3f}"
        )

    return FairnessReport(
        overall_fpr=overall_fpr,
        per_group_fpr=per_group_fpr,
        fpr_parity_ratio=parity_ratio,
        per_group_ece=per_group_ece,
        flagged_disparities=flagged,
        n_per_group=n_per_group,
    )
