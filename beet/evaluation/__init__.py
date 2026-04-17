"""BEET 2.0 evaluation module — dataset loading, metrics, runner, ablation."""
from beet.evaluation.dataset import (
    EvalSample,
    load_dataset,
    save_dataset,
    build_dataset,
)
from beet.evaluation.metrics import (
    auroc,
    ece,
    brier,
    tpr_at_fpr,
    confusion_at_threshold,
    per_tier_breakdown,
    summarize,
)
from beet.evaluation.runner import EvalReport, run_eval
from beet.evaluation.ablation import AblationReport, run_ablation, verdict_for

__all__ = [
    "EvalSample",
    "load_dataset",
    "save_dataset",
    "build_dataset",
    "auroc",
    "ece",
    "brier",
    "tpr_at_fpr",
    "confusion_at_threshold",
    "per_tier_breakdown",
    "summarize",
    "EvalReport",
    "run_eval",
    "AblationReport",
    "run_ablation",
    "verdict_for",
]
