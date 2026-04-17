"""Pure metric functions for evaluation. No I/O, no pipeline coupling."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _nan_if_degenerate(y_true: Sequence[int]) -> bool:
    if len(y_true) == 0:
        return True
    s = set(y_true)
    return len(s) < 2


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if _nan_if_degenerate(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def ece(y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        if not mask.any():
            continue
        bin_conf = y_score[mask].mean()
        bin_acc = y_true[mask].mean()
        total += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(total)


def brier(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


def tpr_at_fpr(y_true: Sequence[int], y_score: Sequence[float], target_fpr: float = 0.01) -> float:
    if _nan_if_degenerate(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # largest tpr whose fpr <= target_fpr
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def confusion_at_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> dict:
    tp = fp = tn = fn = 0
    for yt, ys in zip(y_true, y_score):
        pred = 1 if ys >= threshold else 0
        if yt == 1 and pred == 1:
            tp += 1
        elif yt == 0 and pred == 1:
            fp += 1
        elif yt == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def per_tier_breakdown(
    samples: Sequence,
    predictions: Sequence,
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
) -> dict[str, float]:
    """Group predictions by sample.tier and apply metric_fn per tier.

    `samples` elements may be EvalSample instances or dicts; each exposes
    `.tier`/`tier`, `.id`/`id`, `.label`/`label`. Predictions expose `id`
    and `p_llm`.
    """
    def get(obj, key):
        if hasattr(obj, key):
            return getattr(obj, key)
        return obj[key]

    pred_by_id = {get(p, "id"): get(p, "p_llm") for p in predictions}
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for s in samples:
        tier = get(s, "tier")
        if tier is None:
            continue
        sid = get(s, "id")
        if sid not in pred_by_id:
            continue
        groups[tier].append((int(get(s, "label")), float(pred_by_id[sid])))

    out = {}
    for tier, pairs in groups.items():
        labels = [a for a, _ in pairs]
        scores = [b for _, b in pairs]
        out[tier] = metric_fn(labels, scores)
    return out


def summarize(y_true: Sequence[int], y_score: Sequence[float]) -> dict:
    return {
        "auroc": auroc(y_true, y_score),
        "ece": ece(y_true, y_score),
        "brier": brier(y_true, y_score),
        "tpr_at_fpr_01": tpr_at_fpr(y_true, y_score, target_fpr=0.01),
    }
