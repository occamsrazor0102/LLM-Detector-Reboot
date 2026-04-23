import json
import math

import pytest

from beet.evaluation.runner import EvalReport, eval_report_to_dict


def _mk_report(predictions):
    return EvalReport(
        predictions=predictions,
        metrics={"auroc": 0.85, "ece": 0.07, "brier": 0.12, "tpr_at_fpr_01": 0.6},
        per_tier={"A0": {"auroc": 0.9, "ece": 0.05, "brier": 0.1, "tpr_at_fpr_01": 0.7}},
        n_samples=len(predictions),
        config_hash="abcd1234",
        failed_samples=[{"id": "x", "error": "boom"}],
        per_attack={"paraphrase": {"auroc": 0.8, "ece": 0.1, "brier": 0.15, "tpr_at_fpr_01": 0.5}},
    )


def _preds():
    return [
        {"id": "a", "label": 1, "p_llm": 0.9, "determination": "RED", "tier": "A0"},
        {"id": "b", "label": 0, "p_llm": 0.1, "determination": "GREEN", "tier": "A0"},
        {"id": "c", "label": 1, "p_llm": 0.8, "determination": "RED", "tier": "A0"},
        {"id": "d", "label": 0, "p_llm": 0.6, "determination": "AMBER", "tier": "A0"},
    ]


def test_eval_report_to_dict_has_required_keys():
    out = eval_report_to_dict(_mk_report(_preds()))
    for key in [
        "n_samples", "n_failed", "config_hash", "metrics", "per_tier",
        "per_attack", "confusion", "failed_samples", "predictions",
    ]:
        assert key in out


def test_eval_report_to_dict_is_json_serializable():
    out = eval_report_to_dict(_mk_report(_preds()))
    # Must round-trip through json without error
    serialized = json.dumps(out)
    assert "confusion" in serialized


def test_confusion_and_derived_metrics():
    # At threshold 0.5:
    #   a: p=0.9, label=1 -> predicted 1, actual 1 => tp
    #   b: p=0.1, label=0 -> predicted 0, actual 0 => tn
    #   c: p=0.8, label=1 -> predicted 1, actual 1 => tp
    #   d: p=0.6, label=0 -> predicted 1, actual 0 => fp
    out = eval_report_to_dict(_mk_report(_preds()))
    c = out["confusion"]
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (2, 1, 1, 0)
    assert c["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert c["recall"] == pytest.approx(1.0, abs=1e-3)
    assert c["accuracy"] == pytest.approx(3 / 4, abs=1e-3)


def test_nan_metrics_become_none():
    # degenerate case — all same label, auroc is nan
    rpt = EvalReport(
        predictions=[{"id": "a", "label": 1, "p_llm": 0.9, "determination": "RED", "tier": None}],
        metrics={"auroc": float("nan"), "ece": 0.0, "brier": 0.01, "tpr_at_fpr_01": float("nan")},
        per_tier={}, n_samples=1, config_hash="x",
    )
    out = eval_report_to_dict(rpt)
    assert out["metrics"]["auroc"] is None
    assert out["metrics"]["tpr_at_fpr_01"] is None
    # And it must be JSON-serializable (NaN is not)
    json.dumps(out)


def test_include_predictions_false_empties_list():
    out = eval_report_to_dict(_mk_report(_preds()), include_predictions=False)
    assert out["predictions"] == []
