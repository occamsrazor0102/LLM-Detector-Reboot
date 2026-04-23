"""Sidecar JSON-RPC tests: exercise Sidecar.handle() directly."""
from pathlib import Path

import pytest

from beet.config import load_config
from beet.history import HistoryStore
from beet.monitoring.drift import DriftMonitor
from beet.pipeline import BeetPipeline
from beet.runtime import RuntimeContext
from beet.sidecar import Sidecar, SidecarError


@pytest.fixture
def sidecar(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(BeetPipeline(cfg), "screening", cfg)
    history = HistoryStore(tmp_path / "h.sqlite3")
    drift = DriftMonitor(tmp_path / "drift", cfg)
    return Sidecar(
        ctx=ctx,
        feedback_path=tmp_path / "fb.jsonl",
        history=history,
        drift=drift,
    ), history


def test_analyze_writes_to_history(sidecar):
    sc, hist = sidecar
    res = sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    assert "determination" in res
    assert hist.list()["total"] == 1


def test_analyze_batch_writes_all_with_batch_id(sidecar):
    sc, hist = sidecar
    res = sc.handle("analyze_batch", {"items": [
        {"id": "a", "text": "first text sample"},
        {"id": "b", "text": "second text sample"},
    ]})
    assert len(res["results"]) == 2
    assert res["batch_id"].startswith("batch_")
    assert hist.list(batch_id=res["batch_id"])["total"] == 2


def test_history_list_round_trip(sidecar):
    sc, _ = sidecar
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    res = sc.handle("history_list", {"limit": 10})
    assert res["total"] == 1


def test_history_get_returns_full_payload(sidecar):
    sc, _ = sidecar
    r = sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    got = sc.handle("history_get", {"submission_id": r["submission_id"]})
    assert got["report"]["determination"] == r["determination"]


def test_history_get_missing_raises(sidecar):
    sc, _ = sidecar
    with pytest.raises(SidecarError) as ex:
        sc.handle("history_get", {"submission_id": "does-not-exist"})
    assert ex.value.code == "ERR_NOT_FOUND"


def test_history_delete(sidecar):
    sc, hist = sidecar
    r = sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sc.handle("history_delete", {"submission_id": r["submission_id"]})
    assert hist.get(r["submission_id"]) is None


def test_history_export_json(sidecar):
    sc, _ = sidecar
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    got = sc.handle("history_export", {"format": "json"})
    assert got["mime"] == "application/json"
    assert got["filename"].endswith(".json")


def test_health_reports_history_flag_and_profile(sidecar):
    sc, _ = sidecar
    h = sc.handle("health", {})
    assert h["history_enabled"] is True
    assert h["profile"] == "screening"
    assert h["calibration_status"] in ("heuristic", "fusion-only", "calibrated")


def test_feedback_also_writes_history_feedback(sidecar):
    sc, hist = sidecar
    r = sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sc.handle("feedback", {
        "text": "Certainly! Here is a comprehensive overview.",
        "confirmed_label": 1,
        "submission_id": r["submission_id"],
    })
    got = hist.get(r["submission_id"])
    assert len(got["feedback"]) == 1
    assert got["feedback"][0]["confirmed_label"] == 1


def test_list_profiles_returns_current(sidecar):
    sc, _ = sidecar
    res = sc.handle("list_profiles", {})
    assert res["current"] == "screening"
    names = {p["name"] for p in res["profiles"]}
    assert {"default", "screening", "strict"} <= names


def test_get_config_curated_shape(sidecar):
    sc, _ = sidecar
    got = sc.handle("get_config", {})
    assert got["profile"] == "screening"
    assert "red" in got["thresholds"] and "amber" in got["thresholds"]
    assert isinstance(got["detectors"], list) and got["detectors"]
    for d in got["detectors"]:
        assert {"id", "enabled", "weight"} <= set(d.keys())


def test_switch_profile_changes_active_profile(sidecar):
    sc, _ = sidecar
    res = sc.handle("switch_profile", {"name": "strict"})
    assert res["ok"] is True and res["profile"] == "strict"
    # subsequent health should reflect the swap
    h = sc.handle("health", {})
    assert h["profile"] == "strict"


def test_switch_profile_bad_name_raises(sidecar):
    sc, _ = sidecar
    with pytest.raises(SidecarError) as ex:
        sc.handle("switch_profile", {"name": "no-such-xyz"})
    assert ex.value.code == "ERR_BAD_PROFILE"


def test_monitoring_summary_and_timeline(sidecar):
    sc, _ = sidecar
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sc.handle("analyze", {"text": "quick human note about nothing in particular"})
    summary = sc.handle("monitoring_summary", {})
    assert summary["total"] == 2
    assert isinstance(summary["by_determination"], dict)
    tl = sc.handle("monitoring_timeline", {"limit": 10})
    assert len(tl["items"]) == 2


def test_monitoring_detectors(sidecar):
    sc, _ = sidecar
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    res = sc.handle("monitoring_detectors", {"limit": 100})
    assert isinstance(res["detectors"], list) and res["detectors"]
    d = res["detectors"][0]
    assert {"id", "n", "mean_p_llm", "mean_confidence", "determination_hist"} <= set(d.keys())


def test_monitoring_cascade(sidecar):
    sc, _ = sidecar
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    res = sc.handle("monitoring_cascade", {})
    assert res["n_samples"] >= 1
    assert set(res["phase_counts"].keys()) == {1, 2, 3, 4}
    assert "p_llm_histogram" in res and len(res["p_llm_histogram"]) == 10


def test_run_eval_happy_path(sidecar):
    sc, _ = sidecar
    items = [
        {"id": "a", "text": "Certainly! Here is a comprehensive overview.", "label": 1, "tier": "A0"},
        {"id": "b", "text": "quick human scratch note about nothing", "label": 0, "tier": "A0"},
    ]
    res = sc.handle("run_eval", {"items": items})
    assert res["n_samples"] == 2
    assert "metrics" in res and "confusion" in res
    assert "duration_ms" in res
    assert {p["id"] for p in res["predictions"]} == {"a", "b"}


def test_run_eval_rejects_missing_field(sidecar):
    sc, _ = sidecar
    with pytest.raises(SidecarError) as ex:
        sc.handle("run_eval", {"items": [{"id": "a", "text": "hi"}]})  # no label
    assert ex.value.code == "ERR_BAD_PARAMS"


def test_run_eval_cap_enforced(sidecar):
    sc, _ = sidecar
    items = [
        {"id": f"x{i}", "text": "sample text here", "label": i % 2, "tier": "A0"}
        for i in range(15)
    ]
    with pytest.raises(SidecarError) as ex:
        sc.handle("run_eval", {"items": items, "max_samples": 10})
    assert ex.value.code == "ERR_TOO_LARGE"


def test_disabled_history_raises(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(BeetPipeline(cfg), "screening", cfg)
    sc = Sidecar(ctx=ctx, feedback_path=tmp_path / "fb.jsonl", history=None)
    with pytest.raises(SidecarError) as ex:
        sc.handle("history_list", {})
    assert ex.value.code == "ERR_DISABLED"
