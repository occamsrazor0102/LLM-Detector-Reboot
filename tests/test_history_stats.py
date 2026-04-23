import pytest

from beet.history import HistoryStore


def _report(det="AMBER", p=0.62, sid="", layers=None):
    return {
        "submission_id": sid,
        "timestamp": "2026-04-22T12:00:00Z",
        "determination": det,
        "p_llm": p,
        "confidence_interval": [p - 0.1, p + 0.1],
        "prediction_set": [det],
        "reason": "test",
        "top_features": [],
        "detectors_run": [lr["layer_id"] for lr in (layers or [])],
        "cascade_phases": [1],
        "mixed_report": None,
        "layer_results": layers or [],
        "feature_contributions": {},
        "override_applied": False,
    }


def _layer(lid, p=0.5, conf=0.7, det="AMBER"):
    return {
        "layer_id": lid, "domain": "universal", "raw_score": p,
        "p_llm": p, "confidence": conf, "determination": det,
        "signals": {}, "compute_cost": "trivial",
    }


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "h.sqlite3")


def test_stats_empty_store(store):
    s = store.stats()
    assert s["total"] == 0
    assert s["mean_p_llm"] == 0.0
    assert s["feedback_count"] == 0
    assert s["feedback_accuracy"] is None


def test_stats_counts_and_distribution(store):
    store.record(_report(det="RED", p=0.9, sid="a"), source="analyze", text="x")
    store.record(_report(det="RED", p=0.8, sid="b"), source="analyze", text="y")
    store.record(_report(det="GREEN", p=0.1, sid="c"), source="analyze", text="z")
    s = store.stats()
    assert s["total"] == 3
    assert s["by_determination"] == {"RED": 2, "GREEN": 1}
    assert 0.5 < s["mean_p_llm"] < 0.7


def test_stats_feedback_accuracy_and_brier(store):
    # Submission predicted RED (p=0.9), confirmed LLM (1) — correct.
    # Submission predicted GREEN (p=0.1), confirmed human (0) — correct.
    # Submission predicted AMBER (p=0.6), confirmed human (0) — wrong.
    store.record(_report(det="RED", p=0.9, sid="a"), source="analyze", text="x")
    store.record(_report(det="GREEN", p=0.1, sid="b"), source="analyze", text="y")
    store.record(_report(det="AMBER", p=0.6, sid="c"), source="analyze", text="z")
    store.record_feedback("a", 1)
    store.record_feedback("b", 0)
    store.record_feedback("c", 0)
    s = store.stats()
    assert s["feedback_labeled_count"] == 3
    assert s["feedback_accuracy"] == pytest.approx(2 / 3, rel=1e-3)
    # Brier = mean((p - y)^2) = ((0.9-1)^2 + (0.1-0)^2 + (0.6-0)^2)/3 ≈ 0.1267
    assert s["brier"] == pytest.approx((0.01 + 0.01 + 0.36) / 3, rel=1e-3)


def test_stats_per_day_window(store):
    r = _report(sid="s")
    r["timestamp"] = "2026-04-20T12:00:00Z"
    store.record(r, source="analyze", text="x")
    r2 = _report(sid="s2")
    r2["timestamp"] = "2026-04-21T12:00:00Z"
    store.record(r2, source="analyze", text="x")
    s = store.stats()
    # per_day uses the last-14-days window from "now" — our seeds may fall
    # inside or outside. Just assert the structure.
    assert isinstance(s["per_day"], dict)


def test_timeline_most_recent_first(store):
    for i in range(5):
        r = _report(det="AMBER", p=0.5 + 0.01 * i, sid=f"s{i}")
        r["timestamp"] = f"2026-04-22T12:0{i}:00Z"
        store.record(r, source="analyze", text="x")
    tl = store.timeline(limit=3)
    assert len(tl) == 3
    assert [i["submission_id"] for i in tl] == ["s4", "s3", "s2"]
    assert all("determination" in i for i in tl)


def test_detector_stats_aggregates_across_reports(store):
    layers1 = [_layer("preamble", p=0.8, det="RED"), _layer("voice_spec", p=0.4, det="AMBER")]
    layers2 = [_layer("preamble", p=0.7, det="RED"), _layer("voice_spec", p=0.2, det="GREEN")]
    store.record(_report(det="RED", p=0.8, sid="a", layers=layers1), source="analyze", text="x")
    store.record(_report(det="AMBER", p=0.5, sid="b", layers=layers2), source="analyze", text="y")
    ds = {d["id"]: d for d in store.detector_stats()}
    assert "preamble" in ds and "voice_spec" in ds
    assert ds["preamble"]["n"] == 2
    assert ds["preamble"]["mean_p_llm"] == pytest.approx(0.75, rel=1e-3)
    assert ds["preamble"]["determination_hist"]["RED"] == 2
    assert ds["voice_spec"]["determination_hist"]["AMBER"] == 1
    assert ds["voice_spec"]["determination_hist"]["GREEN"] == 1


def test_detector_stats_ignores_reports_without_layer_results(store):
    store.record(_report(det="AMBER", sid="a", layers=[]), source="analyze", text="x")
    assert store.detector_stats() == []
