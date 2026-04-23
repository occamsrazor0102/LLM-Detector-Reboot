import json
from pathlib import Path

import pytest

from beet.history import HistoryStore, mint_submission_id


def _report(det="AMBER", p=0.62, sid=""):
    return {
        "submission_id": sid,
        "timestamp": "2026-04-22T12:00:00Z",
        "determination": det,
        "p_llm": p,
        "confidence_interval": [p - 0.1, p + 0.1],
        "prediction_set": [det],
        "reason": "test",
        "top_features": [("feat_a", 0.4)],
        "detectors_run": ["preamble", "fingerprint_vocab"],
        "cascade_phases": [1],
        "mixed_report": None,
        "layer_results": [],
        "feature_contributions": {},
        "override_applied": False,
    }


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history.sqlite3")


def test_mint_id_is_unique_and_time_ordered():
    a = mint_submission_id()
    b = mint_submission_id()
    assert a != b
    assert a < b or a.split("_")[1] == b.split("_")[1]


def test_record_and_get_roundtrip(store):
    sid = store.record(_report(), source="analyze", text="hello world")
    got = store.get(sid)
    assert got is not None
    assert got["submission_id"] == sid
    assert got["text"] == "hello world"
    assert got["report"]["determination"] == "AMBER"
    assert got["feedback"] == []


def test_record_honors_provided_submission_id(store):
    sid = store.record(_report(sid="custom-123"), source="analyze", text="x")
    assert sid == "custom-123"
    got = store.get("custom-123")
    assert got and got["submission_id"] == "custom-123"


def test_record_upserts_on_same_id(store):
    store.record(_report(det="AMBER", sid="s1"), source="analyze", text="first")
    store.record(_report(det="RED", sid="s1"), source="analyze", text="second")
    got = store.get("s1")
    assert got["report"]["determination"] == "RED"
    assert got["text"] == "second"
    res = store.list()
    assert res["total"] == 1


def test_retain_text_false_drops_text(tmp_path):
    store = HistoryStore(tmp_path / "h.sqlite3", retain_text=False)
    sid = store.record(_report(), source="analyze", text="sensitive content")
    got = store.get(sid)
    assert got["text"] is None
    assert got["text_hash"] != ""  # hash still stored


def test_list_filters_by_determination(store):
    store.record(_report(det="RED", sid="a"), source="analyze", text="a")
    store.record(_report(det="AMBER", sid="b"), source="analyze", text="b")
    store.record(_report(det="GREEN", sid="c"), source="analyze", text="c")
    res = store.list(determination=["RED", "AMBER"])
    assert res["total"] == 2
    kinds = {i["determination"] for i in res["items"]}
    assert kinds == {"RED", "AMBER"}


def test_list_filters_by_batch(store):
    store.record(_report(sid="a"), source="batch", text="a", batch_id="B1")
    store.record(_report(sid="b"), source="batch", text="b", batch_id="B1")
    store.record(_report(sid="c"), source="analyze", text="c")
    res = store.list(batch_id="B1")
    assert res["total"] == 2


def test_list_pagination_orders_by_recent_first(store):
    for i in range(5):
        r = _report(sid=f"s{i}")
        r["timestamp"] = f"2026-04-22T12:0{i}:00Z"
        store.record(r, source="analyze", text=f"t{i}")
    res = store.list(limit=2, offset=0)
    assert res["total"] == 5
    assert [i["submission_id"] for i in res["items"]] == ["s4", "s3"]
    res2 = store.list(limit=2, offset=2)
    assert [i["submission_id"] for i in res2["items"]] == ["s2", "s1"]


def test_list_search_matches_submission_id_or_text(store):
    store.record(_report(sid="alpha"), source="analyze", text="needle in haystack")
    store.record(_report(sid="beta"), source="analyze", text="no match")
    assert store.list(search="needle")["total"] == 1
    assert store.list(search="beta")["total"] == 1
    assert store.list(search="zzz")["total"] == 0


def test_record_feedback_marks_has_feedback(store):
    sid = store.record(_report(), source="analyze", text="x")
    store.record_feedback(sid, confirmed_label=1, reviewer_notes="clearly llm")
    item = store.list()["items"][0]
    assert item["has_feedback"] is True
    got = store.get(sid)
    assert got["feedback"][0]["confirmed_label"] == 1
    assert got["feedback"][0]["reviewer_notes"] == "clearly llm"


def test_delete_removes_submission_and_feedback(store):
    sid = store.record(_report(), source="analyze", text="x")
    store.record_feedback(sid, 0)
    assert store.delete(sid) is True
    assert store.get(sid) is None
    assert store.list()["total"] == 0


def test_export_json_shape(store):
    store.record(_report(det="AMBER", sid="a"), source="analyze", text="x")
    content, mime, fname = store.export(fmt="json")
    assert mime == "application/json"
    assert fname.endswith(".json")
    data = json.loads(content)
    assert len(data) == 1 and data[0]["submission_id"] == "a"


def test_export_csv_shape(store):
    store.record(_report(det="AMBER", sid="a"), source="analyze", text="x")
    content, mime, fname = store.export(fmt="csv")
    assert mime == "text/csv"
    assert fname.endswith(".csv")
    lines = content.strip().splitlines()
    assert lines[0].startswith("submission_id,")
    assert len(lines) == 2


def test_export_respects_filter(store):
    store.record(_report(det="RED", sid="a"), source="analyze", text="x")
    store.record(_report(det="GREEN", sid="b"), source="analyze", text="y")
    content, _, _ = store.export(fmt="json", determination=["RED"])
    data = json.loads(content)
    assert len(data) == 1 and data[0]["submission_id"] == "a"
