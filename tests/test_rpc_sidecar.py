"""Sidecar JSON-RPC tests: exercise Sidecar.handle() directly."""
from pathlib import Path

import pytest

from beet.history import HistoryStore
from beet.pipeline import BeetPipeline
from beet.sidecar import Sidecar, SidecarError


@pytest.fixture
def sidecar(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = BeetPipeline.from_config_file(config_path)
    history = HistoryStore(tmp_path / "h.sqlite3")
    return Sidecar(
        pipeline,
        feedback_path=tmp_path / "fb.jsonl",
        history=history,
        profile="screening",
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


def test_disabled_history_raises(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = BeetPipeline.from_config_file(config_path)
    sc = Sidecar(pipeline, feedback_path=tmp_path / "fb.jsonl", history=None, profile="screening")
    with pytest.raises(SidecarError) as ex:
        sc.handle("history_list", {})
    assert ex.value.code == "ERR_DISABLED"
