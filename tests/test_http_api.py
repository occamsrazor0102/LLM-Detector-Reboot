"""HTTP API tests: live socket against the stdlib GUI server handler."""
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from beet.config import load_config
from beet.gui.server import _make_handler
from beet.history import HistoryStore
from beet.monitoring.drift import DriftMonitor
from beet.monitoring.meta_detector import MetaDetector
from beet.pipeline import BeetPipeline
from beet.runtime import RuntimeContext


@pytest.fixture
def server(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(BeetPipeline(cfg), "screening", cfg)
    meta = MetaDetector()
    feedback_path = tmp_path / "feedback.jsonl"
    history = HistoryStore(tmp_path / "history.sqlite3")
    drift = DriftMonitor(tmp_path / "drift", cfg)
    handler = _make_handler(
        meta=meta, feedback_path=feedback_path,
        history=history, ctx=ctx, drift=drift,
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", history
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as resp:
        return resp.status, resp.read(), dict(resp.headers)


def _get_json(url, path):
    status, data, _ = _get(url, path)
    return status, json.loads(data)


def _post(url, path, body):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_reports_history_enabled(server):
    url, _ = server
    status, body = _get_json(url, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["history_enabled"] is True
    assert body["profile"] == "screening"
    # Without a trained fusion model the system runs in heuristic mode.
    # The UI relies on this field to show the "untrained" banner.
    assert body["calibration_status"] in ("heuristic", "fusion-only", "calibrated")


def test_index_served(server):
    url, _ = server
    status, data, _ = _get(url, "/")
    assert status == 200
    text = data.decode("utf-8").lower()
    assert "<html" in text and "beet" in text


def test_analyze_writes_history(server):
    url, hist = server
    status, body = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    assert status == 200
    assert "determination" in body
    listed = hist.list()
    assert listed["total"] == 1
    assert listed["items"][0]["submission_id"] == body["submission_id"]


def test_analyze_rejects_empty(server):
    url, _ = server
    status, _ = _post(url, "/analyze", {"text": ""})
    assert status == 400


def test_batch_writes_history_with_batch_id(server):
    url, hist = server
    status, body = _post(url, "/batch", {"items": [
        {"id": "a", "text": "first submission text"},
        {"id": "b", "text": "second submission text"},
    ]})
    assert status == 200
    assert len(body["results"]) == 2
    assert body["batch_id"]
    listed = hist.list(batch_id=body["batch_id"])
    assert listed["total"] == 2


def test_batch_cap_enforced(server):
    url, _ = server
    status, body = _post(url, "/batch", {"items": [{"id": f"i{i}", "text": "x"} for i in range(501)]})
    assert status == 400
    assert "cap" in body["error"].lower()


def test_history_list_filter_by_determination(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    _post(url, "/analyze", {"text": "quick human scratch note about nothing"})
    status, all_items = _post(url, "/history/list", {})
    assert status == 200
    assert all_items["total"] == 2
    dets = {i["determination"] for i in all_items["items"]}
    pick = next(iter(dets))
    status, filtered = _post(url, "/history/list", {"determination": [pick]})
    assert status == 200
    assert all(i["determination"] == pick for i in filtered["items"])


def test_history_get_returns_full_report(server):
    url, _ = server
    _, analyzed = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sid = analyzed["submission_id"]
    status, body = _post(url, "/history/get", {"submission_id": sid})
    assert status == 200
    assert body["report"]["determination"] == analyzed["determination"]
    assert body["text"] is not None


def test_history_delete(server):
    url, hist = server
    _, analyzed = _post(url, "/analyze", {"text": "quick note"})
    sid = analyzed["submission_id"]
    status, body = _post(url, "/history/delete", {"submission_id": sid})
    assert status == 200 and body["ok"] is True
    assert hist.get(sid) is None


def test_history_export_json_download(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, data, headers = _get(url, "/history/export?format=json")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert "attachment" in headers["Content-Disposition"]
    parsed = json.loads(data)
    assert isinstance(parsed, list) and len(parsed) == 1


def test_history_export_csv_download(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, data, headers = _get(url, "/history/export?format=csv")
    assert status == 200
    assert "text/csv" in headers["Content-Type"]
    text = data.decode("utf-8")
    assert text.splitlines()[0].startswith("submission_id,")


def test_config_profiles_lists_repo_profiles(server):
    url, _ = server
    status, body = _get_json(url, "/config/profiles")
    assert status == 200
    names = {p["name"] for p in body["profiles"]}
    assert {"default", "screening", "strict"} <= names
    assert body["current"] == "screening"


def test_config_current_curated_view(server):
    url, _ = server
    status, body = _get_json(url, "/config/current")
    assert status == 200
    assert body["profile"] == "screening"
    assert "red" in body["thresholds"]
    assert isinstance(body["detectors"], list) and body["detectors"]
    # Availability introspection: each detector reports available/reason/requires
    for d in body["detectors"]:
        assert "available" in d and isinstance(d["available"], bool)
        assert "reason" in d
        assert "requires" in d and isinstance(d["requires"], list)


def test_config_switch_updates_active_profile(server):
    url, _ = server
    status, body = _post(url, "/config/switch", {"name": "strict"})
    assert status == 200 and body["ok"] is True and body["profile"] == "strict"
    # health now reports strict
    _, health = _get_json(url, "/health")
    assert health["profile"] == "strict"


def test_config_switch_bad_name_returns_400(server):
    url, _ = server
    status, body = _post(url, "/config/switch", {"name": "no-such-xyz"})
    assert status == 400
    assert body.get("code") == "ERR_BAD_PROFILE"


def test_monitoring_summary_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/summary", {})
    assert status == 200
    assert body["total"] == 1
    assert "mean_p_llm" in body


def test_monitoring_timeline_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/timeline", {"limit": 50})
    assert status == 200
    assert len(body["items"]) == 1
    assert "p_llm" in body["items"][0]


def test_monitoring_detectors_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/detectors", {"limit": 100})
    assert status == 200
    assert isinstance(body["detectors"], list)


def test_monitoring_cascade_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/cascade", {})
    assert status == 200
    assert body["n_samples"] >= 1
    assert "phase_counts" in body
    assert "p_llm_histogram" in body


def test_evaluation_run_happy_path(server):
    url, _ = server
    items = [
        {"id": "a", "text": "Certainly! Here is a comprehensive overview.", "label": 1, "tier": "A0"},
        {"id": "b", "text": "quick human scratch note about nothing", "label": 0, "tier": "A0"},
    ]
    status, body = _post(url, "/evaluation/run", {"items": items})
    assert status == 200
    assert body["n_samples"] == 2
    assert "metrics" in body
    assert "confusion" in body
    assert "duration_ms" in body


def test_evaluation_run_cap_enforced(server):
    url, _ = server
    items = [{"id": f"x{i}", "text": "t" * 5, "label": 0} for i in range(5)]
    status, body = _post(url, "/evaluation/run", {"items": items, "max_samples": 3})
    assert status == 400
    assert body.get("code") == "ERR_TOO_LARGE"


def test_monitoring_drift_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/drift", {})
    assert status == 200
    assert body["n_observations"] >= 1
    assert isinstance(body["alerts"], list)


def test_monitoring_set_baseline_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    _post(url, "/analyze", {"text": "another sample text for baseline seeding"})
    status, body = _post(url, "/monitoring/set-baseline", {"limit": 10})
    assert status == 200
    assert body["ok"] is True and body["n_samples"] == 2
    assert body["baseline_features"]


def test_feedback_records_in_history(server):
    url, hist = server
    _, analyzed = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sid = analyzed["submission_id"]
    status, _ = _post(url, "/feedback", {
        "text": "Certainly! Here is a comprehensive overview.",
        "confirmed_label": 1,
        "submission_id": sid,
        "reviewer_notes": "obvious",
    })
    assert status == 200
    got = hist.get(sid)
    assert len(got["feedback"]) == 1
    assert got["feedback"][0]["confirmed_label"] == 1
