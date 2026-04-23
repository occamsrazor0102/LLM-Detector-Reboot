"""HTTP API tests: live socket against the stdlib GUI server handler."""
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from beet.gui.server import _make_handler
from beet.history import HistoryStore
from beet.monitoring.meta_detector import MetaDetector
from beet.pipeline import BeetPipeline


@pytest.fixture
def server(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = BeetPipeline.from_config_file(config_path)
    meta = MetaDetector()
    feedback_path = tmp_path / "feedback.jsonl"
    history = HistoryStore(tmp_path / "history.sqlite3")
    handler = _make_handler(pipeline, meta, feedback_path, history, profile="screening")
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
