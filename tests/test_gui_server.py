"""GUI server tests using the stdlib HTTP server against a live socket."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from beet.gui.server import _make_handler
from beet.pipeline import BeetPipeline
from http.server import HTTPServer


@pytest.fixture
def server():
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = BeetPipeline.from_config_file(config_path)
    handler = _make_handler(pipeline)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get_json(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def _post_json(url, path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_endpoint(server):
    status, body = _get_json(server, "/health")
    assert status == 200
    assert body["status"] == "ok"


def test_index_served(server):
    with urllib.request.urlopen(server + "/", timeout=5) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "<html" in body.lower()
    assert "BEET" in body


def test_analyze_endpoint(server):
    status, body = _post_json(server, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    assert status == 200
    assert "determination" in body
    assert 0.0 <= body["p_llm"] <= 1.0


def test_analyze_rejects_empty(server):
    status, body = _post_json(server, "/analyze", {"text": ""})
    assert status == 400


def test_batch_endpoint(server):
    status, body = _post_json(server, "/batch", {"items": [
        {"id": "a", "text": "first submission text"},
        {"id": "b", "text": "second submission text"},
    ]})
    assert status == 200
    assert len(body["results"]) == 2
