"""FastAPI REST endpoints tests.

Skips gracefully when FastAPI isn't installed; doesn't require uvicorn at
test time since TestClient drives the app directly.
"""
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient  # noqa: E402

from beet.api import create_app  # noqa: E402


@pytest.fixture
def client():
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    app = create_app(config_path)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "detectors" in data


def test_config_is_redacted(client):
    r = client.get("/config")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_analyze_returns_determination(client):
    r = client.post("/analyze", json={"text": "Certainly! Here is a comprehensive overview."})
    assert r.status_code == 200
    data = r.json()
    assert "determination" in data
    assert "p_llm" in data
    assert 0.0 <= data["p_llm"] <= 1.0


def test_analyze_rejects_empty(client):
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 422


def test_batch_returns_per_item_results(client):
    r = client.post("/batch", json={
        "items": [
            {"id": "a", "text": "first submission text here for analysis."},
            {"id": "b", "text": "second submission text, different content."},
        ],
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    ids = {x["submission_id"] for x in data["results"]}
    assert ids == {"a", "b"}


def test_batch_rejects_empty(client):
    r = client.post("/batch", json={"items": []})
    assert r.status_code == 400
