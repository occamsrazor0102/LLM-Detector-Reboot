"""Drift monitor wiring: end-to-end flow through Sidecar."""
from pathlib import Path

import pytest

from beet.config import load_config
from beet.history import HistoryStore
from beet.monitoring.drift import DriftMonitor
from beet.pipeline import BeetPipeline
from beet.runtime import RuntimeContext
from beet.sidecar import (
    Sidecar,
    SidecarError,
    drift_baseline_from_history,
    drift_from_config,
)


@pytest.fixture
def env(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(BeetPipeline(cfg), "screening", cfg)
    history = HistoryStore(tmp_path / "h.sqlite3")
    drift = DriftMonitor(tmp_path / "drift", cfg)
    sc = Sidecar(
        ctx=ctx,
        feedback_path=tmp_path / "fb.jsonl",
        history=history,
        drift=drift,
    )
    return sc, history, drift


def _seed(history, *, p, det, sid):
    fc = {"fingerprint_vocab": p, "preamble": p - 0.1, "voice_spec": 0.3}
    report = {
        "submission_id": sid,
        "timestamp": "2026-04-23T00:00:00Z",
        "determination": det,
        "p_llm": p,
        "confidence_interval": [p - 0.05, p + 0.05],
        "prediction_set": [det],
        "reason": "seeded",
        "top_features": [],
        "detectors_run": list(fc.keys()),
        "cascade_phases": [1],
        "mixed_report": None,
        "layer_results": [],
        "feature_contributions": fc,
        "override_applied": False,
    }
    history.record(report, source="analyze", text="seed")


def test_drift_from_config_respects_disabled(tmp_path):
    assert drift_from_config({"gui": {"drift": {"enabled": False}}}) is None


def test_drift_from_config_constructs_when_enabled(tmp_path):
    cfg = {"gui": {"drift": {"enabled": True, "store_path": str(tmp_path / "d")}}}
    d = drift_from_config(cfg)
    assert isinstance(d, DriftMonitor)


def test_analyze_records_drift_observation(env):
    sc, _, drift = env
    sc.handle("analyze", {"text": "Certainly! Here is a comprehensive overview."})
    assert len(drift._observations) == 1
    o = drift._observations[0]
    assert "features" in o and "p_llm" in o


def test_set_baseline_from_history_pulls_feature_vectors(env):
    sc, history, drift = env
    for i in range(6):
        _seed(history, p=0.3 + 0.05 * i, det="AMBER", sid=f"s{i}")
    n = drift_baseline_from_history(drift, history, limit=10)
    assert n == 6
    assert drift._baseline_hist  # non-empty
    assert "fingerprint_vocab" in drift._baseline_hist


def test_monitoring_drift_rpc_reports_baseline_and_observations(env):
    sc, history, drift = env
    for i in range(5):
        _seed(history, p=0.4, det="AMBER", sid=f"seed{i}")
    drift_baseline_from_history(drift, history, limit=10)
    sc.handle("analyze", {"text": "short analyze text here"})
    res = sc.handle("monitoring_drift", {})
    assert res["has_baseline"] is True
    assert res["n_observations"] >= 1
    assert isinstance(res["alerts"], list)


def test_set_baseline_rpc_happy_path(env):
    sc, history, _ = env
    for i in range(4):
        _seed(history, p=0.5, det="AMBER", sid=f"b{i}")
    res = sc.handle("monitoring_set_baseline", {"limit": 10})
    assert res["ok"] is True
    assert res["n_samples"] == 4
    assert res["baseline_features"]


def test_population_drift_alert_on_strong_skew(env):
    sc, _, drift = env
    # force window to flush after 30 obs
    drift._window = 30
    for _ in range(30):
        drift.record(p_llm=0.98, determination="RED", feature_vector={"preamble": 0.95})
    # alerts were flushed back; also check _check_drift on a fresh window
    drift._observations = [
        {"p_llm": 0.95, "determination": "RED", "features": {"preamble": 0.95},
         "confirmed": None, "timestamp": "2026-04-23T00:00:00"}
        for _ in range(30)
    ]
    alerts = drift.check_drift()
    assert any("POPULATION_DRIFT" in a for a in alerts)


def test_disabled_drift_raises(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(BeetPipeline(cfg), "screening", cfg)
    sc = Sidecar(ctx=ctx, feedback_path=tmp_path / "fb.jsonl", history=None, drift=None)
    with pytest.raises(SidecarError) as ex:
        sc.handle("monitoring_drift", {})
    assert ex.value.code == "ERR_DISABLED"
