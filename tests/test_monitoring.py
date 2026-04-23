"""Drift monitor + meta-detector tests."""
import pytest

from beet.contracts import LayerResult
from beet.monitoring.drift import DriftMonitor, _kl_symmetric
from beet.monitoring.meta_detector import MetaDetector


def _layer(p, det_id="preamble", det="AMBER"):
    return LayerResult(
        layer_id=det_id, domain="universal", raw_score=p, p_llm=p,
        confidence=0.8, signals={}, determination=det,
        attacker_tiers=["A0"], compute_cost="cheap", min_text_length=0,
    )


def test_kl_symmetric_zero_for_identical():
    p = [1.0, 2.0, 3.0, 4.0]
    assert _kl_symmetric(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_symmetric_positive_for_different():
    assert _kl_symmetric([10, 0, 0, 0], [0, 0, 0, 10]) > 1.0


def test_drift_monitor_fires_on_population_shift(tmp_path):
    import time as _time
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 10}})
    alerts = []
    for _ in range(10):
        alerts.extend(mon.record(0.95, "RED", {}))
    assert any("POPULATION_DRIFT" in a for a in alerts)
    # Alert file is written on a daemon thread; poll briefly.
    path = tmp_path / "alerts.jsonl"
    for _ in range(50):  # up to ~500ms
        if path.exists() and path.stat().st_size > 0:
            break
        _time.sleep(0.01)
    assert path.exists()


def test_drift_monitor_no_alerts_on_balanced(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 20}})
    alerts = []
    for i in range(20):
        alerts.extend(mon.record(0.35 + (0.02 * (i % 5)), "YELLOW", {}))
    assert not any("POPULATION_DRIFT" in a for a in alerts)


def test_drift_monitor_feature_drift(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 10, "kl_threshold": 0.05}})
    baseline = [{"x": 0.1} for _ in range(50)]
    mon.set_baseline(baseline)
    for _ in range(10):
        mon.record(0.5, "AMBER", {"x": 0.9})
    alerts = mon.check_drift()  # window may have flushed; call explicitly if needed
    # After flush, _observations is empty — re-populate and check:
    for _ in range(10):
        mon._observations.append({"p_llm": 0.5, "determination": "AMBER",
                                  "features": {"x": 0.9}, "confirmed": None,
                                  "timestamp": "2026-01-01T00:00:00"})
    alerts = mon.check_drift()
    assert any("FEATURE_DRIFT" in a for a in alerts)


def test_drift_monitor_calibration_drift(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 100, "ece_threshold": 0.10}})
    # Predictions 0.9 with true label 0 → miscalibrated
    for _ in range(30):
        mon._observations.append({"p_llm": 0.9, "determination": "RED",
                                  "features": {}, "confirmed": 0,
                                  "timestamp": "2026-01-01T00:00:00"})
    alerts = mon.check_drift()
    assert any("CALIBRATION_DRIFT" in a for a in alerts)


def test_meta_detector_stable_verdict():
    meta = MetaDetector(window_size=50)
    meta.set_baseline({"preamble": 0.5})
    for _ in range(30):
        meta.record(_layer(0.55))
    health = meta.health()
    assert health["preamble"].verdict == "stable"


def test_meta_detector_flags_drift():
    meta = MetaDetector(window_size=50, mean_shift_threshold=0.10)
    meta.set_baseline({"preamble": 0.5})
    for _ in range(30):
        meta.record(_layer(0.85))
    health = meta.health()
    assert health["preamble"].verdict == "drifting"


def test_meta_detector_flags_degradation():
    meta = MetaDetector(window_size=100, min_accuracy=0.80)
    # High-confidence LLM prediction but all confirmed human
    for _ in range(30):
        meta.record(_layer(0.9), confirmed_label=0)
    health = meta.health()
    assert health["preamble"].verdict == "degrading"
    assert health["preamble"].accuracy_vs_confirmed is not None


def test_meta_detector_ignores_skip():
    meta = MetaDetector()
    meta.record(_layer(0.5, det="SKIP"))
    assert meta.health() == {}
