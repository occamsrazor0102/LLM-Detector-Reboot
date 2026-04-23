import json
import pytest
from beet.report import build_json_report, build_text_report
from beet.contracts import Determination

@pytest.fixture
def sample_determination():
    return Determination(label="AMBER", p_llm=0.62, confidence_interval=(0.48, 0.73),
        prediction_set=["YELLOW", "AMBER"], reason="High binoculars ratio",
        top_features=[("binoculars_ratio", 1.4)], override_applied=False,
        detectors_run=["preamble", "fingerprint_vocab"], cascade_phases=[1], mixed_report=None)

def test_json_report_is_serializable(sample_determination):
    report = build_json_report(sample_determination, submission_id="sub_001")
    json_str = json.dumps(report)
    assert "AMBER" in json_str and "sub_001" in json_str

def test_text_report_is_string(sample_determination):
    report = build_text_report(sample_determination)
    assert "AMBER" in report and "0.62" in report

def test_json_report_has_required_keys(sample_determination):
    report = build_json_report(sample_determination)
    for key in ["determination", "p_llm", "confidence_interval", "detectors_run",
                "layer_results", "feature_contributions"]:
        assert key in report

def test_json_report_serializes_layer_results():
    from beet.contracts import LayerResult
    lr = LayerResult(layer_id="preamble", domain="universal", raw_score=0.8,
        p_llm=0.9, confidence=0.7, signals={"severity": "HIGH"},
        determination="RED", attacker_tiers=["A0"], compute_cost="trivial",
        min_text_length=0)
    det = Determination(label="RED", p_llm=0.9, confidence_interval=(0.8, 0.95),
        prediction_set=["RED"], reason="t", top_features=[("preamble", 1.0)],
        override_applied=True, detectors_run=["preamble"], cascade_phases=[1],
        mixed_report=None, layer_results=[lr],
        feature_contributions={"preamble": 1.0, "fingerprint_vocab": 0.2})
    report = build_json_report(det)
    assert len(report["layer_results"]) == 1
    layer = report["layer_results"][0]
    for key in ["layer_id", "domain", "raw_score", "p_llm", "confidence",
                "determination", "signals", "compute_cost"]:
        assert key in layer
    assert layer["layer_id"] == "preamble"
    assert layer["signals"] == {"severity": "HIGH"}
    assert report["feature_contributions"]["preamble"] == 1.0
    assert report["feature_contributions"]["fingerprint_vocab"] == 0.2
