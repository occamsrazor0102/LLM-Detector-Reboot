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
    for key in ["determination", "p_llm", "confidence_interval", "detectors_run"]:
        assert key in report
