import pytest
from beet.detectors.preamble import PreambleDetector
from tests.fixtures.llm_samples import A0_PREAMBLE
from tests.fixtures.human_samples import CASUAL_SHORT, FORMAL_SOP

@pytest.fixture
def detector():
    return PreambleDetector()

@pytest.fixture
def config():
    return {}

def test_critical_preamble_gives_high_p_llm(detector, config):
    result = detector.analyze(A0_PREAMBLE, config)
    assert result.p_llm > 0.90
    assert result.determination == "RED"
    assert result.signals["severity"] == "CRITICAL"

def test_casual_human_gives_low_p_llm(detector, config):
    result = detector.analyze(CASUAL_SHORT, config)
    assert result.p_llm < 0.15
    assert result.determination in ("GREEN", "SKIP")

def test_formal_sop_no_preamble(detector, config):
    result = detector.analyze(FORMAL_SOP, config)
    assert result.determination in ("GREEN", "SKIP")
    assert result.signals["severity"] in ("NONE", "LOW")

def test_result_has_required_fields(detector, config):
    result = detector.analyze(A0_PREAMBLE, config)
    assert result.layer_id == "preamble"
    assert result.domain == "universal"
    assert result.compute_cost == "trivial"
    assert "A0" in result.attacker_tiers

def test_conversational_opener_is_high_severity(detector, config):
    text = "Sure! Here's a comprehensive guide to pharmacokinetics that you can use:\n\nThe study..."
    result = detector.analyze(text, config)
    assert result.p_llm > 0.70
    assert result.signals["severity"] in ("CRITICAL", "HIGH")
