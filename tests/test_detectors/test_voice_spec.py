import pytest
from beet.detectors.voice_spec import VoiceSpecDetector

@pytest.fixture
def detector():
    return VoiceSpecDetector()

@pytest.fixture
def config():
    return {}

def test_dissonance_mode_high_voice_high_spec(detector, config):
    text = (
        "okay so basically you're gonna need to ensure that the protocol adheres "
        "to all regulatory requirements. you must include a comprehensive statistical "
        "analysis plan. the deliverables should encompass the full range of endpoints "
        "and you'll wanna make sure each one is properly validated. ya know what i mean?"
    )
    result = detector.analyze(text, config)
    assert result.signals["mode"] == "dissonance"
    assert result.p_llm > 0.55

def test_sterile_mode_zero_voice_high_spec(detector, config):
    text = (
        "The protocol must include eligibility criteria, dosing schedule, and sample "
        "collection timepoints. The statistical analysis plan shall utilize non-compartmental "
        "analysis. All deliverables must comply with ICH E6(R2) guidelines. "
        "The output format requires structured sections for each endpoint."
    )
    result = detector.analyze(text, config)
    assert result.signals["mode"] == "sterile"
    assert result.p_llm > 0.45

def test_genuine_casual_scores_low(detector, config):
    text = (
        "okay so i've been working on this for like three weeks and honestly "
        "it's a mess. the reagent concentrations keep drifting — i think it's "
        "temperature. gonna try the cold room next week and see."
    )
    result = detector.analyze(text, config)
    assert result.p_llm < 0.35

def test_result_fields(detector, config):
    result = detector.analyze("The protocol must ensure compliance.", config)
    assert result.layer_id == "voice_spec"
    assert "voice_score" in result.signals
    assert "spec_score" in result.signals
    assert "mode" in result.signals


def test_voice_spec_emits_separate_voice_and_spec_spans(detector, config):
    text = "okay so basically you must ensure the protocol complies with the standard."
    result = detector.analyze(text, config)
    kinds = {s["kind"] for s in result.spans}
    assert "voice_informal" in kinds
    assert "spec" in kinds
    for s in result.spans:
        assert text[s["start"]:s["end"]]
