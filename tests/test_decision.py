import pytest
from beet.decision import DecisionEngine
from beet.contracts import FusionResult, LayerResult

@pytest.fixture
def engine(minimal_config):
    return DecisionEngine(config=minimal_config)

def test_high_p_llm_gives_red(engine):
    fusion = FusionResult(p_llm=0.82, confidence_interval=(0.70, 0.91), prediction_set=["RED"], feature_contributions={}, top_contributors=[])
    assert engine.decide(fusion, layer_results=[]).label == "RED"

def test_medium_p_llm_gives_amber(engine):
    fusion = FusionResult(p_llm=0.58, confidence_interval=(0.44, 0.70), prediction_set=["AMBER"], feature_contributions={}, top_contributors=[])
    assert engine.decide(fusion, layer_results=[]).label == "AMBER"

def test_low_p_llm_gives_green(engine):
    fusion = FusionResult(p_llm=0.08, confidence_interval=(0.03, 0.16), prediction_set=["GREEN"], feature_contributions={}, top_contributors=[])
    assert engine.decide(fusion, layer_results=[]).label == "GREEN"

def test_preamble_critical_override(engine):
    critical_result = LayerResult(layer_id="preamble", domain="universal", raw_score=1.0, p_llm=0.97, confidence=0.95,
        signals={"severity": "CRITICAL", "matched_patterns": ["assistant_ack"]}, determination="RED",
        attacker_tiers=["A0"], compute_cost="trivial", min_text_length=10)
    fusion = FusionResult(p_llm=0.40, confidence_interval=(0.30, 0.55), prediction_set=["YELLOW", "AMBER"], feature_contributions={}, top_contributors=[])
    det = engine.decide(fusion, layer_results=[critical_result])
    assert det.label == "RED"
    assert det.override_applied is True


def test_mixed_boundary_high_probability_triggers_mixed_label(engine):
    mixed_result = LayerResult(
        layer_id="mixed_boundary", domain="prose", raw_score=0.7, p_llm=0.55, confidence=0.6,
        signals={
            "mixed_probability": 0.72,
            "n_boundaries": 2,
            "segment_determinations": ["AMBER", "GREEN"],
            "boundary_positions": [0.35, 0.72],
        },
        determination="AMBER",
        attacker_tiers=["A4"], compute_cost="moderate", min_text_length=100,
    )
    fusion = FusionResult(p_llm=0.55, confidence_interval=(0.40, 0.70),
                          prediction_set=["AMBER", "YELLOW"], feature_contributions={}, top_contributors=[])
    det = engine.decide(fusion, layer_results=[mixed_result])
    assert det.label == "MIXED"
    assert det.override_applied is True
    assert det.mixed_report is not None
    assert det.mixed_report["mixed_probability"] == 0.72


def test_mixed_boundary_low_probability_does_not_trigger_mixed(engine):
    mixed_result = LayerResult(
        layer_id="mixed_boundary", domain="prose", raw_score=0.2, p_llm=0.3, confidence=0.5,
        signals={"mixed_probability": 0.1, "n_boundaries": 0, "segment_determinations": ["GREEN"]},
        determination="GREEN",
        attacker_tiers=["A4"], compute_cost="moderate", min_text_length=100,
    )
    fusion = FusionResult(p_llm=0.3, confidence_interval=(0.2, 0.45),
                          prediction_set=["YELLOW"], feature_contributions={}, top_contributors=[])
    det = engine.decide(fusion, layer_results=[mixed_result])
    assert det.label != "MIXED"
