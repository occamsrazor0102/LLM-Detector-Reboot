# tests/test_contracts.py
from beet.contracts import LayerResult, FusionResult, Determination, RouterDecision

def test_layer_result_defaults():
    r = LayerResult(
        layer_id="test",
        domain="prose",
        raw_score=0.5,
        p_llm=0.5,
        confidence=0.8,
        signals={},
        determination="AMBER",
        attacker_tiers=["A0", "A1"],
        compute_cost="cheap",
        min_text_length=50,
    )
    assert r.layer_id == "test"
    assert r.p_llm == 0.5
    assert r.determination == "AMBER"

def test_layer_result_skip_is_valid():
    r = LayerResult(
        layer_id="skipped",
        domain="prose",
        raw_score=0.0,
        p_llm=0.5,
        confidence=0.0,
        signals={},
        determination="SKIP",
        attacker_tiers=[],
        compute_cost="trivial",
        min_text_length=0,
    )
    assert r.determination == "SKIP"

def test_fusion_result_fields():
    fr = FusionResult(
        p_llm=0.62,
        confidence_interval=(0.48, 0.73),
        prediction_set=["YELLOW", "AMBER"],
        feature_contributions={"binoculars_ratio": 1.4},
        top_contributors=[("binoculars_ratio", 1.4)],
    )
    assert fr.p_llm == 0.62
    assert "AMBER" in fr.prediction_set

def test_determination_fields():
    d = Determination(
        label="AMBER",
        p_llm=0.62,
        confidence_interval=(0.48, 0.73),
        prediction_set=["YELLOW", "AMBER"],
        reason="High binoculars ratio",
        top_features=[("binoculars_ratio", 1.4)],
        override_applied=False,
        detectors_run=["preamble", "fingerprint"],
        cascade_phases=[1],
        mixed_report=None,
    )
    assert d.label == "AMBER"
    assert not d.override_applied
