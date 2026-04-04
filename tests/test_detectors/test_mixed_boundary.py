from beet.detectors.mixed_boundary import MixedBoundaryDetector

MIXED_TEXT = (
    # Human opening
    "I've been analyzing the pharmacokinetics of Drug X for the past two weeks. "
    "The data doesn't quite fit the standard two-compartment model, which is frustrating "
    "but also kind of interesting. Here's what I think is going on.\n\n"
    # LLM-injected section (inserted)
    "**Pharmacokinetic Analysis Framework**\n\n"
    "The analysis must incorporate a comprehensive evaluation of the absorption, distribution, "
    "metabolism, and excretion (ADME) parameters. It is important to note that the non-compartmental "
    "analysis (NCA) approach should be employed to ensure regulatory compliance with ICH guidelines. "
    "Key deliverables include Cmax, Tmax, AUC0-inf, and t1/2 calculations.\n\n"
    # Human closing
    "Anyway, I'll run the NCA this afternoon and see what we get. Probably nothing dramatic."
)

def test_mixed_text_detects_boundaries():
    d = MixedBoundaryDetector()
    result = d.analyze(MIXED_TEXT, {})
    assert result.layer_id == "mixed_boundary"
    assert "n_boundaries" in result.signals
    assert "mixed_probability" in result.signals
    assert 0.0 <= result.p_llm <= 1.0

def test_uniform_human_text_low_boundaries():
    from tests.fixtures.human_samples import CASUAL_SHORT
    d = MixedBoundaryDetector()
    # CASUAL_SHORT is too short — should SKIP
    result = d.analyze(CASUAL_SHORT, {})
    assert result.determination == "SKIP" or result.signals["n_boundaries"] == 0

def test_result_has_segment_determinations():
    d = MixedBoundaryDetector()
    result = d.analyze(MIXED_TEXT, {})
    assert "segment_determinations" in result.signals
