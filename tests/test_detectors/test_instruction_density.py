from beet.detectors.instruction_density import InstructionDensityDetector
from tests.fixtures.llm_samples import A0_CLINICAL_TASK

def test_high_idi_for_llm_task():
    d = InstructionDensityDetector()
    result = d.analyze(A0_CLINICAL_TASK, {})
    assert result.signals["idi"] > 3.0
    assert result.layer_id == "instruction_density"

def test_low_idi_for_casual_text():
    d = InstructionDensityDetector()
    from tests.fixtures.human_samples import CASUAL_SHORT
    result = d.analyze(CASUAL_SHORT, {})
    assert result.signals["idi"] < 5.0
    assert result.p_llm < 0.50


def test_instruction_density_emits_spans():
    d = InstructionDensityDetector()
    text = "Calculate the result. If the value is negative, return zero. Analyze the output."
    result = d.analyze(text, {})
    assert result.spans
    for s in result.spans:
        assert s["kind"] == "instruction"
        assert text[s["start"]:s["end"]]
    # at least one imperative and one conditional should be spans
    notes = " ".join(s["note"] for s in result.spans)
    assert "imperative" in notes
    assert "conditional" in notes
