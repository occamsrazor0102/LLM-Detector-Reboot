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
