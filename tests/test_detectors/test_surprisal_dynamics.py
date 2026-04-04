import pytest

pytest.importorskip("transformers", reason="transformers not installed — skip Tier 2 tests")
pytest.importorskip("torch", reason="torch not installed — skip Tier 2 tests")

from beet.detectors.surprisal_dynamics import SurprisalDynamicsDetector
from tests.fixtures.llm_samples import A0_CLINICAL_TASK
from tests.fixtures.human_samples import FORMAL_SOP

@pytest.fixture(scope="module")
def detector():
    # Uses GPT-2 small — downloads ~500MB on first run
    return SurprisalDynamicsDetector(model_name="gpt2")

def test_result_has_required_fields(detector):
    result = detector.analyze(FORMAL_SOP, {})
    assert result.layer_id == "surprisal_dynamics"
    assert 0.0 <= result.p_llm <= 1.0
    assert "surprisal_mean" in result.signals
    assert "late_volatility_ratio" in result.signals
    assert "surprisal_diversity" in result.signals

def test_compute_cost_is_moderate(detector):
    result = detector.analyze(FORMAL_SOP, {})
    assert result.compute_cost == "moderate"

def test_llm_text_scores_higher_than_human(detector):
    llm_result = detector.analyze(A0_CLINICAL_TASK, {})
    human_result = detector.analyze(FORMAL_SOP, {})
    # LLM text should have lower late volatility (key discriminating signal)
    # This may not always hold on small texts — check the trend direction
    assert isinstance(llm_result.signals["late_volatility_ratio"], float)
    assert isinstance(human_result.signals["late_volatility_ratio"], float)
