import pytest
from beet.detectors.prompt_structure import PromptStructureDetector
from tests.fixtures.llm_samples import A0_CLINICAL_TASK, A0_PREAMBLE
from tests.fixtures.human_samples import CASUAL_SHORT, NURSE_EDUCATOR, FORMAL_SOP

@pytest.fixture
def detector():
    return PromptStructureDetector()

@pytest.fixture
def config():
    return {}

def test_llm_clinical_task_scores_high(detector, config):
    result = detector.analyze(A0_CLINICAL_TASK, config)
    assert result.p_llm > 0.70
    assert result.signals["cfd"] > 0.0

def test_casual_human_scores_low(detector, config):
    result = detector.analyze(CASUAL_SHORT, config)
    assert result.p_llm < 0.30

def test_formal_sop_is_not_flagged_high(detector, config):
    result = detector.analyze(FORMAL_SOP, config)
    assert result.p_llm < 0.75

def test_result_has_cfd_signal(detector, config):
    result = detector.analyze(A0_CLINICAL_TASK, config)
    assert "cfd" in result.signals
    assert "distinct_frames" in result.signals
    assert "framing_completeness" in result.signals
    assert "meta_design_hits" in result.signals

def test_layer_id_and_domain(detector, config):
    result = detector.analyze(A0_CLINICAL_TASK, config)
    assert result.layer_id == "prompt_structure"
    assert result.domain == "prompt"
    assert result.compute_cost == "cheap"
