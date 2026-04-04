# tests/test_router.py
import pytest
from beet.router import TextRouter
from beet.contracts import RouterDecision
from tests.fixtures.human_samples import NURSE_EDUCATOR, FORMAL_SOP, CASUAL_SHORT
from tests.fixtures.llm_samples import A0_CLINICAL_TASK

@pytest.fixture
def router():
    return TextRouter(config={
        "router": {"minimum_words": {"prompt": 30, "prose": 150, "mixed": 300}}
    })

def test_routes_llm_structured_task_as_prompt(router):
    decision = router.route(A0_CLINICAL_TASK)
    assert decision.domain == "prompt"
    assert decision.prompt_score > decision.prose_score

def test_routes_formal_sop_as_prompt(router):
    decision = router.route(FORMAL_SOP)
    assert decision.domain in ("prompt", "mixed")

def test_routes_casual_text_as_prose(router):
    decision = router.route(CASUAL_SHORT)
    assert decision.domain in ("prose", "insufficient")

def test_insufficient_for_very_short_text(router):
    decision = router.route("hello world")
    assert decision.domain == "insufficient"

def test_word_count_is_populated(router):
    decision = router.route(A0_CLINICAL_TASK)
    assert decision.word_count > 50

def test_nurse_educator_routes_as_prompt(router):
    decision = router.route(NURSE_EDUCATOR)
    assert decision.domain in ("prompt", "mixed")

def test_decision_has_recommended_detectors(router):
    decision = router.route(A0_CLINICAL_TASK)
    assert len(decision.recommended_detectors) > 0
