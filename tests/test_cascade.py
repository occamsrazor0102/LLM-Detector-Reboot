import pytest
from beet.cascade import CascadeScheduler
from beet.contracts import LayerResult

def _make_result(layer_id, p_llm, compute_cost, determination):
    return LayerResult(layer_id=layer_id, domain="universal", raw_score=p_llm, p_llm=p_llm,
        confidence=0.8, signals={}, determination=determination, attacker_tiers=["A0"],
        compute_cost=compute_cost, min_text_length=0)

@pytest.fixture
def scheduler(minimal_config):
    return CascadeScheduler(config=minimal_config)

def test_phase1_short_circuits_on_high_p_llm(scheduler):
    results = [_make_result("preamble", 0.97, "trivial", "RED"), _make_result("fingerprint_vocab", 0.90, "trivial", "RED")]
    assert scheduler.should_run_phase2(results) is False

def test_phase1_short_circuits_on_low_p_llm(scheduler):
    results = [_make_result("preamble", 0.05, "trivial", "GREEN"), _make_result("fingerprint_vocab", 0.08, "trivial", "GREEN")]
    assert scheduler.should_run_phase2(results) is False

def test_phase1_inconclusive_runs_phase2(scheduler):
    results = [_make_result("preamble", 0.30, "trivial", "YELLOW"), _make_result("fingerprint_vocab", 0.45, "trivial", "YELLOW")]
    assert scheduler.should_run_phase2(results) is True

def test_aggregate_p_llm_is_weighted_mean(scheduler):
    results = [_make_result("preamble", 0.80, "trivial", "RED"), _make_result("fingerprint_vocab", 0.40, "trivial", "YELLOW")]
    agg = scheduler.aggregate_p_llm(results)
    assert 0.40 < agg < 0.80
