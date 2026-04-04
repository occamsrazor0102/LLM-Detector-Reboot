import pytest
import numpy as np
from beet.fusion.ebm import FeatureAssembler, EBMFusion
from beet.fusion.conformal import ConformalWrapper
from beet.contracts import LayerResult, FusionResult


def _make_layer_result(layer_id: str, p_llm: float, signals: dict = None) -> LayerResult:
    return LayerResult(
        layer_id=layer_id, domain="universal",
        raw_score=p_llm, p_llm=p_llm, confidence=0.8,
        signals=signals or {}, determination="AMBER",
        attacker_tiers=["A0"], compute_cost="cheap",
        min_text_length=0,
    )


def test_feature_assembler_produces_dict():
    assembler = FeatureAssembler()
    results = [
        _make_layer_result("preamble", 0.05, {"severity": "NONE"}),
        _make_layer_result("fingerprint_vocab", 0.65, {"hits_per_1000": 9.3}),
        _make_layer_result("nssi", 0.45, {"n_signals_active": 4}),
    ]
    vec = assembler.assemble(results, word_count=300, domain="prose")
    assert "preamble_p_llm" in vec
    assert "fingerprint_hits_per_1000" in vec
    assert "word_count" in vec


def test_feature_assembler_handles_missing_detectors():
    assembler = FeatureAssembler()
    results = [_make_layer_result("preamble", 0.10)]
    vec = assembler.assemble(results, word_count=200, domain="prompt")
    # Missing detectors → NaN (not KeyError)
    assert "fingerprint_hits_per_1000" in vec
    import math
    assert math.isnan(vec["fingerprint_hits_per_1000"])


def test_ebm_fusion_untrained_falls_back_to_naive():
    fusion = EBMFusion()
    results = [_make_layer_result("preamble", 0.90, {"severity": "CRITICAL"})]
    fr = fusion.fuse(results, word_count=100, domain="prompt")
    assert isinstance(fr, FusionResult)
    assert 0.0 <= fr.p_llm <= 1.0


def test_ebm_fusion_trained_produces_result():
    pytest.importorskip("interpret", reason="interpret not installed — skip EBM training test")
    from beet.fusion.training import train_ebm
    # Toy training data: 10 clear LLM + 10 clear human
    X = []
    y = []
    assembler = FeatureAssembler()
    for _ in range(10):
        r = [_make_layer_result("preamble", 0.90, {"severity": "CRITICAL"}),
             _make_layer_result("fingerprint_vocab", 0.85, {"hits_per_1000": 18.0})]
        X.append(assembler.assemble(r, word_count=200, domain="prompt"))
        y.append(1)
    for _ in range(10):
        r = [_make_layer_result("preamble", 0.05, {"severity": "NONE"}),
             _make_layer_result("fingerprint_vocab", 0.10, {"hits_per_1000": 1.2})]
        X.append(assembler.assemble(r, word_count=250, domain="prose"))
        y.append(0)
    model = train_ebm(X, y)
    fusion = EBMFusion(model=model)
    test_results = [_make_layer_result("preamble", 0.92, {"severity": "CRITICAL"})]
    fr = fusion.fuse(test_results, word_count=200, domain="prompt")
    assert fr.p_llm > 0.5  # should be suspicious


def test_conformal_wrapper():
    # Calibrate with 20 examples
    cal_scores = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.8, 0.7, 0.9, 0.85,
                           0.12, 0.18, 0.22, 0.08, 0.35, 0.75, 0.65, 0.95, 0.88, 0.78])
    cal_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    wrapper = ConformalWrapper(alpha=0.05)
    wrapper.calibrate(cal_scores, cal_labels)
    pset = wrapper.predict_set(0.82)
    assert isinstance(pset, list)
    assert len(pset) >= 1
