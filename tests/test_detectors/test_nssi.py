from beet.detectors.nssi import NSSIDetector
from tests.fixtures.llm_samples import A0_PREAMBLE

def test_nssi_returns_layer_result():
    d = NSSIDetector()
    result = d.analyze(A0_PREAMBLE, {})
    assert result.layer_id == "nssi"
    assert 0.0 <= result.p_llm <= 1.0
    assert "n_signals_active" in result.signals

def test_nssi_signals_dict_has_all_keys():
    d = NSSIDetector()
    result = d.analyze("The approach demonstrates comprehensive understanding. " * 20, {})
    assert "formulaic_density" in result.signals
    assert "power_adj_saturation" in result.signals
    assert "discourse_scaffolding" in result.signals
    assert "sentence_start_monotony" in result.signals
