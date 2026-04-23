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


def test_nssi_emits_boilerplate_spans():
    d = NSSIDetector()
    text = (
        "In conclusion, the comprehensive analysis presents a robust framework. "
        "Furthermore, this approach demonstrates significant innovation. "
        "Moreover, the transformative nature of the work is worth noting."
    )
    result = d.analyze(text, {})
    assert result.spans
    kinds = {s["kind"] for s in result.spans}
    assert kinds == {"boilerplate"}
    for s in result.spans:
        assert text[s["start"]:s["end"]]
