import pytest
from beet.detectors.fingerprint_vocab import FingerprintVocabDetector
from tests.fixtures.llm_samples import A1_CLEANED, A0_CLINICAL_TASK
from tests.fixtures.human_samples import CASUAL_SHORT

@pytest.fixture
def detector():
    return FingerprintVocabDetector()

@pytest.fixture
def config():
    return {}

def test_llm_text_with_many_fingerprints(detector, config):
    result = detector.analyze(A1_CLEANED, config)
    assert result.signals["hits_per_1000"] > 3.0
    assert result.p_llm > 0.50

def test_casual_human_has_low_fingerprints(detector, config):
    result = detector.analyze(CASUAL_SHORT, config)
    assert result.signals["hits_per_1000"] < 5.0
    assert result.p_llm < 0.50

def test_bigrams_are_counted(detector, config):
    text = "It's important to note that this approach plays a crucial role in the process."
    result = detector.analyze(text, config)
    assert result.signals["bigram_hits"] >= 1

def test_result_fields(detector, config):
    result = detector.analyze(A0_CLINICAL_TASK, config)
    assert result.layer_id == "fingerprint_vocab"
    assert result.compute_cost == "trivial"
    assert "A1" in result.attacker_tiers
    assert "hits_per_1000" in result.signals
    assert "matched_words" in result.signals


def test_fingerprint_emits_spans_pointing_at_matches(detector, config):
    text = "It's important to note that this approach plays a crucial role in the process."
    result = detector.analyze(text, config)
    assert result.spans, "text with fingerprint phrases should produce spans"
    for s in result.spans:
        assert {"start", "end", "kind", "note"} <= set(s.keys())
        assert s["kind"] == "fingerprint"
        assert 0 <= s["start"] < s["end"] <= len(text)
        assert text[s["start"]:s["end"]]  # non-empty slice
