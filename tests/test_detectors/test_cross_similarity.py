"""Cross-submission similarity detector tests."""
from beet.detectors.cross_similarity import DETECTOR


def test_registered_in_registry():
    import beet.detectors as registry
    assert "cross_similarity" in registry.get_all_detectors()


def test_single_text_returns_skip():
    r = DETECTOR.analyze("some text", {})
    assert r.determination == "SKIP"


def test_batch_duplicates_produce_high_similarity():
    shared = "the quick brown fox jumps over the lazy dog"
    texts = {
        "a": shared + " one two three",
        "b": shared + " four five six",
        "c": "completely unrelated text about weather patterns and climate",
    }
    out = DETECTOR.analyze_batch(texts, {"shingle_k": 3, "jaccard_threshold": 0.2})
    assert set(out.keys()) == {"a", "b", "c"}
    # a and b share a long prefix → high Jaccard
    assert out["a"].signals["max_jaccard"] > 0.3
    assert out["b"].signals["max_jaccard"] > 0.3
    # c is unrelated → low max
    assert out["c"].signals["max_jaccard"] < 0.2


def test_batch_singleton_skips():
    out = DETECTOR.analyze_batch({"solo": "just one submission"}, {})
    assert out["solo"].determination == "SKIP"


def test_batch_empty_returns_empty():
    assert DETECTOR.analyze_batch({}, {}) == {}


def test_high_similarity_promotes_to_red_or_amber():
    texts = {f"s{i}": "template line one. template line two. template line three." for i in range(3)}
    out = DETECTOR.analyze_batch(texts, {"shingle_k": 3, "jaccard_threshold": 0.3})
    for sid, r in out.items():
        assert r.p_llm >= 0.75
        assert r.determination in ("RED", "AMBER")
