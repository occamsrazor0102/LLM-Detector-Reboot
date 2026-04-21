"""Contributor graph / syndicate detector tests."""
from beet.detectors.contributor_graph import DETECTOR


def test_registered_in_registry():
    import beet.detectors as registry
    assert "contributor_graph" in registry.get_all_detectors()


def test_analyze_single_returns_skip():
    r = DETECTOR.analyze("single text", {})
    assert r.determination == "SKIP"


def test_analyze_contributors_empty():
    assert DETECTOR.analyze_contributors({}, {}) == {}


def test_clusters_similar_contributors_together():
    shared = "template sentence one. template sentence two. template sentence three. template sentence four."
    submissions = {
        "alice": [shared + " alice addition."],
        "bob": [shared + " bob addition."],
        "carol": ["completely different text about weather patterns and cooking recipes."],
    }
    out = DETECTOR.analyze_contributors(submissions, {"shingle_k": 3, "pair_threshold": 0.3})
    assert set(out.keys()) == {"alice", "bob", "carol"}
    assert out["alice"]["cluster_id"] == out["bob"]["cluster_id"]
    assert out["carol"]["cluster_id"] != out["alice"]["cluster_id"]


def test_risk_score_higher_for_clustered():
    shared = "the quick brown fox jumps over the lazy dog this is a template phrase."
    submissions = {
        "a": [shared],
        "b": [shared],
        "c": ["completely unrelated text about something else entirely here today."],
    }
    out = DETECTOR.analyze_contributors(submissions, {"shingle_k": 3, "pair_threshold": 0.3})
    assert out["a"]["risk_score"] > out["c"]["risk_score"]


def test_degree_and_cluster_size_reported():
    shared = "template alpha beta gamma delta epsilon zeta eta theta iota kappa"
    submissions = {cid: [shared] for cid in ("x", "y", "z")}
    out = DETECTOR.analyze_contributors(submissions, {"shingle_k": 2, "pair_threshold": 0.3})
    for cid in ("x", "y", "z"):
        assert out[cid]["cluster_size"] == 3
        assert out[cid]["signals"]["degree"] >= 1
