"""Perturbation detector — stub behavior tests.

Full behavioral tests land alongside the full implementation. These verify
the stub gates correctly and participates in the registry.
"""
from beet.detectors.perturbation import DETECTOR


def test_perturbation_registered():
    import beet.detectors as registry
    all_dets = registry.get_all_detectors()
    assert "perturbation" in all_dets


def test_perturbation_skips_short_text():
    r = DETECTOR.analyze("short sentence.", {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") is True
    assert r.signals.get("reason") == "insufficient_tokens"


def test_perturbation_skip_on_long_text():
    long_text = " ".join(["word"] * 200)
    r = DETECTOR.analyze(long_text, {})
    assert r.determination == "SKIP"
    # Either torch unavailable or not_implemented — both acceptable stub outcomes.
    assert r.signals.get("reason") in {"torch_unavailable", "not_implemented"}


def test_perturbation_placed_in_phase3():
    from beet.cascade import PHASE3_DETECTORS
    assert "perturbation" in PHASE3_DETECTORS
