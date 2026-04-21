"""Contrastive generation detector — stub behavior tests."""
from beet.detectors.contrastive_gen import DETECTOR


def test_contrastive_gen_registered():
    import beet.detectors as registry
    assert "contrastive_gen" in registry.get_all_detectors()


def test_contrastive_gen_skips_without_task_metadata():
    r = DETECTOR.analyze("Some text.", {})
    assert r.determination == "SKIP"
    assert r.signals.get("reason") == "no_task_metadata"


def test_contrastive_gen_skips_without_provider():
    r = DETECTOR.analyze("Some text.", {"task_description": "Write a report."})
    assert r.determination == "SKIP"
    assert r.signals.get("reason") == "no_provider_configured"


def test_contrastive_gen_skip_with_full_config():
    r = DETECTOR.analyze(
        "Some text.",
        {"task_description": "Write a report.", "provider": "anthropic"},
    )
    assert r.determination == "SKIP"
    assert r.signals.get("reason") == "not_implemented"


def test_contrastive_gen_placed_in_phase3():
    from beet.cascade import PHASE3_DETECTORS
    assert "contrastive_gen" in PHASE3_DETECTORS
