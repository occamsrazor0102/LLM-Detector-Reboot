"""Tests for detector availability introspection."""
from pathlib import Path

import pytest

from beet.config import load_config
from beet.pipeline import detector_availability


@pytest.fixture
def screening_cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "screening.yaml")


def test_availability_returns_one_row_per_declared_detector(screening_cfg):
    rows = detector_availability(screening_cfg)
    declared = set(screening_cfg["detectors"].keys())
    assert {r["id"] for r in rows} == declared


def test_availability_row_shape(screening_cfg):
    for r in detector_availability(screening_cfg):
        assert {"id", "enabled", "available", "reason", "requires"} <= set(r.keys())
        assert isinstance(r["enabled"], bool)
        assert isinstance(r["available"], bool)
        assert isinstance(r["requires"], list)


def test_batch_only_detectors_report_unavailable(screening_cfg):
    rows = {r["id"]: r for r in detector_availability(screening_cfg)}
    # cross_similarity and contributor_graph are batch-only; they are
    # declared in default configs but always return SKIP on single text.
    for did in ("cross_similarity", "contributor_graph"):
        if did in rows:
            assert rows[did]["available"] is False
            assert "batch" in rows[did]["reason"].lower() or "disabled" in rows[did]["reason"]


def test_tier1_detectors_are_available_in_screening(screening_cfg):
    rows = {r["id"]: r for r in detector_availability(screening_cfg)}
    # Preamble and fingerprint_vocab are pure-regex and always available.
    for did in ("preamble", "fingerprint_vocab"):
        assert rows[did]["available"] is True, f"{did} should be available: {rows[did]}"


def test_disabled_detector_reports_disabled_reason(screening_cfg):
    rows = {r["id"]: r for r in detector_availability(screening_cfg)}
    # screening.yaml disables surprisal_dynamics/contrastive_lm etc.
    disabled = [did for did, r in rows.items() if not r["enabled"]]
    assert disabled
    for did in disabled:
        r = rows[did]
        assert r["available"] is False
        assert "disabled" in r["reason"].lower()
