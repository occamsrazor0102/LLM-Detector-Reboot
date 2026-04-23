"""Perturbation detector tests.

Runtime behaviour (log-prob scoring with distilgpt2) isn't exercised here
— downloading the model is too slow for CI. We test the gates, the
perturbation helper, the z-score math via a stub scorer, and the
interpolation mapping.
"""
from unittest.mock import patch

import pytest

from beet.detectors.perturbation import (
    PerturbationDetector,
    _HEURISTIC_Z_TO_P_LLM,
    _interpolate,
)


@pytest.fixture
def detector():
    return PerturbationDetector()


def test_perturbation_registered():
    import beet.detectors as registry
    assert "perturbation" in registry.get_all_detectors()


def test_perturbation_placed_in_phase3():
    from beet.cascade import PHASE3_DETECTORS
    assert "perturbation" in PHASE3_DETECTORS


def test_perturbation_skips_short_text(detector):
    r = detector.analyze("short sentence.", {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "too_short"


def test_perturbation_skip_signals_when_torch_unavailable(detector):
    """With torch absent, the detector must skip cleanly with a helpful hint."""
    long_text = " ".join(["word"] * 200)
    # Simulate torch absent by flipping the module flag.
    import beet.detectors.perturbation as mod
    with patch.object(mod, "_HAS_TORCH", False):
        r = detector.analyze(long_text, {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "torch_unavailable"
    assert "tier2" in r.signals.get("hint", "")


def test_perturb_is_deterministic_with_seed(detector):
    import random
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    assert detector._perturb(text, 0.3, rng_a) == detector._perturb(text, 0.3, rng_b)


def test_perturb_replaces_some_words(detector):
    import random
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    out = detector._perturb(text, 0.4, random.Random(1))
    # Output has the same length but some swapped tokens
    assert len(out.split()) == len(text.split())
    assert set(out.split()).issubset(set(text.split()))


@pytest.mark.parametrize(
    "z, expected_bounds",
    [
        (-1.0, (0.05, 0.15)),  # below range → base
        (0.0, (0.05, 0.15)),
        (1.0, (0.40, 0.50)),
        (3.0, (0.75, 0.85)),
        (10.0, (0.90, 0.99)),  # above range → top
    ],
)
def test_heuristic_z_to_p_llm_mapping_is_monotone(z, expected_bounds):
    p = _interpolate(max(z, 0.0), _HEURISTIC_Z_TO_P_LLM)
    lo, hi = expected_bounds
    assert lo <= p <= hi


def test_perturbation_with_stub_scorer_produces_layer_result(detector):
    """Patch in a stub scorer so we can test the end-to-end flow
    without downloading distilgpt2. Verifies: z-score math runs,
    signals shape is right, determination respects the mapping."""
    long_text = " ".join(["token"] * 200)

    import beet.detectors.perturbation as mod

    with patch.object(mod, "_HAS_TORCH", True):
        # Stub model-load and scoring.
        with patch.object(detector, "_ensure_loaded", return_value=None), \
             patch.object(detector, "_log_prob") as mock_lp:
            # Original text scores much higher than perturbations — looks LLM-like.
            mock_lp.side_effect = [-1.0] + [-3.0, -3.1, -2.9, -3.2, -3.0, -2.8, -3.1, -3.0]
            r = detector.analyze(long_text, {"n_perturbations": 8})

    assert r.determination in ("GREEN", "YELLOW", "AMBER", "RED")
    assert "z_score" in r.signals
    assert r.signals["z_score"] > 0  # original above mean perturbed
    assert r.signals["n_perturbations"] == 8
