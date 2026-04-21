# beet/detectors/perturbation.py
"""Perturbation curvature detector (DetectGPT-style).

Theory: LLM-generated text sits at local maxima of log-probability space.
Small mask-fill perturbations reduce log-prob for LLM text more than for
human text. This detector measures that discordance.

Status: registered stub. Requires a reference causal LM and a mask-filling
model. Intended as a Phase 3 cascade detector (expensive, only runs on
borderline cases). Full implementation lands when GPU resources are
available; until then the detector returns SKIP with an explanatory signal.
"""
from __future__ import annotations

from beet.contracts import LayerResult

try:
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class PerturbationDetector:
    id = "perturbation"
    domain = "universal"
    compute_cost = "expensive"

    def analyze(self, text: str, config: dict) -> LayerResult:
        # Gate 1: minimum text length
        n_words = len(text.split())
        if n_words < 100:
            return _skip(self, reason="insufficient_tokens", n_words=n_words)

        # Gate 2: dependency availability
        if not _HAS_TORCH:
            return _skip(self, reason="torch_unavailable")

        # Gate 3: full implementation not yet wired
        return _skip(self, reason="not_implemented")

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, **signals) -> LayerResult:
    return LayerResult(
        layer_id=detector.id,
        domain=detector.domain,
        raw_score=0.0,
        p_llm=0.5,
        confidence=0.0,
        signals={"skipped": True, **signals},
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3"],
        compute_cost=detector.compute_cost,
        min_text_length=100,
    )


DETECTOR = PerturbationDetector()
