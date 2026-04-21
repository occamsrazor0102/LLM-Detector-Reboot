# beet/detectors/contrastive_gen.py
"""Contrastive generation detector.

Theory: instead of asking "does this look like LLM text?" against static
archetypes, ask "does this look like what an LLM would generate for this
exact task?". Generates k baselines via an LLM provider for the same task
description, then compares the submission against them.

Status: registered stub. Returns SKIP when `task_description` is missing
from the detector config (expected path until pipeline plumbs task metadata
and an API provider is configured). Full baseline generation and distance
computation land in a later phase.

PRIVACY: only the task description is ever sent to the provider — never
the submission text itself. That invariant must hold in any full implementation.
"""
from __future__ import annotations

from beet.contracts import LayerResult


class ContrastiveGenDetector:
    id = "contrastive_gen"
    domain = "universal"
    compute_cost = "expensive"

    def analyze(self, text: str, config: dict) -> LayerResult:
        task_description = config.get("task_description")
        if not task_description:
            return _skip(self, reason="no_task_metadata")

        provider = config.get("provider")
        if not provider:
            return _skip(self, reason="no_provider_configured")

        # Full generation + distance computation not yet wired.
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
        min_text_length=0,
    )


DETECTOR = ContrastiveGenDetector()
