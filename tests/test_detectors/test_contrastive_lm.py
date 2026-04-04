import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from beet.detectors.contrastive_lm import ContrastiveLMDetector

def test_result_has_required_fields():
    d = ContrastiveLMDetector()
    from tests.fixtures.human_samples import FORMAL_SOP
    result = d.analyze(FORMAL_SOP, {})
    assert result.layer_id == "contrastive_lm"
    assert "binoculars_ratio" in result.signals
    assert 0.0 <= result.p_llm <= 1.0
    assert result.compute_cost == "moderate"

def test_returns_skip_for_short_text():
    d = ContrastiveLMDetector()
    result = d.analyze("short text", {})
    assert result.determination == "SKIP"
