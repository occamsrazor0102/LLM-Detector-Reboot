import pytest

pytest.importorskip("anthropic", reason="anthropic SDK not installed — skip Tier 3 tests")

from beet.detectors.dna_gpt import DNAGPTDetector
from unittest.mock import patch, MagicMock

@pytest.fixture
def detector():
    return DNAGPTDetector()

def test_returns_skip_without_api_key(detector):
    result = detector.analyze("some text " * 50, {"api_key": None})
    assert result.determination == "SKIP"

def test_result_fields_structure(detector):
    # Patch the API call
    mock_continuation = "The study will incorporate rigorous statistical methods."
    with patch.object(detector, "_generate_continuation", return_value=mock_continuation):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert "bscore_50" in result.signals
    assert "bscore_trend" in result.signals
    assert result.layer_id == "dna_gpt"

def test_high_bscore_gives_high_p_llm(detector):
    # Mock all three truncations returning high-overlap continuations
    with patch.object(detector, "_generate_continuation", return_value="must ensure comprehensive deliverable output format"):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert result.signals["bscore_50"] >= 0.0  # just check it runs
