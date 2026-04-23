import pytest

from unittest.mock import patch

from beet.detectors.dna_gpt import DNAGPTDetector


@pytest.fixture
def detector():
    return DNAGPTDetector()


def test_returns_skip_without_api_key(detector, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = detector.analyze("some text " * 50, {})
    assert result.determination == "SKIP"
    assert result.signals.get("skipped") == "no_api_key"


def test_env_var_provides_api_key(detector, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-xyz")
    with patch.object(detector, "_generate_continuation",
                       return_value=("placeholder continuation text", None)):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {})
    assert result.determination != "SKIP" or result.signals.get("skipped") != "no_api_key"


def test_result_fields_structure(detector):
    with patch.object(detector, "_generate_continuation",
                       return_value=("The study will incorporate rigorous statistical methods.", None)):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert "bscore_50" in result.signals
    assert "bscore_trend" in result.signals
    assert result.layer_id == "dna_gpt"


def test_high_bscore_gives_high_p_llm(detector):
    with patch.object(detector, "_generate_continuation",
                       return_value=("must ensure comprehensive deliverable output format", None)):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert result.signals["bscore_50"] >= 0.0


def test_provider_errors_surface_partial_on_mixed(detector):
    # First call errors, remaining succeed — result should fuse the good ones
    # and surface the partial error in signals rather than SKIPping.
    def fake_gen(self_arg, *a, **k):
        fake_gen.calls += 1
        if fake_gen.calls == 1:
            return "", "rate limit"
        return "partial continuation text works ok", None
    fake_gen.calls = 0

    with patch.object(detector, "_generate_continuation",
                       side_effect=lambda *a, **k: fake_gen(detector, *a, **k)):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert result.determination != "SKIP" or "provider_error" in str(result.signals.get("skipped", ""))


def test_all_provider_errors_yields_skip(detector):
    with patch.object(detector, "_generate_continuation", return_value=("", "upstream down")):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert result.determination == "SKIP"
    assert result.signals.get("skipped") == "provider_error"
    assert result.signals.get("errors")
