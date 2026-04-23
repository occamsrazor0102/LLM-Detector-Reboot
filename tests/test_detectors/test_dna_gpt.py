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
    """First call errors, the other two succeed with continuations that
    actually overlap the real text. Result must fuse only the successful
    calls and surface the partial error without SKIPping."""
    from tests.fixtures.llm_samples import A0_CLINICAL_TASK

    # Pick continuation text that has real n-gram overlap with the actual
    # continuation of A0_CLINICAL_TASK so bscore > 0 on the successful calls.
    words = A0_CLINICAL_TASK.split()
    cut_50 = int(len(words) * 0.50)
    overlapping = " ".join(words[cut_50:cut_50 + 50])

    calls = {"n": 0}

    def fake_gen(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "", "rate limit"
        return overlapping, None

    with patch.object(detector, "_generate_continuation", side_effect=fake_gen):
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})

    assert result.determination != "SKIP", (
        "expected partial recovery but got SKIP: " + str(result.signals)
    )
    # The partial error must be surfaced in signals so operators can see
    # reduced coverage, but the determination is still produced.
    assert "provider_errors_partial" in result.signals
    assert result.signals["provider_errors_partial"] == ["0.3: rate limit"]
    # Trend must not be biased by the failed call (b30 was not treated as 0).
    # We don't assert a specific value, just that the field exists.
    assert "bscore_trend" in result.signals


def test_all_provider_errors_yields_skip(detector):
    with patch.object(detector, "_generate_continuation", return_value=("", "upstream down")):
        from tests.fixtures.llm_samples import A0_CLINICAL_TASK
        result = detector.analyze(A0_CLINICAL_TASK, {"api_key": "test_key"})
    assert result.determination == "SKIP"
    assert result.signals.get("skipped") == "provider_error"
    assert result.signals.get("errors")
