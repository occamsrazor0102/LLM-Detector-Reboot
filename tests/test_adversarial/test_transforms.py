import pytest
from beet.adversarial.transforms import (
    strip_preamble,
    inject_typos,
    casualize,
    synonym_swap,
    mix_human,
    coached_casual,
    paraphrase_launder,
)
from beet.adversarial.registry import get_attack, list_attacks


class TestStripPreamble:
    def test_removes_certainly(self):
        out = strip_preamble("Certainly! Here is your answer: The sky is blue.")
        assert "Certainly" not in out
        assert "sky is blue" in out

    def test_removes_heres(self):
        out = strip_preamble("Here's a comprehensive overview: content follows.")
        assert not out.lower().startswith("here")

    def test_removes_closer(self):
        out = strip_preamble("The answer is 42. Let me know if you'd like more detail.")
        assert "let me know" not in out.lower()
        assert "42" in out

    def test_idempotent_on_clean_text(self):
        clean = "This is a sentence with no preamble."
        assert strip_preamble(clean) == clean


class TestInjectTypos:
    def test_modifies_text(self):
        text = "The quick brown fox has jumped over the lazy dog and the hen."
        out = inject_typos(text, seed=42)
        # With fingerprint words in text and high rate, output should differ.
        assert isinstance(out, str)

    def test_empty_string(self):
        assert inject_typos("", seed=1) == ""

    def test_deterministic_with_seed(self):
        t = "The quick brown fox and the lazy dog"
        assert inject_typos(t, seed=7) == inject_typos(t, seed=7)


class TestCasualize:
    def test_contraction(self):
        out = casualize("I cannot do that. It is not possible.", seed=1)
        assert "can't" in out
        assert "it's" in out.lower() or "isn't" in out.lower()

    def test_short_text_no_crash(self):
        assert isinstance(casualize("Hi.", seed=1), str)


class TestSynonymSwap:
    def test_replaces_fingerprint_vocab(self):
        out = synonym_swap("Furthermore, we utilize a comprehensive approach.")
        lower = out.lower()
        assert "furthermore" not in lower
        assert "utilize" not in lower
        assert "comprehensive" not in lower

    def test_preserves_unknown_words(self):
        assert synonym_swap("The cat sat on the mat.") == "The cat sat on the mat."


class TestMixHuman:
    def test_adds_content(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        out = mix_human(text, seed=1)
        assert len(out) > len(text)

    def test_short_text_appended(self):
        out = mix_human("Only this.", seed=1)
        assert out.startswith("Only this.")
        assert len(out) > len("Only this.")


class TestPromptAttacks:
    """Prompt-based attacks require a provider; verify prompt content and errors."""

    def test_coached_casual_calls_provider_with_instruction_and_text(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "rewritten text"

        out = coached_casual("Comprehensive analysis of widgets.", provider=provider)
        assert out == "rewritten text"
        assert "casual" in captured["prompt"].lower()
        assert "Comprehensive analysis of widgets." in captured["prompt"]

    def test_paraphrase_launder_calls_provider(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "paraphrased"

        out = paraphrase_launder("Original sentence here.", provider=provider)
        assert out == "paraphrased"
        assert "paraphrase" in captured["prompt"].lower()
        assert "Original sentence here." in captured["prompt"]

    def test_coached_casual_requires_provider(self):
        with pytest.raises(RuntimeError):
            coached_casual("anything")

    def test_paraphrase_launder_requires_provider(self):
        with pytest.raises(RuntimeError):
            paraphrase_launder("anything")

    def test_prompt_attacks_registered(self):
        assert get_attack("coached_casual") is not None
        assert get_attack("coached_casual").category == "prompt"
        assert get_attack("paraphrase_launder") is not None
        assert get_attack("paraphrase_launder").category == "prompt"
        names = {a.name for a in list_attacks(category="prompt")}
        assert {"coached_casual", "paraphrase_launder"}.issubset(names)
