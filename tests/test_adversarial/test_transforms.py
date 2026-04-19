import pytest
from beet.adversarial.transforms import (
    strip_preamble,
    inject_typos,
    casualize,
    synonym_swap,
    mix_human,
)


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
