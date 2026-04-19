"""Programmatic post-processing adversarial transforms.

Each transform is a pure function: str -> str. No API calls, no side effects.
"""
from __future__ import annotations

import random
import re

from beet.adversarial.registry import Attack, register


_PREAMBLE_OPENERS = (
    r"(?i)^\s*(certainly!?|sure!?|here(?:'s|\s+is)|of course!?|absolutely!?|"
    r"let\s+me|below\s+is|i'?d\s+be\s+happy\s+to|great\s+question!?)[^\n.!?:]*[.!?:]\s*"
)
_PREAMBLE_CLOSERS = (
    r"(?i)\s*(let\s+me\s+know\s+if\s+you'?d\s+like[^.!?]*[.!?]|"
    r"hope\s+this\s+helps[^.!?]*[.!?]|"
    r"feel\s+free\s+to\s+(?:ask|let\s+me\s+know)[^.!?]*[.!?])\s*$"
)


def strip_preamble(text: str, **_: object) -> str:
    """Remove common LLM opener/closer phrasing."""
    out = re.sub(_PREAMBLE_OPENERS, "", text, count=1)
    out = re.sub(_PREAMBLE_CLOSERS, "", out)
    return out.strip()


_TYPO_SWAPS = {"the": "teh", "and": "adn", "you": "yuo", "with": "wiht",
               "that": "taht", "this": "tihs", "have": "ahve", "from": "form"}


def inject_typos(text: str, *, rate: float = 0.15, seed: int | None = None, **_: object) -> str:
    """Introduce casual misspellings and occasional punctuation drops."""
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for w in words:
        lw = w.lower()
        key = re.sub(r"[^a-z]", "", lw)
        if key in _TYPO_SWAPS and rng.random() < rate * 3:
            swapped = _TYPO_SWAPS[key]
            if w and w[0].isupper():
                swapped = swapped.capitalize()
            trailing = re.sub(r"[a-zA-Z]", "", w)
            out.append(swapped + trailing)
        elif rng.random() < rate / 3 and w.endswith((".", ",", ";")):
            out.append(w[:-1])
        else:
            out.append(w)
    return " ".join(out)


_CONTRACTIONS = [
    (r"\bdo not\b", "don't"),
    (r"\bcannot\b", "can't"),
    (r"\bis not\b", "isn't"),
    (r"\bwill not\b", "won't"),
    (r"\bit is\b", "it's"),
    (r"\bI am\b", "I'm"),
    (r"\byou are\b", "you're"),
    (r"\bthat is\b", "that's"),
    (r"\bare not\b", "aren't"),
    (r"\bhave not\b", "haven't"),
]
_FILLERS = [" honestly", " like", " basically", " kind of"]


def casualize(text: str, *, seed: int | None = None, **_: object) -> str:
    """Lower formality: contractions + sprinkled filler words."""
    rng = random.Random(seed)
    out = text
    for pat, repl in _CONTRACTIONS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # Occasionally splice a filler at sentence starts
    sentences = re.split(r"(?<=[.!?])\s+", out)
    for i in range(len(sentences)):
        if rng.random() < 0.25 and len(sentences[i]) > 10:
            filler = rng.choice(_FILLERS).strip()
            sentences[i] = sentences[i][0].lower() + sentences[i][1:]
            sentences[i] = f"{filler}, " + sentences[i]
    return " ".join(sentences)


_SYNONYMS = {
    "furthermore": "also",
    "moreover": "also",
    "comprehensive": "full",
    "utilize": "use",
    "commence": "start",
    "facilitate": "help",
    "numerous": "many",
    "demonstrate": "show",
    "implement": "do",
    "subsequently": "then",
}


def synonym_swap(text: str, **_: object) -> str:
    """Replace common LLM fingerprint vocabulary with plainer synonyms."""
    out = text
    for word, repl in _SYNONYMS.items():
        out = re.sub(rf"\b{word}\b", repl, out, flags=re.IGNORECASE)
    return out


_HUMAN_FRAGMENTS = [
    "honestly this took me three tries to get right",
    "anyway, moving on",
    "idk if this is what you wanted",
    "i'll circle back to this next week",
    "not sure that matters but whatever",
]


def mix_human(text: str, *, seed: int | None = None, **_: object) -> str:
    """Splice short human-style fragments into LLM text at random positions."""
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return text + " " + rng.choice(_HUMAN_FRAGMENTS) + "."
    n_insertions = max(1, len(sentences) // 4)
    for _ in range(n_insertions):
        pos = rng.randint(1, len(sentences) - 1)
        sentences.insert(pos, rng.choice(_HUMAN_FRAGMENTS) + ".")
    return " ".join(sentences)


# Register all transforms
register(Attack(
    name="strip_preamble",
    category="transform",
    description="Remove 'Certainly!', 'Here's...' openers and closers",
    severity="basic",
    apply=strip_preamble,
))
register(Attack(
    name="inject_typos",
    category="transform",
    description="Random casual misspellings and missing punctuation",
    severity="basic",
    apply=inject_typos,
))
register(Attack(
    name="casualize",
    category="transform",
    description="Lower formality via contractions and filler words",
    severity="moderate",
    apply=casualize,
))
register(Attack(
    name="synonym_swap",
    category="transform",
    description="Replace fingerprint vocabulary with common synonyms",
    severity="moderate",
    apply=synonym_swap,
))
register(Attack(
    name="mix_human",
    category="transform",
    description="Splice human-style sentences into LLM text",
    severity="advanced",
    apply=mix_human,
))
