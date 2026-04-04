# beet/router.py
import re
from beet.contracts import RouterDecision

# Prompt-domain signal patterns
_IMPERATIVE_PATTERN = re.compile(
    r"\b(?:calculate|generate|create|design|develop|write|provide|list|explain|"
    r"identify|analyze|evaluate|assess|ensure|include|specify|describe|outline|"
    r"demonstrate|implement|define|determine|state|show|give|summarize)\b",
    re.IGNORECASE,
)
_SECOND_PERSON_PATTERN = re.compile(
    r"\b(?:you are|your (?:task|role|goal|job|objective|assignment)|"
    r"you will|you must|you should|you need to)\b",
    re.IGNORECASE,
)
_DELIVERABLE_PATTERN = re.compile(
    r"\b(?:output|deliverable|result|response|answer|solution|"
    r"format|structure|submit|provide)\b",
    re.IGNORECASE,
)
_BULLET_STRUCTURE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_SECTION_HEADER = re.compile(
    r"^\s*\*{0,2}(?:context|role|task|constraint|output|evaluation|objective|"
    r"deliverable|requirement|background|instruction)\*{0,2}\s*[:：]",
    re.IGNORECASE | re.MULTILINE,
)

# Prose-domain signal patterns
_FLOWING_PARAGRAPH = re.compile(r"[A-Z][^.!?]*[.!?]\s+[A-Z]")
_FIRST_PERSON_PROSE = re.compile(r"\b(?:I |we |our |my )\b")
_PAST_TENSE = re.compile(r"\b\w+(?:ed|was|were|had)\b")

# Prompt-domain detector IDs
_PROMPT_DETECTORS = [
    "preamble", "fingerprint_vocab", "prompt_structure",
    "voice_spec", "instruction_density",
]
# Prose-domain detector IDs
_PROSE_DETECTORS = [
    "preamble", "fingerprint_vocab", "nssi",
    "surprisal_dynamics", "contrastive_lm", "token_cohesiveness",
]
# Universal detectors (both)
_UNIVERSAL_DETECTORS = [
    "preamble", "fingerprint_vocab", "contrastive_lm",
    "mixed_boundary",
]


class TextRouter:
    def __init__(self, config: dict):
        mins = config.get("router", {}).get("minimum_words", {})
        self._min_prompt = mins.get("prompt", 30)
        self._min_prose = mins.get("prose", 150)

    def route(self, text: str) -> RouterDecision:
        words = text.split()
        word_count = len(words)

        prompt_score = self._score_prompt(text, word_count)
        prose_score = self._score_prose(text, word_count)

        # Determine domain based on scores and thresholds
        if prompt_score > 0.45 and prompt_score > prose_score * 1.5:
            domain = "prompt"
        elif prose_score > 0.45 and prose_score > prompt_score * 1.5:
            domain = "prose"
        elif prompt_score > 0.1 and prose_score > 0.1:
            domain = "mixed"
        elif word_count < self._min_prompt:
            domain = "insufficient"
        elif word_count < self._min_prose:
            domain = "prose"
        else:
            domain = "prose"

        confidence = abs(prompt_score - prose_score) / max(prompt_score + prose_score, 0.01)

        recommended, skip = self._select_detectors(domain)

        return RouterDecision(
            domain=domain,
            confidence=min(confidence, 1.0),
            prompt_score=prompt_score,
            prose_score=prose_score,
            word_count=word_count,
            recommended_detectors=recommended,
            skip_detectors=skip,
        )

    def _score_prompt(self, text: str, word_count: int) -> float:
        if word_count == 0:
            return 0.0
        imperative_density = len(_IMPERATIVE_PATTERN.findall(text)) / word_count * 100
        second_person = min(len(_SECOND_PERSON_PATTERN.findall(text)) * 2, 10)
        deliverables = min(len(_DELIVERABLE_PATTERN.findall(text)) * 1.5, 10)
        bullets = min(len(_BULLET_STRUCTURE.findall(text)) * 1.5, 8)
        numbered = min(len(_NUMBERED_LIST.findall(text)) * 2, 10)
        headers = min(len(_SECTION_HEADER.findall(text)) * 3, 15)
        raw = imperative_density + second_person + deliverables + bullets + numbered + headers
        return min(raw / 40.0, 1.0)

    def _score_prose(self, text: str, word_count: int) -> float:
        if word_count == 0:
            return 0.0
        flowing = min(len(_FLOWING_PARAGRAPH.findall(text)) * 2, 20)
        first_person = min(len(_FIRST_PERSON_PROSE.findall(text)) * 0.5, 10)
        past_tense = min(len(_PAST_TENSE.findall(text)) * 0.1, 5)
        length_bonus = min(word_count / 500 * 5, 10)
        raw = flowing + first_person + past_tense + length_bonus
        return min(raw / 40.0, 1.0)

    def _select_detectors(self, domain: str) -> tuple[list[str], list[str]]:
        if domain == "prompt":
            return _PROMPT_DETECTORS + ["surprisal_dynamics", "contrastive_lm"], ["nssi"]
        elif domain == "prose":
            return _PROSE_DETECTORS, ["prompt_structure", "voice_spec", "instruction_density"]
        elif domain == "mixed":
            return _PROMPT_DETECTORS + _PROSE_DETECTORS, []
        else:  # insufficient
            return ["preamble", "fingerprint_vocab"], []
