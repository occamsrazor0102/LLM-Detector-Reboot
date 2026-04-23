import re
from beet.contracts import LayerResult

_VOICE_PATTERNS = [
    r"\b(?:okay|ok|yeah|ya|gonna|wanna|gotta|kinda|sorta|dunno|nah|yep)\b",
    r"\b(?:basically|honestly|actually|literally|seriously|totally|just)\b",
    r"\b(?:i think|i feel|i believe|i guess|i suppose|i reckon)\b",
    r"\b(?:you know|ya know|know what i mean|right\?|doesn't it)\b",
    r"(?<!\w)lol\b|(?<!\w)haha\b|(?<!\w)ugh\b",
    r"\.\.\.",
    r"[!]{2,}",
]
_VOICE_RE = [re.compile(p, re.IGNORECASE) for p in _VOICE_PATTERNS]

_SPEC_PATTERNS = [
    r"\b(?:must|shall|should|required to|expected to|ensure|guarantee)\b",
    r"\b(?:include|incorporate|provide|specify|define|outline|detail)\b",
    r"\b(?:protocol|procedure|deliverable|output|endpoint|criterion|criteria)\b",
    r"\b(?:comply|adhere|conform|align with|according to|pursuant to)\b",
    r"\b(?:minimum|maximum|threshold|metric|benchmark|standard|target)\b",
    r"\b(?:format|structure|organize|present as|submit|document)\b",
]
_SPEC_RE = [re.compile(p, re.IGNORECASE) for p in _SPEC_PATTERNS]

def _score_voice(text: str, word_count: int, spans: list[dict] | None = None) -> float:
    count = 0
    for p in _VOICE_RE:
        for m in p.finditer(text):
            count += 1
            if spans is not None:
                spans.append({"start": m.start(), "end": m.end(),
                              "kind": "voice_informal",
                              "note": f"informal voice '{m.group(0)}'"})
    return count / max(word_count, 1) * 100

def _score_spec(text: str, word_count: int, spans: list[dict] | None = None) -> float:
    count = 0
    for p in _SPEC_RE:
        for m in p.finditer(text):
            count += 1
            if spans is not None:
                spans.append({"start": m.start(), "end": m.end(),
                              "kind": "spec",
                              "note": f"spec language '{m.group(0)}'"})
    return count / max(word_count, 1) * 100

class VoiceSpecDetector:
    id = "voice_spec"
    domain = "prompt"
    compute_cost = "cheap"

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        word_count = max(len(words), 1)
        spans: list[dict] = []
        voice_score = _score_voice(text, word_count, spans)
        spec_score = _score_spec(text, word_count, spans)
        mode = "normal"
        p_llm = 0.10
        if voice_score >= 1.5 and spec_score >= 4.0:
            mode = "dissonance"
            p_llm = min(0.15 + voice_score * spec_score / 50.0, 0.95)
        elif voice_score < 0.5 and spec_score >= 4.0:
            mode = "sterile"
            p_llm = min(0.30 + spec_score / 30.0, 0.85)
        elif voice_score >= 5.0 and spec_score >= 0.5:
            mode = "informal_excess"
            p_llm = min(0.20 + (voice_score - 5.0) / 20.0, 0.60)
        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=voice_score * spec_score if mode == "dissonance" else spec_score,
            p_llm=p_llm, confidence=min(0.40 + word_count / 800, 0.88),
            signals={"voice_score": round(voice_score, 3), "spec_score": round(spec_score, 3), "mode": mode},
            determination=determination, attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost, min_text_length=30,
            spans=spans,
        )

    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = VoiceSpecDetector()
