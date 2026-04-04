import re
from beet.contracts import LayerResult

_FORMULAIC = re.compile(r"\b(?:in conclusion|in summary|to summarize|it is worth noting|it is important to|as mentioned|as noted above|as discussed|as outlined|needless to say|it goes without saying|at the end of the day|the bottom line|in other words)\b", re.IGNORECASE)
_POWER_ADJ = re.compile(r"\b(?:comprehensive|robust|holistic|transformative|innovative|cutting-edge|state-of-the-art|groundbreaking|revolutionary|unprecedented|meticulous|rigorous|systematic|profound|significant|critical|essential|fundamental)\b", re.IGNORECASE)
_DISCOURSE = re.compile(r"\b(?:firstly|secondly|thirdly|finally|furthermore|moreover|additionally|in addition|on the other hand|by contrast|for instance|for example|specifically|notably|importantly|significantly)\b", re.IGNORECASE)
_DEMONSTRATIVE = re.compile(r"\bThis (?:approach|method|technique|framework|model|study|analysis|result|finding|section|paper|work)\b", re.IGNORECASE)
_SCARE_QUOTES = re.compile(r'"[^"]{2,30}"')
_SENT_SPLIT = re.compile(r"[.!?]\s+")
_THE_THIS_START = re.compile(r"^(?:The |This )", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^#{3,}\s+|^\d+\.\d+\.\d+", re.MULTILINE)
_TRANSITION = re.compile(r"\b(?:having established|building on this|with this in mind|turning now to|it follows that|this brings us to)\b", re.IGNORECASE)

class NSSIDetector:
    id = "nssi"
    domain = "prose"
    compute_cost = "cheap"
    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        word_count = max(len(words), 1)
        formulaic_density = len(_FORMULAIC.findall(text)) / word_count * 100
        power_adj_saturation = len(_POWER_ADJ.findall(text)) / word_count * 100
        discourse_scaffolding = len(_DISCOURSE.findall(text)) / word_count * 100
        demonstrative_monotony = len(_DEMONSTRATIVE.findall(text)) / word_count * 100
        scare_quote_density = len(_SCARE_QUOTES.findall(text)) / word_count * 100
        transition_density = len(_TRANSITION.findall(text)) / word_count * 100
        sentences = _SENT_SPLIT.split(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        the_this_ratio = sum(1 for s in sentences if _THE_THIS_START.match(s)) / max(len(sentences), 1)
        sentence_start_monotony = max(0.0, the_this_ratio - 0.25)
        deep_sections = min(len(_SECTION_HEADER.findall(text)) / 3, 1.0)
        signal_strengths = [min(formulaic_density / 2.0, 1.0), min(power_adj_saturation / 3.0, 1.0),
            min(discourse_scaffolding / 5.0, 1.0), min(demonstrative_monotony / 2.0, 1.0),
            min(scare_quote_density / 1.5, 1.0), min(transition_density / 1.0, 1.0),
            min(sentence_start_monotony / 0.3, 1.0), deep_sections]
        active_signals = sum(1 for s in signal_strengths if s > 0.1)
        mean_strength = sum(signal_strengths) / len(signal_strengths)
        convergence = 1.0 + (active_signals / 8) ** 1.5
        raw_score = mean_strength * convergence
        p_llm = min(raw_score, 1.0)
        if p_llm >= 0.70 and active_signals >= 5: determination = "RED"
        elif p_llm >= 0.45 and active_signals >= 4: determination = "AMBER"
        elif p_llm >= 0.25 and active_signals >= 3: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(layer_id=self.id, domain=self.domain, raw_score=raw_score, p_llm=p_llm,
            confidence=min(0.35 + word_count / 1500, 0.85),
            signals={"formulaic_density": round(formulaic_density, 3), "power_adj_saturation": round(power_adj_saturation, 3),
                "discourse_scaffolding": round(discourse_scaffolding, 3), "demonstrative_monotony": round(demonstrative_monotony, 3),
                "scare_quote_density": round(scare_quote_density, 3), "transition_density": round(transition_density, 3),
                "sentence_start_monotony": round(sentence_start_monotony, 3), "deep_sections": round(deep_sections, 3),
                "n_signals_active": active_signals, "convergence_multiplier": round(convergence, 3)},
            determination=determination, attacker_tiers=["A0", "A1"], compute_cost=self.compute_cost, min_text_length=100)
    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = NSSIDetector()
