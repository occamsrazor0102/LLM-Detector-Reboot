import re
from beet.contracts import LayerResult

_IMPERATIVE_RE = re.compile(
    r"\b(?:calculate|analyze|evaluate|assess|create|design|write|provide|"
    r"list|identify|describe|outline|specify|ensure|include|incorporate|"
    r"demonstrate|implement|define|determine|review|compare|explain|"
    r"justify|document|perform|conduct|apply|use|select|choose)\b", re.IGNORECASE)
_CONDITIONAL_RE = re.compile(r"\b(?:if|when|where|unless|until|provided that|assuming that|given that)\b", re.IGNORECASE)
_BINARY_RE = re.compile(r"\b(?:yes or no|true or false|correct or incorrect|pass or fail)\b", re.IGNORECASE)
_IDI_TO_P_LLM = [(0.0, 0.05), (3.0, 0.20), (6.0, 0.40), (9.0, 0.60), (12.0, 0.75), (16.0, 0.87), (22.0, 0.93)]

def _interpolate(x, table):
    if x <= table[0][0]: return table[0][1]
    if x >= table[-1][0]: return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]; x1, y1 = table[i + 1]
        if x0 <= x <= x1: return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return table[-1][1]

class InstructionDensityDetector:
    id = "instruction_density"
    domain = "prompt"
    compute_cost = "cheap"
    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        word_count = max(len(words), 1)
        imperatives = len(_IMPERATIVE_RE.findall(text))
        conditionals = len(_CONDITIONAL_RE.findall(text))
        binary = len(_BINARY_RE.findall(text))
        idi = (imperatives + conditionals * 0.5 + binary * 2.0) / word_count * 100
        p_llm = _interpolate(idi, _IDI_TO_P_LLM)
        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(layer_id=self.id, domain=self.domain, raw_score=idi, p_llm=p_llm,
            confidence=min(0.4 + word_count / 1000, 0.88),
            signals={"idi": round(idi, 2), "imperatives": imperatives, "conditionals": conditionals, "binary_specs": binary},
            determination=determination, attacker_tiers=["A0", "A1", "A2"], compute_cost=self.compute_cost, min_text_length=30)
    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = InstructionDensityDetector()
