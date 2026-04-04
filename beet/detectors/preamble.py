import re
from beet.contracts import LayerResult
from beet.config import get_pattern_list

_SEVERITY_P_LLM = {"CRITICAL": 0.97, "HIGH": 0.82, "MEDIUM": 0.65, "LOW": 0.30, "NONE": 0.05}

def _compile_patterns(patterns: dict) -> list[tuple[str, re.Pattern, str]]:
    compiled = []
    for severity, items in patterns.items():
        for item in items:
            try:
                compiled.append((item["name"], re.compile(item["pattern"]), severity.upper()))
            except re.error:
                pass
    return compiled

class PreambleDetector:
    id = "preamble"
    domain = "universal"
    compute_cost = "trivial"

    def __init__(self):
        raw = get_pattern_list("preamble_patterns")
        self._patterns = _compile_patterns(raw)

    def analyze(self, text: str, config: dict) -> LayerResult:
        window = text[:500]
        matched_severity = "NONE"
        matched_names = []
        for name, pattern, severity in self._patterns:
            if pattern.search(window):
                matched_names.append(name)
                order = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
                if order.index(severity) > order.index(matched_severity):
                    matched_severity = severity
        p_llm = _SEVERITY_P_LLM[matched_severity]
        if matched_severity == "CRITICAL": determination = "RED"
        elif matched_severity == "HIGH": determination = "AMBER"
        elif matched_severity in ("MEDIUM", "LOW"): determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain, raw_score=float(len(matched_names)),
            p_llm=p_llm, confidence=0.95 if matched_severity != "NONE" else 0.70,
            signals={"severity": matched_severity, "matched_patterns": matched_names, "n_matches": len(matched_names)},
            determination=determination, attacker_tiers=["A0", "A1"],
            compute_cost=self.compute_cost, min_text_length=10,
        )

    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = PreambleDetector()
