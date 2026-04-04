import re
from beet.contracts import LayerResult
from beet.config import get_pattern_list

_TEMPLATE_SECTIONS = re.compile(
    r"\b(?:context|role|task|constraint|output\s+format|evaluation\s+criteria|"
    r"objective|deliverable|background|instruction|requirement)\b", re.IGNORECASE)
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_BOLD_HEADER = re.compile(r"\*{1,2}[A-Z][^*\n]{2,40}\*{1,2}\s*[:：]?")

def _calc_cfd(text: str, word_count: int, frame_patterns: list) -> tuple[float, int, float]:
    total_hits = 0
    distinct = set()
    for item in frame_patterns:
        pattern = re.compile(item["pattern"], re.IGNORECASE)
        hits = len(pattern.findall(text))
        if hits > 0:
            distinct.add(item["name"])
            total_hits += hits * item.get("weight", 1.0)
    cfd = total_hits / max(word_count, 1) * 100
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences: return cfd, len(distinct), 0.0
    multi_frame_sentences = sum(1 for s in sentences if sum(1 for item in frame_patterns if re.search(item["pattern"], s, re.IGNORECASE)) >= 2)
    return cfd, len(distinct), multi_frame_sentences / len(sentences)

class PromptStructureDetector:
    id = "prompt_structure"
    domain = "prompt"
    compute_cost = "cheap"

    def __init__(self):
        data = get_pattern_list("constraint_frames")
        self._frames = data.get("frames", [])
        self._meta_patterns = [re.compile(p, re.IGNORECASE) for p in data.get("meta_design_patterns", [])]

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        word_count = max(len(words), 1)
        cfd, distinct_frames, mfsr = _calc_cfd(text, word_count, self._frames)
        template_hits = len(_TEMPLATE_SECTIONS.findall(text))
        framing_completeness = min(template_hits, 6)
        meta_hits = sum(1 for p in self._meta_patterns if p.search(text))
        bullet_density = (len(_BULLET_LINE.findall(text)) + len(_NUMBERED_LINE.findall(text))) / word_count * 100
        bold_headers = len(_BOLD_HEADER.findall(text))
        composite = (min(cfd / 10.0, 1.0) * 0.25 + mfsr * 0.10 +
                     min(framing_completeness / 5, 1.0) * 0.35 +
                     min(meta_hits / 2, 1.0) * 0.20 + min(bold_headers / 4, 1.0) * 0.10)
        p_llm = min(composite, 1.0)
        if p_llm >= 0.72: determination = "RED"
        elif p_llm >= 0.45: determination = "AMBER"
        elif p_llm >= 0.22: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain, raw_score=cfd, p_llm=p_llm,
            confidence=min(0.4 + word_count / 1000, 0.90),
            signals={"cfd": round(cfd, 3), "distinct_frames": distinct_frames, "mfsr": round(mfsr, 3),
                     "framing_completeness": framing_completeness, "meta_design_hits": meta_hits,
                     "bullet_density": round(bullet_density, 3), "bold_headers": bold_headers, "composite": round(composite, 3)},
            determination=determination, attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost, min_text_length=30,
        )

    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = PromptStructureDetector()
