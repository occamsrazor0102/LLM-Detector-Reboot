# beet/detectors/mixed_boundary.py
"""
CUSUM-based mixed authorship boundary detection.
Applies to texts >= 300 words. Windowed feature extraction + changepoint detection.
"""
import math
import re
from beet.contracts import LayerResult

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FINGERPRINT_WORDS = re.compile(
    r"\b(?:ensure|comprehensive|robust|deliverable|must|shall|leverage|"
    r"utilize|facilitate|adhere|pursuant|holistic|streamline|pivotal)\b",
    re.IGNORECASE,
)
_VOICE_WORDS = re.compile(
    r"\b(?:okay|yeah|basically|honestly|gonna|wanna|kinda|i think|you know)\b",
    re.IGNORECASE,
)


def _sentence_features(sentences: list[str]) -> list[dict]:
    """Extract a feature vector per sentence for CUSUM analysis."""
    features = []
    for s in sentences:
        words = s.split()
        n = max(len(words), 1)
        fp = len(_FINGERPRINT_WORDS.findall(s)) / n * 100
        vc = len(_VOICE_WORDS.findall(s)) / n * 100
        avg_word_len = sum(len(w) for w in words) / n
        features.append({
            "fingerprint_density": fp,
            "voice_density": vc,
            "avg_word_len": avg_word_len,
            "sentence_len": n,
        })
    return features


def _cusum_changepoints(series: list[float], threshold: float = 3.0) -> list[int]:
    """Simple CUSUM: detect points where cumulative sum exceeds threshold."""
    if len(series) < 4:
        return []
    mean = sum(series) / len(series)
    std = math.sqrt(sum((x - mean) ** 2 for x in series) / len(series))
    if std < 1e-6:
        return []
    cusum_pos = 0.0
    cusum_neg = 0.0
    changepoints = []
    for i, x in enumerate(series):
        z = (x - mean) / std
        cusum_pos = max(0, cusum_pos + z - 0.5)
        cusum_neg = max(0, cusum_neg - z - 0.5)
        if cusum_pos > threshold or cusum_neg > threshold:
            changepoints.append(i)
            cusum_pos = 0.0
            cusum_neg = 0.0
    return changepoints


class MixedBoundaryDetector:
    id = "mixed_boundary"
    domain = "prose"
    compute_cost = "moderate"

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        if len(words) < 100:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "too_short"},
                determination="SKIP", attacker_tiers=["A4"],
                compute_cost=self.compute_cost, min_text_length=100,
            )

        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 10]
        if len(sentences) < 6:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "too_few_sentences"},
                determination="SKIP", attacker_tiers=["A4"],
                compute_cost=self.compute_cost, min_text_length=100,
            )

        sent_features = _sentence_features(sentences)
        fp_series = [f["fingerprint_density"] for f in sent_features]
        vc_series = [f["voice_density"] for f in sent_features]
        # Dissonance signal: diff between fp and vc per sentence
        dissonance_series = [f - v for f, v in zip(fp_series, vc_series)]

        changepoints = _cusum_changepoints(dissonance_series, threshold=2.5)
        n_boundaries = len(changepoints)

        # Per-segment determinations (simple: high fp → RED, high vc → GREEN)
        segment_dets = []
        boundaries = [0] + changepoints + [len(sentences)]
        for i in range(len(boundaries) - 1):
            seg = sentences[boundaries[i]:boundaries[i + 1]]
            if not seg: continue
            seg_fp = sum(f["fingerprint_density"] for f in sent_features[boundaries[i]:boundaries[i + 1]]) / max(len(seg), 1)
            seg_vc = sum(f["voice_density"] for f in sent_features[boundaries[i]:boundaries[i + 1]]) / max(len(seg), 1)
            if seg_fp > 3.0: segment_dets.append("AMBER")
            elif seg_vc > 2.0: segment_dets.append("GREEN")
            else: segment_dets.append("YELLOW")

        # Mixed probability: > 1 distinct segment determination
        unique_dets = set(segment_dets)
        mixed_probability = min(n_boundaries / 3, 1.0) if len(unique_dets) > 1 else 0.0
        max_divergence = max(fp_series) - min(fp_series) if fp_series else 0.0

        p_llm = mixed_probability * 0.6 + min(max_divergence / 10.0, 1.0) * 0.4

        if p_llm >= 0.60 and n_boundaries >= 1:
            determination = "AMBER"
        elif p_llm >= 0.30:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        boundary_positions = [cp / max(len(sentences), 1) for cp in changepoints]

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=mixed_probability, p_llm=p_llm,
            confidence=min(0.30 + len(words) / 1000, 0.80),
            signals={
                "n_boundaries": n_boundaries,
                "boundary_positions": boundary_positions,
                "segment_determinations": segment_dets,
                "mixed_probability": round(mixed_probability, 3),
                "max_segment_divergence": round(max_divergence, 3),
                "n_sentences": len(sentences),
            },
            determination=determination,
            attacker_tiers=["A4"],
            compute_cost=self.compute_cost,
            min_text_length=100,
        )

    def calibrate(self, labeled_data: list) -> None: pass


DETECTOR = MixedBoundaryDetector()
