# beet/detectors/token_cohesiveness.py
"""
TOCSIN-inspired: LLM text is more semantically stable under token deletion.
"""
import random
from beet.contracts import LayerResult

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


class TokenCohesivenessDetector:
    id = "token_cohesiveness"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_samples: int = 30):
        if not _HAS_ST:
            raise ImportError("sentence-transformers required. Run: pip install 'beet[tier2]'")
        self._model = SentenceTransformer(model_name)
        self._n_samples = n_samples

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        if len(words) < 50:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "too_short"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=50,
            )

        orig_emb = self._model.encode(text, convert_to_numpy=True)
        n_samples = min(self._n_samples, len(words) // 3)
        sample_indices = random.sample(range(len(words)), n_samples)

        impacts = []
        for idx in sample_indices:
            reduced = " ".join(w for i, w in enumerate(words) if i != idx)
            reduced_emb = self._model.encode(reduced, convert_to_numpy=True)
            # Cosine distance
            cos_sim = float(np.dot(orig_emb, reduced_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(reduced_emb) + 1e-9
            ))
            impacts.append(1.0 - cos_sim)

        mean_impact = sum(impacts) / len(impacts)
        impact_variance = sum((x - mean_impact) ** 2 for x in impacts) / len(impacts)
        low_impact_ratio = sum(1 for x in impacts if x < 0.01) / len(impacts)

        # Low mean_impact → high semantic stability → LLM-like
        p_llm = max(0.0, min(1.0, (0.05 - mean_impact) / 0.05 * 0.8))

        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=mean_impact, p_llm=p_llm,
            confidence=min(0.40 + n_samples / 50, 0.85),
            signals={
                "mean_deletion_impact": round(mean_impact, 5),
                "deletion_impact_variance": round(impact_variance, 6),
                "low_impact_token_ratio": round(low_impact_ratio, 3),
                "n_samples": n_samples,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=50,
        )

    def calibrate(self, labeled_data: list) -> None: pass


if _HAS_ST:
    class _LazyTOCSIN:
        id = "token_cohesiveness"
        domain = "prose"
        compute_cost = "moderate"
        _instance = None

        def analyze(self, text: str, config: dict) -> LayerResult:
            if self._instance is None:
                self._instance = TokenCohesivenessDetector()
            return self._instance.analyze(text, config)

        def calibrate(self, labeled_data: list) -> None: pass

    DETECTOR = _LazyTOCSIN()
