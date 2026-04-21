# beet/detectors/cross_similarity.py
"""Cross-submission similarity detector (batch mode).

Computes pairwise similarity across a batch of submissions using word-shingle
Jaccard and (optionally) sentence-embedding cosine. For each submission,
reports its max similarity to any other submission in the batch. High values
indicate template matches or cross-submission content reuse — a signature of
syndicated LLM attacks.

Single-text mode: returns SKIP. Use `analyze_batch(texts, config)` instead.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from beet.contracts import LayerResult


_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _shingles(tokens: Iterable[str], k: int) -> set[tuple[str, ...]]:
    tokens = list(tokens)
    if len(tokens) < k:
        return set()
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class CrossSimilarityDetector:
    id = "cross_similarity"
    domain = "universal"
    compute_cost = "moderate"

    def analyze(self, text: str, config: dict) -> LayerResult:
        return _skip(self, reason="single_text_mode_use_analyze_batch")

    def analyze_batch(
        self, texts: dict[str, str], config: dict
    ) -> dict[str, LayerResult]:
        """Return per-submission LayerResult with cross-similarity signals.

        texts: {submission_id: text}
        """
        if not texts:
            return {}

        k = int(config.get("shingle_k", 5))
        threshold = float(config.get("jaccard_threshold", 0.40))

        tokens_by_id = {sid: _tokenize(t) for sid, t in texts.items()}
        shingles_by_id = {sid: _shingles(toks, k) for sid, toks in tokens_by_id.items()}
        ids = list(texts.keys())

        # Pairwise Jaccard; for large batches, LSH would replace this
        pairwise: dict[str, list[tuple[str, float]]] = {sid: [] for sid in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sim = _jaccard(shingles_by_id[a], shingles_by_id[b])
                pairwise[a].append((b, sim))
                pairwise[b].append((a, sim))

        out: dict[str, LayerResult] = {}
        for sid in ids:
            neighbors = pairwise[sid]
            if not neighbors:
                out[sid] = _skip(self, reason="singleton_batch")
                continue
            max_sim = max(s for _, s in neighbors)
            mean_sim = sum(s for _, s in neighbors) / len(neighbors)
            n_above = sum(1 for _, s in neighbors if s >= threshold)
            # Simple p_llm mapping: max_sim scaled into [0, 1] with threshold pivot
            # at 0.5. At threshold, p=0.5. At jaccard=1.0, p≈0.95.
            if max_sim >= threshold:
                p_llm = min(0.95, 0.5 + 0.5 * (max_sim - threshold) / max(1e-9, 1.0 - threshold))
                determination = "RED" if p_llm >= 0.75 else "AMBER"
            else:
                p_llm = max(0.05, 0.5 * (max_sim / threshold))
                determination = "YELLOW" if p_llm >= 0.25 else "GREEN"
            confidence = 0.4 + min(0.45, len(neighbors) / 100.0)
            out[sid] = LayerResult(
                layer_id=self.id,
                domain=self.domain,
                raw_score=max_sim,
                p_llm=p_llm,
                confidence=confidence,
                signals={
                    "max_jaccard": round(max_sim, 4),
                    "mean_jaccard": round(mean_sim, 4),
                    "n_above_threshold": n_above,
                    "batch_size": len(ids),
                },
                determination=determination,
                attacker_tiers=["A0", "A1", "A2", "A3", "A5"],
                compute_cost=self.compute_cost,
                min_text_length=0,
            )
        return out

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, **signals) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals={"skipped": True, **signals},
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3", "A5"],
        compute_cost=detector.compute_cost, min_text_length=0,
    )


DETECTOR = CrossSimilarityDetector()
