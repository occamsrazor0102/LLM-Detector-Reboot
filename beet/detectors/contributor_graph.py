# beet/detectors/contributor_graph.py
"""Contributor graph / syndicate detector (batch mode, periodic).

Builds a graph where nodes are contributors and edges weight content overlap
between their submissions. Community detection surfaces clusters that behave
as syndicates — contributors whose submissions resemble each other more than
baseline.

Intended to run nightly/weekly on aggregated submissions, not per-request.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from beet.contracts import LayerResult
from beet.detectors.cross_similarity import _jaccard, _shingles, _tokenize


class ContributorGraphDetector:
    id = "contributor_graph"
    domain = "universal"
    compute_cost = "expensive"

    def analyze(self, text: str, config: dict) -> LayerResult:
        return _skip(self, reason="batch_only_use_analyze_contributors")

    def analyze_contributors(
        self,
        submissions: dict[str, Iterable[str]],
        config: dict,
    ) -> dict[str, dict]:
        """Compute syndicate risk per contributor.

        submissions: {contributor_id: [text, ...]}
        Returns: {contributor_id: {"risk_score", "cluster", "signals"}}
        """
        if not submissions:
            return {}

        k = int(config.get("shingle_k", 5))
        pair_threshold = float(config.get("pair_threshold", 0.30))

        # Aggregate shingles per contributor (union across their submissions)
        contrib_shingles: dict[str, set] = {}
        contrib_counts: dict[str, int] = {}
        for cid, texts in submissions.items():
            pooled: set = set()
            n = 0
            for t in texts:
                pooled |= _shingles(_tokenize(t), k)
                n += 1
            contrib_shingles[cid] = pooled
            contrib_counts[cid] = n

        ids = list(contrib_shingles)
        # Pairwise Jaccard over pooled shingles
        edges: list[tuple[str, str, float]] = []
        neighbor_count: dict[str, int] = defaultdict(int)
        max_sim_per: dict[str, float] = defaultdict(float)
        sum_sim_per: dict[str, float] = defaultdict(float)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sim = _jaccard(contrib_shingles[a], contrib_shingles[b])
                max_sim_per[a] = max(max_sim_per[a], sim)
                max_sim_per[b] = max(max_sim_per[b], sim)
                sum_sim_per[a] += sim
                sum_sim_per[b] += sim
                if sim >= pair_threshold:
                    edges.append((a, b, sim))
                    neighbor_count[a] += 1
                    neighbor_count[b] += 1

        # Connected-component clustering on the thresholded graph
        cluster_of = _connected_components(ids, [(a, b) for a, b, _ in edges])

        # Aggregate cluster sizes
        cluster_size: dict[int, int] = defaultdict(int)
        for cid in ids:
            cluster_size[cluster_of[cid]] += 1

        out: dict[str, dict] = {}
        n = len(ids)
        for cid in ids:
            mean_sim = sum_sim_per[cid] / max(1, n - 1)
            degree = neighbor_count[cid]
            cluster = cluster_of[cid]
            cs = cluster_size[cluster]
            # Risk score: combines max similarity, degree share, and cluster membership.
            degree_share = degree / max(1, n - 1)
            cluster_bonus = 0.0 if cs <= 1 else min(0.25, 0.05 * (cs - 1))
            risk = min(1.0, 0.6 * max_sim_per[cid] + 0.3 * degree_share + cluster_bonus)
            out[cid] = {
                "risk_score": round(risk, 4),
                "cluster_id": cluster,
                "cluster_size": cs,
                "signals": {
                    "max_pair_jaccard": round(max_sim_per[cid], 4),
                    "mean_pair_jaccard": round(mean_sim, 4),
                    "degree": degree,
                    "n_submissions": contrib_counts[cid],
                },
            }
        return out

    def calibrate(self, labeled_data: list) -> None:
        pass


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    roots = {find(n): i for i, n in enumerate(nodes)}
    return {n: roots[find(n)] for n in nodes}


def _skip(detector, **signals) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals={"skipped": True, **signals},
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3", "A5"],
        compute_cost=detector.compute_cost, min_text_length=0,
    )


DETECTOR = ContributorGraphDetector()
