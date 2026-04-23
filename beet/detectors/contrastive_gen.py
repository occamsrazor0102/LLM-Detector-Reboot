"""Contrastive generation detector.

Theory: instead of asking "does this look like LLM text in general?" against
static archetypes, ask "does this look like what an LLM would generate for
*this* task?". Generate k baseline responses via an LLM provider for the
same task description, then measure how close the submission sits to those
baselines.

Implementation:
  * Generate `n_baselines` (default 3) responses via the configured
    provider, using only the task description (NEVER the submission).
  * Measure similarity between the submission and each baseline using
    either word-shingle Jaccard (default, no ML deps) or semantic
    embedding cosine similarity if `sentence-transformers` is available
    and `use_embeddings: true` is set in the config.
  * A high mean similarity (submission ≈ what the LLM would write for
    this task) is evidence of LLM authorship.

PRIVACY: only the task description is ever sent to the provider — the
submission text stays local. This invariant is enforced in code, not
convention.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from beet.contracts import LayerResult


# HEURISTIC mapping from mean baseline similarity → p(LLM). The similarity
# metric is in [0, 1]; a submission that's identical to every baseline
# scores near 1. Hand-picked table — replace via isotonic calibration once
# a labeled dataset exists.
_HEURISTIC_SIM_TO_P_LLM = [
    (0.00, 0.08), (0.10, 0.18), (0.20, 0.32),
    (0.35, 0.55), (0.50, 0.75), (0.70, 0.90), (1.00, 0.97),
]


def _interpolate(x: float, table: list[tuple[float, float]]) -> float:
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return table[-1][1]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(tokens: list[str], k: int = 4) -> set[tuple]:
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _resolve_api_key(config: dict, provider: str) -> str | None:
    key = config.get("api_key") or config.get(f"{provider}_api_key")
    if key:
        return key
    return os.environ.get(f"{provider.upper()}_API_KEY")


def _sanitize_provider_error(exc: Exception, api_key: str | None = None) -> str:
    msg = f"{type(exc).__name__}"
    detail = str(exc)
    if detail:
        if api_key and api_key in detail:
            detail = detail.replace(api_key, "[redacted]")
        detail = detail[:200]
        msg = f"{msg}: {detail}"
    return msg


def _task_description_overlaps_submission(task_description: str, submission: str) -> bool:
    """Privacy guard: reject when the caller passed the submission (or a
    near-duplicate of it) as the task description. Without this check,
    misuse would leak the submission text to the provider."""
    td = task_description.strip()
    sub = submission.strip()
    if not td or not sub:
        return False
    if td == sub:
        return True
    # Task descriptions are typically short (a sentence or two). If someone
    # passes something that's a large fraction of the submission length,
    # treat it as suspicious.
    if len(td) >= max(250, int(len(sub) * 0.5)):
        return True
    # Or if the full task description appears verbatim inside the
    # submission text (suggests the caller confused the fields).
    if len(td) > 40 and td in sub:
        return True
    return False


class ContrastiveGenDetector:
    id = "contrastive_gen"
    domain = "universal"
    compute_cost = "expensive"

    def __init__(self):
        # Cached sentence-transformers model, lazy-loaded on first embedding-
        # similarity call. ~80 MB on disk; reloading per analyze would be
        # gratuitous.
        self._st_model = None

    def analyze(self, text: str, config: dict) -> LayerResult:
        task_description = config.get("task_description")
        if not task_description:
            return _skip(self, {"skipped": "no_task_metadata",
                                "hint": "pass detectors.contrastive_gen.task_description (not the submission)"})
        # Privacy guard: refuse to proceed if the caller confused the
        # task_description and submission fields. Without this check, the
        # submission text would be sent to the provider.
        if _task_description_overlaps_submission(task_description, text):
            return _skip(self, {
                "skipped": "task_description_overlaps_submission",
                "hint": "task_description must be a distinct, short task summary — "
                        "never the submission text itself",
            })
        provider = config.get("provider")
        if not provider:
            return _skip(self, {"skipped": "no_provider_configured"})
        api_key = _resolve_api_key(config, provider)
        if not api_key:
            return _skip(self, {"skipped": "no_api_key",
                                "hint": f"set {provider.upper()}_API_KEY or detectors.contrastive_gen.api_key"})

        words = text.split()
        if len(words) < 50:
            return _skip(self, {"skipped": "too_short", "n_words": len(words)})

        n_baselines = int(config.get("n_baselines", 3))
        model = config.get("model", "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o-mini")

        baselines, errors = self._generate_baselines(
            task_description, api_key, provider, model, n_baselines
        )
        valid = [b for b in baselines if b.strip()]
        if not valid:
            return _skip(self, {"skipped": "provider_error", "errors": errors})

        use_embeddings = bool(config.get("use_embeddings", False))
        similarity_mode = "lexical"
        similarities: list[float]
        if use_embeddings:
            try:
                similarities = self._embedding_similarity(valid, text)
                similarity_mode = "embedding"
            except ImportError:
                similarities = self._lexical_similarities(text, valid)
                similarity_mode = "lexical_fallback"
            except Exception as e:
                similarities = self._lexical_similarities(text, valid)
                similarity_mode = f"lexical_fallback (embed error: {type(e).__name__})"
        else:
            similarities = self._lexical_similarities(text, valid)

        mean_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        p_llm = _interpolate(mean_sim, _HEURISTIC_SIM_TO_P_LLM)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        signals: dict = {
            "n_baselines_generated": len(valid),
            "similarity_mode": similarity_mode,
            "mean_similarity": round(mean_sim, 4),
            "max_similarity": round(max_sim, 4),
            "per_baseline_similarity": [round(s, 4) for s in similarities],
            "provider": provider,
            "model": model,
        }
        if errors:
            signals["provider_errors_partial"] = errors

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=mean_sim, p_llm=p_llm,
            confidence=min(0.4 + len(valid) / 10.0, 0.80),
            signals=signals,
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=50,
        )

    def _lexical_similarities(self, submission: str, baselines: Iterable[str]) -> list[float]:
        sub_shingles = _shingles(_tokenize(submission))
        return [_jaccard(sub_shingles, _shingles(_tokenize(b))) for b in baselines]

    def _embedding_similarity(self, texts: list[str], submission: str) -> list[float]:
        """Cosine similarity via sentence-transformers. Model is cached on
        the instance so we don't reload 80MB on every analyze."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        from sentence_transformers import util  # type: ignore
        sub_emb = self._st_model.encode(
            [submission], convert_to_tensor=True, normalize_embeddings=True
        )
        base_emb = self._st_model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.cos_sim(sub_emb, base_emb)[0].tolist()
        return [max(0.0, min(1.0, s)) for s in sims]

    def _generate_baselines(
        self, task_description: str, api_key: str, provider: str, model: str, n: int,
    ) -> tuple[list[str], list[str]]:
        """Generate `n` LLM responses to the task. Privacy: only the task
        description is sent — never the submission."""
        prompt = (
            "Write a realistic response to the following task. Output only the "
            "response, no preamble or meta-commentary:\n\n"
            f"{task_description}"
        )
        baselines: list[str] = []
        errors: list[str] = []
        if provider == "anthropic":
            try:
                import anthropic  # type: ignore
            except ImportError:
                return [], ["anthropic package not installed (pip install beet[tier3])"]
            try:
                client = anthropic.Anthropic(api_key=api_key)
                for i in range(n):
                    try:
                        msg = client.messages.create(
                            model=model, max_tokens=800,
                            temperature=0.7 + 0.05 * i,  # mild diversity across baselines
                            messages=[{"role": "user", "content": prompt}],
                        )
                        baselines.append(msg.content[0].text)
                    except Exception as e:
                        errors.append(f"baseline {i}: {_sanitize_provider_error(e, api_key)}")
            except Exception as e:
                errors.append(f"client init failed: {_sanitize_provider_error(e, api_key)}")
        elif provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                return [], ["openai package not installed (pip install beet[tier3])"]
            try:
                client = OpenAI(api_key=api_key)
                for i in range(n):
                    try:
                        resp = client.chat.completions.create(
                            model=model, max_tokens=800,
                            temperature=0.7 + 0.05 * i,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        baselines.append(resp.choices[0].message.content or "")
                    except Exception as e:
                        errors.append(f"baseline {i}: {_sanitize_provider_error(e, api_key)}")
            except Exception as e:
                errors.append(f"client init failed: {_sanitize_provider_error(e, api_key)}")
        else:
            errors.append(f"unknown provider {provider!r}")
        return baselines, errors

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3"],
        compute_cost=detector.compute_cost,
        min_text_length=50,
    )


DETECTOR = ContrastiveGenDetector()
