"""Multi-truncation DNA-GPT.

Truncate text at 30%, 50%, 70% and ask an LLM provider for a continuation
of each prefix. Score n-gram overlap between the LLM-generated continuation
and the actual continuation in the text. Texts written by the same family
of LLM tend to have HIGH overlap and an INCREASING trend (later prefixes
are easier to continue predictably); human text has lower overlap and a
flatter / random trend.

Requires a provider API key either in config or via the env var
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Without one, returns SKIP.
"""
from __future__ import annotations

import os

from beet.contracts import LayerResult


def _ngrams(text: str, n: int) -> list[tuple]:
    tokens = text.lower().split()
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_overlap(ref: str, hyp: str, n: int) -> float:
    """Precision of hyp n-grams in ref."""
    ref_ng = set(_ngrams(ref, n))
    hyp_ng = _ngrams(hyp, n)
    if not hyp_ng:
        return 0.0
    return sum(1 for g in hyp_ng if g in ref_ng) / len(hyp_ng)


def _bscore(ref_continuation: str, generated: str) -> float:
    """Weighted F1 of bigram/trigram/4-gram overlap."""
    weights = [(2, 0.4), (3, 0.35), (4, 0.25)]
    return sum(w * _ngram_overlap(ref_continuation, generated, n) for n, w in weights)


def _resolve_api_key(config: dict, provider: str) -> str | None:
    """Look up the provider API key: config → env var."""
    key = config.get("api_key") or config.get(f"{provider}_api_key")
    if key:
        return key
    env = os.environ.get(f"{provider.upper()}_API_KEY")
    return env or None


def _sanitize_provider_error(exc: Exception, api_key: str | None = None) -> str:
    """Trim exception strings so the error signal doesn't leak submission
    text (request bodies) or the API key into the logs / history / UI."""
    msg = f"{type(exc).__name__}"
    detail = str(exc)
    if detail:
        # Drop the key if the SDK embedded it in an error string.
        if api_key and api_key in detail:
            detail = detail.replace(api_key, "[redacted]")
        detail = detail[:200]
        msg = f"{msg}: {detail}"
    return msg


class DNAGPTDetector:
    id = "dna_gpt"
    domain = "prose"
    compute_cost = "expensive"

    def analyze(self, text: str, config: dict) -> LayerResult:
        provider = config.get("provider", "anthropic")
        model = config.get("model", "claude-haiku-4-5-20251001")
        api_key = _resolve_api_key(config, provider)

        if not api_key:
            return _skip(self, signals={"skipped": "no_api_key",
                                        "hint": f"set {provider.upper()}_API_KEY or detectors.dna_gpt.api_key"})

        words = text.split()
        if len(words) < 100:
            return _skip(self, signals={"skipped": "too_short", "n_words": len(words)})

        provider_errors: list[str] = []
        bscores: dict[float, float] = {}
        succeeded: dict[float, bool] = {}
        for frac in (0.30, 0.50, 0.70):
            cut = int(len(words) * frac)
            prefix = " ".join(words[:cut])
            actual_continuation = " ".join(words[cut:cut + 100])
            generated, err = self._generate_continuation(prefix, api_key, provider, model)
            if err:
                provider_errors.append(f"{frac}: {err}")
                succeeded[frac] = False
                bscores[frac] = 0.0
            else:
                succeeded[frac] = True
                bscores[frac] = _bscore(actual_continuation, generated) if generated else 0.0

        ok_fracs = [f for f, ok in succeeded.items() if ok]
        if not ok_fracs:
            return _skip(self, signals={"skipped": "provider_error", "errors": provider_errors})

        # Compute mean/trend only over successful calls. A partial outage
        # must not flip the verdict by treating a failed call as b=0.0.
        ok_bscores = [bscores[f] for f in ok_fracs]
        mean_bscore = sum(ok_bscores) / len(ok_bscores)
        # Trend requires both the first and last fractions to have succeeded;
        # otherwise it's not meaningful. Set to 0 (no bias) when incomplete.
        if succeeded[0.30] and succeeded[0.70]:
            bscore_trend = (bscores[0.70] - bscores[0.30]) / 2
        else:
            bscore_trend = 0.0
        bscore_variance = sum((b - mean_bscore) ** 2 for b in ok_bscores) / len(ok_bscores)
        b30, b50, b70 = bscores[0.30], bscores[0.50], bscores[0.70]

        # HEURISTIC scoring — combine mean overlap with increasing trend.
        # Tuned against the DNA-GPT paper's operating range, NOT empirically
        # calibrated to a labeled dataset.
        p_llm = min(mean_bscore * 3.0 + max(bscore_trend, 0.0) * 5.0, 1.0)

        if p_llm >= 0.60:
            determination = "RED"
        elif p_llm >= 0.36:
            determination = "AMBER"
        elif p_llm >= 0.18:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        signals: dict = {
            "bscore_30": round(b30, 4),
            "bscore_50": round(b50, 4),
            "bscore_70": round(b70, 4),
            "bscore_trend": round(bscore_trend, 4),
            "bscore_variance": round(bscore_variance, 6),
            "mean_bscore": round(mean_bscore, 4),
            "provider": provider,
            "model": model,
        }
        if provider_errors:
            signals["provider_errors_partial"] = provider_errors

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=mean_bscore, p_llm=p_llm,
            confidence=min(0.5 + len(words) / 600, 0.90),
            signals=signals,
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=100,
        )

    def _generate_continuation(
        self, prefix: str, api_key: str, provider: str, model: str
    ) -> tuple[str, str | None]:
        """Return (continuation_text, error_message). Errors surface rather than silently returning ''."""
        prompt = (
            "Continue the following text naturally for approximately 100 words. "
            "Output only the continuation, no preamble:\n\n"
            f"{prefix[-500:]}"
        )
        if provider == "anthropic":
            try:
                import anthropic  # type: ignore
            except ImportError:
                return "", "anthropic package not installed (pip install beet[tier3])"
            try:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text, None
            except Exception as e:
                return "", _sanitize_provider_error(e, api_key)
        if provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                return "", "openai package not installed (pip install beet[tier3])"
            try:
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or "", None
            except Exception as e:
                return "", _sanitize_provider_error(e, api_key)
        return "", f"unknown provider {provider!r}"

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
        compute_cost=detector.compute_cost, min_text_length=100,
    )


DETECTOR = DNAGPTDetector()
