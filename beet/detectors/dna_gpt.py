# beet/detectors/dna_gpt.py
"""
Multi-truncation DNA-GPT: truncate text at 30%, 50%, 70% and measure
n-gram overlap of LLM continuation vs actual text continuation.
"""
import re
from beet.contracts import LayerResult


def _ngram_overlap(ref: str, hyp: str, n: int) -> float:
    """Precision of hyp n-grams in ref."""
    def ngrams(text: str, n: int) -> list[tuple]:
        tokens = text.lower().split()
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ref_ng = set(ngrams(ref, n))
    hyp_ng = ngrams(hyp, n)
    if not hyp_ng: return 0.0
    return sum(1 for g in hyp_ng if g in ref_ng) / len(hyp_ng)


def _bscore(ref_continuation: str, generated: str) -> float:
    """Weighted F1 of bigram/trigram/4-gram overlap."""
    weights = [(2, 0.4), (3, 0.35), (4, 0.25)]
    return sum(w * _ngram_overlap(ref_continuation, generated, n) for n, w in weights)


class DNAGPTDetector:
    id = "dna_gpt"
    domain = "prose"
    compute_cost = "expensive"

    def analyze(self, text: str, config: dict) -> LayerResult:
        api_key = config.get("api_key") or config.get("anthropic_api_key")
        provider = config.get("provider", "anthropic")
        model = config.get("model", "claude-haiku-4-5-20251001")

        if not api_key:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "no_api_key"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=150,
            )

        words = text.split()
        if len(words) < 100:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "too_short"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=100,
            )

        bscores = {}
        for frac in [0.30, 0.50, 0.70]:
            cut = int(len(words) * frac)
            prefix = " ".join(words[:cut])
            actual_continuation = " ".join(words[cut:cut + 100])
            generated = self._generate_continuation(prefix, api_key, provider, model)
            bscores[frac] = _bscore(actual_continuation, generated)

        b30, b50, b70 = bscores[0.30], bscores[0.50], bscores[0.70]
        bscore_trend = (b70 - b30) / 2  # increasing trend → text is increasingly predictable
        bscore_variance = sum((b - (b30 + b50 + b70) / 3) ** 2 for b in [b30, b50, b70]) / 3

        # p_llm: LLM text has high bscores AND increasing trend
        mean_bscore = (b30 + b50 + b70) / 3
        p_llm = min(mean_bscore * 3.0 + max(bscore_trend, 0) * 5.0, 1.0)

        if p_llm >= 0.60: determination = "RED"
        elif p_llm >= 0.36: determination = "AMBER"
        elif p_llm >= 0.18: determination = "YELLOW"
        else: determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=mean_bscore, p_llm=p_llm,
            confidence=min(0.5 + len(words) / 600, 0.90),
            signals={
                "bscore_30": round(b30, 4),
                "bscore_50": round(b50, 4),
                "bscore_70": round(b70, 4),
                "bscore_trend": round(bscore_trend, 4),
                "bscore_variance": round(bscore_variance, 6),
                "mean_bscore": round(mean_bscore, 4),
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=100,
        )

    def _generate_continuation(self, prefix: str, api_key: str, provider: str, model: str) -> str:
        """Generate a ~100-word continuation of prefix using the specified provider."""
        prompt = (
            f"Continue the following text naturally for approximately 100 words. "
            f"Output only the continuation, no preamble:\n\n{prefix[-500:]}"
        )
        if provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}]
                )
                return msg.content[0].text
            except Exception:
                return ""
        elif provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content or ""
            except Exception:
                return ""
        return ""

    def calibrate(self, labeled_data: list) -> None: pass


DETECTOR = DNAGPTDetector()
