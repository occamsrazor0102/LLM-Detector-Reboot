# beet/detectors/contrastive_lm.py
"""
Binoculars-style detector: cross-entropy ratio between two LMs.
Reference: Hans et al. ICML 2024 — "Spotting LLMs With Binoculars"
"""
import math
from beet.contracts import LayerResult

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _cross_entropy(text: str, model, tokenizer, device: str) -> float:
    """Returns mean cross-entropy (bits per token) of text under model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs.loss is mean cross-entropy loss (nats), convert to bits
    return outputs.loss.item() / math.log(2)


class ContrastiveLMDetector:
    id = "contrastive_lm"
    domain = "universal"
    compute_cost = "moderate"

    def __init__(
        self,
        model_a: str = "gpt2",
        model_b: str = "gpt2-medium",
    ):
        if not _HAS_TORCH:
            raise ImportError("torch/transformers required. Run: pip install 'beet[tier2]'")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._tokenizer_a = AutoTokenizer.from_pretrained(model_a)
        self._model_a = AutoModelForCausalLM.from_pretrained(model_a).to(device)
        self._model_a.eval()
        self._tokenizer_b = AutoTokenizer.from_pretrained(model_b)
        self._model_b = AutoModelForCausalLM.from_pretrained(model_b).to(device)
        self._model_b.eval()
        # Empirical calibration: human ratio ~1.0, LLM ratio <0.95 or >1.05
        # These are initial estimates — use calibrate() to refine.
        self._human_mean = 1.0
        self._human_std = 0.08

    def analyze(self, text: str, config: dict) -> LayerResult:
        if len(text.split()) < 50:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "too_short"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=50,
            )

        ce_a = _cross_entropy(text, self._model_a, self._tokenizer_a, self._device)
        ce_b = _cross_entropy(text, self._model_b, self._tokenizer_b, self._device)
        ratio = ce_a / max(ce_b, 1e-9)

        # z-score against human baseline
        z = abs(ratio - self._human_mean) / max(self._human_std, 0.001)
        # High z = text is far from the human distribution → suspicious
        p_llm = min(z / 6.0, 1.0)

        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=ratio,
            p_llm=p_llm,
            confidence=min(0.50 + len(text.split()) / 500, 0.92),
            signals={
                "binoculars_ratio": round(ratio, 4),
                "ce_model_a": round(ce_a, 4),
                "ce_model_b": round(ce_b, 4),
                "z_score": round(z, 3),
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=50,
        )

    def calibrate(self, labeled_data: list) -> None:
        """Fit human_mean/human_std from labeled data."""
        import statistics
        human_ratios = [item["ratio"] for item in labeled_data if item["label"] == 0]
        if human_ratios:
            self._human_mean = statistics.mean(human_ratios)
            self._human_std = statistics.stdev(human_ratios) if len(human_ratios) > 1 else 0.08


if _HAS_TORCH:
    class _LazyContrastiveLM:
        id = "contrastive_lm"
        domain = "universal"
        compute_cost = "moderate"
        _instance = None

        def analyze(self, text: str, config: dict) -> LayerResult:
            if self._instance is None:
                self._instance = ContrastiveLMDetector(
                    model_a=config.get("model_a", "gpt2"),
                    model_b=config.get("model_b", "gpt2-medium"),
                )
            return self._instance.analyze(text, config)

        def calibrate(self, labeled_data: list) -> None: pass

    DETECTOR = _LazyContrastiveLM()
