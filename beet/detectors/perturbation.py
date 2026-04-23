"""Perturbation curvature (DetectGPT-style) detector.

Theory: LLM-generated text sits near local maxima of a language model's
log-probability surface — small edits reduce log-prob noticeably. Human
text is not constrained this way, so perturbation log-prob drops are
smaller and more uniform.

Implementation (simplified vs. the original DetectGPT paper):
  * Scoring LM: `distilgpt2` by default (configurable via `model`).
    Small, CPU-runnable, good enough for relative-score curvature even
    though it's not the model that generated the suspect text.
  * Perturbations: random word swaps with a shared vocabulary sampled
    from the input itself — avoids the separate T5 mask-fill model
    the original paper used, at the cost of less-natural perturbations.
    The signal is still the relative log-prob drop under noise.

Returns SKIP when torch/transformers are unavailable (tier2 extras not
installed) or when the input is too short. All model loading is lazy
and cached on the instance.

PRIVACY: model runs locally — no text leaves the process.
"""
from __future__ import annotations

import math
import random
from typing import Any

from beet.contracts import LayerResult

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Signal magnitude → p(LLM). Hand-picked, NOT calibrated to a labeled
# dataset. The DetectGPT paper reports normalised drops in roughly
# [0, 0.2] for LLM text and near 0 for human — this mapping approximates
# that operating range but will need isotonic calibration for real use.
_HEURISTIC_Z_TO_P_LLM = [
    (0.0, 0.10), (0.5, 0.25), (1.0, 0.45),
    (2.0, 0.65), (3.0, 0.80), (5.0, 0.92),
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


class PerturbationDetector:
    id = "perturbation"
    domain = "universal"
    compute_cost = "expensive"

    def __init__(self):
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_model_name: str | None = None

    def _ensure_loaded(self, model_name: str, device: str) -> None:
        if self._model is not None and self._loaded_model_name == model_name:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self._model.eval()
        self._loaded_model_name = model_name

    def _log_prob(self, text: str, device: str) -> float:
        """Mean per-token log-prob of `text` under the loaded LM."""
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = self._model(input_ids, labels=input_ids)
        # out.loss is mean NLL; log-prob = -NLL
        return float(-out.loss.item())

    def _perturb(self, text: str, rate: float, rng: random.Random) -> str:
        """Swap `rate` fraction of words with others drawn from the same text."""
        words = text.split()
        if len(words) < 10:
            return text
        n_swaps = max(1, int(len(words) * rate))
        vocab = list({w for w in words if w.isalpha() and len(w) > 2})
        if not vocab:
            return text
        out = list(words)
        positions = rng.sample(range(len(words)), min(n_swaps, len(words)))
        for pos in positions:
            replacement = rng.choice(vocab)
            out[pos] = replacement
        return " ".join(out)

    def analyze(self, text: str, config: dict) -> LayerResult:
        n_words = len(text.split())
        if n_words < 100:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})
        if not _HAS_TORCH:
            return _skip(self, {
                "skipped": "torch_unavailable",
                "hint": "pip install beet[tier2] to enable this detector",
            })

        model_name = config.get("model", "distilgpt2")
        n_perturbations = int(config.get("n_perturbations", 8))
        perturbation_rate = float(config.get("perturbation_rate", 0.12))
        seed = config.get("seed", 42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._ensure_loaded(model_name, device)
        except Exception as e:
            return _skip(self, {"skipped": "model_load_failed", "error": str(e)})

        rng = random.Random(seed)
        try:
            original_lp = self._log_prob(text, device)
            perturbed_lps = []
            for _ in range(n_perturbations):
                pert = self._perturb(text, perturbation_rate, rng)
                perturbed_lps.append(self._log_prob(pert, device))
        except Exception as e:
            return _skip(self, {"skipped": "scoring_failed", "error": str(e)})

        # Classic DetectGPT signal: normalized log-prob drop.
        mean_pert = sum(perturbed_lps) / len(perturbed_lps)
        var_pert = sum((p - mean_pert) ** 2 for p in perturbed_lps) / len(perturbed_lps)
        std_pert = math.sqrt(var_pert) if var_pert > 0 else 1e-6
        # Z-score of original vs perturbed distribution — positive means the
        # original is higher log-prob than its perturbations, consistent
        # with sitting near a local maximum (LLM-like).
        z = (original_lp - mean_pert) / std_pert
        raw = float(z)
        p_llm = _interpolate(max(z, 0.0), _HEURISTIC_Z_TO_P_LLM)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=raw, p_llm=p_llm,
            confidence=min(0.4 + n_words / 1000, 0.85),
            signals={
                "original_log_prob": round(original_lp, 4),
                "mean_perturbed_log_prob": round(mean_pert, 4),
                "std_perturbed_log_prob": round(std_pert, 4),
                "z_score": round(z, 4),
                "n_perturbations": n_perturbations,
                "perturbation_rate": perturbation_rate,
                "model": model_name,
                "device": device,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=100,
        )

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
        min_text_length=100,
    )


DETECTOR = PerturbationDetector()
