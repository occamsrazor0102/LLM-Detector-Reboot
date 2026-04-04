# beet/detectors/surprisal_dynamics.py
import math
from beet.contracts import LayerResult

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_token_surprisals(text: str, model, tokenizer, device: str) -> list[float]:
    """Returns per-token surprisal (negative log-prob) values."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token log-probs
        logits = outputs.logits[0, :-1]  # shift
        targets = inputs["input_ids"][0, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs[range(len(targets)), targets]
        return [-lp.item() for lp in token_lp]


def _windowed_variance(values: list[float], window_size: int) -> list[float]:
    """Returns variance in each sliding window."""
    if len(values) < window_size:
        return []
    variances = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        mean = sum(window) / window_size
        var = sum((x - mean) ** 2 for x in window) / window_size
        variances.append(var)
    return variances


class SurprisalDynamicsDetector:
    id = "surprisal_dynamics"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self, model_name: str = "gpt2"):
        if not _HAS_TORCH:
            raise ImportError("torch/transformers required. Run: pip install 'beet[tier2]'")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def analyze(self, text: str, config: dict) -> LayerResult:
        surprisals = _get_token_surprisals(text, self._model, self._tokenizer, self._device)
        if len(surprisals) < 20:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "insufficient_tokens"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=150,
            )

        n = len(surprisals)
        third = n // 3
        early = surprisals[:third]
        late = surprisals[2 * third:]

        # Key signals
        surprisal_mean = sum(surprisals) / n
        surprisal_std = math.sqrt(sum((x - surprisal_mean) ** 2 for x in surprisals) / n)
        early_var = sum((x - sum(early) / len(early)) ** 2 for x in early) / max(len(early), 1)
        late_var = sum((x - sum(late) / len(late)) ** 2 for x in late) / max(len(late), 1)
        late_volatility_ratio = late_var / max(early_var, 0.001)

        # Surprisal diversity: entropy of binned distribution
        bins = [0] * 20
        for s in surprisals:
            bin_idx = min(int(s / 1.5), 19)
            bins[bin_idx] += 1
        total = sum(bins)
        entropy = -sum((b / total) * math.log(b / total + 1e-9) for b in bins if b > 0)
        surprisal_diversity = entropy / math.log(20)  # normalize to 0–1

        # Windowed trajectory slope
        win_vars = _windowed_variance(surprisals, window_size=20)
        if len(win_vars) > 2:
            n_w = len(win_vars)
            x_mean = (n_w - 1) / 2
            slope = sum((i - x_mean) * (v - sum(win_vars) / n_w) for i, v in enumerate(win_vars))
            slope /= max(sum((i - x_mean) ** 2 for i in range(n_w)), 1e-9)
        else:
            slope = 0.0

        # Kurtosis excess
        mu = surprisal_mean
        m4 = sum((x - mu) ** 4 for x in surprisals) / n
        kurtosis_excess = m4 / max(surprisal_std ** 4, 1e-9) - 3

        # Combine into p_llm:
        # LLM text: low late_volatility_ratio (<0.7), high surprisal diversity, negative slope
        lv_score = max(0.0, (0.85 - late_volatility_ratio) / 0.85)   # 0 if ratio>0.85, 1 if ratio=0
        div_score = max(0.0, (0.65 - surprisal_diversity) / 0.65)    # lower diversity = more LLM-like
        slope_score = max(0.0, -slope * 5)                            # negative slope = LLM-like
        p_llm = min((lv_score * 0.5 + div_score * 0.3 + slope_score * 0.2), 1.0)

        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=late_volatility_ratio,
            p_llm=p_llm,
            confidence=min(0.40 + n / 500, 0.88),
            signals={
                "surprisal_mean": round(surprisal_mean, 3),
                "surprisal_std": round(surprisal_std, 3),
                "late_volatility_ratio": round(late_volatility_ratio, 3),
                "surprisal_diversity": round(surprisal_diversity, 3),
                "window_trajectory_slope": round(slope, 5),
                "kurtosis_excess": round(kurtosis_excess, 3),
                "n_tokens": n,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=150,
        )

    def calibrate(self, labeled_data: list) -> None: pass


if _HAS_TORCH:
    # Lazy instantiation: don't load model at import time
    class _LazyDetector:
        id = "surprisal_dynamics"
        domain = "prose"
        compute_cost = "moderate"
        _instance = None

        def analyze(self, text: str, config: dict) -> LayerResult:
            if self._instance is None:
                model_name = config.get("model_name", "gpt2")
                self._instance = SurprisalDynamicsDetector(model_name)
            return self._instance.analyze(text, config)

        def calibrate(self, labeled_data: list) -> None: pass

    DETECTOR = _LazyDetector()
