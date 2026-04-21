# beet/fusion/ebm.py
import math
from typing import Any
from beet.contracts import LayerResult, FusionResult


# --- Feature Assembler ---

_NAN = float("nan")

class FeatureAssembler:
    """Assembles layer results into a flat feature dict for the EBM."""

    def assemble(self, results: list[LayerResult], word_count: int, domain: str) -> dict:
        by_id = {r.layer_id: r for r in results if r.determination != "SKIP"}

        def p(layer_id: str) -> float:
            return by_id[layer_id].p_llm if layer_id in by_id else _NAN

        def sig(layer_id: str, key: str) -> float:
            if layer_id not in by_id: return _NAN
            v = by_id[layer_id].signals.get(key, _NAN)
            return float(v) if v is not None else _NAN

        def sig_str(layer_id: str, key: str, mapping: dict) -> float:
            if layer_id not in by_id: return _NAN
            v = by_id[layer_id].signals.get(key)
            return float(mapping.get(v, _NAN))

        severity_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        vec = {
            # Preamble
            "preamble_p_llm": p("preamble"),
            "preamble_severity": sig_str("preamble", "severity", severity_map),
            "preamble_n_matches": sig("preamble", "n_matches"),
            # Fingerprint vocab
            "fingerprint_p_llm": p("fingerprint_vocab"),
            "fingerprint_hits_per_1000": sig("fingerprint_vocab", "hits_per_1000"),
            "fingerprint_bigram_hits": sig("fingerprint_vocab", "bigram_hits"),
            # Prompt structure
            "ps_p_llm": p("prompt_structure"),
            "ps_cfd": sig("prompt_structure", "cfd"),
            "ps_distinct_frames": sig("prompt_structure", "distinct_frames"),
            "ps_mfsr": sig("prompt_structure", "mfsr"),
            "ps_framing_completeness": sig("prompt_structure", "framing_completeness"),
            "ps_meta_design_hits": sig("prompt_structure", "meta_design_hits"),
            # Voice-spec
            "vs_p_llm": p("voice_spec"),
            "vs_voice_score": sig("voice_spec", "voice_score"),
            "vs_spec_score": sig("voice_spec", "spec_score"),
            # IDI
            "idi_p_llm": p("instruction_density"),
            "idi_score": sig("instruction_density", "idi"),
            # NSSI
            "nssi_p_llm": p("nssi"),
            "nssi_n_signals_active": sig("nssi", "n_signals_active"),
            "nssi_formulaic_density": sig("nssi", "formulaic_density"),
            "nssi_discourse_scaffolding": sig("nssi", "discourse_scaffolding"),
            # Tier 2
            "surprisal_p_llm": p("surprisal_dynamics"),
            "surprisal_late_volatility": sig("surprisal_dynamics", "late_volatility_ratio"),
            "surprisal_diversity": sig("surprisal_dynamics", "surprisal_diversity"),
            "binoculars_p_llm": p("contrastive_lm"),
            "binoculars_ratio": sig("contrastive_lm", "binoculars_ratio"),
            "tocsin_p_llm": p("token_cohesiveness"),
            "tocsin_mean_deletion_impact": sig("token_cohesiveness", "mean_deletion_impact"),
            # Metadata
            "word_count": float(word_count),
            "domain_prompt": 1.0 if domain == "prompt" else 0.0,
            "domain_prose": 1.0 if domain == "prose" else 0.0,
        }
        return vec


# --- EBM Fusion ---

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_INTERPRET = True
except ImportError:
    _HAS_INTERPRET = False


class EBMFusion:
    """Fusion using a trained EBM. Falls back to naive fusion if untrained."""

    def __init__(self, model: Any = None, conformal: Any = None):
        self._model = model
        self._conformal = conformal
        self._assembler = FeatureAssembler()
        self._feature_names: list[str] | None = None
        if model is not None:
            names = getattr(model, "_beet_feature_names", None)
            if names:
                self._feature_names = list(names)

    def fuse(self, layer_results: list[LayerResult], word_count: int = 0, domain: str = "prose") -> FusionResult:
        if self._model is None:
            return self._naive_fuse(layer_results)
        return self._ebm_fuse(layer_results, word_count, domain)

    def _ebm_fuse(self, results: list[LayerResult], word_count: int, domain: str) -> FusionResult:
        import numpy as np
        vec = self._assembler.assemble(results, word_count, domain)
        features = self._feature_names or list(vec.keys())
        X = np.array([[vec.get(f, _NAN) for f in features]])
        # Replace NaN with column means (fitted during training) or 0.5
        X = np.where(np.isnan(X), 0.5, X)
        proba = self._model.predict_proba(X)[0]
        p_llm = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Per-feature contributions from EBM
        try:
            local = self._model.explain_local(X)
            contribs = dict(zip(features, local.data(0)["scores"]))
            top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        except Exception:
            contribs = {}
            top = []

        ci = (max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10))
        if self._conformal is not None:
            prediction_set = self._conformal.predict_set(p_llm)
        else:
            prediction_set = _p_llm_to_labels(p_llm, 0.10)
        return FusionResult(p_llm=p_llm, confidence_interval=ci,
                            prediction_set=prediction_set,
                            feature_contributions=contribs, top_contributors=top)

    def _naive_fuse(self, results: list[LayerResult]) -> FusionResult:
        """Fallback when no EBM is trained."""
        active = [r for r in results if r.determination != "SKIP" and r.confidence > 0]
        if not active:
            return FusionResult(p_llm=0.5, confidence_interval=(0.25, 0.75),
                                prediction_set=["YELLOW", "AMBER"],
                                feature_contributions={}, top_contributors=[])
        w_sum = sum(r.confidence for r in active)
        p_llm = sum(r.p_llm * r.confidence for r in active) / w_sum
        spread = max(r.p_llm for r in active) - min(r.p_llm for r in active)
        hw = max(0.05, spread / 2)
        ci = (max(0.0, p_llm - hw), min(1.0, p_llm + hw))
        contribs = {r.layer_id: (r.p_llm - 0.5) * r.confidence for r in active}
        top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        if self._conformal is not None:
            prediction_set = self._conformal.predict_set(p_llm)
        else:
            prediction_set = _p_llm_to_labels(p_llm, hw)
        return FusionResult(p_llm=p_llm, confidence_interval=ci,
                            prediction_set=prediction_set,
                            feature_contributions=contribs, top_contributors=top)


def _p_llm_to_labels(p_llm: float, uncertainty: float) -> list[str]:
    bands = [("RED", 0.75, 1.01), ("AMBER", 0.50, 0.75),
             ("YELLOW", 0.25, 0.50), ("GREEN", 0.0, 0.25)]
    labels = []
    for label, lo, hi in bands:
        if not (p_llm + uncertainty < lo or p_llm - uncertainty >= hi):
            labels.append(label)
    return labels or ["UNCERTAIN"]


# NaiveFusion alias preserved for backwards compatibility
class NaiveFusion(EBMFusion):
    pass


DEFAULT_FUSION = EBMFusion()
