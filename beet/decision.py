from beet.contracts import LayerResult, FusionResult, Determination

class DecisionEngine:
    def __init__(self, config: dict):
        decision = config.get("decision", {})
        self._red = decision.get("red_threshold", 0.75)
        self._amber = decision.get("amber_threshold", 0.50)
        self._yellow = decision.get("yellow_threshold", 0.25)
        abstention = decision.get("abstention", {})
        self._abstain = abstention.get("enabled", True)
        self._max_pred_set = abstention.get("max_prediction_set_size", 3)

    def decide(self, fusion: FusionResult, layer_results: list[LayerResult]) -> Determination:
        mixed_report: dict | None = None
        for result in layer_results:
            if result.layer_id == "preamble" and result.signals.get("severity") == "CRITICAL":
                return Determination(label="RED", p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
                    prediction_set=["RED"], reason="Critical LLM preamble detected — raw assistant output present.",
                    top_features=list(fusion.top_contributors), override_applied=True,
                    detectors_run=[r.layer_id for r in layer_results], cascade_phases=[], mixed_report=None,
                    layer_results=list(layer_results),
                    feature_contributions=dict(fusion.feature_contributions),
                    conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
                    conformal_alpha=fusion.conformal_alpha,
                    fusion_mode=fusion.fusion_mode)
            if result.layer_id == "mixed_boundary":
                mixed_prob = float(result.signals.get("mixed_probability", 0.0) or 0.0)
                if mixed_prob >= 0.60 and result.determination != "SKIP":
                    mixed_report = {
                        "mixed_probability": mixed_prob,
                        "n_boundaries": result.signals.get("n_boundaries", 0),
                        "segment_determinations": result.signals.get("segment_determinations", []),
                    }
        if mixed_report is not None:
            return Determination(
                label="MIXED", p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
                prediction_set=["MIXED"],
                reason=(
                    f"Mixed-authorship signals: probability {mixed_report['mixed_probability']:.2f} "
                    f"across {mixed_report['n_boundaries']} style boundaries. "
                    "Some segments look human, others look LLM."
                ),
                top_features=list(fusion.top_contributors), override_applied=True,
                detectors_run=[r.layer_id for r in layer_results], cascade_phases=[], mixed_report=mixed_report,
                layer_results=list(layer_results),
                feature_contributions=dict(fusion.feature_contributions),
                conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
                conformal_alpha=fusion.conformal_alpha,
                fusion_mode=fusion.fusion_mode,
            )
        if self._abstain and len(fusion.prediction_set) >= self._max_pred_set:
            label = "UNCERTAIN"
            reason = f"Prediction set spans {len(fusion.prediction_set)} severity bands ({', '.join(fusion.prediction_set)}). Human review recommended."
        else:
            label = self._p_llm_to_label(fusion.p_llm)
            reason = self._generate_reason(label, fusion)
        return Determination(label=label, p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
            prediction_set=fusion.prediction_set, reason=reason, top_features=list(fusion.top_contributors),
            override_applied=False, detectors_run=[r.layer_id for r in layer_results], cascade_phases=[], mixed_report=None,
            layer_results=list(layer_results),
            feature_contributions=dict(fusion.feature_contributions),
            conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
            conformal_alpha=fusion.conformal_alpha,
            fusion_mode=fusion.fusion_mode)

    def _p_llm_to_label(self, p_llm: float) -> str:
        if p_llm >= self._red: return "RED"
        if p_llm >= self._amber: return "AMBER"
        if p_llm >= self._yellow: return "YELLOW"
        return "GREEN"

    def _generate_reason(self, label: str, fusion: FusionResult) -> str:
        top = fusion.top_contributors[:2]
        feat_str = "; ".join(f"{f}: {v:.2f}" for f, v in top) if top else "no dominant features"
        return f"{label} (P(LLM)={fusion.p_llm:.2f}, CI=[{fusion.confidence_interval[0]:.2f}, {fusion.confidence_interval[1]:.2f}]). Top: {feat_str}."
