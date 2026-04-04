from beet.contracts import FusionResult

_FEATURE_DESCRIPTIONS = {
    "binoculars_ratio": "Cross-entropy ratio between observer LMs",
    "late_volatility_ratio": "Surprisal variance drops in second half",
    "fingerprint_hits_per_1000": "Density of LLM-signature vocabulary",
    "ps_cfd": "Constraint frame density",
    "idi_score": "Instruction density index",
    "vs_spec_score": "Specification density",
    "nssi_n_signals_active": "N-gram self-similarity convergence",
    "preamble_severity": "Preamble severity",
    "tocsin_mean_deletion_impact": "Token cohesiveness",
}

def explain(fusion: FusionResult, occupation: str = "") -> str:
    lines = [f"P(LLM) = {fusion.p_llm:.2f}  CI: [{fusion.confidence_interval[0]:.2f}, {fusion.confidence_interval[1]:.2f}]",
        f"Prediction set: {{{', '.join(fusion.prediction_set)}}}", ""]
    if fusion.top_contributors:
        lines.append("Top contributing signals:")
        for i, (feat, contrib) in enumerate(fusion.top_contributors[:3], 1):
            desc = _FEATURE_DESCRIPTIONS.get(feat, feat)
            direction = "suspicious" if contrib > 0 else "reassuring"
            lines.append(f"  {i}. {desc}  ({direction}, contribution: {contrib:+.3f})")
    if occupation: lines.append(f"\nOccupation context: {occupation}")
    return "\n".join(lines)
