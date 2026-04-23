import datetime
from beet.contracts import Determination

def build_json_report(determination: Determination, submission_id: str = "") -> dict:
    return {
        "submission_id": submission_id, "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "determination": determination.label, "p_llm": round(determination.p_llm, 4),
        "confidence_interval": [round(x, 4) for x in determination.confidence_interval],
        "prediction_set": determination.prediction_set, "reason": determination.reason,
        "top_features": [{"feature": f, "contribution": round(c, 4)} for f, c in determination.top_features],
        "override_applied": determination.override_applied,
        "detectors_run": determination.detectors_run, "cascade_phases": determination.cascade_phases,
        "mixed_report": determination.mixed_report,
        "layer_results": [
            {
                "layer_id": lr.layer_id,
                "domain": lr.domain,
                "raw_score": round(float(lr.raw_score), 4),
                "p_llm": round(float(lr.p_llm), 4),
                "confidence": round(float(lr.confidence), 4),
                "determination": lr.determination,
                "signals": lr.signals,
                "compute_cost": lr.compute_cost,
                "spans": list(getattr(lr, "spans", None) or []),
            }
            for lr in determination.layer_results
        ],
        "feature_contributions": {
            k: round(float(v), 4) for k, v in determination.feature_contributions.items()
        },
    }

def build_text_report(determination: Determination) -> str:
    ci = determination.confidence_interval
    lines = [f"DETERMINATION: {determination.label}",
        f"P(LLM) = {determination.p_llm:.2f}  CI: [{ci[0]:.2f}, {ci[1]:.2f}]",
        f"Prediction set: {{{', '.join(determination.prediction_set)}}}", "",
        determination.reason, ""]
    if determination.top_features:
        lines.append("Top contributing signals:")
        for i, (feat, contrib) in enumerate(determination.top_features[:3], 1):
            lines.append(f"  {i}. {feat}: {contrib:+.3f}")
    lines.extend(["", f"Detectors run: {', '.join(determination.detectors_run)}"])
    if determination.override_applied:
        lines.append("NOTE: Override rule applied — preamble bypass.")
    return "\n".join(lines)

def build_csv_row(determination: Determination, submission_id: str = "") -> dict:
    return {"submission_id": submission_id, "determination": determination.label,
        "p_llm": determination.p_llm, "ci_lower": determination.confidence_interval[0],
        "ci_upper": determination.confidence_interval[1], "override_applied": determination.override_applied,
        "detectors_run": "|".join(determination.detectors_run)}
