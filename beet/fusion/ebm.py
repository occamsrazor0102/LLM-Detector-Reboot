from beet.contracts import LayerResult, FusionResult

class NaiveFusion:
    def fuse(self, layer_results: list[LayerResult]) -> FusionResult:
        active = [r for r in layer_results if r.determination != "SKIP" and r.confidence > 0]
        if not active:
            return FusionResult(p_llm=0.5, confidence_interval=(0.25, 0.75),
                prediction_set=["YELLOW", "AMBER", "GREEN"], feature_contributions={}, top_contributors=[])
        total_weight = sum(r.confidence for r in active)
        p_llm = sum(r.p_llm * r.confidence for r in active) / total_weight
        p_values = [r.p_llm for r in active]
        spread = max(p_values) - min(p_values)
        half_width = max(0.05, spread / 2)
        ci = (max(0.0, p_llm - half_width), min(1.0, p_llm + half_width))
        contribs = {r.layer_id: r.p_llm for r in active}
        top = sorted(contribs.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:5]
        prediction_set = _p_llm_to_prediction_set(p_llm, half_width)
        return FusionResult(p_llm=p_llm, confidence_interval=ci, prediction_set=prediction_set,
            feature_contributions=contribs, top_contributors=top)

def _p_llm_to_prediction_set(p_llm: float, uncertainty: float) -> list[str]:
    labels = []
    bands = [("RED", 0.75, 1.01), ("AMBER", 0.50, 0.75), ("YELLOW", 0.25, 0.50), ("GREEN", 0.0, 0.25)]
    for label, lo, hi in bands:
        if not (p_llm + uncertainty < lo or p_llm - uncertainty >= hi):
            labels.append(label)
    return labels or ["UNCERTAIN"]

DEFAULT_FUSION = NaiveFusion()
