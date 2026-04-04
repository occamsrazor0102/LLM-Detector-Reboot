from beet.contracts import LayerResult

PHASE1_DETECTORS = {"preamble", "fingerprint_vocab", "prompt_structure", "voice_spec", "instruction_density", "nssi"}
PHASE2_DETECTORS = {"surprisal_dynamics", "contrastive_lm", "token_cohesiveness"}
PHASE3_DETECTORS = {"perturbation", "contrastive_gen", "dna_gpt"}
PHASE4_DETECTORS = {"cross_similarity", "mixed_boundary"}

class CascadeScheduler:
    def __init__(self, config: dict):
        cascade = config.get("cascade", {})
        self._p1_high = cascade.get("phase1_short_circuit_high", 0.85)
        self._p1_low = cascade.get("phase1_short_circuit_low", 0.10)
        self._p2_high = cascade.get("phase2_short_circuit_high", 0.80)
        self._p2_low = cascade.get("phase2_short_circuit_low", 0.15)
        self._force_phase3 = cascade.get("phase3_always_run", False)

    def aggregate_p_llm(self, results: list[LayerResult]) -> float:
        active = [r for r in results if r.determination != "SKIP" and r.confidence > 0]
        if not active: return 0.5
        total_weight = sum(r.confidence for r in active)
        return sum(r.p_llm * r.confidence for r in active) / total_weight

    def should_run_phase2(self, phase1_results: list[LayerResult]) -> bool:
        agg = self.aggregate_p_llm(phase1_results)
        return self._p1_low < agg < self._p1_high

    def should_run_phase3(self, all_results: list[LayerResult]) -> bool:
        if self._force_phase3: return True
        agg = self.aggregate_p_llm(all_results)
        return self._p2_low < agg < self._p2_high

    def detectors_for_phase(self, phase: int) -> set[str]:
        return {1: PHASE1_DETECTORS, 2: PHASE2_DETECTORS, 3: PHASE3_DETECTORS, 4: PHASE4_DETECTORS}.get(phase, set())
