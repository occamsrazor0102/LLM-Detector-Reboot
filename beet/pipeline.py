from pathlib import Path
from beet.config import load_config
from beet.contracts import Determination, LayerResult
from beet.normalizer import normalize_text
from beet.router import TextRouter
from beet.cascade import CascadeScheduler
from beet.fusion.ebm import NaiveFusion
from beet.decision import DecisionEngine
import beet.detectors as detector_registry

class BeetPipeline:
    def __init__(self, config: dict):
        self._config = config
        self._router = TextRouter(config)
        self._cascade = CascadeScheduler(config)
        self._fusion = NaiveFusion()
        self._decision = DecisionEngine(config)
        self._detectors = detector_registry.get_all_detectors()

    @classmethod
    def from_config_file(cls, path: Path | str) -> "BeetPipeline":
        return cls(load_config(Path(path)))

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        text = normalize_text(text)
        router_decision = self._router.route(text)
        cfg = self._config
        results: list[LayerResult] = []
        phase1_results = self._run_phase(1, text, cfg, router_decision.skip_detectors)
        results.extend(phase1_results)
        if self._cascade.should_run_phase2(phase1_results):
            phase2_results = self._run_phase(2, text, cfg, router_decision.skip_detectors)
            results.extend(phase2_results)
            if self._cascade.should_run_phase3(results):
                phase3_results = self._run_phase(3, text, cfg, router_decision.skip_detectors)
                results.extend(phase3_results)
        fusion_result = self._fusion.fuse(results)
        determination = self._decision.decide(fusion_result, results)
        determination.cascade_phases = [1]
        if len(results) > len(phase1_results):
            determination.cascade_phases.append(2)
        return determination

    def _run_phase(self, phase: int, text: str, config: dict, skip: list[str]) -> list[LayerResult]:
        phase_detector_ids = self._cascade.detectors_for_phase(phase)
        results = []
        detector_cfg = config.get("detectors", {})
        for det_id in phase_detector_ids:
            if det_id in skip: continue
            det_config = detector_cfg.get(det_id, {})
            if not det_config.get("enabled", True): continue
            detector = self._detectors.get(det_id)
            if detector is None: continue
            try:
                result = detector.analyze(text, det_config)
                results.append(result)
            except Exception:
                pass
        return results
