import pytest
from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.ablation import run_ablation, AblationReport


def _det(p):
    return Determination(
        label="RED" if p >= 0.5 else "GREEN",
        p_llm=p,
        confidence_interval=(p, p),
        prediction_set=["LLM" if p >= 0.5 else "HUMAN"],
        reason="mock",
        top_features=[],
        override_applied=False,
        detectors_run=[],
        cascade_phases=[1],
        mixed_report=None,
    )


class MockPipeline:
    """Pipeline whose output depends on which detectors are enabled.
    Disabling detector_a halves the score; disabling detector_b has no effect.
    """
    def __init__(self, config: dict):
        self._config = config
        dets = config.get("detectors", {})
        self._a_on = dets.get("detector_a", {}).get("enabled", True)
        self._b_on = dets.get("detector_b", {}).get("enabled", True)

    @property
    def config(self) -> dict:
        return self._config

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        # Humans: low score; LLMs: high when detector_a is on, mid otherwise.
        is_llm = text.startswith("llm")
        if is_llm:
            score = 0.9 if self._a_on else 0.55
        else:
            score = 0.1
        # detector_b is a no-op — ablating it should produce ~0 delta
        return _det(score)


def _build_mock_pipeline(config: dict):
    return MockPipeline(config)


@pytest.fixture
def base_config():
    return {
        "detectors": {
            "detector_a": {"enabled": True},
            "detector_b": {"enabled": True},
        }
    }


@pytest.fixture
def dataset():
    return [
        EvalSample(id="h1", text="human-1", label=0, tier="human"),
        EvalSample(id="h2", text="human-2", label=0, tier="human"),
        EvalSample(id="l1", text="llm-1", label=1, tier="A0"),
        EvalSample(id="l2", text="llm-2", label=1, tier="A0"),
    ]


class TestRunAblation:
    def test_report_structure(self, base_config, dataset):
        report = run_ablation(base_config, dataset, pipeline_factory=_build_mock_pipeline)
        assert isinstance(report, AblationReport)
        assert "detector_a" in report.per_detector
        assert "detector_b" in report.per_detector
        assert set(report.deltas.keys()) == {"detector_a", "detector_b"}

    def test_ranked_sorted_descending(self, base_config, dataset):
        report = run_ablation(base_config, dataset, pipeline_factory=_build_mock_pipeline)
        deltas = [abs_delta for _, abs_delta in report.ranked]
        assert deltas == sorted(deltas, reverse=True)

    def test_load_bearing_detector_ranked_first(self, base_config, dataset):
        report = run_ablation(base_config, dataset, pipeline_factory=_build_mock_pipeline)
        assert report.ranked[0][0] == "detector_a"

    def test_explicit_detectors_filter(self, base_config, dataset):
        report = run_ablation(
            base_config, dataset,
            pipeline_factory=_build_mock_pipeline,
            detectors=["detector_a"],
        )
        assert set(report.per_detector.keys()) == {"detector_a"}
