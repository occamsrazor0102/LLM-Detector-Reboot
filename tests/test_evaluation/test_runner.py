import pytest
from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.runner import run_eval, EvalReport


def _make_det(p_llm: float, label: str = "RED") -> Determination:
    return Determination(
        label=label,
        p_llm=p_llm,
        confidence_interval=(p_llm, p_llm),
        prediction_set=["LLM"],
        reason="mock",
        top_features=[],
        override_applied=False,
        detectors_run=[],
        cascade_phases=[1],
        mixed_report=None,
    )


class MockPipeline:
    def __init__(self, score_map: dict[str, float], config: dict | None = None, fail_on: set | None = None):
        self._scores = score_map
        self._config = config or {"profile": "mock"}
        self._fail_on = fail_on or set()

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        if text in self._fail_on:
            raise RuntimeError("boom")
        return _make_det(self._scores.get(text, 0.5))

    @property
    def config(self) -> dict:
        return self._config


@pytest.fixture
def tiny_dataset():
    return [
        EvalSample(id="h1", text="human-1", label=0, tier="human"),
        EvalSample(id="h2", text="human-2", label=0, tier="human"),
        EvalSample(id="l1", text="llm-1", label=1, tier="A0"),
        EvalSample(id="l2", text="llm-2", label=1, tier="A0"),
    ]


class TestRunEval:
    def test_report_structure(self, tiny_dataset):
        pipeline = MockPipeline({"human-1": 0.1, "human-2": 0.2, "llm-1": 0.8, "llm-2": 0.9})
        report = run_eval(pipeline, tiny_dataset)
        assert isinstance(report, EvalReport)
        assert report.n_samples == 4
        assert len(report.predictions) == 4
        assert {p["id"] for p in report.predictions} == {"h1", "h2", "l1", "l2"}
        for key in ("auroc", "ece", "brier", "tpr_at_fpr_01"):
            assert key in report.metrics
        assert report.metrics["auroc"] == pytest.approx(1.0)

    def test_per_tier_breakdown(self, tiny_dataset):
        pipeline = MockPipeline({"human-1": 0.1, "human-2": 0.2, "llm-1": 0.8, "llm-2": 0.9})
        report = run_eval(pipeline, tiny_dataset)
        assert set(report.per_tier.keys()) == {"human", "A0"}
        assert "auroc" in report.per_tier["human"]

    def test_config_hash_stable(self, tiny_dataset):
        p1 = MockPipeline({"human-1": 0.1, "human-2": 0.2, "llm-1": 0.8, "llm-2": 0.9})
        p2 = MockPipeline({"human-1": 0.1, "human-2": 0.2, "llm-1": 0.8, "llm-2": 0.9})
        r1 = run_eval(p1, tiny_dataset)
        r2 = run_eval(p2, tiny_dataset)
        assert r1.config_hash == r2.config_hash
        assert len(r1.config_hash) == 12

    def test_failed_samples_collected(self, tiny_dataset):
        pipeline = MockPipeline(
            {"human-1": 0.1, "llm-1": 0.8, "llm-2": 0.9},
            fail_on={"human-2"},
        )
        report = run_eval(pipeline, tiny_dataset)
        assert report.n_samples == 3
        assert len(report.failed_samples) == 1
        assert report.failed_samples[0]["id"] == "h2"
        assert "boom" in report.failed_samples[0]["error"]

    def test_prediction_fields(self, tiny_dataset):
        pipeline = MockPipeline({"human-1": 0.1, "human-2": 0.2, "llm-1": 0.8, "llm-2": 0.9})
        report = run_eval(pipeline, tiny_dataset)
        p = report.predictions[0]
        for k in ("id", "label", "p_llm", "determination", "tier"):
            assert k in p
