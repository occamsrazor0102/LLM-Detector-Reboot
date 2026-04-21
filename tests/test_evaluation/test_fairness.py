"""Fairness evaluation tests using a synthetic pipeline."""
import pytest

from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.fairness import run_fairness_eval


class BiasedPipeline:
    """Pipeline that over-predicts LLM for the 'clinical' source but not 'casual'."""

    def __init__(self):
        self._config = {}

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        p = 0.80 if "clinical" in text.lower() else 0.30
        return Determination(
            label="RED" if p > 0.75 else "YELLOW",
            p_llm=p,
            confidence_interval=(max(0, p - 0.1), min(1, p + 0.1)),
            prediction_set=["RED"] if p > 0.75 else ["YELLOW"],
            reason="fake",
            top_features=[],
            override_applied=False,
            detectors_run=["fake"],
            cascade_phases=[1],
            mixed_report=None,
        )


@pytest.fixture
def mixed_source_dataset():
    return [
        EvalSample(id=f"c{i}", text="clinical human note", label=0, source="clinical")
        for i in range(6)
    ] + [
        EvalSample(id=f"k{i}", text="casual human comment", label=0, source="casual")
        for i in range(6)
    ]


def test_fairness_reports_per_group_fpr(mixed_source_dataset):
    pipe = BiasedPipeline()
    report = run_fairness_eval(pipe, mixed_source_dataset, group_key="source", threshold=0.5)
    assert set(report.per_group_fpr.keys()) == {"clinical", "casual"}
    assert report.per_group_fpr["clinical"] > report.per_group_fpr["casual"]


def test_fairness_flags_parity_violation(mixed_source_dataset):
    pipe = BiasedPipeline()
    report = run_fairness_eval(
        pipe, mixed_source_dataset, group_key="source",
        threshold=0.5, max_fpr_ratio=1.1,
    )
    assert report.fpr_parity_ratio > 1.1
    assert report.flagged_disparities
    assert "parity" in report.flagged_disparities[0].lower()


def test_fairness_no_flag_when_within_ratio():
    # Uniform predictor → FPR parity ratio 1.0
    class Uniform:
        _config = {}

        def analyze(self, text, task_metadata=None):
            return Determination(
                label="GREEN", p_llm=0.1,
                confidence_interval=(0.0, 0.2), prediction_set=["GREEN"],
                reason="", top_features=[], override_applied=False,
                detectors_run=[], cascade_phases=[1], mixed_report=None,
            )

    data = [
        EvalSample(id=f"a{i}", text="x", label=0, source="A") for i in range(5)
    ] + [
        EvalSample(id=f"b{i}", text="x", label=0, source="B") for i in range(5)
    ]
    report = run_fairness_eval(Uniform(), data, group_key="source", threshold=0.5)
    assert not report.flagged_disparities


def test_fairness_handles_single_group():
    class Uniform:
        _config = {}

        def analyze(self, text, task_metadata=None):
            return Determination(
                label="GREEN", p_llm=0.1,
                confidence_interval=(0.0, 0.2), prediction_set=["GREEN"],
                reason="", top_features=[], override_applied=False,
                detectors_run=[], cascade_phases=[1], mixed_report=None,
            )

    data = [EvalSample(id=f"a{i}", text="x", label=0, source="only") for i in range(3)]
    report = run_fairness_eval(Uniform(), data, group_key="source")
    assert report.fpr_parity_ratio == 1.0
    assert not report.flagged_disparities
