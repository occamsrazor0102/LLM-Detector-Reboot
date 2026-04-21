"""Robustness evaluation tests using a synthetic pipeline."""
import pytest

from beet.contracts import Determination
from beet.evaluation.dataset import EvalSample
from beet.evaluation.robustness import run_robustness_eval


class FakePipeline:
    """Returns a determination whose p_llm depends on whether text contains a fingerprint."""

    def __init__(self, drop_on_attack: float = 0.3):
        self._drop = drop_on_attack
        self._config = {"fake": True}

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        # Samples transformed by strip_preamble/synonym_swap will lose
        # "comprehensive" / "Certainly!" / etc. Detect simple fingerprint.
        has_fp = any(
            tok in text.lower()
            for tok in ("certainly!", "comprehensive", "furthermore", "utilize")
        )
        p = 0.95 if has_fp else 0.55
        return Determination(
            label="RED" if p > 0.75 else "AMBER",
            p_llm=p,
            confidence_interval=(max(0, p - 0.1), min(1, p + 0.1)),
            prediction_set=["RED"] if p > 0.75 else ["AMBER"],
            reason="fake",
            top_features=[],
            override_applied=False,
            detectors_run=["fake"],
            cascade_phases=[1],
            mixed_report=None,
        )


@pytest.fixture
def clean_dataset():
    return [
        EvalSample(id="h1", text="I wrote this myself with my own words.", label=0, tier="human"),
        EvalSample(id="h2", text="Another human-authored sentence, casual and plain.", label=0, tier="human"),
        EvalSample(id="l1", text="Certainly! Here is a comprehensive overview.", label=1, tier="A0"),
        EvalSample(id="l2", text="Furthermore, we must utilize the comprehensive framework.", label=1, tier="A0"),
    ]


def test_robustness_report_produces_baseline_and_per_attack(clean_dataset):
    pipe = FakePipeline()
    report = run_robustness_eval(pipe, clean_dataset, attacks=["strip_preamble", "synonym_swap"])
    assert report.baseline.n_samples == 4
    assert set(report.per_attack.keys()) == {"strip_preamble", "synonym_swap"}
    for name in ("strip_preamble", "synonym_swap"):
        assert name in report.attack_deltas


def test_robustness_ranking_sorted_by_auroc_delta(clean_dataset):
    pipe = FakePipeline()
    report = run_robustness_eval(pipe, clean_dataset, attacks=["strip_preamble", "synonym_swap"])
    # Most negative delta first
    deltas_in_rank_order = [d for _, d in report.vulnerability_ranking]
    assert deltas_in_rank_order == sorted(deltas_in_rank_order)


def test_robustness_handles_empty_attack_list(clean_dataset):
    pipe = FakePipeline()
    report = run_robustness_eval(pipe, clean_dataset, attacks=[])
    assert report.per_attack == {}
    assert report.vulnerability_ranking == []
    assert report.baseline.n_samples == 4
