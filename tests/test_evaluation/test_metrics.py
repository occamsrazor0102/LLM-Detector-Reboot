import math
import pytest
from beet.evaluation.metrics import (
    auroc,
    ece,
    brier,
    tpr_at_fpr,
    confusion_at_threshold,
    per_tier_breakdown,
    summarize,
)


class TestAuroc:
    def test_perfect_ranking(self):
        assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_inverted_ranking(self):
        assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_random_ranking(self):
        val = auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert val == pytest.approx(0.5)

    def test_all_same_label_returns_nan(self):
        assert math.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))
        assert math.isnan(auroc([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_empty_returns_nan(self):
        assert math.isnan(auroc([], []))


class TestEce:
    def test_perfect_calibration(self):
        # label matches score exactly at 0 and 1
        y_true = [0, 0, 1, 1]
        y_score = [0.0, 0.0, 1.0, 1.0]
        assert ece(y_true, y_score, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_worst_calibration(self):
        y_true = [0, 0, 1, 1]
        y_score = [1.0, 1.0, 0.0, 0.0]
        assert ece(y_true, y_score, n_bins=10) == pytest.approx(1.0, abs=1e-9)

    def test_empty_returns_nan(self):
        assert math.isnan(ece([], []))


class TestBrier:
    def test_perfect(self):
        assert brier([0, 1], [0.0, 1.0]) == pytest.approx(0.0)

    def test_worst(self):
        assert brier([0, 1], [1.0, 0.0]) == pytest.approx(1.0)

    def test_known_value(self):
        # two samples, err = 0.2 and 0.3 -> mean squared = (0.04+0.09)/2 = 0.065
        assert brier([0, 1], [0.2, 0.7]) == pytest.approx(0.065)

    def test_empty_returns_nan(self):
        assert math.isnan(brier([], []))


class TestTprAtFpr:
    def test_perfect_separation(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert tpr_at_fpr(y_true, y_score, target_fpr=0.01) == pytest.approx(1.0)

    def test_all_same_label_returns_nan(self):
        assert math.isnan(tpr_at_fpr([1, 1, 1], [0.1, 0.2, 0.3]))


class TestConfusionAtThreshold:
    def test_basic(self):
        # threshold = 0.5
        # sample 0: label 0, score 0.2 -> TN
        # sample 1: label 0, score 0.7 -> FP
        # sample 2: label 1, score 0.8 -> TP
        # sample 3: label 1, score 0.3 -> FN
        r = confusion_at_threshold([0, 0, 1, 1], [0.2, 0.7, 0.8, 0.3], threshold=0.5)
        assert r == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


class TestPerTierBreakdown:
    def test_groups_and_computes(self):
        samples = [
            {"id": "a", "tier": "A0", "label": 1},
            {"id": "b", "tier": "A0", "label": 0},
            {"id": "c", "tier": "human", "label": 0},
            {"id": "d", "tier": "human", "label": 1},
        ]
        preds = [
            {"id": "a", "p_llm": 0.9},
            {"id": "b", "p_llm": 0.1},
            {"id": "c", "p_llm": 0.2},
            {"id": "d", "p_llm": 0.8},
        ]
        result = per_tier_breakdown(samples, preds, auroc)
        assert set(result.keys()) == {"A0", "human"}
        assert result["A0"] == pytest.approx(1.0)
        assert result["human"] == pytest.approx(1.0)


class TestSummarize:
    def test_returns_all_keys(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        out = summarize(y_true, y_score)
        for k in ("auroc", "ece", "brier", "tpr_at_fpr_01"):
            assert k in out
        assert out["auroc"] == pytest.approx(1.0)
