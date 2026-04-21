"""Unit tests for calibration infrastructure (Phase 1).

These tests use synthetic fixtures only; real dataset calibration is Phase 7.
"""
from pathlib import Path

import numpy as np
import pytest

from beet.calibration import DetectorCalibrator
from beet.fusion.conformal import ConformalWrapper
from beet.fusion.ebm import EBMFusion
from beet.contracts import LayerResult


def _layer(p, layer_id="preamble", conf=0.8):
    return LayerResult(
        layer_id=layer_id, domain="universal",
        raw_score=p, p_llm=p, confidence=conf, signals={},
        determination="AMBER", attacker_tiers=["A0"],
        compute_cost="cheap", min_text_length=0,
    )


def test_detector_calibrator_fit_transform_monotone():
    cal = DetectorCalibrator()
    xs = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    ys = [0, 0, 0, 1, 1, 1]
    cal.fit("preamble", xs, ys)
    assert cal.has("preamble")
    low = cal.transform("preamble", 0.15)
    high = cal.transform("preamble", 0.85)
    assert 0.0 <= low <= high <= 1.0


def test_detector_calibrator_passthrough_when_unfit():
    cal = DetectorCalibrator()
    assert cal.transform("unknown", 0.73) == pytest.approx(0.73)


def test_detector_calibrator_skips_degenerate_labels():
    cal = DetectorCalibrator()
    cal.fit("preamble", [0.1, 0.2, 0.3], [1, 1, 1])  # no variation
    assert not cal.has("preamble")


def test_detector_calibrator_save_load_roundtrip(tmp_path):
    cal = DetectorCalibrator()
    cal.fit("preamble", [0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1])
    cal.fit("nssi", [0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    p = tmp_path / "cal.json"
    cal.save(p)

    cal2 = DetectorCalibrator()
    cal2.load(p)
    for x in (0.15, 0.5, 0.85):
        assert cal2.transform("preamble", x) == pytest.approx(cal.transform("preamble", x))
        assert cal2.transform("nssi", x) == pytest.approx(cal.transform("nssi", x))


def test_conformal_wrapper_save_load_roundtrip(tmp_path):
    w = ConformalWrapper(alpha=0.1)
    scores = np.linspace(0.05, 0.95, 20)
    labels = (scores > 0.5).astype(int)
    w.calibrate(scores, labels)
    p = tmp_path / "conf.json"
    w.save(p)

    w2 = ConformalWrapper()
    w2.load(p)
    for s in (0.2, 0.5, 0.8):
        assert w2.predict_set(s) == w.predict_set(s)


def test_conformal_save_requires_calibration(tmp_path):
    w = ConformalWrapper()
    with pytest.raises(RuntimeError):
        w.save(tmp_path / "x.json")


def test_ebm_fusion_applies_conformal_in_naive_path():
    """When only conformal is provided (no trained EBM), prediction_set must
    come from the conformal wrapper, not the naive heuristic."""
    w = ConformalWrapper(alpha=0.1)
    scores = np.linspace(0.05, 0.95, 20)
    labels = (scores > 0.5).astype(int)
    w.calibrate(scores, labels)

    fusion_with = EBMFusion(conformal=w)
    fusion_without = EBMFusion()

    r_with = fusion_with.fuse([_layer(0.82)], word_count=100, domain="prose")
    r_without = fusion_without.fuse([_layer(0.82)], word_count=100, domain="prose")

    expected = w.predict_set(r_with.p_llm)
    assert r_with.prediction_set == expected
    # Sanity: the naive heuristic path yields a different (or at least
    # independent) prediction_set computation — they may coincide by chance
    # but the code path is distinct.
    assert fusion_with._conformal is w
    assert fusion_without._conformal is None


def test_pipeline_falls_back_to_naive_when_model_path_missing(tmp_path):
    from beet.config import load_config
    from beet.pipeline import BeetPipeline
    cfg = load_config(Path("configs/screening.yaml"))
    cfg["fusion"] = {"model_path": str(tmp_path / "does_not_exist.pkl")}
    pipeline = BeetPipeline(cfg)
    # DEFAULT_FUSION is the naive fallback (model is None)
    assert pipeline._fusion._model is None


def test_pipeline_analyze_detailed_returns_triple():
    from beet.pipeline import BeetPipeline
    from tests.fixtures.llm_samples import A0_CLINICAL_TASK
    pipeline = BeetPipeline.from_config_file("configs/screening.yaml")
    det, results, rd = pipeline.analyze_detailed(A0_CLINICAL_TASK)
    assert det.p_llm is not None
    assert isinstance(results, list) and len(results) > 0
    assert rd.word_count > 0
