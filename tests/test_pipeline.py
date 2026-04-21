import pytest
from beet.pipeline import BeetPipeline
from pathlib import Path
from tests.fixtures.llm_samples import A0_PREAMBLE, A0_CLINICAL_TASK, A1_CLEANED
from tests.fixtures.human_samples import CASUAL_SHORT, FORMAL_SOP

@pytest.fixture
def pipeline():
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    return BeetPipeline.from_config_file(config_path)

def test_a0_preamble_gives_red(pipeline):
    det = pipeline.analyze(A0_PREAMBLE)
    assert det.label == "RED"

def test_a0_clinical_task_gives_red_or_amber(pipeline):
    det = pipeline.analyze(A0_CLINICAL_TASK)
    assert det.label in ("RED", "AMBER", "UNCERTAIN")

def test_casual_human_gives_green_or_yellow(pipeline):
    det = pipeline.analyze(CASUAL_SHORT)
    assert det.label in ("GREEN", "YELLOW", "UNCERTAIN")

def test_determination_has_detectors_run(pipeline):
    det = pipeline.analyze(A0_CLINICAL_TASK)
    assert len(det.detectors_run) > 0

def test_p_llm_is_between_0_and_1(pipeline):
    det = pipeline.analyze(A1_CLEANED)
    assert 0.0 <= det.p_llm <= 1.0

def test_formal_sop_not_flagged_red(pipeline):
    det = pipeline.analyze(FORMAL_SOP)
    assert det.label != "RED"


def test_fusion_receives_word_count_and_domain(pipeline, monkeypatch):
    """Pipeline must forward router word_count/domain to fusion.fuse()."""
    captured = {}
    original_fuse = pipeline._fusion.fuse

    def spy_fuse(results, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return original_fuse(results, *args, **kwargs)

    monkeypatch.setattr(pipeline._fusion, "fuse", spy_fuse)
    pipeline.analyze(A0_CLINICAL_TASK)
    wc = captured["kwargs"].get("word_count")
    dom = captured["kwargs"].get("domain")
    assert wc is not None and wc > 0
    assert dom in ("prompt", "prose", "mixed", "insufficient")


def test_cascade_phase3_tracked_when_forced():
    """Phase 3 must appear in cascade_phases when forced via config."""
    from beet.config import load_config
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    cfg.setdefault("cascade", {})["phase3_always_run"] = True
    pipeline = BeetPipeline(cfg)
    det = pipeline.analyze(A0_CLINICAL_TASK)
    assert 1 in det.cascade_phases
    assert 3 in det.cascade_phases


def test_missing_detectors_api_exists():
    """Registry exposes get_missing_detectors() for diagnostics."""
    import beet.detectors as registry
    missing = registry.get_missing_detectors()
    assert isinstance(missing, dict)
