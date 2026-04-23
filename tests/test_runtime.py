from pathlib import Path

import pytest

from beet.config import list_profiles, load_config
from beet.pipeline import BeetPipeline
from beet.runtime import RuntimeContext


@pytest.fixture
def ctx():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "screening.yaml")
    return RuntimeContext(BeetPipeline(cfg), "screening", cfg)


def test_context_exposes_pipeline_profile_config(ctx):
    assert ctx.profile == "screening"
    assert ctx.config.get("decision") is not None
    assert ctx.pipeline is not None


def test_switch_profile_replaces_pipeline(ctx):
    original = ctx.pipeline
    ctx.switch_profile("strict")
    assert ctx.profile == "strict"
    assert ctx.pipeline is not original


def test_switch_profile_unknown_raises_and_preserves_state(ctx):
    before_pipeline = ctx.pipeline
    before_profile = ctx.profile
    with pytest.raises(FileNotFoundError):
        ctx.switch_profile("no-such-profile-xyz")
    assert ctx.pipeline is before_pipeline
    assert ctx.profile == before_profile


def test_resolve_profile_path_rejects_traversal():
    from beet.config import ConfigError, resolve_profile_path

    for bad in ["../evil", "../../etc/passwd", "a/b", "a\\b", "/abs", ""]:
        with pytest.raises(ConfigError):
            resolve_profile_path(bad)


def test_switch_profile_traversal_is_rejected(ctx):
    before_profile = ctx.profile
    with pytest.raises(Exception):
        ctx.switch_profile("../../../tmp/evil")
    assert ctx.profile == before_profile


def test_list_profiles_includes_repo_profiles():
    names = {p["name"] for p in list_profiles()}
    # sanity: at least default + screening + strict exist
    assert {"default", "screening", "strict"} <= names


def test_list_profiles_skips_underscore_files(tmp_path, monkeypatch):
    # list_profiles reads CONFIGS_DIR — verify the filter catches _private.yaml
    from beet import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path)
    (tmp_path / "public.yaml").write_text("_profile: public\ndetectors: {}\ncascade: {}\ndecision: {}\n")
    (tmp_path / "_private.yaml").write_text("_profile: private\n")
    names = {p["name"] for p in list_profiles()}
    assert "public" in names
    assert "private" not in names
