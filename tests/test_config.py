# tests/test_config.py
import pytest
from beet.config import load_config, get_pattern_list, ConfigError

def test_load_default_config(default_config_path):
    cfg = load_config(default_config_path)
    assert "detectors" in cfg
    assert "cascade" in cfg
    assert "decision" in cfg

def test_load_strict_profile(tmp_path):
    from pathlib import Path
    strict = Path(__file__).parent.parent / "configs" / "strict.yaml"
    cfg = load_config(strict)
    assert cfg["decision"]["red_threshold"] < 0.75  # strict = lower threshold

def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detectors: {}\n")
    with pytest.raises(ConfigError):
        load_config(bad)

def test_get_pattern_list_returns_strings():
    patterns = get_pattern_list("fingerprint_words")
    assert isinstance(patterns, list)
    assert "delve" in patterns
    assert "utilize" in patterns

def test_get_pattern_list_preamble():
    patterns = get_pattern_list("preamble_patterns")
    assert isinstance(patterns, dict)
    assert "critical" in patterns
