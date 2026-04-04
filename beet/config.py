# beet/config.py
from pathlib import Path
import yaml
import copy

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PATTERNS_DIR = CONFIGS_DIR / "patterns"

REQUIRED_KEYS = {"detectors", "cascade", "decision"}

class ConfigError(Exception):
    pass

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key.startswith("_"):
            continue  # skip meta-keys
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result

def load_config(path: Path | str) -> dict:
    path = Path(path)
    raw = _load_yaml(path)

    # Handle _extends
    if "_extends" in raw:
        base_name = raw["_extends"]
        base_path = CONFIGS_DIR / f"{base_name}.yaml"
        base = load_config(base_path)
        cfg = _deep_merge(base, raw)
    else:
        cfg = raw

    missing = REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ConfigError(f"Config missing required keys: {missing}")

    return cfg

def get_pattern_list(name: str) -> list | dict:
    """Load a pattern file from configs/patterns/."""
    path = PATTERNS_DIR / f"{name}.yaml"
    data = _load_yaml(path)
    # Return the most useful top-level structure
    if "words" in data:
        return data["words"]
    return data
