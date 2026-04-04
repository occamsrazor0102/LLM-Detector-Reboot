# tests/conftest.py
import pytest
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

@pytest.fixture
def default_config_path():
    return CONFIGS_DIR / "default.yaml"

@pytest.fixture
def minimal_config():
    return {
        "detectors": {},
        "cascade": {
            "phase1_short_circuit_high": 0.85,
            "phase1_short_circuit_low": 0.10,
            "phase2_short_circuit_high": 0.80,
            "phase2_short_circuit_low": 0.15,
        },
        "decision": {
            "red_threshold": 0.75,
            "amber_threshold": 0.50,
            "yellow_threshold": 0.25,
        },
    }
