"""
Auto-discovery registry for BEET detectors.
Each module in this package should expose a DETECTOR module-level attribute
implementing the Detector protocol.
"""
import importlib
import pkgutil
from pathlib import Path
from beet.contracts import Detector

_registry: dict[str, Detector] = {}

def _discover() -> None:
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name == "__init__":
            continue
        try:
            mod = importlib.import_module(f"beet.detectors.{module_info.name}")
            if hasattr(mod, "DETECTOR"):
                d = mod.DETECTOR
                _registry[d.id] = d
        except ImportError:
            pass

def get_detector(detector_id: str) -> Detector | None:
    if not _registry:
        _discover()
    return _registry.get(detector_id)

def get_all_detectors() -> dict[str, Detector]:
    if not _registry:
        _discover()
    return dict(_registry)
