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
_missing: dict[str, str] = {}
_discovered = False


def _discover() -> None:
    global _discovered
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name == "__init__" or module_info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"beet.detectors.{module_info.name}")
            if hasattr(mod, "DETECTOR"):
                d = mod.DETECTOR
                _registry[d.id] = d
        except ImportError as e:
            _missing[module_info.name] = str(e)
    _discovered = True


def get_detector(detector_id: str) -> Detector | None:
    if not _discovered:
        _discover()
    return _registry.get(detector_id)


def get_all_detectors() -> dict[str, Detector]:
    if not _discovered:
        _discover()
    return dict(_registry)


def get_missing_detectors() -> dict[str, str]:
    if not _discovered:
        _discover()
    return dict(_missing)
