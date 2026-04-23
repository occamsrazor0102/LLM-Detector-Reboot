"""Runtime context: holds the live pipeline, profile, and config with
thread-safe hot-swap for profile switching.

The swap strategy is optimistic: we build the new pipeline outside the
lock (so failures don't block in-flight RPCs), then briefly acquire the
lock only to reassign the three fields. If pipeline construction fails,
the old context is untouched.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from beet.config import load_config, resolve_profile_path

if TYPE_CHECKING:
    from beet.pipeline import BeetPipeline


class RuntimeContext:
    def __init__(self, pipeline: "BeetPipeline", profile: str | None, config: dict):
        self._lock = threading.Lock()
        self._pipeline = pipeline
        self._profile = profile
        self._config = config

    @property
    def pipeline(self) -> "BeetPipeline":
        with self._lock:
            return self._pipeline

    @property
    def profile(self) -> str | None:
        with self._lock:
            return self._profile

    @property
    def config(self) -> dict:
        with self._lock:
            return self._config

    def switch_profile(self, name: str) -> dict:
        """Load the named profile's config, build a fresh pipeline, swap.

        Returns the new profile summary. Raises on failure with the old
        context intact.
        """
        from beet.pipeline import BeetPipeline

        path = resolve_profile_path(name)
        if not Path(path).exists():
            raise FileNotFoundError(f"no config for profile '{name}' at {path}")
        new_config = load_config(path)
        new_pipeline = BeetPipeline(new_config)

        with self._lock:
            self._pipeline = new_pipeline
            self._profile = name
            self._config = new_config
        return {"profile": name, "path": str(path)}
