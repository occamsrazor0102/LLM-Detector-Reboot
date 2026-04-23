# Profiles & Settings — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 3 of 6

## Goal

Make the active detector profile a first-class runtime concern. Let the user
see which profile is loaded, what thresholds and detectors it implies, and
switch to another profile without restarting the sidecar.

## Non-Goals

- Edit-in-place config authoring (users still edit YAML on disk for
  persistent changes — the UI is read-only).
- Per-submission profile override.
- Detector toggle UI (that's really a config edit — out of scope here).

## Load-Bearing Concern: Runtime Swap Safety

The current HTTP server is single-threaded (stdlib `HTTPServer`) and the
sidecar is strictly sequential over stdio, so concurrent requests during a
swap aren't an issue today. But a swap must still be atomic at the object
reference level: callers that snapshot `ctx.pipeline` must get either the
old or new pipeline, never a half-constructed one. A lock around the swap
suffices.

## Architecture

### `beet/runtime.py` (new)

```python
class RuntimeContext:
    """Thread-safe holder for the active pipeline + profile + config."""
    def __init__(self, pipeline, profile, config): ...

    @property
    def pipeline(self) -> BeetPipeline: ...
    @property
    def profile(self) -> str | None: ...
    @property
    def config(self) -> dict: ...

    def switch_profile(self, name: str) -> dict:
        """Load new config, build new pipeline, swap under lock.
        Raises ConfigError/FileNotFoundError on failure — old pipeline stays."""
```

Swap is optimistic: we build the new pipeline *outside* the lock (so
failures don't hold up in-flight RPCs), then lock only long enough to
reassign the three fields. If pipeline construction raises, nothing
changes.

### `beet/config.py`

Add `list_profiles() -> list[dict]`: scans `configs/*.yaml` (skipping
subdirs and files starting with `_`), opens each, extracts `_profile`
or filename, and returns `[{name, path, extends, description}]`.

### Sidecar + HTTP server

Both replace the current `pipeline: BeetPipeline` field with a
`ctx: RuntimeContext`. Every place that previously read `self._pipeline`
now reads `self._ctx.pipeline`. No behavior change for existing methods.

New RPC methods on both transports:

| Method          | Params       | Returns                                             |
|-----------------|--------------|-----------------------------------------------------|
| `list_profiles` | none         | `{profiles: [{name, path, description, extends}], current}` |
| `get_config`    | none         | `{profile, config, thresholds, detectors: [...]}` |
| `switch_profile`| `{name}`     | `{ok, profile, detectors_enabled}`                  |

`get_config` returns a curated view suitable for the UI — not the full
raw config dict — so the shape is stable across refactors. Shape:

```jsonc
{
  "profile": "screening",
  "path": "configs/screening.yaml",
  "thresholds": {
    "red": 0.75, "amber": 0.50, "yellow": 0.25,
    "abstention": { "enabled": true, "max_prediction_set_size": 3 }
  },
  "detectors": [
    { "id": "preamble", "enabled": true, "weight": 1.0 },
    ...
  ],
  "history": { "enabled": true, "retain_text": true, "db_path": "..." }
}
```

### HTTP endpoints

Added to `beet/gui/server.py`:

- `GET  /config/profiles` → profile list + current name
- `GET  /config/current`  → current config (curated shape above)
- `POST /config/switch`   → `{name}` → `{ok, profile, detectors_enabled}`

### Health + existing endpoints

Unchanged — they continue to read `ctx.pipeline`/`ctx.profile`.

## Frontend

Add a **Settings** tab (4th tab in the shell).

Layout:

- **Active profile** row: current profile name, dropdown of available
  profiles, "Switch" button; status line below.
- **Thresholds** panel: read-only list — red / amber / yellow p(LLM)
  boundaries plus abstention config.
- **Detectors** panel: table — id · enabled (badge) · weight. Disabled
  detectors dimmed.
- **History** panel: read-only — enabled y/n, retain_text y/n, db_path.
- **Actions** row: "Reload" re-fetches `/config/current`.

Switching a profile:
1. Disable the switch button, show "Switching…"
2. Call `switch_profile`; on success, refresh both `get_config` and the
   health indicator in the header.
3. On error, surface the message inline.

The header's existing `mode` indicator (tauri/http) gains a second line
showing the active profile — so the user always sees which profile is in
force, even outside the Settings tab.

## Testing

- `tests/test_runtime.py` (new):
  - `RuntimeContext.switch_profile` happy path replaces the pipeline.
  - Switching to an unknown profile raises `ConfigError`/`FileNotFoundError`
    and leaves the old pipeline in place.
- Extend `tests/test_config.py`:
  - `list_profiles` returns expected names for the repo's `configs/` dir.
- Extend `tests/test_rpc_sidecar.py`:
  - `list_profiles`, `get_config`, `switch_profile` round-trips.
  - `switch_profile` updates `health.profile`.
  - `switch_profile` to bad name returns `ERR_BAD_PROFILE`.
- Extend `tests/test_http_api.py` similarly.

## Back-compat

- `Sidecar.__init__` grows a `ctx` param; `run()` builds the ctx if not
  supplied (for the single-shot CLI launch path). Existing callers that
  pass a pipeline will be supported via a convenience wrapper.
- Actually: rather than optional param shuffle, replace `pipeline` param
  with `ctx` directly. Only one caller (`main()` in sidecar.py, the HTTP
  `serve()` fn) is affected, and both are under our control.

## Build Order

1. `beet/runtime.py` + `list_profiles` + unit tests.
2. Refactor Sidecar + HTTP to use `RuntimeContext`. Tests pass unchanged.
3. Add `list_profiles`/`get_config`/`switch_profile` RPC methods + HTTP
   endpoints. Tests.
4. Frontend Settings tab.
5. Manual QA: switch between profiles, verify detectors_run list changes
   on analyze, confirm header indicator updates.
