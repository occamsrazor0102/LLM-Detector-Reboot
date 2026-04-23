# Drift Monitor Wiring — Design

**Date:** 2026-04-23
**Status:** Draft
**Scope:** GUI post-roadmap enhancement (part of item "b")

## Goal

Wire the existing `beet.monitoring.drift.DriftMonitor` into the live
analyze path and surface its baseline status + active alerts on the
Monitoring tab. Today the monitor is instantiable but nothing feeds it
observations at runtime.

## Non-Goals

- Persisting drift observations across sidecar restarts. The monitor
  is in-memory; baselines are re-established on demand from the
  HistoryStore (which *is* persistent).
- Editing drift thresholds in the UI (they come from config).
- Per-detector drift views (future — covered by the monitor's
  feature-level histograms but not surfaced yet).

## Architecture

### `RuntimeContext` → `Sidecar` hold a `DriftMonitor`

`RuntimeContext` gains no new fields — keep its scope to
pipeline/profile/config. The drift instance lives on `Sidecar` /
HTTP handler alongside `HistoryStore`. Shared `drift_from_config()`
helper in `beet/sidecar.py` constructs it from `config["drift_monitoring"]`
(defaults in the DriftMonitor constructor are fine if unset).

### Wiring analyze

After `_analyze` / `_analyze_batch` successfully produces a report,
if `drift is not None`, call:

```python
drift.record(
  p_llm=det.p_llm,
  determination=det.label,
  feature_vector=report["feature_contributions"],
  confirmed_label=None,
)
```

`feature_contributions` is already a float-valued dict — exactly the
shape DriftMonitor expects. `record()` returns the alert list for the
just-flushed window; we don't act on it inline (the UI polls).

Wiring `feedback`: when a label is recorded, thread `confirmed_label`
into the current window by calling `record(...)` again with the same
feature vector + the confirmed label. Simpler alternative — let
`feedback` flip the most-recent matching observation's `confirmed`
field; but DriftMonitor has no such API, and a duplicate record is
harmless given the window-based flushing. We'll use the first approach
only if it's trivial to thread the feature vector; otherwise skip.
For v0, **skip feedback→drift wiring** — feedback still writes
history+jsonl, and drift computes ECE only from observations with
`confirmed` set, which will simply be empty. Not a regression.

### Baseline from history

New method on the shared helper:

```python
def drift_baseline_from_history(drift, history, *, limit=500) -> int:
    """Pull recent submissions' feature_contributions from history and
    set them as the baseline. Returns the count used."""
```

Reads `limit` recent submissions, parses each stored report's
`feature_contributions`, passes the list of feature-vector dicts to
`drift.set_baseline()`.

### RPC surface

Added to both transports:

- `monitoring_drift` (no params) → `{has_baseline, baseline_features,
  n_observations, alerts, summary}`. `summary` is `DriftMonitor.get_summary()`.
- `monitoring_set_baseline` (`{limit?: int = 500}`) →
  `{ok, n_samples, baseline_features}`. If history is empty, return
  `{ok: false, n_samples: 0}`.

HTTP mirrors: `POST /monitoring/drift`, `POST /monitoring/set-baseline`.

## Frontend

Monitoring tab gets a new "Drift" panel positioned between the
determination-distribution bar and the per-day volume sparkline:

- Row 1: baseline state — "Set from N samples" / "Not set".
  Observation count in current window.
- Row 2: alerts list — each alert in a tinted `<div>`; colorless if
  none.
- Action button: "Set baseline from recent history (500 samples)".
  On success, refresh the panel.

Pulled via `monitoring_drift` in the existing `loadMonitoring()` flow.

## Testing

- `tests/test_drift_wiring.py` (new):
  - Drift records on analyze when enabled.
  - `drift_baseline_from_history` reads recent reports, sets baseline,
    returns correct count.
  - A synthetic population skew (all p=0.95) triggers a
    `POPULATION_DRIFT` alert.
- Extend `tests/test_rpc_sidecar.py` with the two new methods.
- Extend `tests/test_http_api.py` with the two new endpoints.

## Build Order

1. `drift_from_config` + `drift_baseline_from_history` helpers.
2. Sidecar + HTTP handler hold a DriftMonitor, wire analyze/batch.
3. New RPC methods + endpoints.
4. Frontend drift panel.
5. Tauri commands (on disk, will be included in next scaffold update).
