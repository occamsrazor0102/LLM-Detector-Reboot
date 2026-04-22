# Per-Detector Deep-Dive — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 1 of 5 (deep-dive · batch/history · settings · monitoring · eval runner)

## Goal

Surface the per-detector data the pipeline already computes but discards after
fusion, so a reviewer inspecting a single submission can see *why* each layer
voted the way it did — not just the final fused verdict and top-3 contributors.

## Non-Goals

- In-text span highlighting (tracked as a follow-on; most detectors don't emit
  spans today — separate cross-detector effort).
- New pages, tabs, or routes.
- Changes to the Tauri sidecar command surface — it already proxies the full
  `/analyze` JSON verbatim.
- Persistence of deep-dive data (that lands with phase 2, history store).

## Backend Changes

### `beet/contracts.py` — `Determination`

Add two fields:

```python
layer_results: list[LayerResult] = field(default_factory=list)
feature_contributions: dict[str, float] = field(default_factory=dict)
```

`layer_results` is the list of per-detector results the pipeline currently
computes and discards after fusion. `feature_contributions` is the full map
from `FusionResult.feature_contributions` (already computed; currently only the
top-3 projection survives as `top_features`).

`top_features` stays as-is for back-compat with existing consumers
(`build_text_report`, `build_csv_row`, tests).

### `beet/pipeline.py`

Where the pipeline constructs the `Determination`, pass the per-layer results
list and the full contribution dict through. No new computation — just stop
throwing them away.

### `beet/report.py` — `build_json_report`

Serialize each `LayerResult` as:

```json
{
  "layer_id": "...",
  "domain": "prompt|prose|universal",
  "raw_score": 0.0,
  "p_llm": 0.0,
  "confidence": 0.0,
  "determination": "RED|AMBER|...",
  "signals": { ... },
  "compute_cost": "trivial|cheap|moderate|expensive"
}
```

Fields `attacker_tiers` and `min_text_length` on `LayerResult` are static
detector metadata, not per-run output; omit from the report.

Add two top-level keys to the report:

- `"layer_results"`: list of the per-layer objects above.
- `"feature_contributions"`: the full dict.

`build_text_report` and `build_csv_row` are unchanged.

## Frontend Changes

All changes confined to `beet/gui/static/index.html`.

### Layout additions

Under the existing "Top contributing signals" block, add two collapsible
sections (both collapsed by default):

1. **Detector breakdown** — compact table, one row per entry in
   `layer_results`:
   - columns: `id` · `domain` · `raw` (2dp) · `p_llm` (3dp) · `conf` (2dp) ·
     determination badge (reuses existing badge CSS).
   - Row click toggles an expansion beneath it showing the `signals` dict as
     a `<dl>`-style key/value list. Values that are lists or dicts are
     rendered as compact JSON.

2. **All feature contributions** — sortable list from
   `feature_contributions`:
   - default sort: `|contribution|` descending.
   - each row: feature name · signed value (3dp) · horizontal bar scaled to
     the max absolute value in the set.

### Interaction

- Disclosure summary uses `<details>`/`<summary>` for zero-JS toggling.
- Sort toggle on the contributions section: click a column header to flip
  between abs-desc, signed-desc, signed-asc, alpha.

### Styling

Reuse existing `--panel`, `--line`, `--muted`, badge classes. Table uses
`border-collapse: collapse` with a 1px dashed bottom border per row to match
the existing `.contribs .row` aesthetic.

## Testing

- `tests/test_api.py` (or equivalent report-shape test): assert
  `layer_results` and `feature_contributions` keys are present on the analyze
  response, that `len(layer_results) == len(detectors_run)`, and that each
  layer object has the documented fields.
- Extend `tests/test_pipeline.py` to assert the `Determination` now carries
  a non-empty `layer_results` list after `analyze()`.
- No automated UI test (no framework in the repo). Manual verification on a
  mixed-signal submission: confirm both collapsibles render, table sort
  works, and signals expansion shows reasonable content for each detector.

## Back-compat & Migration

- `Determination` gains fields with defaults — existing constructors still
  work.
- Report gains keys; existing consumers ignore unknown keys.
- No schema/storage migrations (no persistence yet).

## Build Order

1. Contracts + pipeline plumbing + report serialization (backend).
2. Backend tests.
3. Frontend: detector breakdown table + signals expansion.
4. Frontend: all-contributions section with sort.
5. Manual QA pass on varied texts (short/long, mixed, clean human, clean LLM).
