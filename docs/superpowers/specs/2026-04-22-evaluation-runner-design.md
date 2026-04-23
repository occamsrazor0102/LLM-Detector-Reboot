# Evaluation Runner — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 5 of 6

## Goal

Let a reviewer paste or upload a labeled JSONL dataset, run the active
pipeline against it, and see AUROC / ECE / Brier / TPR@FPR plus per-tier
and per-attack breakdowns — without dropping to the CLI.

## Non-Goals (deliberate scope control)

- **Background/cancellable jobs.** The v0 runner is synchronous with a
  hard cap on sample count (200 by default, configurable up to 1000).
  Anything bigger belongs to `beet eval` on the CLI. This avoids a job
  registry, polling, timeout handling, and cancellation semantics —
  features we don't need until someone wants to eval thousands of
  samples from the UI.
- Streaming progress updates. Single request, single response. The UI
  shows "Running…" while waiting.
- Dataset builder (composing from source files) — the existing
  `beet.evaluation.dataset.build_dataset` does this; users edit JSONL
  outside the UI.

## Backend

### `beet/sidecar.py` — new RPC method

```
run_eval({
  items: [{id, text, label, tier?, source?, attack_name?, ...}, ...],
  max_samples?: int        # hard cap; default 200, max 1000
}) -> {
  metrics: {...},          # from summarize()
  per_tier: {...},
  per_attack: {...},
  confusion: {tp, fp, tn, fn},  # at threshold 0.5
  n_samples, n_failed,
  failed_samples: [{id, error}, ...],
  config_hash,
  predictions: [{id, label, p_llm, determination, tier}, ...],
  duration_ms
}
```

Under the hood: builds a `list[EvalSample]`, calls
`beet.evaluation.runner.run_eval(pipeline, dataset)`, serializes the
`EvalReport` via a helper `_eval_report_to_dict`, and returns. Any
sample missing `id` / `text` / `label` is rejected with
`ERR_BAD_PARAMS`.

The cap is enforced before running:
- default 200; caller can raise up to 1000.
- exceeding returns `ERR_TOO_LARGE` with the cap in the message.

### HTTP endpoint

`POST /evaluation/run` — same body shape as the RPC.

### `beet/evaluation/dataset.py`

No change.

### `beet/evaluation/runner.py`

Add a small helper `eval_report_to_dict(report, *, include_predictions=True)`
that produces the JSON shape above. Kept in the module so the sidecar
doesn't duplicate serialization.

## Frontend

New **Evaluation** tab — inserted between Monitoring and Settings.

Layout:

- **Dataset input** panel:
  - Three tabs (same pattern as Batch): "Paste JSONL", "Upload file",
    "Sample recent history" (pulls recent analyzed submissions with
    feedback labels — bootstrapping QA without a separate dataset).
  - Validation ticks: sample count · label distribution (counts of 0
    and 1) · tier distribution if present.
  - Max samples input (default 200; warns if user exceeds the
    server-cap of 1000).
- **Run** button.
- **Results** panel (hidden until run):
  - **Summary cards**: n_samples · AUROC · ECE · Brier · TPR@FPR 1%.
  - **Confusion matrix** at p≥0.5: 2×2 grid with tp/fp/tn/fn, and
    derived precision/recall/F1.
  - **Per-tier table**: tier · n · AUROC · ECE · Brier · TPR@FPR.
  - **Per-attack table** (same columns) — rendered only if at least
    one sample had `attack_name`.
  - **Failed samples** section (collapsible): id · error.

## Testing

- `tests/test_eval_serialization.py` (new): `eval_report_to_dict`
  produces a JSON-serializable dict with the documented keys.
- `tests/test_rpc_sidecar.py`: `run_eval` happy path, cap enforcement,
  rejecting items missing required fields.
- `tests/test_http_api.py`: `/evaluation/run` round-trip.

## Back-compat

Purely additive.

## Build Order

1. `eval_report_to_dict` + unit test.
2. Sidecar `run_eval` + HTTP `/evaluation/run` + tests.
3. Frontend Evaluation tab.
4. Tauri command (on disk).
