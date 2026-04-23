# Monitoring Dashboard — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 4 of 6

## Goal

Surface aggregate behavior across submissions: volume, verdict
distribution, recent-verdict timeline, per-detector score
distributions, and feedback-derived calibration metrics.

## Non-Goals

- Streaming/push updates — reviewer pulls with a refresh button.
- Alerts and drift-vs-baseline (require a DriftMonitor baseline, which
  isn't set up in typical operational use — wire up later).
- Per-user segmentation (single-user app today).

## Data Source

Everything here reads from the SQLite history store introduced in phase 2.
No new persistence, no dependency on DriftMonitor or MetaDetector being
initialized. If a user's history is empty or history is disabled, the
dashboard shows empty-state placeholders.

## Backend

### `beet/history.py` — new methods

```python
def stats(self, *, since: str | None = None) -> dict:
    """Aggregate over the submission window.

    Returns:
      total, by_determination, mean_p_llm,
      feedback_count, feedback_accuracy (if any labels),
      per_day (last 14 days: date -> count)
    """

def timeline(self, *, limit: int = 200) -> list[dict]:
    """Recent submissions as (recorded_at, p_llm, determination,
    submission_id). For timeline scatter."""

def detector_stats(self, *, limit: int = 500) -> list[dict]:
    """Aggregate per-detector across the last N submissions.

    For each detector seen in the window:
      id, n (sample count), mean_p_llm, mean_confidence,
      determination_hist (RED/AMBER/... counts)
    """
```

### RPC methods

Added to both transports:

- `monitoring_summary` → `{stats, feedback_accuracy, per_day}`
- `monitoring_timeline` → `{items: [...]}`
- `monitoring_detectors` → `{detectors: [...]}`

HTTP mirror:

- `POST /monitoring/summary`
- `POST /monitoring/timeline`
- `POST /monitoring/detectors`

### Feedback accuracy computation

From `feedback_log` joined to `submissions`, compute Brier and simple
accuracy against the confirmed label. AUROC is skipped for v0 (fine for
a dashboard; avoids introducing sklearn-ish logic here — the eval runner
in phase 5 owns rigorous metrics).

## Frontend

New **Monitoring** tab (5th in the shell).

Layout (top to bottom):

1. **Summary cards** row: total · last 24h · last 7d · mean p(LLM) ·
   feedback count · feedback accuracy.
2. **Determination distribution**: horizontal stacked bar with per-label
   counts and percentages.
3. **Per-day volume**: 14-day sparkline (SVG).
4. **Recent timeline** (last 200): SVG scatter — x = timestamp,
   y = p(LLM), dot color = determination. Hover shows submission id.
   Click jumps to Analyze tab for that submission.
5. **Per-detector table**: id · n · mean p(LLM) · mean conf ·
   determination histogram (mini stacked bar).

Each section has its own loader; the "Refresh" button at the top re-runs
all three RPCs in parallel.

## Testing

- `tests/test_history_stats.py` (new): `stats()`, `timeline()`,
  `detector_stats()` with a seeded store.
- Extend `tests/test_rpc_sidecar.py` and `tests/test_http_api.py` with
  round-trip assertions for the three new endpoints.

## Back-compat

Purely additive — no contract or endpoint changes to existing methods.

## Build Order

1. HistoryStore methods + unit tests.
2. RPC wiring + HTTP endpoints + tests.
3. Frontend Monitoring tab.
4. Tauri commands (on disk).
