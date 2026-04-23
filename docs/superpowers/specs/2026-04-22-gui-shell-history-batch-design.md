# GUI Shell + History + Batch — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 2 of 6

## Goal

Turn the single-page analyze GUI into a three-tab shell (**Analyze**,
**Batch**, **History**) and introduce the queryable submission log every
downstream feature (monitoring, eval runner, drift review) will read from.

## Non-Goals

- Rewrite as SPA framework. Stay vanilla JS + single `index.html`.
- Replace `PrivacyStore` or `AuditChain`. The new SQLite log sits alongside
  them as a queryable index.
- Profile switching at runtime (phase 3).
- Monitoring/eval runner views (phases 4–5).
- Authentication, multi-user, or remote storage.

## Architecture

### Submission log (new substrate)

A SQLite database at `data/beet_history.sqlite3` (configurable). Schema:

```sql
CREATE TABLE IF NOT EXISTS submissions (
  submission_id  TEXT PRIMARY KEY,
  recorded_at    TEXT NOT NULL,        -- ISO-8601 UTC with trailing Z
  text_hash      TEXT NOT NULL,        -- sha256 of raw text
  text           TEXT,                 -- nullable; retained by default
  determination  TEXT NOT NULL,
  p_llm          REAL NOT NULL,
  ci_lower       REAL NOT NULL,
  ci_upper       REAL NOT NULL,
  prediction_set TEXT NOT NULL,        -- JSON array
  detectors_run  TEXT NOT NULL,        -- JSON array
  report         TEXT NOT NULL,        -- full JSON report
  profile        TEXT,                 -- config profile name when known
  batch_id       TEXT,                 -- groups batch-mode submissions
  source         TEXT NOT NULL         -- 'analyze' | 'batch'
);

CREATE INDEX IF NOT EXISTS idx_submissions_recorded_at ON submissions(recorded_at);
CREATE INDEX IF NOT EXISTS idx_submissions_determination ON submissions(determination);
CREATE INDEX IF NOT EXISTS idx_submissions_batch_id ON submissions(batch_id);

CREATE TABLE IF NOT EXISTS feedback_log (
  submission_id   TEXT NOT NULL,
  confirmed_label INTEGER NOT NULL,    -- 0 human, 1 LLM
  reviewer_notes  TEXT,
  recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_submission ON feedback_log(submission_id);
```

New module `beet/history.py` exposes `HistoryStore`:

- `record(report: dict, *, source: str, text: str | None, profile: str | None,
  batch_id: str | None) -> None`
- `list(limit, offset, determination, since, until, batch_id, search) -> dict`
  returns `{items: [...], total: N}` where items are projection rows (no
  full `report` blob for listing efficiency; UI fetches the full report on
  row open).
- `get(submission_id) -> dict | None` — full report plus text.
- `record_feedback(submission_id, label, notes) -> None`
- `export(format: 'json'|'csv', filter: {...}) -> str` — serializes filtered
  rows to a string the server returns as file download.

Store opens a per-call `sqlite3.connect` (short-lived) so it's safe to share
across threads/processes (sidecar and HTTP server are separate).

ID generation: if the caller provides `submission_id`, use it (upsert on
conflict — new row, same id replaces). Otherwise mint a short ULID-style id
`f"sub_{timestamp_ms:x}_{rand4}"` for lexicographic-by-time ordering.

### Wiring analyze / batch / feedback into the store

Both `beet/gui/server.py` and `beet/sidecar.py` construct a `HistoryStore`
pointed at the same DB. After every `/analyze`, every item in `/batch`, and
every `/feedback` call, they write to the store. A new top-level `source`
and optional `batch_id` parameter is threaded through the RPC envelope (no
breaking changes for existing callers — defaults apply).

### New endpoints / RPC methods

Added to both HTTP and sidecar transports:

| Method            | Params                                            | Returns                                 |
|-------------------|---------------------------------------------------|-----------------------------------------|
| `history_list`    | `limit, offset, determination, since, until, search, batch_id` | `{items: [...], total: N}` |
| `history_get`     | `submission_id`                                   | `{report, text}` or `null`              |
| `history_export`  | `format, filter`                                  | HTTP: download. Sidecar: `{content, filename}` |
| `history_delete`  | `submission_id`                                   | `{ok: true}`                            |

No new methods for analyze/batch/feedback — same signatures; the write to
the history store happens server-side.

### Config

`configs/*.yaml` gains a `gui` section (optional; defaults below):

```yaml
gui:
  history:
    enabled: true
    db_path: data/beet_history.sqlite3
    retain_text: true        # if false, text column stays null
```

Server reads these at startup. If `enabled: false`, history is a no-op and
the History tab shows a friendly placeholder.

## Frontend — Tabbed shell

Single `index.html`, vanilla JS + DOM; no SPA framework.

```
+---- header ----------+
|  BEET 2.0            |
+---- tabs ------------+
| Analyze | Batch | History
+---- view container --+
|  (one of three panes)
+----------------------+
```

Tab switching: plain click handlers toggle which `<section class="view">`
is visible. Hash routing (`#/analyze`, `#/batch`, `#/history`) for deep
links; default `#/analyze`.

### Analyze tab

The existing markup, unchanged — single textarea + result panel. The only
addition is a muted "Viewing submission `<id>` from history" banner when
the user opens a history row, with a "Return" button that clears back to
a fresh analyze state.

### Batch tab

- **Input modes:**
  - Paste textarea, one text per line (trimmed; blank lines skipped).
  - File upload: `.txt` (one per line), `.json` (array of `{id, text}` or
    array of strings), `.csv` (columns `id,text` — id optional).
- Run button → fires `/batch` with a generated `batch_id` (ULID-style).
- Results render as a compact table: index · submission_id · determination
  badge · p_llm · detectors count. Row click → opens that submission in
  Analyze tab.
- Export button: JSON/CSV of the whole batch (from the history store,
  filtered by `batch_id`).
- Hard cap: 500 items per submit; server echoes the cap in an error if
  exceeded.

### History tab

- Filter bar: determination checkboxes (RED/AMBER/YELLOW/GREEN/UNCERTAIN/MIXED),
  date range (two `<input type="date">`), free-text search (matches
  `submission_id` or, if text is retained, a substring of the text).
- Paginated table (25/page): recorded_at · determination · p_llm · detectors
  count · feedback indicator (✓ if feedback recorded, else —).
- Row click → opens in Analyze tab (pre-fills the text if retained, renders
  the stored report without re-running the pipeline). An explicit "Re-run"
  button on the Analyze banner lets the user re-analyze with the current
  pipeline (useful when profile or detector set changes between sessions).
- Export button: JSON/CSV of the current filtered set.

### Shared UI elements

- `<nav class="tabs">` with aria-selected buttons.
- All three tabs reuse the badge palette, panel styling, and contribs/layers
  rendering already in place.
- Loading states: a small spinner (`.loading`) replaces the Run buttons
  while the request is in flight.

## Error handling

- History writes are best-effort — a failed DB write logs to stderr and
  does NOT fail the analyze/batch RPC. The ground-truth response is the
  report itself.
- History reads that hit a corrupt/missing DB return `{items: [], total: 0}`
  with a one-line warning banner in the UI.
- Bulk batch with malformed items: skip silently (as current `/batch` does)
  but return a `skipped: [...]` list so the UI can show a notice.

## Testing

- `tests/test_history.py` (new):
  - round-trip `record` + `list` + `get` + `delete`
  - filter by determination / date range / batch_id
  - text-retention flag toggles `text` column write
  - `record_feedback` links to submission; `list` reports `has_feedback`
  - `export` JSON/CSV shape smoke test
- `tests/test_gui_server.py` (restore — it was deleted; reinstate minimal
  coverage for `/analyze`, `/batch`, `/feedback`, and the new
  `/history/*` endpoints)
- `tests/test_sidecar.py`: extend (or add) for `history_list`/`history_get`
  methods
- No UI test framework; manual QA for tab flow, batch upload parsing,
  history row → analyze reload.

## Back-compat

- New fields on `Determination` were already added in phase 1; no more
  contract changes here.
- Old callers of `/analyze` and `/batch` keep working; history capture is
  side-effect only.
- Older `data/reviewer_feedback.jsonl` file stays the source of truth for
  training-label persistence. The SQLite `feedback_log` table is a redundant
  mirror for UI queries; we write to both to avoid coupling.

## Build Order

1. `beet/history.py` + DB schema + unit tests.
2. Wire `HistoryStore` into `beet/gui/server.py` and `beet/sidecar.py` (write
   path only; reads come next). Tests assert rows land.
3. Add `history_list`, `history_get`, `history_delete`, `history_export` to
   both transports. Tests.
4. Frontend: tabbed shell skeleton; port the existing analyze markup into
   an `Analyze` view; no new content yet.
5. Frontend: History view — list/filter/paginate/open-row.
6. Frontend: Batch view — three input modes, run, results table, export.
7. Manual QA pass; commit.
