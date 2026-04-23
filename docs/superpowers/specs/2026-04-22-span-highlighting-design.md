# Span Highlighting — Design

**Date:** 2026-04-22
**Status:** Draft
**Scope:** GUI phase 6 of 6 (final)

## Goal

Let detectors emit character spans pointing to the evidence behind
their verdict, and render those spans as colored highlights over the
submission text in the Analyze view.

## Non-Goals

- Force *every* detector to produce spans. This is opt-in; detectors
  whose signal is distributional or structural (not positional) leave
  spans empty.
- Cross-highlight overlap resolution beyond simple sort-by-start (the
  two v0 detectors don't overlap each other; we'll revisit if future
  detectors do).

## Contract

Add to `LayerResult`:

```python
spans: list[dict] = field(default_factory=list)
# Each span: {"start": int, "end": int, "kind": str, "note": str}
# start, end: python slice bounds over the *original* submission text
# kind: short identifier for styling (e.g., "preamble", "fingerprint")
# note: human-readable hint shown on hover
```

Serialized through `build_json_report` as `layer_results[*].spans`.

## v0 Emitters

Two detectors where spans are cheap:

1. **`preamble`** — regex-driven. Each match.span() inside the 500-char
   preamble window becomes a span with `kind="preamble"`,
   `note=f"matched pattern '{name}' ({severity})"`.

2. **`fingerprint_vocab`** — regex-driven. Word and bigram matches
   become spans with `kind="fingerprint"`,
   `note=f"fingerprint word '{word}'"` or bigram equivalent.

Other detectors unchanged; their `spans` stays `[]`.

## Frontend

Add a **"Highlighted text"** collapsible section to the Analyze result
panel (between `All feature contributions` and the detectors line).

Rendering:

- Build a flat list of spans from `r.layer_results` where
  `spans.length > 0`.
- Sort by `start` ascending.
- Walk the original text, emitting alternating plain text and
  `<mark class="span-<kind>" data-note="…">…</mark>`.
- Overlap handling for v0: if span B starts before span A ends, skip B
  (logged to console for diagnostics). Fine for the two initial
  emitters whose patterns don't overlap each other in practice.
- Each `<mark>` shows its `note` as a native tooltip (`title` attr).

Styling: two faint tinted backgrounds
- `.span-preamble` — reddish
- `.span-fingerprint` — amber
Border-bottom dotted so the highlight is readable without looking
bannered.

If no detector emitted spans, the section shows a muted message
"No spans available for this submission."

## Testing

- Extend `tests/test_detectors/` (or add one):
  - preamble: known preamble text produces at least one span whose
    slice matches a pattern string
  - fingerprint_vocab: known fingerprint word text produces a span at
    the right offset
- Extend `tests/test_report.py`:
  - `spans` is present (possibly empty) in serialized `layer_results`
- No UI test; manual QA via Analyze view.

## Back-compat

`LayerResult` gains a field with a default. Nothing else changes.

## Build Order

1. Contract + report serialization + tests (minimal change; existing
   tests should pass unchanged).
2. preamble and fingerprint_vocab emit spans + tests.
3. Frontend highlighted-text section.
