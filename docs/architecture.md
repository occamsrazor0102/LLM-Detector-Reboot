# BEET 2.0 — Architecture

This doc describes the runtime pipeline, the detector contract, and how
cascade / fusion / decision compose to produce a verdict.

## Pipeline

```
                ┌──────────┐
  text in ─────►│  Router  │─── {prompt | prose | mixed | insufficient}
                └──────────┘
                     │
                     ▼
             ┌────────────────┐
             │ CascadeScheduler│
             └────────────────┘
                     │
                     ▼
   Phase 1 (always) ─► Phase 2 (if inconclusive) ─► Phase 3 (if still inconclusive)
     trivial/cheap       moderate (statistical)       expensive (LLM-backed)
        │                      │                             │
        └──────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │   EBM Fusion    │
                             └─────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────┐
                      │ Conformal prediction set    │
                      └────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────┐
                           │  DecisionEngine  │
                           └──────────────────┘
                                      │
                                      ▼
                              Determination
```

`beet/pipeline.py::BeetPipeline.analyze(text)` wires these together and
returns a `Determination` object (defined in `beet/contracts.py`).

## Core contracts

`beet/contracts.py` defines three dataclasses plus a detector protocol:

### `LayerResult`
One detector's output.
```python
layer_id: str
domain: "prompt" | "prose" | "universal"
raw_score: float
p_llm: float          # calibrated P(LLM), 0.0-1.0
confidence: float     # detector's trust in its own output
signals: dict         # detector-specific key/value evidence
determination: str    # per-detector verdict (RED/AMBER/...)
attacker_tiers: list[str]
compute_cost: "trivial" | "cheap" | "moderate" | "expensive"
min_text_length: int
spans: list[dict]     # optional character-level evidence
```

### `FusionResult`
Aggregated across all layers.
```python
p_llm: float
confidence_interval: (lo, hi)
prediction_set: list[str]
feature_contributions: dict[str, float]
top_contributors: list[(name, value)]
```

### `Determination`
The final verdict returned by `pipeline.analyze`.
```python
label: str
p_llm: float
confidence_interval: (lo, hi)
prediction_set: list[str]
reason: str
top_features: list[(name, value)]
override_applied: bool
detectors_run: list[str]
cascade_phases: list[int]
mixed_report: dict | None
layer_results: list[LayerResult]
feature_contributions: dict[str, float]
```

### `Detector` protocol
```python
class Detector(Protocol):
    id: str
    domain: str
    compute_cost: str
    def analyze(self, text: str, config: dict) -> LayerResult: ...
    def calibrate(self, labeled_data: list) -> None: ...
```

## Router

`beet/router.py` classifies incoming text into one of four domains:
`prompt`, `prose`, `mixed`, `insufficient`. The result decides which
detectors are *recommended* versus *skipped* — e.g. prompt-shaped
detectors (`instruction_density`, `voice_spec`, `prompt_structure`) are
routed around when the input is clearly prose.

Routing is conservative: when confidence is low, all detectors run.

## Cascade

`beet/cascade.py::CascadeScheduler` orchestrates four phases with
short-circuit logic driven by configurable thresholds
(`cascade.phase1_short_circuit_{high,low}`, etc.):

| Phase | Detectors | When it runs |
|------:|-----------|--------------|
| 1 | preamble, fingerprint_vocab, prompt_structure, voice_spec, instruction_density, nssi | always |
| 2 | surprisal_dynamics, contrastive_lm, token_cohesiveness | if aggregated Phase 1 p(LLM) falls in the uncertain band |
| 3 | perturbation, contrastive_gen, dna_gpt | if aggregated Phase 1+2 is still uncertain |
| 4 | cross_similarity, mixed_boundary, contributor_graph | batch-only (requires multiple submissions) |

`phase3_always_run: true` forces every detector regardless of
intermediate confidence — useful for evaluation.

## Fusion (EBM)

`beet/fusion/ebm.py::EBMFusion` takes the list of `LayerResult`s,
builds a feature vector (per-detector `p_llm` × weight, plus a few
derived features like detectors-active count), and asks an Explainable
Boosting Machine for:

1. A fused `p_llm`
2. Per-feature contributions (additive, interpretable)
3. A prediction set via conformal prediction (`beet/fusion/conformal.py`)

If no fusion model is loaded (common in development), the fusion falls
back to a naive weighted mean — less accurate but deterministic and
runnable without training.

## Decision engine

`beet/decision.py::DecisionEngine` maps `FusionResult` → `Determination`:

1. **Override rules** — if a detector returned a `CRITICAL` severity
   signal (today: `preamble` only), short-circuit to RED with
   `override_applied=True`.
2. **Abstention** — if the conformal prediction set spans too many
   severity bands (`decision.abstention.max_prediction_set_size`),
   abstain to UNCERTAIN.
3. **Threshold mapping** — fused `p_llm` → RED / AMBER / YELLOW /
   GREEN via `red_threshold`, `amber_threshold`, `yellow_threshold`.

## Evaluation

`beet/evaluation/` is separable from the live pipeline:

- `dataset.py` — JSONL loader / saver / builder for `EvalSample` lists
- `metrics.py` — pure functions for AUROC, ECE, Brier, TPR@FPR,
  confusion_at_threshold
- `runner.py` — `run_eval(pipeline, dataset) -> EvalReport`;
  `eval_report_to_dict` for UI-safe JSON serialization
- `ablation.py` — leave-one-out detector ablation
- `fairness.py` — per-group metric breakdown
- `robustness.py` — per-attack breakdown (requires adversarial transforms)

## Monitoring

`beet/monitoring/drift.py::DriftMonitor` tracks feature histograms,
population p(LLM), and (if reviewer labels are recorded) calibration
via ECE. Baselines are rebuilt from the history store on demand.

`beet/monitoring/meta_detector.py::MetaDetector` records per-detector
agreement with reviewer feedback to flag degradation over time.

## Runtime shells

All three shells consume the same `index.html` and the same RPC
surface:

| Shell | Backend | Transport |
|-------|---------|-----------|
| `beet gui` | stdlib HTTP in `beet/gui/server.py` | `fetch()` |
| `beet serve` | FastAPI in `beet.api.create_app` | `fetch()` |
| Tauri desktop | `beet.sidecar` subprocess | stdio JSON-RPC + `invoke()` |

The frontend picks `fetch` vs `invoke` via `window.__TAURI_INTERNALS__`
detection, so one `index.html` works everywhere.

`beet/runtime.py::RuntimeContext` holds `{pipeline, profile, config}`
and provides a thread-safe `switch_profile()` so the UI can hot-swap
the active config without restarting the process. The new pipeline is
built outside the lock; the assignment is atomic; a failed switch
leaves the old context untouched.

## Privacy and provenance

- `beet/privacy/store.py::PrivacyStore` — feature-vector store with
  optional raw-text vault and an access log.
- `beet/privacy/retention.py` — purge-by-age with a `--confirm` gate.
- `beet/provenance/chain.py::AuditChain` — hash-chained JSONL where
  each entry's `_hash` incorporates the previous entry's hash. Any
  tamper breaks the chain and `validate()` surfaces it.

## History store

`beet/history.py::HistoryStore` is a SQLite-backed queryable log that
sits alongside PrivacyStore and AuditChain. Every analyze / batch /
feedback call writes through it so the History, Batch, Monitoring, and
Evaluation tabs can filter, sort, paginate, and export across the
entire submission history without re-running the pipeline.

Schema lives in `beet/history.py::SCHEMA`. Two tables: `submissions`
(one row per analyze) and `feedback_log` (0..N per submission).
