# BEET 2.0 — Evaluation Module + Ablation Harness

**Status:** Approved design, ready for implementation planning
**Date:** 2026-04-07
**Scope:** A new `beet/evaluation/` package providing dataset loading, metrics, an evaluation runner, and a leave-one-out ablation harness, plus two new CLI commands.

## Goal

BEET 2.0 has 91 passing tests and 12 working detectors, but no way to measure how well the pipeline actually performs on labeled data, and no way to know which detectors are pulling weight. This module fills both gaps.

## Non-goals

- `robustness.py` and `fairness.py` from the original plan are **deferred**. Robustness needs an adversarial perturbation harness (separate feature) and fairness needs a sensitive-attribute schema we don't have yet.
- No parallelism in the runner (premature).
- No HTML report (separate feature).
- No integration with `monitoring/drift.py` (separate feature).
- No new datasets — the user will supply labeled data; this work defines the format.

## Architecture

A new package with five small, single-purpose modules. None mutate the existing pipeline — they call it.

```
beet/evaluation/
├── __init__.py          # public re-exports: load_dataset, run_eval, run_ablation, EvalSample, EvalReport, AblationReport
├── dataset.py           # JSONL loader + schema validation
├── metrics.py           # pure metric functions (auroc, ece, brier, tpr_at_fpr, ...)
├── runner.py            # run_eval(pipeline, dataset) → EvalReport
└── ablation.py          # run_ablation(base_config, dataset) → AblationReport
```

**Key constraint:** `run_eval` takes a `Pipeline` instance — it does not construct one. Callers configure the pipeline however they like. `run_ablation` constructs N+1 pipelines from a base config (one baseline + one per ablated detector).

## Data contracts

```python
@dataclass(frozen=True)
class EvalSample:
    id: str                    # stable identifier
    text: str                  # raw input
    label: int                 # 1 = LLM, 0 = human
    tier: str | None = None    # "A0" | "A1" | "A2" | "human" | None
    source: str | None = None  # provenance string, optional

@dataclass(frozen=True)
class EvalReport:
    predictions: list[dict]    # [{id, label, p_llm, determination, tier}, ...]
    metrics: dict              # {auroc, ece, brier, tpr_at_fpr_01, ...}
    per_tier: dict[str, dict]  # {"A0": {auroc, ...}, "A1": {...}, ...}
    n_samples: int
    config_hash: str           # sha256(config_json)[:12]
    failed_samples: list[dict] # [{id, error}, ...] — samples the pipeline raised on

@dataclass(frozen=True)
class AblationReport:
    baseline: EvalReport
    per_detector: dict[str, EvalReport]   # detector_name → ablated report
    deltas: dict[str, dict]               # detector_name → {delta_auroc, delta_ece, ...}
    ranked: list[tuple[str, float]]       # [(detector, abs_delta_auroc), ...] descending
```

### Dataset format on disk

JSONL, one sample per line:

```json
{"id": "human_001", "text": "...", "label": 0, "tier": "human", "source": "reddit_r_writing"}
{"id": "llm_a0_001", "text": "...", "label": 1, "tier": "A0", "source": "claude_default_prompt"}
```

`load_dataset(path)` validates required fields (`id`, `text`, `label`) and raises `ValueError` with line numbers on bad rows.

## Metrics module

Pure functions in `metrics.py` — no I/O, no pipeline coupling. All take arrays/lists and return numbers or dicts.

```python
def auroc(y_true: list[int], y_score: list[float]) -> float
def ece(y_true: list[int], y_score: list[float], n_bins: int = 10) -> float
def brier(y_true: list[int], y_score: list[float]) -> float
def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.01) -> float
def confusion_at_threshold(y_true, y_score, threshold: float) -> dict  # tp, fp, tn, fn
def per_tier_breakdown(samples, predictions, metric_fn) -> dict[str, float]
def summarize(y_true, y_score) -> dict  # convenience: returns all of the above
```

**Edge cases:** when AUROC is undefined (all labels identical) return `float('nan')`, never crash.

**Dependencies:** uses `scikit-learn` (already in `fusion` extras) and `numpy`. We fold evaluation deps into the existing `fusion` extras rather than creating a new extras group, since they overlap.

**Why pure functions, not a class:** trivial tests, reusable from notebooks and `ablation.py` without instantiation.

## Runner

```python
def run_eval(
    pipeline: Pipeline,
    dataset: list[EvalSample],
    *,
    progress: bool = False,
) -> EvalReport
```

**Behavior:**

1. For each sample, call `pipeline.analyze(sample.text)` → get `FusionResult` + `Determination`.
2. Record `{id, label, p_llm, determination, tier}` per sample.
3. After all samples processed, compute overall metrics via `metrics.summarize`, plus a `per_tier` breakdown grouped by `sample.tier`.
4. Hash the pipeline config (stringify config dict deterministically + sha256, first 12 chars) into `config_hash`.
5. Return `EvalReport`.

**Error handling:** if `pipeline.analyze` raises on a sample, log the sample id + error and continue. The report's `failed_samples` field collects these. One bad sample must not kill a 500-sample run.

**Determinism:** the runner is deterministic given a deterministic pipeline. Tier 2/3 detectors that hit LLMs aren't deterministic — that's the caller's problem (use frozen configs or seeds).

## Ablation harness

```python
def run_ablation(
    base_config: dict,
    dataset: list[EvalSample],
    *,
    detectors: list[str] | None = None,   # None = ablate all enabled detectors
    progress: bool = False,
) -> AblationReport
```

**Algorithm:**

1. Build baseline `Pipeline` from `base_config`, run `run_eval` → `baseline: EvalReport`.
2. Determine which detectors to ablate (default: every detector currently enabled in `base_config`).
3. For each detector `d`:
   - Deep-copy `base_config`, disable detector `d`.
   - Build a fresh `Pipeline`, run `run_eval` → `ablated: EvalReport`.
   - Compute deltas: `delta_auroc = baseline.metrics["auroc"] - ablated.metrics["auroc"]` (positive = detector helped).
4. Rank detectors by `abs(delta_auroc)` descending → `ranked`.
5. Return `AblationReport`.

**Disabling a detector:** the current config schema may not have a clean per-detector enable toggle. Confirm during implementation. If absent, the smallest change is to add an `enabled: bool = True` field per detector in config and have the cascade skip disabled ones. This is a one-line schema addition.

**Cost warning:** ablating Tier 2/3 detectors over a 500-sample dataset means hundreds of GPT-2 / Claude calls per ablation run. The CLI prints an estimated call count and requires `--confirm` for runs above a threshold (default: >1000 model calls).

**Output table format:**

```
Detector              ΔAUROC    ΔECE    Verdict
preamble              +0.082    -0.011  load-bearing
fingerprint_vocab     +0.041    -0.004  helpful
nssi                  +0.018    +0.002  marginal
voice_spec            +0.003    +0.001  negligible
prompt_structure      -0.005    +0.008  hurting (?)
```

Verdict thresholds (tunable): `load-bearing` ≥ 0.05, `helpful` ≥ 0.02, `marginal` ≥ 0.01, `negligible` < 0.01, `hurting` < 0.

## CLI

Two new commands in `beet/cli.py`:

```bash
beet eval <dataset.jsonl> [--config <path>] [--profile <name>] [--output <json|text>] [--out <file>]
beet ablation <dataset.jsonl> [--config <path>] [--profile <name>] [--detectors d1,d2] [--confirm] [--out <file>]
```

Both commands print a human-readable summary by default and write structured JSON when `--out` is given.

## Testing strategy

`tests/test_evaluation/`:

- `test_metrics.py` — pure-function tests with hand-computed expected values. Edge cases: all-same labels → AUROC = NaN (no crash), perfect ranking → AUROC = 1.0, etc.
- `test_dataset.py` — loads `eval_mini.jsonl`, asserts schema validation rejects bad rows with line numbers.
- `test_runner.py` — runs a 3-sample dataset through a real pipeline (Tier 1 only, for speed), asserts report structure and that `config_hash` is stable across runs.
- `test_ablation.py` — runs ablation on a 3-sample dataset with 2 enabled detectors, asserts both ablated reports exist and `ranked` is sorted.

**Fixtures:**

- `tests/fixtures/eval_mini.jsonl` — ~10 samples reusing existing fixture texts (`A0_PREAMBLE`, `A1_CLEANED`, `CASUAL_SHORT`, etc.). No new content needed.

**AUROC on tiny fixtures is unstable.** Tests assert *structure* (keys present, ranges sane, sortedness) more than exact metric values.

## Risks and known unknowns

1. **Detector disable mechanism may need a small config schema addition.** Confirm during implementation; if needed, add `enabled: bool = True` per detector and have the cascade skip disabled ones.
2. **AUROC on tiny test fixtures is unstable** — tests assert structure, not values.
3. **Eval deps overlap with `fusion` extras** — fold in rather than creating a new extras group.

## Out of scope (deferred to future work)

- `robustness.py` (adversarial perturbation harness)
- `fairness.py` (FPR / calibration parity across sensitive attributes)
- Parallelism in the runner
- HTML evaluation reports
- Drift integration with `monitoring/drift.py`
- Building the labeled dataset itself (user will supply)
