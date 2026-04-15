# BEET 2.0 — Evaluation, Adversarial Testing & Benchmarks

**Status:** Approved design, ready for implementation planning
**Date:** 2026-04-14
**Supersedes:** `2026-04-07-evaluation-module-design.md` (folded in with revisions)
**Scope:** Three independent modules — `beet/evaluation/`, `beet/adversarial/`, `beet/benchmarks/` — that share a data format (JSONL + `EvalReport`) but have no cross-imports.

## Goal

BEET 2.0 has 91 passing tests and 12 working detectors, but no way to:

1. **Measure** how well the pipeline performs on labeled data
2. **Stress-test** the pipeline against real-world evasion attacks
3. **Compare** BEET against existing open-source and commercial detectors

This spec defines three modules that fill all three gaps. They compose through a shared data format but are built, tested, and shipped independently.

## Non-goals

- `robustness.py` adversarial perturbation harness (gradient-based/theoretical ML attacks) — deferred
- `fairness.py` (sensitive-attribute schema not available) — deferred
- Parallelism in any runner — premature
- HTML reports — separate feature
- Integration with `monitoring/drift.py` — separate feature
- Building the labeled dataset itself — the user supplies data; this work defines the format and tooling

## Architecture Overview

```
Source texts (domain data)
    |
    +---> beet/evaluation/dataset.py ---> labeled dataset (JSONL)
    |                                         |
    |                                         +---> beet eval ---> EvalReport (BEET)
    |                                         |
    |                                         +---> beet benchmark ---> EvalReport (per competitor)
    |
    +---> beet/adversarial/generator.py ---> adversarial dataset (JSONL)
                                                  |
                                                  +---> beet eval ---> EvalReport (BEET on attacks)
                                                  |
                                                  +---> beet benchmark ---> EvalReport (competitors on attacks)
```

No module imports from another. They communicate through JSONL files and the `EvalReport` dataclass.

**Build order:** Evaluation first, adversarial second, benchmarks third.

---

## Module 1: Evaluation — `beet/evaluation/`

### Package structure

```
beet/evaluation/
├── __init__.py          # public re-exports: load_dataset, build_dataset, run_eval, run_ablation, EvalSample, EvalReport, AblationReport
├── dataset.py           # JSONL loader + schema validation + dataset builder
├── metrics.py           # pure metric functions (auroc, ece, brier, tpr_at_fpr, ...)
├── runner.py            # run_eval(pipeline, dataset) -> EvalReport
└── ablation.py          # run_ablation(base_config, dataset) -> AblationReport
```

### Data contracts

```python
@dataclass(frozen=True)
class EvalSample:
    id: str                          # stable identifier
    text: str                        # raw input
    label: int                       # 1 = LLM, 0 = human
    tier: str | None = None          # "A0" | "A1" | "A2" | "human" | None
    source: str | None = None        # provenance string, optional
    attack_name: str | None = None   # e.g. "inject_typos" — None for clean samples
    attack_category: str | None = None  # "transform" | "prompt" | None
    source_id: str | None = None     # links to pre-attack sample id

@dataclass(frozen=True)
class EvalReport:
    predictions: list[dict]          # [{id, label, p_llm, determination, tier}, ...]
    metrics: dict                    # {auroc, ece, brier, tpr_at_fpr_01, ...}
    per_tier: dict[str, dict]        # {"A0": {auroc, ...}, "A1": {...}, ...}
    per_attack: dict[str, dict]      # {"inject_typos": {auroc, ...}, ...} — empty for clean datasets
    n_samples: int
    config_hash: str                 # sha256(config_json)[:12]
    failed_samples: list[dict]       # [{id, error}, ...] — samples the pipeline raised on

@dataclass(frozen=True)
class AblationReport:
    baseline: EvalReport
    per_detector: dict[str, EvalReport]   # detector_name -> ablated report
    deltas: dict[str, dict]               # detector_name -> {delta_auroc, delta_ece, ...}
    ranked: list[tuple[str, float]]       # [(detector, abs_delta_auroc), ...] descending
```

### Dataset format on disk

JSONL, one sample per line:

```json
{"id": "human_001", "text": "...", "label": 0, "tier": "human", "source": "reddit_r_writing"}
{"id": "llm_a0_001", "text": "...", "label": 1, "tier": "A0", "source": "claude_default_prompt"}
{"id": "adv_001", "text": "...", "label": 1, "tier": "A0", "source": "claude_default", "attack_name": "inject_typos", "attack_category": "transform", "source_id": "llm_a0_001"}
```

`load_dataset(path)` validates required fields (`id`, `text`, `label`) and raises `ValueError` with line numbers on bad rows. Optional fields default to `None`.

### Dataset builder

```python
def build_dataset(
    sources: list[dict],   # [{"path": "...", "label": 1, "tier": "A0", "source": "claude"}, ...]
    *,
    id_prefix: str = "",
) -> list[EvalSample]
```

Each source dict points to a directory of `.txt` files or a single text file. The builder reads them, assigns sequential IDs (e.g., `{id_prefix}_{index:04d}`), and returns a validated `EvalSample` list. Convenience function — hand-writing JSONL is always an option.

### Metrics module

Pure functions — no I/O, no pipeline coupling. All take arrays/lists and return numbers or dicts.

```python
def auroc(y_true: list[int], y_score: list[float]) -> float
def ece(y_true: list[int], y_score: list[float], n_bins: int = 10) -> float
def brier(y_true: list[int], y_score: list[float]) -> float
def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.01) -> float
def confusion_at_threshold(y_true, y_score, threshold: float) -> dict  # tp, fp, tn, fn
def per_tier_breakdown(samples, predictions, metric_fn) -> dict[str, float]
def per_attack_breakdown(samples, predictions, metric_fn) -> dict[str, float]
def summarize(y_true, y_score) -> dict  # convenience: returns all of the above
```

**Edge cases:** when AUROC is undefined (all labels identical) return `float('nan')`, never crash.

**Dependencies:** `scikit-learn` and `numpy`, folded into the existing `fusion` extras group.

### Runner

```python
def run_eval(
    pipeline: BeetPipeline,
    dataset: list[EvalSample],
    *,
    progress: bool = False,
) -> EvalReport
```

**Behavior:**

1. For each sample, call `pipeline.analyze(sample.text)` to get a `Determination`.
2. Record `{id, label, p_llm, determination, tier}` per sample.
3. After all samples, compute overall metrics via `metrics.summarize`, plus `per_tier` breakdown grouped by `sample.tier` and `per_attack` breakdown grouped by `sample.attack_name`.
4. Hash the pipeline config deterministically (stringify + sha256, first 12 chars) into `config_hash`.
5. Return `EvalReport`.

**Error handling:** if `pipeline.analyze` raises on a sample, log the sample id + error and continue. The report's `failed_samples` field collects these. One bad sample does not kill a 500-sample run.

### Ablation harness

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

1. Build baseline `BeetPipeline` from `base_config`, run `run_eval` -> `baseline: EvalReport`.
2. Determine which detectors to ablate (default: every detector with `enabled: true` in `base_config`).
3. For each detector `d`:
   - Deep-copy `base_config`, set `detectors.{d}.enabled = false`.
   - Build a fresh `BeetPipeline`, run `run_eval` -> `ablated: EvalReport`.
   - Compute deltas: `delta_auroc = baseline.metrics["auroc"] - ablated.metrics["auroc"]` (positive = detector helped).
4. Rank detectors by `abs(delta_auroc)` descending -> `ranked`.
5. Return `AblationReport`.

**Disabling a detector:** the config schema already has `enabled: bool` per detector, and `pipeline.py:50` skips disabled detectors. No schema changes needed.

**Output table format:**

```
Detector              AUROC     ECE     Verdict
preamble              +0.082    -0.011  load-bearing
fingerprint_vocab     +0.041    -0.004  helpful
nssi                  +0.018    +0.002  marginal
voice_spec            +0.003    +0.001  negligible
prompt_structure      -0.005    +0.008  hurting (?)
```

Verdict thresholds: `load-bearing` >= 0.05, `helpful` >= 0.02, `marginal` >= 0.01, `negligible` < 0.01, `hurting` < 0.

### CLI additions

```bash
beet eval <dataset.jsonl> [--config <path>] [--profile <name>] [--output <json|text>] [--out <file>]
beet ablation <dataset.jsonl> [--config <path>] [--profile <name>] [--detectors d1,d2] [--confirm] [--out <file>]
```

Both print a human-readable summary by default and write structured JSON when `--out` is given.

**Cost warning for ablation:** the CLI prints an estimated call count and requires `--confirm` for runs above 1000 model calls.

### Testing

`tests/test_evaluation/`:

- `test_metrics.py` — pure-function tests with hand-computed expected values. Edge cases: all-same labels -> AUROC = NaN (no crash), perfect ranking -> AUROC = 1.0.
- `test_dataset.py` — loads `eval_mini.jsonl`, asserts schema validation rejects bad rows with line numbers.
- `test_runner.py` — runs a 3-sample dataset through a real pipeline (Tier 1 only, for speed), asserts report structure and that `config_hash` is stable across runs.
- `test_ablation.py` — runs ablation on a 3-sample dataset with 2 enabled detectors, asserts both ablated reports exist and `ranked` is sorted.

**Fixtures:**

- `tests/fixtures/eval_mini.jsonl` — ~10 samples reusing existing fixture texts. No new content needed.

**Note:** AUROC on tiny fixtures is unstable. Tests assert structure (keys present, ranges sane, sortedness) more than exact metric values.

---

## Module 2: Adversarial Testing — `beet/adversarial/`

### Package structure

```
beet/adversarial/
├── __init__.py          # public re-exports: generate, list_attacks, Attack
├── registry.py          # attack registry — discover + list available attacks
├── transforms.py        # programmatic post-processing attacks
├── prompts.py           # evasion-prompted LLM generation attack templates
└── generator.py         # orchestrator: source text + attack -> adversarial samples
```

### Attack protocol

```python
@dataclass
class Attack:
    name: str                    # unique identifier, e.g. "inject_typos"
    category: str                # "transform" | "prompt"
    description: str             # human-readable, one line
    severity: str                # "basic" | "moderate" | "advanced"

    def apply(self, text: str, **kwargs) -> str:
        """Return the modified text."""
        ...
```

### Transforms — `transforms.py`

Programmatic post-processing attacks that take existing LLM-generated text and modify it:

| Attack | Description | Severity |
|--------|-------------|----------|
| `strip_preamble` | Remove "Certainly!", "Here's...", "Sure," openers and closers | basic |
| `inject_typos` | Random character swaps, missing punctuation, casual misspellings | basic |
| `casualize` | Lower formality: contractions, shorter sentences, filler words | moderate |
| `mix_human` | Splice human-written sentences into LLM text at random positions | advanced |
| `synonym_swap` | Replace fingerprint vocabulary with common synonyms | moderate |

Each transform is a pure function: `text in -> text out`. No API calls, no side effects.

### Evasion prompts — `prompts.py`

Prompt templates that instruct LLMs to generate text designed to evade detection:

| Attack | Prompt strategy | Severity |
|--------|----------------|----------|
| `casual_voice` | "Write as if you're texting a friend" | basic |
| `mimic_student` | "Write like a college freshman, include minor grammar mistakes" | moderate |
| `anti_pattern` | "Avoid formal transitions, don't use 'furthermore' or 'moreover'" | moderate |
| `hybrid_instruction` | "Write the first paragraph, then I'll write the second" (simulated mixed authorship) | advanced |

These attacks require an LLM provider. The `apply` method takes the original text's topic/prompt and generates a new evasion-style response. The generator passes through the configured provider (defaults to the same provider in BEET config).

**Provider handling:** `prompts.py` uses a simple callable interface `(system_prompt: str, user_prompt: str) -> str` so it works with any LLM backend. The generator injects the callable based on config. No direct provider coupling.

### Registry — `registry.py`

```python
_attacks: dict[str, Attack] = {}

def register(attack: Attack) -> Attack: ...
def get_attack(name: str) -> Attack | None: ...
def list_attacks(category: str | None = None) -> list[Attack]: ...
```

All attacks in `transforms.py` and `prompts.py` register themselves at import time via the `@register` decorator pattern (same approach as `beet/detectors/__init__.py` auto-discovery).

### Generator — `generator.py`

```python
def generate(
    source_dataset: list[EvalSample],
    attacks: list[str],
    *,
    provider: Callable | None = None,   # required for prompt-based attacks
    seed: int | None = None,            # for reproducible transforms
) -> list[EvalSample]
```

**Behavior:**

1. Filter `source_dataset` to LLM-labeled samples only (`label == 1`). Transforms on human text are meaningless.
2. For each source sample x each requested attack:
   - Apply the attack to the source text.
   - Produce a new `EvalSample` with:
     - `id`: `"{source.id}__{attack_name}"`
     - `label`: 1 (still LLM-generated, just evaded)
     - `attack_name`: the attack name
     - `attack_category`: "transform" or "prompt"
     - `source_id`: the original sample's id
     - All other fields inherited from the source sample.
3. Return the list of adversarial samples.

**Output format:** Standard `EvalSample` list. The caller can serialize to JSONL with `save_dataset()` or feed directly into `run_eval()`.

### CLI

```bash
beet attack <source.jsonl> --attacks strip_preamble,inject_typos --out adversarial.jsonl
beet attack <source.jsonl> --all-transforms --out adversarial.jsonl
beet attack <source.jsonl> --all-prompts --provider anthropic --out adversarial.jsonl
beet attack --list
```

### Testing

`tests/test_adversarial/`:

- `test_transforms.py` — each transform modifies text (output != input), output is still a string, no crashes on edge cases (empty string, very short text).
- `test_prompts.py` — prompt templates render correctly, contain the expected instruction fragments. Actual LLM generation is mocked.
- `test_generator.py` — generator produces correct number of samples (n_sources x n_attacks), all output samples have `attack_name` and `source_id` set, ids follow the naming convention.
- `test_registry.py` — all built-in attacks are discoverable, `list_attacks(category="transform")` filters correctly.

---

## Module 3: Benchmarks — `beet/benchmarks/`

### Package structure

```
beet/benchmarks/
├── __init__.py          # public re-exports: run_benchmark, list_adapters, BaseAdapter
├── adapter.py           # BaseAdapter protocol + adapter registry
├── runner.py            # run_benchmark(adapters, dataset) -> dict[str, EvalReport]
└── adapters/
    ├── __init__.py      # auto-discovery of adapters
    ├── binoculars.py    # open-source, local (Phase 1)
    ├── fast_detectgpt.py # open-source, local (Phase 1)
    ├── gptzero.py       # commercial API (Phase 2)
    └── originality.py   # commercial API (Phase 2)
```

### Adapter protocol

```python
class BaseAdapter(Protocol):
    name: str
    requires_api_key: bool
    requires_gpu: bool

    def score(self, text: str) -> float:
        """Return p(LLM-generated) in [0, 1]."""
        ...

    def is_available(self) -> bool:
        """Check if the adapter's dependencies/API keys are present."""
        ...
```

Every competitor detector implements this interface. The `is_available()` method lets the runner skip adapters whose dependencies aren't installed or whose API keys are missing, rather than crashing.

### Phase 1 adapters (open-source, local)

**Binoculars** (`binoculars.py`):
- Uses two language models to compute cross-perplexity ratio.
- Dependencies: `transformers`, `torch`. Included in a `benchmarks` extras group.
- `requires_gpu: True` (runs on CPU but very slow).
- `is_available()` checks that `transformers` is installed and model weights are accessible.

**Fast-DetectGPT** (`fast_detectgpt.py`):
- Conditional probability curvature estimation without perturbations.
- Dependencies: `transformers`, `torch`. Same extras group.
- `requires_gpu: True`.
- `is_available()` checks same.

### Phase 2 adapters (commercial API) — stubs only

**GPTZero** (`gptzero.py`) and **Originality.ai** (`originality.py`):
- Implemented as stubs with `is_available() -> False` and a `NotImplementedError` in `score()`.
- When implemented: read API key from environment variable, make HTTP request, parse response to extract p(LLM).
- `requires_api_key: True`.

### Runner

```python
def run_benchmark(
    adapters: list[BaseAdapter],
    dataset: list[EvalSample],
    *,
    include_beet: bool = True,
    beet_config: dict | None = None,
    progress: bool = False,
) -> dict[str, EvalReport]
```

**Behavior:**

1. If `include_beet` is True, build a `BeetPipeline` from `beet_config` (or default config) and run it through `run_eval` from `beet/evaluation/`. This is the only place where benchmarks imports from evaluation — a single function call.
2. For each adapter where `is_available()` returns True:
   - For each sample, call `adapter.score(sample.text)` to get `p_llm`.
   - Collect predictions, compute metrics using `beet.evaluation.metrics.summarize`.
   - Package as an `EvalReport`.
3. Return dict keyed by adapter name (and "BEET" if included).

**Error handling:** if `adapter.score()` raises on a sample, log it and continue. Same pattern as `run_eval`.

**Note on evaluation import:** The benchmark runner imports `EvalReport`, `EvalSample`, and `metrics.summarize` from `beet.evaluation`. This is a read-only dependency on data contracts and pure functions — not a coupling between systems. The evaluation module does not know benchmarks exist.

### Comparison output

```
Detector          AUROC   ECE     TPR@1%FPR   Brier
BEET (default)    0.94    0.031   0.72        0.048
Binoculars        0.89    0.055   0.58        0.071
Fast-DetectGPT    0.85    0.068   0.49        0.089
```

When run on adversarial samples, the same table is also broken down per attack.

### CLI

```bash
beet benchmark <dataset.jsonl> --adapters binoculars,fast_detectgpt [--config <path>] [--out <file>]
beet benchmark <dataset.jsonl> --all-available [--config <path>] [--out <file>]
beet benchmark --list
```

### Dependencies

Open-source adapters need `transformers` and `torch`. These go in a new `benchmarks` extras group in `pyproject.toml`:

```toml
[project.optional-dependencies]
benchmarks = ["transformers>=4.40", "torch>=2.0"]
```

Not installed by default. `beet benchmark --list` shows which adapters are available vs. missing dependencies.

### Testing

`tests/test_benchmarks/`:

- `test_adapter.py` — mock adapter conforms to protocol, `is_available()` returns expected values.
- `test_runner.py` — runs benchmark with 2 mock adapters on 3-sample dataset, asserts dict has correct keys and each value is a valid `EvalReport`.
- `test_comparison.py` — comparison table formatter produces expected output with column alignment.

Real adapter tests (Binoculars, Fast-DetectGPT) are integration tests gated behind a `--run-slow` pytest marker since they require model downloads.

---

## Composition and Workflow

### Typical usage

1. Assemble labeled dataset from domain sources -> `dataset.jsonl`
2. `beet eval dataset.jsonl` -> baseline BEET performance
3. `beet ablation dataset.jsonl` -> which detectors matter
4. `beet attack dataset.jsonl --all-transforms --out adversarial.jsonl` -> generate evasion samples
5. `beet eval adversarial.jsonl` -> BEET on adversarial data (per-attack breakdown)
6. `beet benchmark dataset.jsonl --adapters binoculars,fast_detectgpt` -> competitors on clean data
7. `beet benchmark adversarial.jsonl --adapters binoculars,fast_detectgpt` -> competitors on attacks

### Cross-module data flow

- **Adversarial -> Evaluation:** adversarial module outputs JSONL in `EvalSample` schema. Feed it to `beet eval` unchanged.
- **Evaluation -> Benchmarks:** benchmark runner imports `EvalReport`, `EvalSample`, and `metrics.summarize` from evaluation. Read-only dependency on data contracts and pure functions.
- **Adversarial -> Benchmarks:** adversarial JSONL is fed to `beet benchmark` the same way as any other dataset. No direct dependency.

### Build order

1. **Evaluation** — ships first. Measurement infrastructure everything else depends on.
2. **Adversarial** — ships second. Generates attack data for stress testing.
3. **Benchmarks** — ships third. Compares against competitors.

Each module has its own test suite and can be merged independently.

---

## Risks and known unknowns

1. **Evasion prompt effectiveness is unknown.** The prompts may or may not produce text that actually evades BEET. That's the point of measuring — but don't expect all attacks to be effective.
2. **AUROC on tiny test fixtures is unstable.** Tests assert structure, not values.
3. **Binoculars / Fast-DetectGPT model weights are large.** First benchmark run requires a multi-GB download. The CLI should warn about this.
4. **Commercial API adapters (Phase 2) depend on third-party pricing and rate limits.** Stubs only until needed.
5. **Prompt-based attacks require an LLM provider.** Running `--all-prompts` costs money. The CLI prints estimated call count.

## Out of scope (deferred)

- `robustness.py` (gradient-based adversarial perturbation harness)
- `fairness.py` (FPR / calibration parity across sensitive attributes)
- Parallelism in any runner
- HTML evaluation reports
- Drift integration with `monitoring/drift.py`
- Building the labeled dataset itself (user supplies data)
