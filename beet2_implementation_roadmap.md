# BEET 2.0 — Complete Implementation Roadmap

**Document date:** 2026-04-19
**Baseline:** 93 files, 11 working detectors, naive fusion, 0 trained models
**Target:** Fully operational pipeline with trained EBM, conformal prediction, 15 detectors, adversarial evaluation, and production CLI/API

---

## Current State Summary

### What works

The package scaffolding, all contracts, config system with profile inheritance, text router, cascade scheduler, normalizer, 11 detectors (6 Tier 1 + 3 Tier 2 + 1 Tier 3 + 1 Tier 4), naive fusion fallback, decision engine with override rules and abstention, CLI with analyze/batch/eval/ablation commands, evaluation harness with AUROC/ECE/Brier/TPR@FPR metrics plus per-tier and per-attack breakdowns, adversarial transform framework (5 transforms), privacy store with access logging, provenance manifests with hash signing, and drift monitor stubs.

### What doesn't

The EBM fusion model is untrained — the pipeline runs on confidence-weighted averaging with no learned interactions and no real feature-level attribution. Conformal prediction is implemented but never wired into the pipeline. Four detectors (`perturbation`, `contrastive_gen`, `cross_similarity`, `contributor_graph`) exist only as config entries. No labeled dataset exists for calibration. `pipeline.py` doesn't pass `word_count` or `domain` to the fusion layer. Phase 3 cascade tracking is broken. No robustness or fairness evaluation modules. No retention purging for the privacy vault. No hash-chained provenance audit log. No REST API. No GUI.

---

## Roadmap Overview

```
Phase 0  Bug Fixes & Wiring (1–2 days)
  ├── Fix pipeline → fusion argument passing
  ├── Fix cascade phase tracking
  ├── Add detector-availability logging
  └── Clean up naive fusion contribution reporting

Phase 1  Calibration Infrastructure (3–5 days)
  ├── Implement detector-level calibration machinery (isotonic regression code)
  ├── Implement EBM training script (not executed — no dataset yet)
  ├── Wire conformal prediction into pipeline (load-path only)
  ├── Add fusion.model_path config + graceful fallback to naive fusion
  └── Unit-test calibration/training/conformal code with synthetic data
  NOTE: Dataset construction and actual model fitting moved to Phase 7.

Phase 2  Missing Tier 2–3 Detectors (5–8 days)
  ├── Implement perturbation curvature detector (DetectGPT-style)
  ├── Implement contrastive generation engine
  ├── Add prompt-based adversarial attacks to transform registry
  └── Expand labeled dataset with A3 (paraphrased) samples

Phase 3  Robustness & Fairness Evaluation (3–5 days)
  ├── Implement robustness evaluation module
  ├── Implement fairness evaluation module
  ├── Wire adversarial → evaluation pipeline
  ├── Add robustness and fairness CLI commands
  └── Run first public benchmark (RAID or HC3)

Phase 4  Batch-Mode Detectors (5–7 days)
  ├── Implement cross-submission similarity detector
  ├── Implement contributor graph / syndicate detection
  ├── Add batch analysis CLI command
  └── Integrate batch detectors into pipeline batch mode

Phase 5  Privacy, Provenance & Monitoring Hardening (3–4 days)
  ├── Implement privacy retention / auto-purge
  ├── Implement hash-chained provenance audit log
  ├── Upgrade drift monitor with feature-distribution tracking
  ├── Wire drift monitor into pipeline
  └── Implement meta-detector (signal degradation alerting)

Phase 6  API & GUI (5–8 days)
  ├── Implement FastAPI REST server
  ├── Build web GUI (single-file embedded SPA or React artifact)
  ├── Integrate privacy store and provenance into API
  └── Production deployment configuration

Phase 7  Dataset Construction, Training & Continuous Improvement (ongoing)
  ├── Build seed labeled dataset (100+ samples across A0–A2 + human)  [was 1.1]
  ├── Run detector-level calibration against real data                 [was 1.2 exec]
  ├── Train first EBM model                                            [was 1.3 exec]
  ├── Calibrate conformal predictor                                    [was 1.4 exec]
  ├── End-to-end calibration validation (ECE/coverage/AUROC gates)     [was 1.5]
  ├── Expand labeled dataset to 500+ samples across A0–A5
  ├── Retrain EBM with expanded feature set (Tier 2–3 detectors)
  ├── Occupation-stratified calibration
  ├── Ablation-guided detector pruning
  └── Monthly recalibration protocol
```

---

## Phase 0: Bug Fixes & Wiring

**Goal:** Make the existing code do what it claims. No new features — just fix silent failures.
**Duration:** 1–2 days
**Prerequisites:** None
**Test gate:** All existing tests still pass; new tests cover the fixed behavior.

---

### Task 0.1: Fix pipeline → fusion argument passing

**Problem:** `pipeline.py` calls `self._fusion.fuse(results)` without passing `word_count` or `domain`. The `FeatureAssembler` always receives `word_count=0` and `domain="prose"`, making the metadata features (`word_count`, `domain_prompt`, `domain_prose`) useless.

**Files to modify:**
- `beet/pipeline.py`

**Changes:**

In `pipeline.py`, the `analyze` method currently does:
```python
fusion_result = self._fusion.fuse(results)
```

Change to:
```python
fusion_result = self._fusion.fuse(results, word_count=router_decision.word_count, domain=router_decision.domain)
```

**Test:**

Add to `tests/test_pipeline.py`:
```python
def test_fusion_receives_word_count_and_domain(pipeline):
    """Verify the pipeline passes routing metadata to the fusion layer."""
    # Analyze a prompt-like text
    det = pipeline.analyze(A0_CLINICAL_TASK_TEXT)
    # The pipeline should have passed non-zero word_count
    # (We verify indirectly: if the feature assembler received word_count=0,
    # the metadata features would be zero, which affects scoring)
    assert det.p_llm != 0.5  # not the degenerate "no data" default
```

**Verification:** `pytest tests/test_pipeline.py -v` — all pass including the new test.

---

### Task 0.2: Fix cascade phase tracking

**Problem:** `pipeline.py` only appends phases 1 and 2 to `cascade_phases`. Phase 3 is never recorded even when it runs.

**Files to modify:**
- `beet/pipeline.py`

**Changes:**

Current code after the cascade block:
```python
determination.cascade_phases = [1]
if len(results) > len(phase1_results):
    determination.cascade_phases.append(2)
```

Replace with tracking during execution:
```python
phases_run = [1]
phase1_results = self._run_phase(1, text, cfg, router_decision.skip_detectors)
results.extend(phase1_results)

if self._cascade.should_run_phase2(phase1_results):
    phase2_results = self._run_phase(2, text, cfg, router_decision.skip_detectors)
    results.extend(phase2_results)
    phases_run.append(2)
    if self._cascade.should_run_phase3(results):
        phase3_results = self._run_phase(3, text, cfg, router_decision.skip_detectors)
        results.extend(phase3_results)
        phases_run.append(3)

# ... fusion, decision ...
determination.cascade_phases = phases_run
```

**Test:**

Add to `tests/test_pipeline.py` a case that forces Phase 3 via config (`phase3_always_run: true`) and asserts `3 in det.cascade_phases`.

---

### Task 0.3: Add detector-availability logging

**Problem:** When `beet[core]` is installed without ML dependencies, Tier 2 detectors silently have no `DETECTOR` attribute. The registry skips them. The cascade says "run Phase 2" but Phase 2 has zero detectors. No log message, no feedback.

**Files to modify:**
- `beet/detectors/__init__.py`
- `beet/pipeline.py`

**Changes in `detectors/__init__.py`:**

Add a `_missing` dict that tracks detectors that failed to import:

```python
_missing: dict[str, str] = {}  # detector_id → error message

def _discover() -> None:
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name == "__init__":
            continue
        try:
            mod = importlib.import_module(f"beet.detectors.{module_info.name}")
            if hasattr(mod, "DETECTOR"):
                d = mod.DETECTOR
                _registry[d.id] = d
            # else: module exists but has no DETECTOR (conditional on import)
        except ImportError as e:
            _missing[module_info.name] = str(e)

def get_missing_detectors() -> dict[str, str]:
    if not _registry and not _missing:
        _discover()
    return dict(_missing)
```

**Changes in `pipeline.py`:**

After building the pipeline, log missing detectors:

```python
import logging
logger = logging.getLogger("beet")

class BeetPipeline:
    def __init__(self, config: dict):
        # ... existing init ...
        missing = detector_registry.get_missing_detectors()
        for name, err in missing.items():
            logger.info(f"Detector '{name}' unavailable: {err}")
```

**Test:** Mock a missing import and verify `get_missing_detectors()` returns it.

---

### Task 0.4: Fix naive fusion contribution reporting

**Problem:** `_naive_fuse` reports `p_llm` values as "contributions," creating a false impression of feature-level attribution. A detector with `p_llm=0.80` shows as "contribution: 0.80" which looks like a log-odds contribution but is actually just the detector's standalone estimate.

**Files to modify:**
- `beet/fusion/ebm.py`

**Changes:**

In `_naive_fuse`, change the `top_contributors` to report *deviation from the prior* (0.5) weighted by confidence, which is at least directionally correct:

```python
contribs = {r.layer_id: (r.p_llm - 0.5) * r.confidence for r in active}
top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
```

This makes contributions signed (+0.24 means "this detector pulled toward LLM") and weighted (low-confidence detectors contribute less). It's still not EBM-quality attribution, but it's honest about what it is.

**Test:** Unit test in `tests/test_fusion.py` asserting that a detector with `p_llm=0.5` contributes ~0.0 and a detector with `p_llm=0.9, confidence=0.8` contributes a positive value.

---

## Phase 1: Calibration Infrastructure

**Goal:** Land all the code, scripts, wiring, and config for detector calibration, EBM fusion, and conformal prediction — without requiring a labeled dataset yet. The pipeline continues running on naive fusion until Phase 7 builds the dataset and fits the models.
**Duration:** 3–5 days
**Prerequisites:** Phase 0
**Test gate:** Calibration/training/conformal code is covered by unit tests using synthetic data; pipeline loads `fusion.model_path` when present and falls back to naive fusion when absent; no regressions.

**Scope note:** Tasks 1.1 (seed dataset construction), the *execution* of 1.2–1.4 against real data, and 1.5 (end-to-end calibration validation) are moved to Phase 7. Phase 1 here delivers *infrastructure only*: the code/scripts exist and are tested with synthetic fixtures, but no `models/*.pkl` or `data/seed_dataset.jsonl` artifacts are produced.

---

### Task 1.1: Build seed labeled dataset

**Files to create:**
- `data/seed/human/` — directory of confirmed human-authored text files
- `data/seed/llm_a0/` — directory of raw LLM output (A0)
- `data/seed/llm_a1/` — directory of lightly cleaned LLM output (A1)
- `data/seed/llm_a2/` — directory of prompt-coached LLM output (A2)
- `data/seed_dataset.jsonl` — assembled JSONL
- `scripts/build_seed_dataset.py` — assembly script

**Approach:**

1. **Human samples (50+):** Collect from confirmed human-authored sources: pre-2022 task prompts (before widespread LLM adoption), published SOPs, clinical protocols, educator assignments, researcher notes. Draw from multiple occupations: clinical pharmacologist, nurse educator, research scientist, engineer, data analyst.

2. **LLM A0 samples (30+):** Generate task prompts using Claude and GPT-4 with no evasion instructions. Topics should match the human samples' occupations. Include the raw preamble ("Here's a comprehensive...").

3. **LLM A1 samples (20+):** Take A0 samples and manually strip preamble, swap 2–3 fingerprint words, add a personal sentence at top.

4. **LLM A2 samples (15+):** Generate task prompts with explicit instructions: "Write casually, include occasional typos, avoid numbered lists, don't use words like 'ensure' or 'comprehensive'."

5. **Assembly script:** Uses `beet.evaluation.dataset.build_dataset()` to create the JSONL with proper tier labels.

**Minimum dataset size:** 115 samples (50 human + 30 A0 + 20 A1 + 15 A2). Target 150+.

**Verification:** `python scripts/build_seed_dataset.py && beet eval data/seed_dataset.jsonl --output text` runs without errors.

---

### Task 1.2: Implement detector-level calibration

**Files to modify:**
- `beet/fusion/training.py`
- Each detector's `calibrate()` method (currently all `pass`)

**Files to create:**
- `beet/calibration.py` — shared calibration utilities
- `scripts/calibrate_detectors.py`

**Approach:**

Implement isotonic regression calibration per detector. The `calibrate()` method on each detector will fit a mapping from `raw_score → p_llm` using labeled data.

**`beet/calibration.py`:**

```python
from sklearn.isotonic import IsotonicRegression
import numpy as np
import json
from pathlib import Path

class DetectorCalibrator:
    """Fits and stores isotonic regression for raw_score → p_llm mapping."""
    
    def __init__(self):
        self._models: dict[str, IsotonicRegression] = {}
    
    def fit(self, detector_id: str, raw_scores: list[float], labels: list[int]) -> None:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(np.array(raw_scores), np.array(labels, dtype=float))
        self._models[detector_id] = ir
    
    def transform(self, detector_id: str, raw_score: float) -> float:
        if detector_id not in self._models:
            return raw_score  # uncalibrated fallback
        return float(self._models[detector_id].predict([[raw_score]])[0])
    
    def save(self, path: Path) -> None:
        # Serialize isotonic models as (X_, y_) pairs
        data = {}
        for det_id, ir in self._models.items():
            data[det_id] = {"X": ir.X_.tolist(), "y": ir.y_.tolist()}
        path.write_text(json.dumps(data))
    
    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        for det_id, params in data.items():
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.X_ = np.array(params["X"])
            ir.y_ = np.array(params["y"])
            ir.X_min_ = ir.X_[0]
            ir.X_max_ = ir.X_[-1]
            ir.f_ = None  # will use X_, y_ directly
            self._models[det_id] = ir
```

**`scripts/calibrate_detectors.py`:**

1. Load seed dataset.
2. Run each detector (Tier 1 only initially — no ML model dependency) on every sample.
3. Collect `(raw_score, label)` pairs per detector.
4. Fit isotonic regression per detector.
5. Save calibration parameters to `models/detector_calibration.json`.

**Integration:** The pipeline loads calibration on startup and applies it after each detector's `analyze()` call, overwriting the detector's self-assigned `p_llm` with the calibrated value.

**Verification:** Plot reliability diagrams for each detector showing calibrated p_llm vs. observed frequency. ECE per detector < 0.15.

---

### Task 1.3: Train first EBM model

**Files to modify:**
- `beet/fusion/training.py` (already has `train_ebm`)
- `beet/fusion/ebm.py` (load trained model on startup)

**Files to create:**
- `scripts/train_fusion.py`
- `models/ebm_v1.pkl`

**Approach:**

1. Run the full pipeline (Tier 1 detectors) on the seed dataset.
2. Extract feature vectors from `FeatureAssembler.assemble()` for each sample.
3. Train EBM using `train_ebm(X, y)`.
4. Evaluate on held-out split (80/20 stratified by tier).
5. Save model to `models/ebm_v1.pkl`.

**`scripts/train_fusion.py`:**

```python
"""Train EBM fusion model from labeled dataset."""
from pathlib import Path
from beet.evaluation.dataset import load_dataset
from beet.pipeline import BeetPipeline
from beet.fusion.ebm import FeatureAssembler
from beet.fusion.training import train_ebm, save_model
from sklearn.model_selection import train_test_split

dataset = load_dataset("data/seed_dataset.jsonl")
pipeline = BeetPipeline.from_config_file("configs/screening.yaml")  # Tier 1 only
assembler = FeatureAssembler()

# Run detectors and assemble features
X, y = [], []
for sample in dataset:
    det = pipeline.analyze(sample.text)
    # Need to access intermediate results — add a method to pipeline
    # that returns (determination, layer_results, router_decision)
    ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = train_ebm(X_train, y_train)
save_model(model, Path("models/ebm_v1.pkl"))
```

**Pipeline modification required:** Add `analyze_detailed()` method to `BeetPipeline` that returns `(Determination, list[LayerResult], RouterDecision)` so the training script can access layer results and router metadata:

```python
def analyze_detailed(self, text: str) -> tuple[Determination, list[LayerResult], RouterDecision]:
    text = normalize_text(text)
    router_decision = self._router.route(text)
    results = []
    # ... run cascade ...
    fusion_result = self._fusion.fuse(results, router_decision.word_count, router_decision.domain)
    determination = self._decision.decide(fusion_result, results)
    return determination, results, router_decision
```

**Config modification:** Add `fusion.model_path` to config:

```yaml
fusion:
  model_path: "models/ebm_v1.pkl"  # null = use naive fusion
```

**EBM loading in `pipeline.py`:**

```python
from beet.fusion.training import load_model

class BeetPipeline:
    def __init__(self, config: dict):
        # ...
        model_path = config.get("fusion", {}).get("model_path")
        if model_path and Path(model_path).exists():
            model = load_model(Path(model_path))
            self._fusion = EBMFusion(model=model)
        else:
            self._fusion = DEFAULT_FUSION
```

**Verification:** `beet eval data/seed_dataset.jsonl` shows AUROC ≥ 0.85 and ECE ≤ 0.10.

---

### Task 1.4: Wire conformal prediction into pipeline

**Files to modify:**
- `beet/pipeline.py`
- `beet/fusion/ebm.py`
- `beet/fusion/conformal.py`

**Files to create:**
- `scripts/calibrate_conformal.py`
- `models/conformal_cal.json`

**Approach:**

1. **Calibration script:** Takes a held-out calibration set (20% of seed data, separate from EBM training split), runs the trained pipeline, collects `(p_llm, label)` pairs, and calls `ConformalWrapper.calibrate()`.

2. **Integration:** The `EBMFusion` class wraps its output through the `ConformalWrapper` to produce statistically valid prediction sets.

3. **Serialization:** Save/load the conformal threshold alongside the EBM model.

**`scripts/calibrate_conformal.py`:**

```python
from beet.fusion.conformal import ConformalWrapper
import numpy as np, json
from pathlib import Path

# ... load pipeline, run on calibration set ...
wrapper = ConformalWrapper(alpha=0.05)
wrapper.calibrate(np.array(p_llms), np.array(labels))
# Save threshold
Path("models/conformal_cal.json").write_text(
    json.dumps({"alpha": 0.05, "threshold": wrapper._threshold})
)
```

**Modification to `EBMFusion`:**

Add conformal wrapper as an optional component:

```python
class EBMFusion:
    def __init__(self, model=None, conformal: ConformalWrapper | None = None):
        self._model = model
        self._conformal = conformal
        # ...
    
    def fuse(self, ...):
        # ... compute p_llm via EBM or naive ...
        if self._conformal is not None:
            prediction_set = self._conformal.predict_set(p_llm)
        else:
            prediction_set = _p_llm_to_labels(p_llm, uncertainty)
        # ...
```

**Verification:** On the calibration set, verify: `sum(true_label in prediction_set) / n >= 0.95`. This is the coverage guarantee.

---

### Task 1.5: End-to-end calibration validation

**Files to create:**
- `scripts/validate_calibration.py`
- `tests/test_calibration.py`

**This task runs after 1.2–1.4 are complete.**

**Validation checks:**

1. **ECE ≤ 0.10** on held-out data. If the system says P(LLM)=0.6, approximately 60% of such samples should actually be LLM-generated.

2. **Conformal coverage ≥ 95%** at α=0.05. The true label should fall within the prediction set at least 95% of the time.

3. **AUROC ≥ 0.85** on the seed dataset (entire dataset, cross-validated).

4. **Regression test update:** Update `tests/test_regression.py` p_llm assertions to match calibrated behavior. The calibrated model will produce different p_llm values than the naive fusion.

5. **Reliability diagrams:** Generate and visually inspect reliability diagrams showing calibration quality per detector and for the overall fusion.

**Verification:** All checks pass. Commit calibrated models to `models/`.

---

## Phase 2: Missing Tier 2–3 Detectors

**Goal:** Implement the two highest-value missing detectors and expand the adversarial testing framework.
**Duration:** 5–8 days
**Prerequisites:** Phase 1 (calibration pipeline exists to calibrate new detectors)
**Test gate:** New detectors pass unit tests; EBM retrained with expanded feature set; AUROC improves.

---

### Task 2.1: Implement perturbation curvature detector

**Files to create:**
- `beet/detectors/perturbation.py`
- `tests/test_detectors/test_perturbation.py`

**Theory:** DetectGPT shows that LLM-generated text occupies local maxima of log-probability space. Small perturbations (via mask-filling model) decrease log-probability for LLM text but have mixed effects on human text.

**Implementation:**

```python
class PerturbationDetector:
    id = "perturbation"
    domain = "universal"
    compute_cost = "expensive"
    
    def __init__(self, ref_model="gpt2", perturb_model="t5-small", n_perturbations=25):
        # Load reference LM (reuse from surprisal_dynamics if available)
        # Load T5-small for mask-filling perturbations
        ...
    
    def analyze(self, text: str, config: dict) -> LayerResult:
        # 1. Compute log_prob(original) under reference model
        # 2. Generate k perturbations via T5 mask-filling:
        #    - Randomly mask 15% of tokens
        #    - Fill with T5
        #    - Compute log_prob(perturbation) for each
        # 3. perturbation_discordance = mean(log_prob(original) - log_prob(perturbation))
        # 4. Positive discordance → text at local max → likely LLM
        ...
```

**Key implementation detail:** Share the reference model with `SurprisalDynamicsDetector` to avoid loading GPT-2 twice. Add a model registry/cache in `beet/detectors/` or pass a shared model instance via config.

**Model sharing approach:**

Create `beet/detectors/_model_cache.py`:
```python
_cache = {}

def get_model(name: str):
    if name not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _cache[name] = {
            "model": AutoModelForCausalLM.from_pretrained(name),
            "tokenizer": AutoTokenizer.from_pretrained(name)
        }
    return _cache[name]
```

**Features emitted:** `perturbation_discordance`, `perturbation_variance`, `curvature_z_score`

**Config:**
```yaml
perturbation:
  enabled: false  # opt-in (expensive)
  weight: 0.80
  n_perturbations: 25
  ref_model: "gpt2"
  perturb_model: "t5-small"
```

**Tests:**
- Unit test with a known LLM text → positive discordance
- Unit test with human text → near-zero or negative discordance
- Test that SKIP is returned for text < 100 words
- Test that the detector respects `n_perturbations` config

**Verification:** `pytest tests/test_detectors/test_perturbation.py -v`

---

### Task 2.2: Implement contrastive generation engine

**Files to create:**
- `beet/detectors/contrastive_gen.py`
- `tests/test_detectors/test_contrastive_gen.py`

**Theory:** Instead of asking "does this look like LLM text?" (static archetypes), ask "does this look different from what an LLM would generate for this exact task?" Compare the submission against dynamically generated baselines for the same task.

**Implementation:**

```python
class ContrastiveGenDetector:
    id = "contrastive_gen"
    domain = "universal"
    compute_cost = "expensive"
    
    def analyze(self, text: str, config: dict) -> LayerResult:
        task_description = config.get("task_description")
        if not task_description:
            return LayerResult(..., determination="SKIP", 
                              signals={"skipped": "no_task_metadata"})
        
        # 1. Generate k=3 synthetic baselines for this task
        baselines = self._generate_baselines(task_description, config)
        
        # 2. Compute distances
        #    a. Sentence-embedding cosine distance (semantic)
        #    b. N-gram overlap (BScore-style surface)
        #    c. Structural edit distance (section ordering)
        
        # 3. Minimum distance = primary signal
        #    Low distance to any baseline → suspicious
        ...
    
    def _generate_baselines(self, task_desc: str, config: dict) -> list[str]:
        provider = config.get("provider", "anthropic")
        model = config.get("model", "claude-sonnet-4-6")
        n = config.get("n_baselines", 3)
        # Generate with varied temperatures for diversity
        ...
```

**Critical privacy constraint:** The *task description* goes to the API, never the submission text itself. Document this in the config comments.

**Integration with pipeline:** The `analyze()` method on `BeetPipeline` needs to pass `task_metadata` through to detectors. Modify `_run_phase`:

```python
def _run_phase(self, phase, text, config, skip, task_metadata=None):
    # ...
    det_config = detector_cfg.get(det_id, {})
    if task_metadata:
        det_config = {**det_config, **task_metadata}
    result = detector.analyze(text, det_config)
```

**Features emitted:** `min_semantic_distance`, `mean_semantic_distance`, `ngram_overlap_max`, `structural_edit_distance`, `feature_vector_distance`

**Tests:**
- Test with `task_description` → produces distances
- Test without `task_description` → returns SKIP
- Test that submission text is NOT included in API calls (mock the API and assert the prompt only contains the task description)

---

### Task 2.3: Add prompt-based adversarial attacks

**Files to modify:**
- `beet/adversarial/transforms.py`

**Currently the adversarial framework has 5 transforms (all post-processing). Add prompt-based attacks that generate evasion text via LLM:**

```python
def coached_casual(text: str, *, provider: Callable, seed: int | None = None, **_) -> str:
    """Re-generate the content with casual coaching."""
    prompt = (
        "Rewrite the following text in a casual, conversational tone. "
        "Include occasional typos and informal language. "
        "Avoid numbered lists, bullet points, and words like 'ensure', "
        "'comprehensive', 'robust', or 'facilitate'. "
        "Make it sound like a real person wrote it quickly.\n\n"
        f"{text}"
    )
    return provider(prompt)

def paraphrase_launder(text: str, *, provider: Callable, seed: int | None = None, **_) -> str:
    """Paraphrase to remove stylistic fingerprints."""
    prompt = (
        "Paraphrase the following text completely. Use different sentence "
        "structures, vocabulary, and phrasing while preserving the meaning. "
        "Do not add any preamble.\n\n"
        f"{text}"
    )
    return provider(prompt)

register(Attack(name="coached_casual", category="prompt", 
                description="LLM re-generates with casual coaching",
                severity="moderate", apply=coached_casual))
register(Attack(name="paraphrase_launder", category="prompt",
                description="LLM paraphrases to remove fingerprints",
                severity="advanced", apply=paraphrase_launder))
```

These attacks require a `provider` callable — the generator harness already supports this via the `category="prompt"` check.

---

### Task 2.4: Expand labeled dataset with A3 samples

**Files to create:**
- `data/seed/llm_a3/` — paraphrase-laundered samples
- `scripts/generate_a3_samples.py`

**Approach:**
1. Take A0 samples.
2. Run through `paraphrase_launder` attack using Claude Sonnet.
3. Verify manually that they're meaningfully paraphrased (not just synonym swaps).
4. Add to dataset as `tier=A3`.
5. Rebuild the JSONL.
6. Retrain EBM with the expanded dataset and new detector features.

---

### Task 2.5: Retrain EBM with expanded feature set

After Tasks 2.1–2.4, the feature vector has grown from 28 to ~33 features (adding perturbation and contrastive generation signals). Retrain:

1. Run `scripts/calibrate_detectors.py` — recalibrate all detectors including new ones.
2. Run `scripts/train_fusion.py` — retrain EBM on expanded features.
3. Run `scripts/calibrate_conformal.py` — recalibrate conformal sets.
4. Run `beet eval data/seed_dataset.jsonl` — verify AUROC improved.
5. Run `beet ablation data/seed_dataset.jsonl --confirm` — verify new detectors are contributing.
6. Update regression test assertions.

---

## Phase 3: Robustness & Fairness Evaluation

**Goal:** Implement the evaluation modules that answer "how well does this work under attack?" and "is it fair?"
**Duration:** 3–5 days
**Prerequisites:** Phase 2 (adversarial framework has prompt-based attacks)
**Test gate:** Robustness report generates; fairness report generates; no crashes on public benchmarks.

---

### Task 3.1: Implement robustness evaluation module

**Files to create:**
- `beet/evaluation/robustness.py`
- `tests/test_evaluation/test_robustness.py`

**Design:**

```python
@dataclass(frozen=True)
class RobustnessReport:
    baseline_metrics: dict              # metrics on unattacked data
    per_attack: dict[str, dict]         # attack_name → metrics
    per_tier: dict[str, dict]           # tier → metrics
    attack_deltas: dict[str, dict]      # attack → delta from baseline
    vulnerability_ranking: list[tuple]  # (attack, delta_auroc) sorted by impact

def run_robustness_eval(
    pipeline,
    clean_dataset: list[EvalSample],
    attacks: list[str],
    *,
    provider: Callable | None = None,
    seed: int = 42,
    progress: bool = False,
) -> RobustnessReport:
    """
    1. Evaluate on clean dataset → baseline metrics
    2. For each attack:
       a. Generate adversarial samples from LLM-labeled data
       b. Combine with original human data (attacks don't apply to human text)
       c. Evaluate → per-attack metrics
       d. Compute delta from baseline
    3. Rank attacks by impact
    """
```

**CLI command:**
```bash
beet robustness data/seed_dataset.jsonl \
  --attacks strip_preamble,inject_typos,casualize,synonym_swap,mix_human \
  --output json --out robustness_report.json
```

---

### Task 3.2: Implement fairness evaluation module

**Files to create:**
- `beet/evaluation/fairness.py`
- `tests/test_evaluation/test_fairness.py`

**Design:**

```python
@dataclass(frozen=True)
class FairnessReport:
    overall_fpr: float
    per_group_fpr: dict[str, float]     # group → FPR
    fpr_parity_ratio: float             # max_fpr / min_fpr
    calibration_per_group: dict[str, float]  # group → ECE
    flagged_disparities: list[str]      # human-readable disparity warnings

def run_fairness_eval(
    pipeline,
    dataset: list[EvalSample],
    *,
    group_key: str = "tier",           # which field to stratify by
    threshold: float = 0.50,            # decision threshold for FPR
    max_fpr_ratio: float = 2.0,        # alert if any group's FPR is 2x another
) -> FairnessReport:
```

**Group keys:** Initially stratify by `tier` (attacker tier) and `source` (occupation/domain). Future: add language proficiency proxies (vocabulary diversity, avg word length as crude bins).

---

### Task 3.3: Wire adversarial → evaluation in CLI

**Files to modify:**
- `beet/cli.py`

**Add commands:**

```bash
# Robustness evaluation
beet robustness data/seed_dataset.jsonl --attacks all --confirm

# Fairness evaluation
beet fairness data/seed_dataset.jsonl --group-by source --threshold 0.50
```

---

### Task 3.4: Run first public benchmark

**Files to create:**
- `scripts/run_benchmark.py`
- `scripts/download_hc3.py` (or `download_raid.py`)

**Approach:**

Start with **HC3** (Human ChatGPT Comparison Corpus) — it's small, well-labeled, and available. Convert HC3's format to BEET's `EvalSample` JSONL, run the pipeline, report metrics.

**Expected outcome:** This is the first externally verifiable accuracy number. Whatever it is, report it honestly. The system isn't designed for general AI text detection (it's tuned for prompt provenance), so HC3 performance may be moderate. That's fine — it establishes a baseline.

If HC3 is not available or not representative enough, use **RAID** (larger, adversarial-focused, more representative of the threat model).

**Verification:** Benchmark report generated with AUROC, ECE, TPR@0.01% FPR. Committed to `docs/benchmarks/`.

---

## Phase 4: Batch-Mode Detectors

**Goal:** Implement cross-submission similarity and contributor graph analysis for batch/periodic operation.
**Duration:** 5–7 days
**Prerequisites:** Phase 1 (privacy store exists for feature vectors)
**Test gate:** Batch analysis produces meaningful results on synthetic corpora.

---

### Task 4.1: Implement cross-submission similarity detector

**Files to create:**
- `beet/detectors/cross_similarity.py`
- `tests/test_detectors/test_cross_similarity.py`

**Design:**

This detector is different — it operates on a corpus, not a single text. It needs a `BatchDetector` protocol extension:

```python
class CrossSimilarityDetector:
    id = "cross_similarity"
    domain = "universal"
    compute_cost = "batch"
    
    def analyze(self, text: str, config: dict) -> LayerResult:
        # Single-text mode: return SKIP
        return LayerResult(..., determination="SKIP")
    
    def analyze_batch(self, texts: dict[str, str], config: dict) -> dict[str, LayerResult]:
        """
        texts: {submission_id: text} for an entire batch
        Returns: {submission_id: LayerResult} with similarity signals
        
        1. Compute word-shingle Jaccard (surface similarity)
        2. Compute sentence-embedding cosine (semantic similarity)
        3. Compute structural fingerprint distance (formatting, section structure)
        4. For each text, report max similarity to any other text
        5. Detect template matches (multiple texts matching known LLM structural templates)
        """
```

**Similarity computation uses LSH for efficiency** on batches > 100 texts.

**Config:**
```yaml
cross_similarity:
  enabled: true  # in batch mode
  weight: 0.70
  jaccard_threshold: 0.40
  semantic_threshold: 0.85
  use_lsh: true
  lsh_num_perm: 128
```

---

### Task 4.2: Implement contributor graph / syndicate detection

**Files to create:**
- `beet/detectors/contributor_graph.py`
- `tests/test_detectors/test_contributor_graph.py`

**Design:**

```python
class ContributorGraphDetector:
    id = "contributor_graph"
    domain = "universal"
    compute_cost = "batch"
    
    def analyze_contributors(
        self,
        submissions: dict[str, list[dict]],  # contributor_id → [submission records]
        config: dict,
    ) -> dict[str, dict]:
        """
        1. Build graph: nodes = contributors, edges = similarity between submission sets
        2. Node features: avg p_llm, flag rate, submission velocity, vocabulary diversity
        3. Edge features: semantic overlap, structural similarity, temporal patterns
        4. Community detection: Louvain clustering
        5. Anomaly scoring: syndicate_risk_score per contributor
        """
```

**Dependencies:** `networkx`, `scikit-learn` (Louvain from `sklearn.cluster` or `community` package).

**This runs periodically (nightly/weekly), not per-submission.**

---

### Task 4.3: Add batch analysis to pipeline and CLI

**Files to modify:**
- `beet/pipeline.py` — add `analyze_batch()` method
- `beet/cli.py` — upgrade `batch` command

**Pipeline batch mode:**

```python
def analyze_batch(self, texts: dict[str, str], task_metadata: dict = None) -> dict[str, Determination]:
    """
    1. Run per-submission analysis on each text
    2. Run cross-submission similarity across the batch
    3. Merge batch-level signals into per-submission results
    4. Re-run fusion and decision with enriched features
    """
```

**CLI:**
```bash
beet batch input.xlsx --prompt-col prompt --attempter-col attempter \
  --cross-similarity --output results.csv
```

---

## Phase 5: Privacy, Provenance & Monitoring Hardening

**Goal:** Complete the privacy, provenance, and monitoring subsystems from stubs to production-ready.
**Duration:** 3–4 days
**Prerequisites:** Phase 1 (pipeline produces real results worth storing)
**Test gate:** Retention purge works; audit chain validates; drift alerts fire on synthetic drift.

---

### Task 5.1: Implement privacy retention / auto-purge

**Files to create:**
- `beet/privacy/retention.py`
- `tests/test_privacy_retention.py`

**Design:**

```python
class RetentionManager:
    def __init__(self, vault_dir: Path, retention_days: int = 90):
        self._vault = vault_dir
        self._retention = retention_days
    
    def purge_expired(self) -> int:
        """Delete vault files older than retention_days. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=self._retention)
        count = 0
        for f in self._vault.glob("*.json"):
            record = json.loads(f.read_text())
            stored_at = datetime.fromisoformat(record["stored_at"])
            if stored_at < cutoff:
                f.unlink()
                count += 1
        return count
    
    def schedule(self, interval_hours: int = 24):
        """Register a background purge schedule (for long-running API server)."""
        ...
```

**CLI command:**
```bash
beet privacy purge --vault-dir /path/to/vault --dry-run
beet privacy purge --vault-dir /path/to/vault --confirm
```

---

### Task 5.2: Implement hash-chained provenance audit log

**Files to create:**
- `beet/provenance/chain.py`
- `tests/test_provenance_chain.py`

**Design:**

```python
class AuditChain:
    def __init__(self, log_path: Path):
        self._path = log_path
    
    def append(self, manifest: dict) -> dict:
        """
        1. Read the last entry to get its hash
        2. Set manifest["_prev_hash"] = that hash
        3. Compute manifest's own hash
        4. Append to the log
        """
        prev_hash = self._get_last_hash()
        manifest["_prev_hash"] = prev_hash
        content = json.dumps(manifest, sort_keys=True).encode()
        manifest["_hash"] = hashlib.sha256(content).hexdigest()
        with open(self._path, "a") as f:
            f.write(json.dumps(manifest) + "\n")
        return manifest
    
    def validate(self) -> tuple[bool, list[str]]:
        """Verify the entire chain. Returns (valid, list of error messages)."""
        entries = [json.loads(line) for line in open(self._path)]
        errors = []
        for i, entry in enumerate(entries):
            if i == 0:
                if entry.get("_prev_hash") != "genesis":
                    errors.append(f"Entry 0: expected genesis, got {entry.get('_prev_hash')}")
            else:
                expected_prev = entries[i-1]["_hash"]
                if entry.get("_prev_hash") != expected_prev:
                    errors.append(f"Entry {i}: chain broken")
        return len(errors) == 0, errors
```

---

### Task 5.3: Upgrade drift monitor

**Files to modify:**
- `beet/monitoring/drift.py`

**Current state:** Basic mean-p_llm alerting. Upgrade to:

1. **Per-feature distribution tracking:** Store running statistics (mean, variance) for each feature. Alert when KL divergence exceeds threshold.

2. **Calibration drift:** Track predicted P(LLM) vs. confirmed outcomes over time. Alert when ECE exceeds threshold.

3. **Signal correlation monitoring:** Track pairwise correlation between detector p_llm values. Alert when correlation structure changes (detectors becoming redundant or diverging).

**Implementation adds:**

```python
def _check_feature_drift(self) -> list[str]:
    """Compare current window's feature distributions against stored baselines."""
    ...

def _check_calibration_drift(self, confirmed_labels: list[int]) -> list[str]:
    """Compare predicted p_llm against confirmed outcomes."""
    ...
```

---

### Task 5.4: Wire monitoring into pipeline

**Files to modify:**
- `beet/pipeline.py`

**Add optional monitoring hook:**

```python
class BeetPipeline:
    def __init__(self, config: dict):
        # ...
        self._monitor = None
        if config.get("monitoring", {}).get("enabled"):
            from beet.monitoring.drift import DriftMonitor
            self._monitor = DriftMonitor(Path("data/monitoring"), config)
    
    def analyze(self, text: str, ...):
        # ... existing analysis ...
        if self._monitor:
            feature_vec = self._fusion._assembler.assemble(results, ...)
            self._monitor.record(determination.p_llm, determination.label, feature_vec)
        return determination
```

---

### Task 5.5: Implement meta-detector

**Files to create:**
- `beet/monitoring/meta_detector.py`
- `tests/test_monitoring.py`

**Design:**

The meta-detector monitors individual detector performance over time:

```python
class MetaDetector:
    """Monitors whether individual detectors are degrading."""
    
    def __init__(self, window_size: int = 500):
        self._per_detector: dict[str, list[dict]] = defaultdict(list)
    
    def record(self, layer_result: LayerResult, confirmed_label: int | None = None):
        self._per_detector[layer_result.layer_id].append({
            "p_llm": layer_result.p_llm,
            "determination": layer_result.determination,
            "confirmed": confirmed_label,
        })
    
    def check_degradation(self) -> dict[str, str]:
        """
        For each detector, check:
        1. Is its p_llm distribution shifting?
        2. If confirmed labels exist, is its accuracy declining?
        3. Is it firing on known-human text more often?
        Returns: {detector_id: "degrading" | "stable" | "unknown"}
        """
```

---

## Phase 6: API & GUI

**Goal:** Expose the pipeline via a REST API and provide a web interface for non-technical users.
**Duration:** 5–8 days
**Prerequisites:** Phases 0–5
**Test gate:** API serves requests; GUI renders and produces results; both match CLI output for same input.

---

### Task 6.1: Implement FastAPI REST server

**Files to create:**
- `beet/api.py`
- `tests/test_api.py`

**Endpoints:**

```
POST /analyze          — single text analysis
POST /batch            — batch analysis (JSON array or CSV upload)
GET  /health           — health check + model version
GET  /config           — current config profile
POST /eval             — run evaluation on uploaded JSONL
```

**`POST /analyze` request/response:**

```json
// Request
{
  "text": "Here's a comprehensive...",
  "task_metadata": {"task_description": "...", "occupation": "..."},
  "config_profile": "default"
}

// Response
{
  "determination": "RED",
  "p_llm": 0.92,
  "confidence_interval": [0.85, 0.96],
  "prediction_set": ["RED"],
  "reason": "...",
  "top_features": [...],
  "detectors_run": [...],
  "cascade_phases": [1],
  "override_applied": true
}
```

**Implementation uses FastAPI with Pydantic models for request/response validation.** The pipeline is a singleton initialized on startup.

---

### Task 6.2: Build web GUI

**Files to create:**
- `beet/gui/` — directory
- `beet/gui/server.py` — serves the embedded SPA
- `beet/gui/static/index.html` — single-file embedded SPA

**Design principles (from the spec):**
- Accessible, polished GUI suitable for non-technical users (QA reviewers, project managers, content operations staff)
- Zero external network dependencies (fully self-contained)
- Shows: text input, determination badge, P(LLM) gauge, confidence interval, prediction set, top contributing signals with explanations, per-detector breakdown
- Supports batch upload (CSV/XLSX) with downloadable results
- Color-coded results: RED/AMBER/YELLOW/GREEN/UNCERTAIN

**Implementation approach:** Single HTML file with embedded JS/CSS (like BEET v1's GUI). Served by a lightweight Python HTTP server. Calls the pipeline directly (no API dependency, though the API can be used as backend if available).

**CLI:**
```bash
beet gui --port 8877  # auto-opens browser
```

---

### Task 6.3: Production deployment configuration

**Files to create:**
- `configs/production.yaml` — production profile
- `Dockerfile`
- `docker-compose.yaml`
- `docs/deployment.md`

**Production config:**
```yaml
_profile: production
_extends: default

fusion:
  model_path: "/app/models/ebm_v1.pkl"
  conformal_path: "/app/models/conformal_cal.json"

monitoring:
  enabled: true
  store_path: "/data/monitoring"

privacy:
  raw_text_retention_days: 90
  access_logging: true
  vault_dir: "/data/vault"
```

---

## Phase 7: Second-Generation Training & Continuous Improvement

**Goal:** Expand the dataset, retrain with full feature set, establish recurring calibration cadence.
**Duration:** Ongoing
**Prerequisites:** Phases 0–6

---

### Task 7.1: Expand labeled dataset to 500+ samples

**Target composition:**

| Tier | Count | Source |
|------|-------|--------|
| Human | 200 | Vetted contributors, pre-LLM era archives, multiple occupations |
| A0 | 80 | Raw Claude/GPT-4/Gemini output, various task types |
| A1 | 60 | Light cleanup of A0 (manual preamble removal, word swaps) |
| A2 | 60 | Prompt-coached generation ("write casually, avoid...") |
| A3 | 50 | Paraphrase-laundered via second LLM |
| A4 | 30 | Manually constructed mixed-authorship (human scaffold + LLM fill) |
| A5 | 20 | Adversarial expert edits (knowing detection methods) |

**A4 construction:** Manually write human intro/transitions, insert LLM-generated body sections, record boundary positions as ground truth for mixed-boundary detector evaluation.

**A5 construction:** Review the detector signal documentation, then write text that deliberately avoids all signals while maintaining LLM-generated content. This tests the system's theoretical ceiling.

---

### Task 7.2: Retrain EBM with full feature set

After all Tier 1–3 detectors are implemented and calibrated:

1. Run full pipeline (all detectors) on the expanded dataset.
2. Feature vector now has ~35 features (Tier 1 + Tier 2 + Tier 3 + metadata).
3. Train EBM with `interactions=10` (increased from 5 for richer interaction learning).
4. Cross-validate with stratified 5-fold.
5. Run ablation to identify redundant detectors.
6. If any detector has `verdict="hurting"` in ablation, investigate and consider disabling.

---

### Task 7.3: Occupation-stratified calibration

**Problem:** IDI=12 is suspicious for creative writing but normal for clinical SOPs. The EBM partially handles this via `domain_prompt`/`domain_prose` features, but occupation-level calibration needs explicit support.

**Approach:**

1. Add `occupation` as a feature in the EBM feature vector (encoded as a categorical via target encoding or occupation-group mapping).
2. Alternatively, train separate calibration curves per occupation group:
   - Clinical (pharmacologist, clinician, nurse)
   - Engineering (software, biomedical, chemical)
   - Research (scientist, data analyst)
   - Education (educator, instructor)
3. The pipeline selects the appropriate calibration based on `task_metadata["occupation"]`.

**Implementation:** Add `occupation_group` as a field in `FeatureAssembler.assemble()` and let the EBM learn the interaction naturally. This is simpler than separate models and the EBM's interaction terms handle it.

---

### Task 7.4: Monthly recalibration protocol

**Document:** `docs/recalibration_protocol.md`

**Monthly checklist:**

1. **Collect new confirmed labels:** Review AMBER/UNCERTAIN determinations from the past month. Get human review confirmations. Add to the labeled dataset.
2. **Check drift alerts:** Review `monitoring/alerts.jsonl` for any drift alerts.
3. **Run evaluation:** `beet eval data/labeled_dataset.jsonl --output json --out reports/eval_YYYY-MM.json`
4. **Check calibration:** ECE still ≤ 0.10? Conformal coverage still ≥ 95%?
5. **Retrain if needed:** If ECE > 0.10 or AUROC dropped > 0.02 from previous month, retrain EBM.
6. **Generate new LLM samples:** Use the latest models (GPT-5, Claude next, etc.) to generate fresh A0–A2 samples. Check if existing detectors catch them.
7. **Run robustness evaluation:** `beet robustness data/labeled_dataset.jsonl --attacks all --confirm`
8. **Update regression tests:** Adjust p_llm assertions if calibration shifted.

---

## Dependency Graph

```
Phase 0 ─── Bug Fixes
  │
  ▼
Phase 1 ─── Calibration Pipeline
  │
  ├──────────────────────────┐
  ▼                          ▼
Phase 2                    Phase 5
Missing Detectors          Privacy/Provenance/Monitoring
  │                          │
  ▼                          │
Phase 3                      │
Robustness/Fairness          │
  │                          │
  └──────────┬───────────────┘
             ▼
           Phase 6
           API & GUI
             │
             ▼
           Phase 7
           Second-Gen Training (ongoing)
```

Phases 2 and 5 are independent and can be parallelized. Phase 3 depends on Phase 2 (adversarial attacks). Phase 6 depends on everything being stable. Phase 7 is ongoing.

---

## Effort Estimates

| Phase | Effort | Calendar Time | Dependencies |
|-------|--------|---------------|-------------|
| Phase 0 | 1–2 days | Week 1 | None |
| Phase 1 | 3–5 days | Weeks 1–2 | Phase 0 |
| Phase 2 | 5–8 days | Weeks 2–4 | Phase 1 |
| Phase 3 | 3–5 days | Weeks 4–5 | Phase 2 |
| Phase 4 | 5–7 days | Weeks 3–5 | Phase 1 |
| Phase 5 | 3–4 days | Weeks 3–4 | Phase 1 |
| Phase 6 | 5–8 days | Weeks 5–7 | Phases 0–5 |
| Phase 7 | Ongoing | Monthly | Phase 6 |
| **Total to production** | **~30–40 days** | **~7–8 weeks** | |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Seed dataset too small for reliable EBM training | EBM overfits, calibration unreliable | Medium | Start with naive fusion as baseline; only switch to EBM when ablation shows improvement; use cross-validation to detect overfitting |
| Perturbation detector too slow for practical use | Phase 2 stalls or detector is unusable | Medium | Make it cascade Phase 3 (only runs on borderline cases); reduce default n_perturbations from 25 to 10; profile and optimize |
| Contrastive generation leaks sensitive task metadata | Privacy violation | Low | Document in privacy manifest; provide "no-API" mode; only send task description, never submission text |
| EBM learns spurious features from small dataset | False confidence in calibration | Medium | Run ablation after every training; monitor ECE on held-out data; retrain monthly; keep naive fusion as fallback |
| Public benchmark performance is poor | Credibility issue | Medium | BEET is designed for prompt provenance, not general AI detection; report benchmark results honestly with this caveat; focus on in-domain evaluation |
| Tier 2 detectors require GPU | Limits deployment environments | High | Support CPU-only mode with degraded performance; quantize models; provide screening profile that skips Tier 2 |
| Contributor graph GNN is overkill for current data volume | Over-engineering | Medium | Start with simple community detection (Louvain); defer GNN until data volume justifies it |

---

## File Creation/Modification Summary

### New files to create (across all phases):

```
beet/calibration.py                          Phase 1
beet/detectors/_model_cache.py               Phase 2
beet/detectors/perturbation.py               Phase 2
beet/detectors/contrastive_gen.py            Phase 2
beet/detectors/cross_similarity.py           Phase 4
beet/detectors/contributor_graph.py          Phase 4
beet/evaluation/robustness.py                Phase 3
beet/evaluation/fairness.py                  Phase 3
beet/privacy/retention.py                    Phase 5
beet/provenance/chain.py                     Phase 5
beet/monitoring/meta_detector.py             Phase 5
beet/api.py                                  Phase 6
beet/gui/                                    Phase 6
beet/gui/server.py                           Phase 6
beet/gui/static/index.html                   Phase 6

scripts/build_seed_dataset.py                Phase 1
scripts/calibrate_detectors.py               Phase 1
scripts/train_fusion.py                      Phase 1
scripts/calibrate_conformal.py               Phase 1
scripts/validate_calibration.py              Phase 1
scripts/generate_a3_samples.py               Phase 2
scripts/run_benchmark.py                     Phase 3
scripts/download_hc3.py                      Phase 3

data/seed/human/                             Phase 1
data/seed/llm_a0/                            Phase 1
data/seed/llm_a1/                            Phase 1
data/seed/llm_a2/                            Phase 1
data/seed/llm_a3/                            Phase 2
data/seed_dataset.jsonl                      Phase 1

models/detector_calibration.json             Phase 1
models/ebm_v1.pkl                            Phase 1
models/conformal_cal.json                    Phase 1

tests/test_calibration.py                    Phase 1
tests/test_detectors/test_perturbation.py    Phase 2
tests/test_detectors/test_contrastive_gen.py Phase 2
tests/test_detectors/test_cross_similarity.py Phase 4
tests/test_detectors/test_contributor_graph.py Phase 4
tests/test_evaluation/test_robustness.py     Phase 3
tests/test_evaluation/test_fairness.py       Phase 3
tests/test_privacy_retention.py              Phase 5
tests/test_provenance_chain.py               Phase 5
tests/test_monitoring.py                     Phase 5
tests/test_api.py                            Phase 6

configs/production.yaml                      Phase 6
Dockerfile                                   Phase 6
docker-compose.yaml                          Phase 6

docs/deployment.md                           Phase 6
docs/recalibration_protocol.md               Phase 7
docs/benchmarks/                             Phase 3
```

### Files to modify:

```
beet/pipeline.py              Phases 0, 1, 4, 5
beet/fusion/ebm.py            Phases 0, 1
beet/fusion/conformal.py      Phase 1
beet/detectors/__init__.py    Phase 0
beet/adversarial/transforms.py Phase 2
beet/monitoring/drift.py      Phase 5
beet/cli.py                   Phases 3, 4, 5, 6
beet/config.py                Phase 1 (fusion.model_path)
configs/default.yaml          Phase 1 (fusion section)
tests/test_pipeline.py        Phase 0
tests/test_fusion.py          Phase 0
tests/test_regression.py      Phase 1 (recalibrated assertions)
pyproject.toml                Phase 6 (add api deps)
```

---

## Success Criteria (Per Phase)

| Phase | Gate | Metric |
|-------|------|--------|
| 0 | All existing tests pass; new bug-fix tests pass | Zero regressions |
| 1 | EBM trained, conformal calibrated | AUROC ≥ 0.85, ECE ≤ 0.10, coverage ≥ 95% |
| 2 | New detectors pass tests; EBM retrained | AUROC improves over Phase 1 baseline |
| 3 | Robustness and fairness reports generate | Per-attack AUROC reported; FPR parity ratio < 2.0 |
| 4 | Batch analysis produces meaningful cross-submission signals | Template matches detected in synthetic corpora |
| 5 | Retention purge works; chain validates; drift alerts fire | 100% chain validation on synthetic log; drift alert fires on injected drift |
| 6 | API serves requests; GUI renders | API response matches CLI for same input |
| 7 | Expanded dataset trained; occupation calibration active | AUROC ≥ 0.90 on expanded dataset; ECE ≤ 0.08 |
