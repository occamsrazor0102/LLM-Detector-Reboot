# Benchmarks

This directory is the landing zone for metrics reports produced by
`scripts/run_benchmark.py`. Results are tracked here so anyone can see
how the current pipeline performs on public data — not just the
synthetic fixtures bundled with the repo.

## Status

**No external benchmark has been run yet against the current pipeline.**
The system is in heuristic mode (no trained EBM fusion, no calibrated
conformal wrapper). Any numbers recorded in this directory prior to
training the fusion should be treated as a lower bound on what the
system can do once calibrated.

## HC3 — Human ChatGPT Comparison Corpus

[Hugging Face link](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

One-time install:

```bash
pip install datasets
```

Download a subset:

```bash
python scripts/download_hc3.py \
    --subset finance \
    --out data/hc3_finance.jsonl \
    --max-per-class 200
```

Subsets that ship with HC3: `all`, `finance`, `medicine`, `open_qa`,
`reddit_eli5`, `wiki_csai`. `--max-per-class` caps each of human /
LLM to keep runs manageable.

Run the pipeline against it:

```bash
python scripts/run_benchmark.py \
    --dataset data/hc3_finance.jsonl \
    --config configs/default.yaml \
    --out docs/benchmarks/hc3_finance.json \
    --progress
```

The output JSON has the full EvalReport shape (AUROC, ECE, Brier,
TPR@FPR, per-tier, per-attack, confusion at threshold 0.5 + derived
precision/recall/F1).

Smoke benchmark (100 items, a minute or two):

```bash
python scripts/run_benchmark.py \
    --dataset data/hc3_finance.jsonl \
    --config configs/screening.yaml \
    --out docs/benchmarks/hc3_finance_smoke.json \
    --limit 100
```

`screening` profile keeps it to Tier-1 detectors only, so no
transformers download is required.

## RAID — Robust AI Detector benchmark

[Hugging Face link](https://huggingface.co/datasets/liamdugan/raid)

RAID isn't included in `scripts/download_*`; convert manually:

```python
from datasets import load_dataset
from beet.evaluation.dataset import EvalSample, save_dataset

raid = load_dataset("liamdugan/raid", split="train[:1000]")
samples = [
    EvalSample(
        id=f"raid_{i:06d}",
        text=row["generation"],
        label=0 if row["model"] == "human" else 1,
        tier=row.get("attack") or "raid_base",
        attack_name=row.get("attack"),
        attack_category=row.get("domain"),
    )
    for i, row in enumerate(raid)
]
save_dataset(samples, "data/raid_smoke.jsonl")
```

Then the same `run_benchmark.py` invocation as for HC3.

## Submitting results

After a benchmark run, commit the output JSON to this directory. The
file name should include the dataset name, the profile, and whether
it's a smoke run:

```
docs/benchmarks/
  hc3_finance_default.json
  hc3_finance_screening_smoke.json
  raid_default.json
```

Include the git SHA in the commit message so results can be pinned to
a code version. Do not overwrite historical results — add a dated
suffix if you need to re-run the same combination.

## Interpreting the numbers

In heuristic mode (no trained fusion / conformal):

- **AUROC** is the most meaningful number — it measures ranking ability
  independent of the threshold. Values above ~0.85 suggest the
  ensemble is discriminating; values near 0.5 mean the detector
  stack is adding noise at best.
- **ECE** and **Brier** reflect calibration quality. In heuristic mode
  these will be bad — the per-detector p(LLM) mappings are
  hand-picked. Don't read them as "the system's calibration"; they're
  what's left over after the guesses.
- **TPR@1%FPR** is the operationally relevant number if you care about
  minimising false positives (e.g. academic integrity use).
- **Confusion matrix** at threshold 0.5 is diagnostic, not the
  operating point — tune `decision.*_threshold` in the config to the
  working threshold you actually want.

After training a fusion model and a conformal wrapper, re-run every
benchmark and replace the entries here. At that point ECE and Brier
become meaningful.
