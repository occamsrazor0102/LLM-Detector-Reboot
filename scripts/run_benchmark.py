"""Run BEET on a labeled benchmark JSONL and emit a metrics report.

Usage:
    python scripts/run_benchmark.py \
        --dataset data/hc3_finance.jsonl \
        --config configs/default.yaml \
        --out docs/benchmarks/hc3_finance.json

Any JSONL conforming to EvalSample works (HC3, RAID converted, custom).
Use --limit to sub-sample a dataset for a quick smoke benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from beet.evaluation import load_dataset, run_eval
from beet.evaluation.runner import eval_report_to_dict
from beet.pipeline import BeetPipeline


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "   n/a"
    return f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--progress", action="store_true",
                    help="Show a tqdm progress bar during evaluation.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Random-subsample the dataset to N items (deterministic with --seed).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    if args.limit is not None and args.limit < len(dataset):
        rng = random.Random(args.seed)
        dataset = rng.sample(dataset, args.limit)

    pipeline = BeetPipeline.from_config_file(args.config)
    t0 = time.monotonic()
    report = run_eval(pipeline, dataset, progress=args.progress)
    duration = time.monotonic() - t0

    # JSON-safe serialization (NaN → None, rounded floats, confusion matrix
    # and derived precision/recall/F1).
    out = eval_report_to_dict(report, include_predictions=False)
    out["dataset"] = str(args.dataset)
    out["config"] = str(args.config)
    out["duration_seconds"] = round(duration, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    m = out["metrics"]
    print(f"benchmark: {args.dataset.name}")
    print(f"  samples:        {out['n_samples']}")
    print(f"  failed:         {out['n_failed']}")
    print(f"  AUROC:          {_fmt(m.get('auroc'))}")
    print(f"  ECE:            {_fmt(m.get('ece'))}")
    print(f"  Brier:          {_fmt(m.get('brier'))}")
    print(f"  TPR@1%FPR:      {_fmt(m.get('tpr_at_fpr_01'))}")
    c = out["confusion"]
    print(f"  accuracy@0.5:   {_fmt(c.get('accuracy'))} (precision {_fmt(c.get('precision'))}, recall {_fmt(c.get('recall'))}, F1 {_fmt(c.get('f1'))})")
    print(f"  duration:       {duration:.1f}s")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
