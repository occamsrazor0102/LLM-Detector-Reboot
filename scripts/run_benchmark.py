"""Run BEET on a labeled benchmark JSONL and emit a metrics report.

Usage:
    python scripts/run_benchmark.py \
        --dataset data/hc3_finance.jsonl \
        --config configs/default.yaml \
        --out docs/benchmarks/hc3_finance.json

Any JSONL conforming to EvalSample works (HC3, RAID converted, custom).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from beet.evaluation import load_dataset, run_eval
from beet.pipeline import BeetPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    pipeline = BeetPipeline.from_config_file(args.config)
    report = run_eval(pipeline, dataset, progress=args.progress)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    m = report.metrics
    print(f"benchmark: {args.dataset.name}")
    print(f"  samples: {report.n_samples}")
    print(f"  AUROC:   {m['auroc']:.4f}")
    print(f"  ECE:     {m['ece']:.4f}")
    print(f"  Brier:   {m['brier']:.4f}")
    print(f"  TPR@1%FPR: {m['tpr_at_fpr_01']:.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
