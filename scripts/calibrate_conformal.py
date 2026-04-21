"""Calibrate conformal prediction threshold from a held-out dataset.

Usage:
    python scripts/calibrate_conformal.py \
        --dataset data/seed_conformal.jsonl \
        --config configs/default.yaml \
        --alpha 0.05 \
        --out models/conformal_cal.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beet.evaluation.dataset import load_dataset
from beet.fusion.conformal import ConformalWrapper
from beet.pipeline import BeetPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pipeline = BeetPipeline.from_config_file(args.config)
    samples = load_dataset(args.dataset)

    scores: list[float] = []
    labels: list[int] = []
    for s in samples:
        det = pipeline.analyze(s.text)
        scores.append(float(det.p_llm))
        labels.append(int(s.label))

    wrapper = ConformalWrapper(alpha=args.alpha)
    wrapper.calibrate(np.asarray(scores), np.asarray(labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(args.out)
    print(f"calibrated on {len(scores)} samples (alpha={args.alpha}); wrote {args.out}")


if __name__ == "__main__":
    main()
