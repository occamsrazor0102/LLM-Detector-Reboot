"""Fit per-detector isotonic calibration from a labeled JSONL dataset.

Usage:
    python scripts/calibrate_detectors.py \
        --dataset data/seed_dataset.jsonl \
        --config configs/default.yaml \
        --out models/detector_calibration.json

Runs the pipeline on each sample, collects (raw_score, label) per detector,
and fits a DetectorCalibrator. Execution of this script depends on a labeled
dataset being available (Phase 7).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from beet.calibration import DetectorCalibrator
from beet.evaluation.dataset import load_dataset
from beet.pipeline import BeetPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pipeline = BeetPipeline.from_config_file(args.config)
    samples = load_dataset(args.dataset)

    scores: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        _, results, _ = pipeline.analyze_detailed(s.text)
        for r in results:
            if r.determination == "SKIP":
                continue
            scores[r.layer_id].append(float(r.raw_score))
            labels[r.layer_id].append(int(s.label))

    cal = DetectorCalibrator()
    for det_id in scores:
        cal.fit(det_id, scores[det_id], labels[det_id])
        print(f"fit {det_id}: n={len(scores[det_id])} has={cal.has(det_id)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cal.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
