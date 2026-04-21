"""Train the EBM fusion model from a labeled JSONL dataset.

Usage:
    python scripts/train_fusion.py \
        --dataset data/seed_dataset.jsonl \
        --config configs/default.yaml \
        --out models/ebm_v1.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from beet.evaluation.dataset import load_dataset
from beet.fusion.ebm import FeatureAssembler
from beet.fusion.training import save_model, train_ebm
from beet.pipeline import BeetPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    pipeline = BeetPipeline.from_config_file(args.config)
    samples = load_dataset(args.dataset)
    assembler = FeatureAssembler()

    X: list[dict] = []
    y: list[int] = []
    for s in samples:
        _, results, rd = pipeline.analyze_detailed(s.text)
        vec = assembler.assemble(results, word_count=rd.word_count, domain=rd.domain)
        X.append(vec)
        y.append(int(s.label))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    model = train_ebm(X_train, y_train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print(f"trained on {len(X_train)}, held out {len(X_test)}, wrote {args.out}")


if __name__ == "__main__":
    main()
