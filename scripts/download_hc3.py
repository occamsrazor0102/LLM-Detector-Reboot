"""Download and convert HC3 (Human ChatGPT Comparison Corpus) to BEET JSONL.

HC3 is hosted on Hugging Face (Hello-SimpleAI/HC3). This script requires the
'datasets' package and internet access. Converts HC3's {question, human_answers,
chatgpt_answers} schema to BEET's EvalSample JSONL.

Usage:
    python scripts/download_hc3.py --subset finance --out data/hc3_finance.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from beet.evaluation.dataset import EvalSample, save_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all",
                    help="HC3 subset: all, finance, medicine, open_qa, reddit_eli5, wiki_csai")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-per-class", type=int, default=200,
                    help="Cap samples per (human/llm) class for manageable size")
    args = ap.parse_args()

    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "HC3 download requires 'datasets'. Install: pip install datasets"
        ) from e

    ds = hf_load("Hello-SimpleAI/HC3", args.subset, split="train")

    samples: list[EvalSample] = []
    human_count = llm_count = 0
    for i, row in enumerate(ds):
        human_answers = row.get("human_answers", []) or []
        chatgpt_answers = row.get("chatgpt_answers", []) or []
        for j, ans in enumerate(human_answers):
            if human_count >= args.max_per_class:
                break
            if not isinstance(ans, str) or not ans.strip():
                continue
            samples.append(EvalSample(
                id=f"hc3_{args.subset}_h_{i}_{j}",
                text=ans.strip(),
                label=0,
                tier="human",
                source=f"hc3_{args.subset}",
            ))
            human_count += 1
        for j, ans in enumerate(chatgpt_answers):
            if llm_count >= args.max_per_class:
                break
            if not isinstance(ans, str) or not ans.strip():
                continue
            samples.append(EvalSample(
                id=f"hc3_{args.subset}_l_{i}_{j}",
                text=ans.strip(),
                label=1,
                tier="A0",
                source=f"hc3_{args.subset}",
            ))
            llm_count += 1
        if human_count >= args.max_per_class and llm_count >= args.max_per_class:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, args.out)
    print(f"wrote {len(samples)} samples (human={human_count}, llm={llm_count}) to {args.out}")


if __name__ == "__main__":
    main()
