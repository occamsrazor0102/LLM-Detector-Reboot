"""Download and convert HC3 (Human ChatGPT Comparison Corpus) to BEET JSONL.

HC3 is hosted on Hugging Face as `Hello-SimpleAI/HC3`. The repo publishes
per-subset JSONL files; this script grabs them directly via the Hub API
(the dataset-script loader was deprecated in datasets >= 4.0).

Converts HC3's {question, human_answers, chatgpt_answers} schema to BEET's
EvalSample JSONL.

Usage:
    python scripts/download_hc3.py --subset finance --out data/hc3_finance.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beet.evaluation.dataset import EvalSample, save_dataset


_SUBSET_FILES = {
    "all": "all.jsonl",
    "finance": "finance.jsonl",
    "medicine": "medicine.jsonl",
    "open_qa": "open_qa.jsonl",
    "reddit_eli5": "reddit_eli5.jsonl",
    "wiki_csai": "wiki_csai.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="finance",
                    choices=sorted(_SUBSET_FILES.keys()),
                    help="HC3 subset.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-per-class", type=int, default=200,
                    help="Cap samples per (human/llm) class.")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "HC3 download requires `huggingface_hub`. "
            "Install: pip install huggingface_hub"
        ) from e

    filename = _SUBSET_FILES[args.subset]
    print(f"fetching Hello-SimpleAI/HC3/{filename} …")
    path = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename=filename,
        repo_type="dataset",
    )

    samples: list[EvalSample] = []
    human_count = llm_count = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
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
