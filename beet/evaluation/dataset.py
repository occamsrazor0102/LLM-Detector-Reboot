from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class EvalSample:
    id: str
    text: str
    label: int  # 1 = LLM, 0 = human
    tier: str | None = None
    source: str | None = None
    attack_name: str | None = None
    attack_category: str | None = None
    source_id: str | None = None

def load_dataset(path: Path | str) -> list[EvalSample]:
    """Parse JSONL, validate required fields, raise ValueError with line numbers on bad rows."""
    path = Path(path)
    samples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno}: invalid JSON: {e}")
            for field in ("id", "text", "label"):
                if field not in row:
                    raise ValueError(f"line {lineno}: missing required field '{field}'")
            samples.append(EvalSample(
                id=row["id"],
                text=row["text"],
                label=int(row["label"]),
                tier=row.get("tier"),
                source=row.get("source"),
                attack_name=row.get("attack_name"),
                attack_category=row.get("attack_category"),
                source_id=row.get("source_id"),
            ))
    return samples

def save_dataset(samples: list[EvalSample], path: Path | str) -> None:
    """Serialize to JSONL, omitting None optional fields."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            d = {k: v for k, v in asdict(s).items() if v is not None}
            f.write(json.dumps(d) + "\n")

def build_dataset(sources: list[dict], *, id_prefix: str = "") -> list[EvalSample]:
    """Build EvalSample list from source definitions.

    Each source dict has: path (str), label (int), tier (str|None), source (str|None).
    If path is a directory, reads every *.txt inside (sorted); if a file, reads that file.
    """
    samples = []
    index = 0
    for src in sources:
        src_path = Path(src["path"])
        label = src["label"]
        tier = src.get("tier")
        source = src.get("source")
        if src_path.is_dir():
            files = sorted(src_path.glob("*.txt"))
        else:
            files = [src_path]
        for fp in files:
            text = fp.read_text(encoding="utf-8")
            tag = tier or "sample"
            sample_id = f"{id_prefix}{tag}_{index:04d}"
            samples.append(EvalSample(
                id=sample_id,
                text=text,
                label=label,
                tier=tier,
                source=source,
            ))
            index += 1
    # Validate via load/save roundtrip logic (check all required fields present)
    for s in samples:
        if not s.id or not s.text:
            raise ValueError(f"build_dataset produced invalid sample: {s.id}")
    return samples
