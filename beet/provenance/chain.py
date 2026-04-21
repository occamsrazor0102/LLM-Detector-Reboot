"""Hash-chained audit log for provenance manifests.

Each appended entry records the hash of the previous entry, producing a
tamper-evident chain: modifying an earlier record invalidates every hash
downstream. `validate()` walks the chain and reports any break.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GENESIS = "genesis"


def _hash_content(entry: dict[str, Any]) -> str:
    payload = {k: v for k, v in entry.items() if k != "_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class AuditChain:
    def __init__(self, log_path: Path):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def _last_hash(self) -> str:
        entries = self._read_all()
        if not entries:
            return GENESIS
        return entries[-1].get("_hash", GENESIS)

    def append(self, manifest: dict) -> dict:
        entry = dict(manifest)
        entry["_prev_hash"] = self._last_hash()
        entry["_hash"] = _hash_content(entry)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        entries = self._read_all()
        for i, entry in enumerate(entries):
            if i == 0:
                if entry.get("_prev_hash") != GENESIS:
                    errors.append(f"entry 0: expected _prev_hash={GENESIS!r}, got {entry.get('_prev_hash')!r}")
            else:
                expected_prev = entries[i - 1].get("_hash")
                if entry.get("_prev_hash") != expected_prev:
                    errors.append(f"entry {i}: _prev_hash mismatch")
            recomputed = _hash_content(entry)
            if recomputed != entry.get("_hash"):
                errors.append(f"entry {i}: _hash does not match recomputed content")
        return not errors, errors

    def __len__(self) -> int:
        return len(self._read_all())
