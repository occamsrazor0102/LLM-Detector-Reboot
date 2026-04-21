"""Retention manager: purges privacy-vault records older than a threshold.

The vault holds raw-text records written via `PrivacyStore.save_raw_text`.
Their JSON payload carries a `stored_at` ISO timestamp; this module compares
against a cutoff and deletes expired files. Every purge is logged to the
store's access log.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


class RetentionManager:
    def __init__(self, vault_dir: Path, retention_days: int = 90, access_log: Path | None = None):
        self._vault = Path(vault_dir)
        self._retention = int(retention_days)
        self._access_log = Path(access_log) if access_log else None

    def _cutoff(self, now: datetime | None = None) -> datetime:
        base = now or datetime.now(timezone.utc).replace(tzinfo=None)
        return base - timedelta(days=self._retention)

    def scan_expired(self, *, now: datetime | None = None) -> list[Path]:
        """Return vault files whose stored_at is older than cutoff."""
        cutoff = self._cutoff(now)
        expired: list[Path] = []
        if not self._vault.exists():
            return expired
        for f in self._vault.glob("*.json"):
            try:
                record = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            stored_at = record.get("stored_at")
            if not isinstance(stored_at, str):
                continue
            try:
                ts = datetime.fromisoformat(stored_at.replace("Z", ""))
            except ValueError:
                continue
            if ts < cutoff:
                expired.append(f)
        return expired

    def purge_expired(self, *, dry_run: bool = False, now: datetime | None = None) -> int:
        expired = self.scan_expired(now=now)
        if dry_run:
            return len(expired)
        count = 0
        for f in expired:
            try:
                sid = f.stem
                f.unlink()
                count += 1
                self._log_purge(sid)
            except OSError:
                continue
        return count

    def _log_purge(self, submission_id: str) -> None:
        if self._access_log is None:
            return
        entry = {
            "event": "PURGE",
            "submission_id": submission_id,
            "accessor": "retention_manager",
            "reason": f"retention_days={self._retention}",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        }
        with open(self._access_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
