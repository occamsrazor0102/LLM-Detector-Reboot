"""Retention manager tests."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from beet.privacy.retention import RetentionManager


def _write_record(vault, sid, stored_at_iso):
    (vault / f"{sid}.json").write_text(json.dumps({
        "submission_id": sid, "text": "x", "stored_at": stored_at_iso,
    }))


def _now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_purge_removes_old_records(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    old = (_now_utc() - timedelta(days=120)).isoformat()
    new = (_now_utc() - timedelta(days=5)).isoformat()
    _write_record(vault, "old1", old)
    _write_record(vault, "fresh1", new)

    rm = RetentionManager(vault, retention_days=90)
    n = rm.purge_expired()
    assert n == 1
    assert not (vault / "old1.json").exists()
    assert (vault / "fresh1.json").exists()


def test_dry_run_does_not_delete(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    old = (_now_utc() - timedelta(days=120)).isoformat()
    _write_record(vault, "old1", old)
    rm = RetentionManager(vault, retention_days=90)
    n = rm.purge_expired(dry_run=True)
    assert n == 1
    assert (vault / "old1.json").exists()


def test_scan_expired_handles_bad_records(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "bad.json").write_text("not valid json")
    (vault / "nostored.json").write_text(json.dumps({"submission_id": "n"}))
    old = (_now_utc() - timedelta(days=100)).isoformat()
    _write_record(vault, "old1", old)
    rm = RetentionManager(vault, retention_days=90)
    expired = rm.scan_expired()
    assert len(expired) == 1
    assert expired[0].name == "old1.json"


def test_missing_vault_dir_returns_empty(tmp_path):
    rm = RetentionManager(tmp_path / "does_not_exist", retention_days=30)
    assert rm.scan_expired() == []
    assert rm.purge_expired() == 0


def test_purge_logs_to_access_log(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    log = tmp_path / "access.jsonl"
    old = (_now_utc() - timedelta(days=200)).isoformat()
    _write_record(vault, "expired1", old)

    rm = RetentionManager(vault, retention_days=90, access_log=log)
    rm.purge_expired()
    assert log.exists()
    entry = json.loads(log.read_text().strip().splitlines()[0])
    assert entry["event"] == "PURGE"
    assert entry["submission_id"] == "expired1"
