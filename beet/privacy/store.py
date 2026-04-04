import json
import datetime
from pathlib import Path

class PrivacyStore:
    def __init__(self, base_dir: Path):
        self._base = Path(base_dir)
        self._features_dir = self._base / "feature_store"
        self._results_dir = self._base / "result_store"
        self._vault_dir = self._base / "raw_text_vault"
        self._access_log = self._base / "access_log.jsonl"
        for d in [self._features_dir, self._results_dir, self._vault_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_features(self, submission_id: str, feature_vector: dict, text_hash: str, determination: str) -> None:
        record = {"submission_id": submission_id, "text_hash": text_hash, "determination": determination,
            "feature_vector": feature_vector, "saved_at": datetime.datetime.utcnow().isoformat()}
        (self._features_dir / f"{submission_id}.json").write_text(json.dumps(record, indent=2))

    def get_features(self, submission_id: str) -> dict | None:
        path = self._features_dir / f"{submission_id}.json"
        if not path.exists(): return None
        return json.loads(path.read_text())

    def save_raw_text(self, submission_id: str, text: str, reason: str) -> None:
        record = {"submission_id": submission_id, "text": text, "reason": reason,
            "stored_at": datetime.datetime.utcnow().isoformat()}
        (self._vault_dir / f"{submission_id}.json").write_text(json.dumps(record, indent=2))
        self._log_access("WRITE", submission_id, reason)

    def get_raw_text(self, submission_id: str, accessor: str, reason: str) -> str | None:
        path = self._vault_dir / f"{submission_id}.json"
        self._log_access("READ", submission_id, reason, accessor=accessor)
        if not path.exists(): return None
        return json.loads(path.read_text()).get("text")

    def _log_access(self, event: str, submission_id: str, reason: str, accessor: str = "system") -> None:
        entry = {"event": event, "submission_id": submission_id, "accessor": accessor,
            "reason": reason, "timestamp": datetime.datetime.utcnow().isoformat()}
        with open(self._access_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
