import json
from datetime import datetime
from pathlib import Path

class AlertManager:
    def __init__(self, log_path: Path):
        self._log = Path(log_path)

    def fire(self, alert_type: str, message: str, data: dict = None) -> None:
        entry = {"timestamp": datetime.utcnow().isoformat(), "type": alert_type,
            "message": message, "data": data or {}}
        with open(self._log, "a") as f:
            f.write(json.dumps(entry) + "\n")
