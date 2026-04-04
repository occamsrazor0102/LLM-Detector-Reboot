import datetime
import json
import hashlib
from beet.contracts import Determination

def build_manifest(submission_id: str, contributor_id: str, task_id: str, occupation: str,
    text_hash: str, determination: Determination, pipeline_version: str = "2.0.0") -> dict:
    return {
        "manifest_version": "2.0", "submission_id": submission_id, "contributor_id": contributor_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z", "task_id": task_id, "occupation": occupation,
        "provenance_signals": {
            "text_hash": text_hash, "determination": determination.label,
            "p_llm": round(determination.p_llm, 4),
            "confidence_interval": list(determination.confidence_interval),
            "prediction_set": determination.prediction_set,
            "detectors_run": determination.detectors_run,
            "top_features": [{"feature": f, "contribution": round(c, 4)} for f, c in determination.top_features],
            "pipeline_version": pipeline_version,
        },
        "audit_trail": [
            {"event": "submitted", "timestamp": datetime.datetime.utcnow().isoformat()},
            {"event": "analyzed", "timestamp": datetime.datetime.utcnow().isoformat(), "pipeline_version": pipeline_version},
        ],
    }

def sign_manifest(manifest: dict) -> dict:
    content = json.dumps(manifest, sort_keys=True).encode()
    manifest["_hash"] = hashlib.sha256(content).hexdigest()
    return manifest
