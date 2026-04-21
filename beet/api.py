"""FastAPI REST server for BEET.

Endpoints:
    POST /analyze     Single-text analysis
    POST /batch       Batch analysis with optional cross-similarity
    GET  /health      Liveness + pipeline model info
    GET  /config      Current config (with secrets redacted)

The pipeline is a singleton initialised on startup. FastAPI is an optional
dependency; import errors surface at server-construction time so unit tests
that don't hit the API do not pay the cost.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from beet.pipeline import BeetPipeline
from beet.report import build_json_report

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def _redact(config: dict) -> dict:
    """Strip API keys, paths under /data, etc., from config for /config endpoint."""
    import copy
    redacted = copy.deepcopy(config)
    for section in ("privacy", "fusion", "monitoring"):
        if section in redacted and isinstance(redacted[section], dict):
            for key in list(redacted[section].keys()):
                if any(tok in key.lower() for tok in ("secret", "api_key", "token")):
                    redacted[section][key] = "[redacted]"
    return redacted


def create_app(config_path: Path | str | None = None) -> Any:
    """Construct a FastAPI app bound to a pipeline instance.

    Called lazily so the import-time cost of FastAPI/Pydantic is only paid
    when the REST server is actually being started.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI and Pydantic required for the REST API. "
            "Install with: pip install 'beet[api]'"
        )

    cfg_path = Path(config_path or os.environ.get("BEET_CONFIG") or DEFAULT_CONFIG_PATH)
    pipeline = BeetPipeline.from_config_file(cfg_path)

    app = FastAPI(title="BEET 2.0 API", version="2.0.0")

    class AnalyzeRequest(BaseModel):
        text: str = Field(..., min_length=1)
        task_metadata: dict | None = None
        submission_id: str | None = None

    class BatchItem(BaseModel):
        id: str
        text: str
        task_metadata: dict | None = None

    class BatchRequest(BaseModel):
        items: list[BatchItem]
        cross_similarity: bool = False

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "2.0.0",
            "model_loaded": pipeline._fusion._model is not None,
            "conformal_loaded": pipeline._fusion._conformal is not None,
            "detectors": list(pipeline._detectors.keys()),
        }

    @app.get("/config")
    def get_config() -> dict:
        return _redact(pipeline._config)

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        try:
            det = pipeline.analyze(req.text, task_metadata=req.task_metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return build_json_report(det, submission_id=req.submission_id or "")

    @app.post("/batch")
    def batch(req: BatchRequest) -> dict:
        if not req.items:
            raise HTTPException(status_code=400, detail="items list is empty")
        if req.cross_similarity:
            texts = {it.id: it.text for it in req.items}
            dets = pipeline.analyze_batch(texts)
            return {
                "results": [
                    build_json_report(det, submission_id=sid)
                    for sid, det in dets.items()
                ],
            }
        out = []
        for it in req.items:
            det = pipeline.analyze(it.text, task_metadata=it.task_metadata)
            out.append(build_json_report(det, submission_id=it.id))
        return {"results": out}

    return app
