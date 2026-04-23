"""BEET pipeline sidecar: line-delimited JSON-RPC over stdio.

Spawned by the Tauri shell as a long-running subprocess. Reads one JSON
object per line from stdin, writes one JSON object per line to stdout.
All diagnostic logging goes to stderr so stdout stays pure JSON-RPC.

Request envelope:  {"id": <any>, "method": <str>, "params": {...}}
Response (ok):     {"id": <same>, "result": {...}}
Response (err):    {"id": <same or null>, "error": {"code": <str>, "message": <str>}}

Methods:
  analyze        {text, submission_id?}                 -> report dict
  analyze_batch  {items: [{id, text}, ...]}             -> {results: [report, ...]}
  feedback       {text, confirmed_label,
                  submission_id?, reviewer_notes?}      -> {ok, submission_id, recorded_at}
  health         {}                                     -> {status, model_loaded, conformal_loaded}
  shutdown       {}                                     -> {ok: true}  (then exits)

On startup, before reading stdin, emits one line:
  {"event": "ready", "pid": <pid>, "protocol_version": 1}

Run:
  python -m beet.sidecar --config configs/default.yaml
  python -m beet.sidecar --profile default
  beet sidecar
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from beet.config import list_profiles, load_config
from beet.history import HistoryStore, mint_submission_id
from beet.pipeline import BeetPipeline
from beet.report import build_json_report
from beet.monitoring.meta_detector import MetaDetector
from beet.runtime import RuntimeContext


PROTOCOL_VERSION = 1
DEFAULT_FEEDBACK_PATH = Path("data") / "reviewer_feedback.jsonl"


def _curated_config_view(profile: str | None, config: dict) -> dict:
    """A UI-friendly projection of the config.

    Kept separate from the raw dict so the RPC shape stays stable even if
    the internal config layout changes.
    """
    decision = (config or {}).get("decision") or {}
    abst = decision.get("abstention") or {}
    detectors_raw = (config or {}).get("detectors") or {}
    detectors = [
        {
            "id": did,
            "enabled": bool((d or {}).get("enabled", True)),
            "weight": float((d or {}).get("weight", 1.0)),
        }
        for did, d in detectors_raw.items()
    ]
    history = ((config or {}).get("gui") or {}).get("history") or {}
    return {
        "profile": profile,
        "thresholds": {
            "red": float(decision.get("red_threshold", 0.75)),
            "amber": float(decision.get("amber_threshold", 0.50)),
            "yellow": float(decision.get("yellow_threshold", 0.25)),
            "abstention": {
                "enabled": bool(abst.get("enabled", True)),
                "max_prediction_set_size": int(abst.get("max_prediction_set_size", 3)),
            },
        },
        "detectors": detectors,
        "history": {
            "enabled": bool(history.get("enabled", True)),
            "retain_text": bool(history.get("retain_text", True)),
            "db_path": history.get("db_path", "data/beet_history.sqlite3"),
        },
    }


def history_from_config(config: dict | None) -> HistoryStore | None:
    gui = (config or {}).get("gui") or {}
    hist = gui.get("history") or {}
    if hist.get("enabled", True) is False:
        return None
    db_path = Path(hist.get("db_path", "data/beet_history.sqlite3"))
    retain = bool(hist.get("retain_text", True))
    try:
        return HistoryStore(db_path, retain_text=retain)
    except Exception as e:
        log.warning("history store disabled: %s", e)
        return None

log = logging.getLogger("beet.sidecar")


class SidecarError(Exception):
    """Raised for method-level failures. `code` surfaces to the caller."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def record_feedback(
    pipeline: BeetPipeline,
    meta: MetaDetector,
    feedback_path: Path,
    params: dict,
    history: HistoryStore | None = None,
) -> dict:
    """Shared feedback handler used by both the HTTP server and the sidecar.

    Re-runs the pipeline to get per-detector layer results, records them
    against a confirmed label via MetaDetector, and appends the raw submission
    to the JSONL. Pipeline failures are logged but do not block JSONL
    persistence — the ground-truth label is the authoritative output.
    """
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SidecarError("ERR_BAD_PARAMS", "text required (non-empty string)")
    label = params.get("confirmed_label")
    if label not in (0, 1):
        raise SidecarError("ERR_BAD_PARAMS", "confirmed_label must be 0 (human) or 1 (LLM)")
    submission_id = str(params.get("submission_id") or "")
    notes = params.get("reviewer_notes")
    if notes is not None and not isinstance(notes, str):
        raise SidecarError("ERR_BAD_PARAMS", "reviewer_notes must be a string")

    try:
        _det, layer_results, _rd = pipeline.analyze_detailed(text)
        for lr in layer_results:
            try:
                meta.record(lr, confirmed_label=int(label))
            except Exception as e:
                log.warning("meta record failed for %s: %s", lr.layer_id, e)
    except Exception as e:
        log.warning("feedback re-analysis failed; persisting label only: %s", e)

    recorded_at = datetime.datetime.utcnow().isoformat() + "Z"
    row: dict[str, Any] = {
        "submission_id": submission_id,
        "text": text,
        "confirmed_label": int(label),
        "recorded_at": recorded_at,
    }
    if notes:
        row["reviewer_notes"] = notes
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    if history is not None and submission_id:
        try:
            history.record_feedback(submission_id, int(label), notes)
        except Exception as e:
            log.warning("history feedback write failed: %s", e)

    return {"ok": True, "submission_id": submission_id, "recorded_at": recorded_at}


class Sidecar:
    def __init__(
        self,
        pipeline: BeetPipeline | None = None,
        feedback_path: Path | None = None,
        history: HistoryStore | None = None,
        profile: str | None = None,
        ctx: RuntimeContext | None = None,
    ):
        if ctx is None:
            if pipeline is None:
                raise ValueError("Sidecar requires either ctx or pipeline")
            ctx = RuntimeContext(pipeline, profile, {})
        self._ctx = ctx
        self._feedback_path = feedback_path or DEFAULT_FEEDBACK_PATH
        self._history = history
        self._meta = MetaDetector()
        self._running = True

    @property
    def _pipeline(self) -> BeetPipeline:
        return self._ctx.pipeline

    @property
    def _profile(self) -> str | None:
        return self._ctx.profile

    def handle(self, method: str, params: dict) -> dict:
        if method == "analyze":
            return self._analyze(params)
        if method == "analyze_batch":
            return self._analyze_batch(params)
        if method == "feedback":
            return self._feedback(params)
        if method == "health":
            return self._health()
        if method == "history_list":
            return self._history_list(params)
        if method == "history_get":
            return self._history_get(params)
        if method == "history_delete":
            return self._history_delete(params)
        if method == "history_export":
            return self._history_export(params)
        if method == "list_profiles":
            return self._list_profiles()
        if method == "get_config":
            return self._get_config()
        if method == "switch_profile":
            return self._switch_profile(params)
        if method == "monitoring_summary":
            return self._monitoring_summary(params)
        if method == "monitoring_timeline":
            return self._monitoring_timeline(params)
        if method == "monitoring_detectors":
            return self._monitoring_detectors(params)
        if method == "shutdown":
            self._running = False
            return {"ok": True}
        raise SidecarError("ERR_METHOD_NOT_FOUND", f"unknown method '{method}'")

    @property
    def running(self) -> bool:
        return self._running

    def _analyze(self, params: dict) -> dict:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SidecarError("ERR_BAD_PARAMS", "text required (non-empty string)")
        submission_id = str(params.get("submission_id") or "").strip() or mint_submission_id()
        try:
            det = self._pipeline.analyze(text)
        except Exception as e:
            raise SidecarError("ERR_PIPELINE", f"analyze failed: {e}") from e
        report = build_json_report(det, submission_id=submission_id)
        self._record(report, source="analyze", text=text)
        return report

    def _analyze_batch(self, params: dict) -> dict:
        items = params.get("items") or []
        if not isinstance(items, list) or not items:
            raise SidecarError("ERR_BAD_PARAMS", "items required (non-empty list)")
        if len(items) > 500:
            raise SidecarError("ERR_BAD_PARAMS", "batch exceeds 500-item cap")
        batch_id = str(params.get("batch_id") or mint_submission_id().replace("sub_", "batch_"))
        texts: dict[str, str] = {}
        order: list[tuple[int, str]] = []
        skipped: list[int] = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                skipped.append(idx)
                continue
            sid = str(it.get("id") or f"{batch_id}_{idx:04d}")
            txt = it.get("text")
            if not isinstance(txt, str) or not txt.strip():
                skipped.append(idx)
                continue
            texts[sid] = txt
            order.append((idx, sid))
        if not texts:
            raise SidecarError("ERR_BAD_PARAMS", "no valid items in batch")
        try:
            dets = self._pipeline.analyze_batch(texts)
        except Exception as e:
            raise SidecarError("ERR_PIPELINE", f"analyze_batch failed: {e}") from e
        results = []
        for _idx, sid in order:
            det = dets.get(sid)
            if det is None:
                continue
            report = build_json_report(det, submission_id=sid)
            self._record(report, source="batch", text=texts[sid], batch_id=batch_id)
            results.append(report)
        return {"results": results, "skipped": skipped, "batch_id": batch_id}

    def _feedback(self, params: dict) -> dict:
        return record_feedback(
            self._pipeline, self._meta, self._feedback_path, params, history=self._history
        )

    def _health(self) -> dict:
        fusion = self._pipeline._fusion
        return {
            "status": "ok",
            "model_loaded": fusion._model is not None,
            "conformal_loaded": fusion._conformal is not None,
            "protocol_version": PROTOCOL_VERSION,
            "history_enabled": self._history is not None,
            "profile": self._profile,
        }

    def _history_list(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return self._history.list(
            limit=int(params.get("limit", 25)),
            offset=int(params.get("offset", 0)),
            determination=params.get("determination") or None,
            since=params.get("since"),
            until=params.get("until"),
            batch_id=params.get("batch_id"),
            search=params.get("search"),
        )

    def _history_get(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        sid = str(params.get("submission_id") or "").strip()
        if not sid:
            raise SidecarError("ERR_BAD_PARAMS", "submission_id required")
        got = self._history.get(sid)
        if got is None:
            raise SidecarError("ERR_NOT_FOUND", f"no submission '{sid}'")
        return got

    def _history_delete(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        sid = str(params.get("submission_id") or "").strip()
        if not sid:
            raise SidecarError("ERR_BAD_PARAMS", "submission_id required")
        return {"ok": self._history.delete(sid)}

    def _history_export(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        fmt = str(params.get("format", "json")).lower()
        if fmt not in ("json", "csv"):
            raise SidecarError("ERR_BAD_PARAMS", "format must be 'json' or 'csv'")
        content, mime, filename = self._history.export(
            fmt=fmt,
            determination=params.get("determination") or None,
            since=params.get("since"),
            until=params.get("until"),
            batch_id=params.get("batch_id"),
            search=params.get("search"),
        )
        return {"content": content, "mime": mime, "filename": filename}

    def _list_profiles(self) -> dict:
        return {"profiles": list_profiles(), "current": self._ctx.profile}

    def _get_config(self) -> dict:
        return _curated_config_view(self._ctx.profile, self._ctx.config)

    def _switch_profile(self, params: dict) -> dict:
        name = str(params.get("name") or "").strip()
        if not name:
            raise SidecarError("ERR_BAD_PARAMS", "name required")
        try:
            self._ctx.switch_profile(name)
        except FileNotFoundError as e:
            raise SidecarError("ERR_BAD_PROFILE", str(e)) from e
        except Exception as e:
            raise SidecarError("ERR_BAD_PROFILE", f"failed to switch profile: {e}") from e
        view = _curated_config_view(self._ctx.profile, self._ctx.config)
        return {
            "ok": True,
            "profile": view["profile"],
            "detectors_enabled": [d["id"] for d in view["detectors"] if d["enabled"]],
        }

    def _monitoring_summary(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return self._history.stats(since=params.get("since"))

    def _monitoring_timeline(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return {"items": self._history.timeline(limit=int(params.get("limit", 200)))}

    def _monitoring_detectors(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return {"detectors": self._history.detector_stats(limit=int(params.get("limit", 500)))}

    def _record(
        self,
        report: dict,
        *,
        source: str,
        text: str | None,
        batch_id: str | None = None,
    ) -> None:
        if self._history is None:
            return
        try:
            self._history.record(
                report, source=source, text=text, profile=self._profile, batch_id=batch_id
            )
        except Exception as e:
            log.warning("history write failed: %s", e)


def _write(obj: dict, stream=None) -> None:
    stream = stream or sys.stdout
    stream.write(json.dumps(obj, separators=(",", ":")) + "\n")
    stream.flush()


def run(
    pipeline: BeetPipeline | None = None,
    feedback_path: Path | None = None,
    stdin=None,
    stdout=None,
    history: HistoryStore | None = None,
    profile: str | None = None,
    ctx: RuntimeContext | None = None,
) -> int:
    """Drive the JSON-RPC loop. Returns the process exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    sidecar = Sidecar(
        pipeline=pipeline,
        feedback_path=feedback_path,
        history=history,
        profile=profile,
        ctx=ctx,
    )
    _write({"event": "ready", "pid": os.getpid(), "protocol_version": PROTOCOL_VERSION}, stream=stdout)

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _write({"id": None, "error": {"code": "ERR_BAD_JSON", "message": str(e)}}, stream=stdout)
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if not isinstance(method, str):
            _write(
                {"id": req_id, "error": {"code": "ERR_BAD_REQUEST", "message": "method must be a string"}},
                stream=stdout,
            )
            continue
        if not isinstance(params, dict):
            _write(
                {"id": req_id, "error": {"code": "ERR_BAD_REQUEST", "message": "params must be an object"}},
                stream=stdout,
            )
            continue

        try:
            result = sidecar.handle(method, params)
            _write({"id": req_id, "result": result}, stream=stdout)
        except SidecarError as e:
            _write(
                {"id": req_id, "error": {"code": e.code, "message": e.message}}, stream=stdout
            )
        except Exception as e:
            log.exception("unhandled error in method %s", method)
            _write(
                {"id": req_id, "error": {"code": "ERR_INTERNAL", "message": str(e)}}, stream=stdout
            )

        if not sidecar.running:
            return 0

    return 0


def _resolve_config_path(config: str | None, profile: str) -> Path:
    if config:
        return Path(config)
    return Path(__file__).parent.parent / "configs" / f"{profile}.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beet-sidecar", description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None, help="Path to a config YAML")
    parser.add_argument("--profile", "-p", default="default", help="Profile name in configs/ (when --config is unset)")
    parser.add_argument(
        "--feedback-path",
        default=str(DEFAULT_FEEDBACK_PATH),
        help="JSONL path for reviewer feedback (default: data/reviewer_feedback.jsonl)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="[%(name)s] %(levelname)s %(message)s",
    )

    config_path = _resolve_config_path(args.config, args.profile)
    config = load_config(config_path)
    pipeline = BeetPipeline(config)
    feedback_path = Path(args.feedback_path)
    history = history_from_config(config)
    ctx = RuntimeContext(pipeline, args.profile, config)

    return run(feedback_path=feedback_path, history=history, ctx=ctx)


if __name__ == "__main__":
    sys.exit(main())
