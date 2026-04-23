"""Minimal, self-contained HTTP server that serves the embedded SPA and
proxies /analyze, /batch, /feedback, and /history/* to the local BEET
pipeline. No external API dependency; calls the pipeline directly.
"""
from __future__ import annotations

import json
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from beet.config import list_profiles
from beet.history import HistoryStore, mint_submission_id
from beet.monitoring.drift import DriftMonitor
from beet.monitoring.meta_detector import MetaDetector
from beet.pipeline import BeetPipeline
from beet.report import build_json_report
from beet.runtime import RuntimeContext
from beet.sidecar import (
    DEFAULT_FEEDBACK_PATH,
    SidecarError,
    _curated_config_view,
    drift_baseline_from_history,
    drift_from_config,
    record_feedback,
)


STATIC_DIR = Path(__file__).parent / "static"


def _history_from_config(config: dict | None) -> HistoryStore | None:
    gui = (config or {}).get("gui") or {}
    hist = gui.get("history") or {}
    if hist.get("enabled", True) is False:
        return None
    db_path = Path(hist.get("db_path", "data/beet_history.sqlite3"))
    retain = bool(hist.get("retain_text", True))
    try:
        return HistoryStore(db_path, retain_text=retain)
    except Exception as e:
        sys.stderr.write(f"[gui] history store disabled: {e}\n")
        return None


def _make_handler(
    pipeline: BeetPipeline | None = None,
    meta: MetaDetector | None = None,
    feedback_path: Path | None = None,
    history: HistoryStore | None = None,
    profile: str | None = None,
    ctx: RuntimeContext | None = None,
    drift: DriftMonitor | None = None,
):
    if ctx is None:
        if pipeline is None:
            raise ValueError("_make_handler requires either ctx or pipeline")
        ctx = RuntimeContext(pipeline, profile, {})
    meta = meta or MetaDetector()
    feedback_path = feedback_path or DEFAULT_FEEDBACK_PATH

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            sys.stdout.write("[gui] " + (format % args) + "\n")

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_attachment(self, content: str, mime: str, filename: str) -> None:
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
            elif path == "/health":
                pipe = ctx.pipeline
                self._send_json(200, {
                    "status": "ok",
                    "model_loaded": pipe._fusion._model is not None,
                    "conformal_loaded": pipe._fusion._conformal is not None,
                    "history_enabled": history is not None,
                    "profile": ctx.profile,
                })
            elif path == "/config/profiles":
                self._send_json(200, {
                    "profiles": list_profiles(),
                    "current": ctx.profile,
                })
            elif path == "/config/current":
                self._send_json(200, _curated_config_view(ctx.profile, ctx.config))
            elif path == "/history/export":
                if history is None:
                    self._send_json(503, {"error": "history disabled"})
                    return
                q = parse_qs(parsed.query)
                fmt = (q.get("format", ["json"])[0] or "json").lower()
                if fmt not in ("json", "csv"):
                    self._send_json(400, {"error": "format must be 'json' or 'csv'"})
                    return
                content, mime, filename = history.export(
                    fmt=fmt,
                    determination=q.get("determination") or None,
                    since=q.get("since", [None])[0],
                    until=q.get("until", [None])[0],
                    batch_id=q.get("batch_id", [None])[0],
                    search=q.get("search", [None])[0],
                )
                self._send_attachment(content, mime, filename)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            if path == "/analyze":
                self._handle_analyze(body)
            elif path == "/batch":
                self._handle_batch(body)
            elif path == "/feedback":
                self._handle_feedback(body)
            elif path == "/history/list":
                self._handle_history_list(body)
            elif path == "/history/get":
                self._handle_history_get(body)
            elif path == "/history/delete":
                self._handle_history_delete(body)
            elif path == "/config/switch":
                self._handle_switch_profile(body)
            elif path == "/monitoring/summary":
                self._handle_monitoring_summary(body)
            elif path == "/monitoring/timeline":
                self._handle_monitoring_timeline(body)
            elif path == "/monitoring/detectors":
                self._handle_monitoring_detectors(body)
            elif path == "/evaluation/run":
                self._handle_run_eval(body)
            elif path == "/monitoring/drift":
                self._handle_monitoring_drift(body)
            elif path == "/monitoring/set-baseline":
                self._handle_monitoring_set_baseline(body)
            else:
                self.send_error(404)

        def _handle_analyze(self, body: dict) -> None:
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text required"})
                return
            sid = str(body.get("submission_id") or "").strip() or mint_submission_id()
            try:
                det = ctx.pipeline.analyze(text)
                report = build_json_report(det, submission_id=sid)
                self._record_history(report, source="analyze", text=text)
                self._send_json(200, report)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_batch(self, body: dict) -> None:
            items = body.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json(400, {"error": "items required"})
                return
            if len(items) > 500:
                self._send_json(400, {"error": "batch exceeds 500-item cap"})
                return
            batch_id = str(body.get("batch_id") or mint_submission_id().replace("sub_", "batch_"))
            results, skipped = [], []
            for idx, it in enumerate(items):
                if not isinstance(it, dict):
                    skipped.append(idx)
                    continue
                sid = str(it.get("id") or f"{batch_id}_{idx:04d}")
                txt = it.get("text", "")
                if not isinstance(txt, str) or not txt.strip():
                    skipped.append(idx)
                    continue
                try:
                    det = ctx.pipeline.analyze(txt)
                    report = build_json_report(det, submission_id=sid)
                    self._record_history(report, source="batch", text=txt, batch_id=batch_id)
                    results.append(report)
                except Exception as e:
                    skipped.append(idx)
                    sys.stderr.write(f"[gui] batch item {idx} failed: {e}\n")
            self._send_json(200, {"results": results, "skipped": skipped, "batch_id": batch_id})

        def _handle_feedback(self, body: dict) -> None:
            try:
                result = record_feedback(ctx.pipeline, meta, feedback_path, body)
                if history is not None:
                    sid = str(body.get("submission_id") or result.get("submission_id") or "").strip()
                    if sid:
                        try:
                            history.record_feedback(
                                sid,
                                int(body.get("confirmed_label", 0)),
                                body.get("reviewer_notes"),
                            )
                        except Exception as e:
                            sys.stderr.write(f"[gui] history feedback write failed: {e}\n")
                self._send_json(200, result)
            except SidecarError as e:
                self._send_json(400, {"error": e.message, "code": e.code})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_history_list(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                res = history.list(
                    limit=int(body.get("limit", 25)),
                    offset=int(body.get("offset", 0)),
                    determination=body.get("determination") or None,
                    since=body.get("since"),
                    until=body.get("until"),
                    batch_id=body.get("batch_id"),
                    search=body.get("search"),
                )
                self._send_json(200, res)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_history_get(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            sid = str(body.get("submission_id") or "").strip()
            if not sid:
                self._send_json(400, {"error": "submission_id required"})
                return
            got = history.get(sid)
            if got is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, got)

        def _handle_history_delete(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            sid = str(body.get("submission_id") or "").strip()
            if not sid:
                self._send_json(400, {"error": "submission_id required"})
                return
            ok = history.delete(sid)
            self._send_json(200, {"ok": ok})

        def _handle_switch_profile(self, body: dict) -> None:
            name = str(body.get("name") or "").strip()
            if not name:
                self._send_json(400, {"error": "name required"})
                return
            try:
                ctx.switch_profile(name)
            except FileNotFoundError as e:
                self._send_json(400, {"error": str(e), "code": "ERR_BAD_PROFILE"})
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json(400, {"error": str(e), "code": "ERR_BAD_PROFILE"})
                return
            view = _curated_config_view(ctx.profile, ctx.config)
            self._send_json(200, {
                "ok": True,
                "profile": view["profile"],
                "detectors_enabled": [d["id"] for d in view["detectors"] if d["enabled"]],
            })

        def _handle_monitoring_summary(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                self._send_json(200, history.stats(since=body.get("since")))
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_timeline(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                limit = int(body.get("limit", 200))
                self._send_json(200, {"items": history.timeline(limit=limit)})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_detectors(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                limit = int(body.get("limit", 500))
                self._send_json(200, {"detectors": history.detector_stats(limit=limit)})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_run_eval(self, body: dict) -> None:
            import time
            from beet.evaluation.dataset import EvalSample
            from beet.evaluation.runner import eval_report_to_dict, run_eval

            items = body.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json(400, {"error": "items required"})
                return
            cap = max(1, min(int(body.get("max_samples", 200)), 1000))
            if len(items) > cap:
                self._send_json(400, {
                    "error": f"dataset has {len(items)} samples, cap is {cap} (max 1000)",
                    "code": "ERR_TOO_LARGE",
                })
                return
            samples = []
            for idx, it in enumerate(items):
                if not isinstance(it, dict):
                    self._send_json(400, {"error": f"item {idx} is not an object"})
                    return
                for required in ("id", "text", "label"):
                    if required not in it:
                        self._send_json(400, {"error": f"item {idx} missing '{required}'"})
                        return
                if not isinstance(it["text"], str) or not it["text"].strip():
                    self._send_json(400, {"error": f"item {idx} has empty text"})
                    return
                try:
                    samples.append(EvalSample(
                        id=str(it["id"]),
                        text=str(it["text"]),
                        label=int(it["label"]),
                        tier=it.get("tier"),
                        source=it.get("source"),
                        attack_name=it.get("attack_name"),
                        attack_category=it.get("attack_category"),
                        source_id=it.get("source_id"),
                    ))
                except (TypeError, ValueError) as e:
                    self._send_json(400, {"error": f"item {idx}: {e}"})
                    return
            t0 = time.monotonic()
            try:
                report = run_eval(ctx.pipeline, samples)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
                return
            result = eval_report_to_dict(report)
            result["duration_ms"] = int((time.monotonic() - t0) * 1000)
            self._send_json(200, result)

        def _handle_monitoring_drift(self, body: dict) -> None:
            if drift is None:
                self._send_json(503, {"error": "drift monitor disabled"})
                return
            try:
                alerts = drift.check_drift()
                summary = drift.get_summary()
                self._send_json(200, {
                    "has_baseline": bool(drift._baseline_hist),
                    "baseline_features": list(drift._baseline_hist.keys()),
                    "n_observations": len(drift._observations),
                    "alerts": alerts,
                    "summary": summary,
                })
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_set_baseline(self, body: dict) -> None:
            if drift is None:
                self._send_json(503, {"error": "drift monitor disabled"})
                return
            if history is None:
                self._send_json(503, {"error": "history required for baseline"})
                return
            limit = int(body.get("limit", 500))
            try:
                n = drift_baseline_from_history(drift, history, limit=limit)
                self._send_json(200, {
                    "ok": n > 0,
                    "n_samples": n,
                    "baseline_features": list(drift._baseline_hist.keys()),
                })
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _record_history(
            self,
            report: dict,
            *,
            source: str,
            text: str | None,
            batch_id: str | None = None,
        ) -> None:
            if history is not None:
                try:
                    history.record(
                        report, source=source, text=text, profile=ctx.profile, batch_id=batch_id
                    )
                except Exception as e:
                    sys.stderr.write(f"[gui] history write failed: {e}\n")
            if drift is not None:
                try:
                    drift.record(
                        p_llm=float(report.get("p_llm", 0.0)),
                        determination=str(report.get("determination", "")),
                        feature_vector=dict(report.get("feature_contributions") or {}),
                    )
                except Exception as e:
                    sys.stderr.write(f"[gui] drift record failed: {e}\n")

        def _serve_static(self, name: str) -> None:
            path = STATIC_DIR / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            ctype = "text/html" if name.endswith(".html") else "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(
    pipeline: BeetPipeline,
    host: str = "127.0.0.1",
    port: int = 8877,
    open_browser: bool = True,
    feedback_path: Path | None = None,
    config: dict | None = None,
    profile: str | None = None,
) -> None:
    meta = MetaDetector()
    fp = feedback_path or DEFAULT_FEEDBACK_PATH
    history = _history_from_config(config)
    drift = drift_from_config(config)
    ctx = RuntimeContext(pipeline, profile, config or {})
    handler = _make_handler(meta=meta, feedback_path=fp, history=history, ctx=ctx, drift=drift)
    httpd = HTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    hist_status = "enabled" if history is not None else "disabled"
    print(f"[gui] serving on {url}  (history {hist_status}, Ctrl-C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gui] stopping")
        httpd.server_close()
