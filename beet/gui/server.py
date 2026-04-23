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

from beet.history import HistoryStore, mint_submission_id
from beet.monitoring.meta_detector import MetaDetector
from beet.pipeline import BeetPipeline
from beet.report import build_json_report
from beet.sidecar import DEFAULT_FEEDBACK_PATH, SidecarError, record_feedback


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
    pipeline: BeetPipeline,
    meta: MetaDetector,
    feedback_path: Path,
    history: HistoryStore | None,
    profile: str | None,
):

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
                self._send_json(200, {
                    "status": "ok",
                    "model_loaded": pipeline._fusion._model is not None,
                    "conformal_loaded": pipeline._fusion._conformal is not None,
                    "history_enabled": history is not None,
                    "profile": profile,
                })
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
            else:
                self.send_error(404)

        def _handle_analyze(self, body: dict) -> None:
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text required"})
                return
            sid = str(body.get("submission_id") or "").strip() or mint_submission_id()
            try:
                det = pipeline.analyze(text)
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
                    det = pipeline.analyze(txt)
                    report = build_json_report(det, submission_id=sid)
                    self._record_history(report, source="batch", text=txt, batch_id=batch_id)
                    results.append(report)
                except Exception as e:
                    skipped.append(idx)
                    sys.stderr.write(f"[gui] batch item {idx} failed: {e}\n")
            self._send_json(200, {"results": results, "skipped": skipped, "batch_id": batch_id})

        def _handle_feedback(self, body: dict) -> None:
            try:
                result = record_feedback(pipeline, meta, feedback_path, body)
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

        def _record_history(
            self,
            report: dict,
            *,
            source: str,
            text: str | None,
            batch_id: str | None = None,
        ) -> None:
            if history is None:
                return
            try:
                history.record(report, source=source, text=text, profile=profile, batch_id=batch_id)
            except Exception as e:
                sys.stderr.write(f"[gui] history write failed: {e}\n")

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
    handler = _make_handler(pipeline, meta, fp, history, profile)
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
