"""Minimal, self-contained HTTP server that serves the embedded SPA and
proxies /analyze and /batch to the local BEET pipeline. No external API
dependency; calls the pipeline directly.
"""
from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from beet.pipeline import BeetPipeline
from beet.report import build_json_report


STATIC_DIR = Path(__file__).parent / "static"


def _make_handler(pipeline: BeetPipeline):

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

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
            elif path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "model_loaded": pipeline._fusion._model is not None,
                    "conformal_loaded": pipeline._fusion._conformal is not None,
                })
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
                text = body.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    self._send_json(400, {"error": "text required"})
                    return
                try:
                    det = pipeline.analyze(text)
                    self._send_json(200, build_json_report(det))
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
            elif path == "/batch":
                items = body.get("items") or []
                if not isinstance(items, list) or not items:
                    self._send_json(400, {"error": "items required"})
                    return
                results = []
                for it in items:
                    sid = str(it.get("id", ""))
                    txt = it.get("text", "")
                    if not isinstance(txt, str) or not txt.strip():
                        continue
                    det = pipeline.analyze(txt)
                    results.append(build_json_report(det, submission_id=sid))
                self._send_json(200, {"results": results})
            else:
                self.send_error(404)

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


def serve(pipeline: BeetPipeline, host: str = "127.0.0.1", port: int = 8877, open_browser: bool = True) -> None:
    httpd = HTTPServer((host, port), _make_handler(pipeline))
    url = f"http://{host}:{port}/"
    print(f"[gui] serving on {url}  (Ctrl-C to stop)")
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
