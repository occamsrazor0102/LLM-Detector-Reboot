"""Queryable submission log.

SQLite-backed index alongside PrivacyStore and AuditChain. Every analyze/
batch/feedback call writes here so the GUI can list, filter, sort, and open
past submissions without re-running the pipeline.

Writes are best-effort: a failed DB write must not fail the pipeline RPC.
Callers should catch and log.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import random
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  submission_id  TEXT PRIMARY KEY,
  recorded_at    TEXT NOT NULL,
  text_hash      TEXT NOT NULL,
  text           TEXT,
  determination  TEXT NOT NULL,
  p_llm          REAL NOT NULL,
  ci_lower       REAL NOT NULL,
  ci_upper       REAL NOT NULL,
  prediction_set TEXT NOT NULL,
  detectors_run  TEXT NOT NULL,
  report         TEXT NOT NULL,
  profile        TEXT,
  batch_id       TEXT,
  source         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_recorded_at ON submissions(recorded_at);
CREATE INDEX IF NOT EXISTS idx_submissions_determination ON submissions(determination);
CREATE INDEX IF NOT EXISTS idx_submissions_batch_id ON submissions(batch_id);

CREATE TABLE IF NOT EXISTS feedback_log (
  submission_id   TEXT NOT NULL,
  confirmed_label INTEGER NOT NULL,
  reviewer_notes  TEXT,
  recorded_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_submission ON feedback_log(submission_id);
"""


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def mint_submission_id() -> str:
    """Short, lexicographically-time-ordered id."""
    ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    rand = random.randint(0, 0xFFFF)
    return f"sub_{ms:011x}_{rand:04x}"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HistoryStore:
    """Thin SQLite wrapper. One connection per call — short-lived, thread-safe."""

    def __init__(self, db_path: Path, *, retain_text: bool = True):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retain_text = retain_text
        self._init_lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(
        self,
        report: dict,
        *,
        source: str,
        text: str | None = None,
        profile: str | None = None,
        batch_id: str | None = None,
    ) -> str:
        sid = str(report.get("submission_id") or "").strip() or mint_submission_id()
        recorded_at = report.get("timestamp") or _now_iso()
        ci = report.get("confidence_interval") or [0.0, 0.0]
        ci_lo, ci_hi = (float(ci[0]), float(ci[1])) if len(ci) >= 2 else (0.0, 0.0)
        text_hash = _hash_text(text) if text else ""
        stored_text = text if (self._retain_text and text is not None) else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO submissions (
                  submission_id, recorded_at, text_hash, text, determination,
                  p_llm, ci_lower, ci_upper, prediction_set, detectors_run,
                  report, profile, batch_id, source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(submission_id) DO UPDATE SET
                  recorded_at=excluded.recorded_at,
                  text_hash=excluded.text_hash,
                  text=excluded.text,
                  determination=excluded.determination,
                  p_llm=excluded.p_llm,
                  ci_lower=excluded.ci_lower,
                  ci_upper=excluded.ci_upper,
                  prediction_set=excluded.prediction_set,
                  detectors_run=excluded.detectors_run,
                  report=excluded.report,
                  profile=excluded.profile,
                  batch_id=excluded.batch_id,
                  source=excluded.source
                """,
                (
                    sid, recorded_at, text_hash, stored_text,
                    report.get("determination", ""), float(report.get("p_llm", 0.0)),
                    ci_lo, ci_hi,
                    json.dumps(report.get("prediction_set", [])),
                    json.dumps(report.get("detectors_run", [])),
                    json.dumps(report),
                    profile, batch_id, source,
                ),
            )
        return sid

    def record_feedback(
        self, submission_id: str, confirmed_label: int, reviewer_notes: str | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_log (submission_id, confirmed_label, reviewer_notes, recorded_at) "
                "VALUES (?,?,?,?)",
                (submission_id, int(confirmed_label), reviewer_notes, _now_iso()),
            )

    def list(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        determination: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        batch_id: str | None = None,
        search: str | None = None,
    ) -> dict:
        where, params = self._build_filter(determination, since, until, batch_id, search)
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM submissions s {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT s.submission_id, s.recorded_at, s.determination, s.p_llm,
                       s.ci_lower, s.ci_upper, s.detectors_run, s.source,
                       s.batch_id, s.profile,
                       EXISTS(SELECT 1 FROM feedback_log f
                              WHERE f.submission_id = s.submission_id) AS has_feedback
                FROM submissions s
                {where}
                ORDER BY s.recorded_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, int(limit), int(offset)),
            ).fetchall()
        items = [self._row_to_summary(r) for r in rows]
        return {"items": items, "total": int(total)}

    def get(self, submission_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM submissions WHERE submission_id=?", (submission_id,)
            ).fetchone()
            if not row:
                return None
            feedback = conn.execute(
                "SELECT confirmed_label, reviewer_notes, recorded_at FROM feedback_log "
                "WHERE submission_id=? ORDER BY recorded_at DESC",
                (submission_id,),
            ).fetchall()
        report = json.loads(row["report"])
        return {
            "submission_id": row["submission_id"],
            "recorded_at": row["recorded_at"],
            "text": row["text"],
            "text_hash": row["text_hash"],
            "profile": row["profile"],
            "batch_id": row["batch_id"],
            "source": row["source"],
            "report": report,
            "feedback": [
                {"confirmed_label": f["confirmed_label"],
                 "reviewer_notes": f["reviewer_notes"],
                 "recorded_at": f["recorded_at"]}
                for f in feedback
            ],
        }

    def delete(self, submission_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM submissions WHERE submission_id=?", (submission_id,)
            )
            conn.execute(
                "DELETE FROM feedback_log WHERE submission_id=?", (submission_id,)
            )
            return cur.rowcount > 0

    def export(
        self,
        *,
        fmt: str = "json",
        determination: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        batch_id: str | None = None,
        search: str | None = None,
    ) -> tuple[str, str, str]:
        """Return (content, mime, filename)."""
        where, params = self._build_filter(determination, since, until, batch_id, search)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM submissions s {where} ORDER BY s.recorded_at DESC",
                params,
            ).fetchall()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if fmt == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow([
                "submission_id", "recorded_at", "source", "batch_id", "profile",
                "determination", "p_llm", "ci_lower", "ci_upper",
                "detectors_run", "text_hash",
            ])
            for r in rows:
                w.writerow([
                    r["submission_id"], r["recorded_at"], r["source"],
                    r["batch_id"] or "", r["profile"] or "",
                    r["determination"], f"{r['p_llm']:.4f}",
                    f"{r['ci_lower']:.4f}", f"{r['ci_upper']:.4f}",
                    r["detectors_run"], r["text_hash"],
                ])
            return buf.getvalue(), "text/csv", f"beet-history-{stamp}.csv"
        payload = [
            {
                "submission_id": r["submission_id"],
                "recorded_at": r["recorded_at"],
                "source": r["source"],
                "batch_id": r["batch_id"],
                "profile": r["profile"],
                "report": json.loads(r["report"]),
                "text_hash": r["text_hash"],
            }
            for r in rows
        ]
        return json.dumps(payload, indent=2), "application/json", f"beet-history-{stamp}.json"

    @staticmethod
    def _build_filter(
        determination: list[str] | None,
        since: str | None,
        until: str | None,
        batch_id: str | None,
        search: str | None,
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        if determination:
            marks = ",".join("?" * len(determination))
            clauses.append(f"s.determination IN ({marks})")
            params.extend(determination)
        if since:
            clauses.append("s.recorded_at >= ?")
            params.append(since)
        if until:
            clauses.append("s.recorded_at <= ?")
            params.append(until)
        if batch_id:
            clauses.append("s.batch_id = ?")
            params.append(batch_id)
        if search:
            clauses.append("(s.submission_id LIKE ? OR s.text LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, tuple(params)

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> dict:
        return {
            "submission_id": row["submission_id"],
            "recorded_at": row["recorded_at"],
            "determination": row["determination"],
            "p_llm": round(float(row["p_llm"]), 4),
            "confidence_interval": [
                round(float(row["ci_lower"]), 4),
                round(float(row["ci_upper"]), 4),
            ],
            "detectors_run": json.loads(row["detectors_run"]),
            "source": row["source"],
            "batch_id": row["batch_id"],
            "profile": row["profile"],
            "has_feedback": bool(row["has_feedback"]),
        }
