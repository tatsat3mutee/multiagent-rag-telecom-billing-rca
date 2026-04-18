"""
Lightweight SQLite inference logger — every UI-driven RCA call gets a row.

Used in Streamlit pages to demonstrate "production-readiness" / live monitoring
during the thesis viva. Keeps a single file at <project_root>/inferences.db.

Table schema:
    timestamp     TEXT  -- ISO 8601 UTC
    anomaly_id    TEXT
    anomaly_type  TEXT
    severity      TEXT
    root_cause    TEXT
    confidence    REAL
    latency_ms    REAL
    provider      TEXT  -- e.g. 'groq', 'kimi'
    model         TEXT
    source        TEXT  -- 'ui_single' | 'ui_batch' | 'cli'
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parents[2] / "inferences.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    anomaly_id    TEXT,
    anomaly_type  TEXT,
    severity      TEXT,
    root_cause    TEXT,
    confidence    REAL,
    latency_ms    REAL,
    provider      TEXT,
    model         TEXT,
    source        TEXT
);
CREATE INDEX IF NOT EXISTS idx_inferences_ts ON inferences(timestamp);
CREATE INDEX IF NOT EXISTS idx_inferences_type ON inferences(anomaly_type);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.executescript(_SCHEMA)
    return conn


def log_inference(
    anomaly_id: str,
    anomaly_type: str,
    severity: str,
    root_cause: str,
    confidence: float,
    latency_ms: float,
    source: str = "ui_single",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Append a row. Best-effort — never raises into the UI."""
    try:
        if provider is None or model is None:
            try:
                from config import LLM_PROVIDER, LLM_MODEL
                provider = provider or LLM_PROVIDER
                model = model or LLM_MODEL
            except Exception:
                provider = provider or "unknown"
                model = model or "unknown"
        with _conn() as c:
            c.execute(
                """INSERT INTO inferences
                   (timestamp, anomaly_id, anomaly_type, severity, root_cause,
                    confidence, latency_ms, provider, model, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    str(anomaly_id),
                    str(anomaly_type),
                    str(severity),
                    str(root_cause)[:2000],
                    float(confidence) if confidence is not None else None,
                    float(latency_ms) if latency_ms is not None else None,
                    provider,
                    model,
                    source,
                ),
            )
    except Exception as e:
        # Logging must never break inference
        print(f"[inference_log] write failed: {e}")


def fetch_recent(limit: int = 50):
    """Return most-recent N rows as list of dicts (for the dashboard view)."""
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM inferences ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[inference_log] read failed: {e}")
        return []


def stats():
    """Aggregate stats for the dashboard."""
    try:
        with _conn() as c:
            total = c.execute("SELECT COUNT(*) FROM inferences").fetchone()[0]
            avg_lat = c.execute("SELECT AVG(latency_ms) FROM inferences").fetchone()[0]
            by_type = c.execute(
                "SELECT anomaly_type, COUNT(*) AS n, AVG(latency_ms) AS lat "
                "FROM inferences GROUP BY anomaly_type ORDER BY n DESC"
            ).fetchall()
            return {
                "total": total,
                "avg_latency_ms": avg_lat,
                "by_type": [
                    {"type": r[0], "count": r[1], "avg_latency_ms": r[2]} for r in by_type
                ],
            }
    except Exception as e:
        print(f"[inference_log] stats failed: {e}")
        return {"total": 0, "avg_latency_ms": None, "by_type": []}
