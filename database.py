"""
database.py – Ultra-fast, zero-latency database layer for DRVision.
Uses local SQLite by default for instant (<1ms) reads/writes and maximum reliability.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime
from config import Config, BASE_DIR

logger = logging.getLogger(__name__)

SQLITE_PATH = os.path.join(BASE_DIR, "drvision.db")


def _parse_json_field(value) -> list:
    """Safely parse a DB field that may be a list, JSON string, or Python repr string."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        val = value.strip()
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        import ast
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [val] if val else []
    return []


def get_connection():
    """Return an ultra-fast local SQLite connection."""
    conn = sqlite3.connect(SQLITE_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Ensure tables and performance indexes exist in SQLite."""
    ddl = """
    CREATE TABLE IF NOT EXISTS predictions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename      TEXT        NOT NULL,
        heatmap_filename    TEXT,
        original_filename   TEXT,
        predicted_grade     INTEGER     NOT NULL,
        confidence_score    REAL        NOT NULL,
        grade_label         TEXT        NOT NULL,
        risk_level          TEXT        NOT NULL,
        recommendation      TEXT        NOT NULL,
        findings            TEXT,
        clinical_notes      TEXT,
        lesion_locations    TEXT,
        groq_raw_response   TEXT,
        created_at          TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_grade ON predictions(predicted_grade);
    CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC);
    """
    try:
        with get_connection() as conn:
            conn.executescript(ddl)
        logger.info(f"SQLite database initialized at {SQLITE_PATH}")
    except Exception as e:
        logger.exception(f"Database init failed: {e}")


# Initialize schema immediately on module load
init_db()


def save_prediction(data: dict) -> int:
    """Insert a new prediction record into SQLite instantly."""
    data = data.copy()
    findings_json = json.dumps(data.get("findings", []))
    lesions_json = json.dumps(data.get("lesion_locations", []))

    sql = """
    INSERT INTO predictions (
        image_filename, heatmap_filename, original_filename,
        predicted_grade, confidence_score, grade_label,
        risk_level, recommendation, findings,
        clinical_notes, lesion_locations, groq_raw_response
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (
            data["image_filename"],
            data.get("heatmap_filename"),
            data.get("original_filename"),
            data["predicted_grade"],
            float(data["confidence_score"]),
            data["grade_label"],
            data["risk_level"],
            data["recommendation"],
            findings_json,
            data.get("clinical_notes"),
            lesions_json,
            data.get("groq_raw_response")
        ))
        conn.commit()
        return cur.lastrowid


def get_all_predictions(page: int = 1, per_page: int = 300) -> dict:
    """Return paginated prediction history instantly."""
    offset = (page - 1) * per_page
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions;")
        total = cur.fetchone()[0]

        sql = """
        SELECT id, image_filename, heatmap_filename, original_filename,
               predicted_grade, confidence_score, grade_label,
               risk_level, recommendation, findings, clinical_notes,
               lesion_locations, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ? OFFSET ?;
        """
        cur.execute(sql, (per_page, offset))
        rows = [dict(r) for r in cur.fetchall()]

    records = []
    for r in rows:
        r["findings"] = _parse_json_field(r.get("findings"))
        r["lesion_locations"] = _parse_json_field(r.get("lesion_locations"))
        records.append(r)

    return {
        "records": records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, -(-total // per_page)) if total > 0 else 1,
    }


def get_prediction_by_id(pred_id: int) -> dict | None:
    """Return a single prediction record by id instantly."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM predictions WHERE id = ?;", (pred_id,))
        row = cur.fetchone()
        row = dict(row) if row else None

    if not row:
        return None

    row["findings"] = _parse_json_field(row.get("findings"))
    row["lesion_locations"] = _parse_json_field(row.get("lesion_locations"))
    return row


def get_grade_statistics() -> list[dict]:
    """Return count of predictions per grade."""
    with get_connection() as conn:
        cur = conn.cursor()
        sql = """
        SELECT predicted_grade, grade_label, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_grade, grade_label
        ORDER BY predicted_grade;
        """
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]
