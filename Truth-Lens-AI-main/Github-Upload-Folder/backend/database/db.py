"""
TruthLens AI — SQLite Database Layer
Handles connection, schema creation, and CRUD helpers.
"""

import sqlite3
import json
import os
from pathlib import Path

DB_PATH = Path(__file__).parent / "truthlens.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print("[DB] Initialized TruthLens database.")


def insert_report(
    input_type: str,
    input_summary: str,
    verdict: str,
    confidence: float,
    credibility_score: int | None,
    signals: list[str],
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO reports (input_type, input_summary, verdict, confidence, credibility_score, signals)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (input_type, input_summary[:300] if input_summary else "", verdict, confidence, credibility_score, json.dumps(signals)),
        )
        conn.commit()
        rid = cur.lastrowid
    finally:
        conn.close()

    # Update keyword frequency if text input
    if input_type == "text":
        _update_keywords(input_summary)

    return rid


def _update_keywords(text: str):
    """Extract and count keywords from submitted text."""
    import re
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","can",
        "this","that","these","those","i","we","you","he","she","it","they","and",
        "or","but","if","in","on","at","to","for","of","with","by","from","as","into",
        "about","after","before","between","during","under","over","above","below",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in stopwords]
    freq: dict[str, int] = {}
    for kw in keywords:
        freq[kw] = freq.get(kw, 0) + 1

    conn = get_connection()
    for kw, count in freq.items():
        existing = conn.execute("SELECT id, frequency FROM keywords WHERE keyword=?", (kw,)).fetchone()
        if existing:
            conn.execute("UPDATE keywords SET frequency=?, last_seen=CURRENT_TIMESTAMP WHERE id=?",
                         (existing["frequency"] + count, existing["id"]))
        else:
            conn.execute("INSERT INTO keywords (keyword, frequency) VALUES (?, ?)", (kw, count))
    conn.commit()
    conn.close()


def get_dashboard_data() -> dict:
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as c FROM reports").fetchone()["c"]
    fake = conn.execute("SELECT COUNT(*) as c FROM reports WHERE verdict='FAKE'").fetchone()["c"]
    real = conn.execute("SELECT COUNT(*) as c FROM reports WHERE verdict='REAL'").fetchone()["c"]
    suspicious = conn.execute("SELECT COUNT(*) as c FROM reports WHERE verdict='SUSPICIOUS'").fetchone()["c"]

    recent_rows = conn.execute(
        "SELECT input_type, input_summary, verdict, confidence, created_at FROM reports ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    recent = [dict(r) for r in recent_rows]

    keyword_rows = conn.execute(
        "SELECT keyword, frequency FROM keywords ORDER BY frequency DESC LIMIT 30"
    ).fetchall()
    keywords = [dict(r) for r in keyword_rows]

    conn.close()
    return {
        "total": total,
        "fake_count": fake,
        "real_count": real,
        "suspicious_count": suspicious,
        "recent": recent,
        "trending_keywords": keywords,
    }


def log_request(endpoint: str, status_code: int, processing_time_ms: float):
    conn = get_connection()
    conn.execute(
        "INSERT INTO logs (endpoint, status_code, processing_time_ms) VALUES (?,?,?)",
        (endpoint, status_code, processing_time_ms),
    )
    conn.commit()
    conn.close()
