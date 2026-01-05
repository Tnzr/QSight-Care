"""Lightweight SQLite helper utilities for the Streamlit diabetic retinopathy app."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DB_PATH = Path("data") / "qsight_app.db"


def init_db() -> None:
    """Initialise database schema if it does not yet exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                weight_kg REAL,
                height_cm REAL,
                bmi REAL,
                obesity_flag INTEGER,
                insulin INTEGER,
                smoking INTEGER,
                alcohol INTEGER,
                vascular_disease INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                inference_mode TEXT,
                final_label TEXT,
                final_confidence REAL,
                head_fullres_conf REAL,
                head_comp_conf REAL,
                head_quantum_conf REAL,
                ensemble_weights TEXT,
                metadata TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection() -> Iterable[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _boolean(value: bool) -> int:
    return int(bool(value))


def create_patient(profile: Dict[str, Any]) -> int:
    now = datetime.utcnow().isoformat()
    payload = profile.copy()
    payload.update({"created_at": now, "updated_at": now})
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(["?" for _ in payload])
    values = list(payload.values())
    with get_connection() as conn:
        cur = conn.execute(
            f"INSERT INTO patients ({columns}) VALUES ({placeholders})", values
        )
        conn.commit()
        return cur.lastrowid


def update_patient(patient_id: int, profile: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    payload = profile.copy()
    payload["updated_at"] = now
    assignments = ", ".join([f"{key} = ?" for key in payload.keys()])
    values = list(payload.values()) + [patient_id]
    with get_connection() as conn:
        conn.execute(
            f"UPDATE patients SET {assignments} WHERE id = ?",
            values,
        )
        conn.commit()


def delete_patient(patient_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()


def list_patients() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
    return dict(row) if row else None


def record_assessment(patient_id: int, payload: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    data = {
        "patient_id": patient_id,
        "created_at": now,
        "inference_mode": payload.get("inference_mode"),
        "final_label": payload.get("final_label"),
        "final_confidence": payload.get("final_confidence"),
        "head_fullres_conf": payload.get("head_fullres_conf"),
        "head_comp_conf": payload.get("head_comp_conf"),
        "head_quantum_conf": payload.get("head_quantum_conf"),
        "ensemble_weights": json.dumps(payload.get("ensemble_weights", [])),
        "metadata": json.dumps(payload.get("metadata", {})),
    }
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    values = list(data.values())
    with get_connection() as conn:
        conn.execute(
            f"INSERT INTO assessments ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


def list_assessments(patient_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM assessments
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        if data.get("ensemble_weights"):
            data["ensemble_weights"] = json.loads(data["ensemble_weights"])
        if data.get("metadata"):
            data["metadata"] = json.loads(data["metadata"])
        results.append(data)
    return results
