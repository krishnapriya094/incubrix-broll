"""Job store: SQLite-backed persistence for background jobs.

Integration Standard point 3 (background jobs) and point 4 (safe retries) rely
on this: a job is created once per Idempotency-Key and its state is persisted
across restarts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:20]


class JobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    request TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    warnings TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    # -- writes ----------------------------------------------------------- #
    def create(self, request: dict, idempotency_key: str | None) -> dict:
        with self._lock, self._connect() as conn:
            if idempotency_key:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if row:
                    return self._row_to_dict(row)
            job_id = new_job_id()
            now = time.time()
            conn.execute(
                "INSERT INTO jobs (job_id, idempotency_key, status, progress, request, "
                "created_at, updated_at) VALUES (?, ?, 'queued', 0, ?, ?, ?)",
                (job_id, idempotency_key, json.dumps(request), now, now),
            )
            conn.commit()
            return self.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"status", "progress", "result", "error", "warnings"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        values = []
        for col in cols:
            value = fields[col]
            if col in {"result", "error", "warnings"} and value is not None:
                value = json.dumps(value)
            values.append(value)
        values.append(time.time())
        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(f'{c} = ?' for c in cols)}, updated_at = ? WHERE job_id = ?"
        with self._lock, self._connect() as conn:
            conn.execute(sql, values)
            conn.commit()

    # -- reads ------------------------------------------------------------ #
    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("request", "result", "error", "warnings"):
            if data.get(key):
                try:
                    data[key] = json.loads(data[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return data
