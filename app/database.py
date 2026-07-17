from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL CHECK(subject IN ('A', 'B')),
    vector BLOB NOT NULL,
    quality REAL NOT NULL,
    detection_confidence REAL NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'enrollment',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(subject, source_sha256)
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    predicted_label TEXT NOT NULL,
    score_a REAL NOT NULL,
    score_b REAL NOT NULL,
    quality REAL NOT NULL,
    detection_confidence REAL NOT NULL,
    corrected_label TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()

    def add_embedding(self, *, subject: str, vector: bytes, quality: float, detection_confidence: float, source_sha256: str, source_kind: str = "enrollment") -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO embeddings
                (subject, vector, quality, detection_confidence, source_sha256, source_kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (subject, vector, quality, detection_confidence, source_sha256, source_kind, self.now()),
            )
            return cursor.rowcount == 1

    def list_embeddings(self, subject: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM embeddings WHERE subject = ? AND active = 1 ORDER BY created_at DESC", (subject,)))

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute("SELECT subject, COUNT(*) AS total FROM embeddings WHERE active = 1 GROUP BY subject").fetchall()
        counts = {"A": 0, "B": 0}
        counts.update({row["subject"]: int(row["total"]) for row in rows})
        return counts

    def save_prediction(self, *, prediction_id: str, vector: bytes, predicted_label: str, score_a: float, score_b: float, quality: float, detection_confidence: float) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO predictions
                (id, vector, predicted_label, score_a, score_b, quality, detection_confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (prediction_id, vector, predicted_label, score_a, score_b, quality, detection_confidence, self.now()),
            )

    def get_prediction(self, prediction_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()

    def correct_prediction(self, prediction_id: str, corrected_label: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE predictions SET corrected_label = ? WHERE id = ?", (corrected_label, prediction_id))
            connection.execute(
                "INSERT INTO audit_events(event_type, detail, created_at) VALUES (?, ?, ?)",
                ("prediction_corrected", f"{prediction_id}:{corrected_label}", self.now()),
            )

    def purge_old_predictions(self, retention_days: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM predictions WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

    def erase_all(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM embeddings")
            connection.execute("DELETE FROM predictions")
            connection.execute("DELETE FROM audit_events")
