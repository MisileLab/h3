import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)


class AudioCache:
    """Persistent cache for audio hash to transcript text."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_cache (
                    audio_hash TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    text TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (audio_hash, target_lang)
                )
                """
            )
            self._conn.commit()

    def get(self, audio_hash: str, target_lang: str) -> str | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT text FROM audio_cache WHERE audio_hash = ? AND target_lang = ?",
                (audio_hash, target_lang),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set(self, audio_hash: str, target_lang: str, text: str) -> None:
        if not text:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO audio_cache (audio_hash, target_lang, text, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(audio_hash, target_lang)
                DO UPDATE SET text = excluded.text, updated_at = excluded.updated_at
                """,
                (audio_hash, target_lang, text, now),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
