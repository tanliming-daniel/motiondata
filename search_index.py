from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable


class SearchIndex:
    """Persistent FTS5 candidate index for the lexical motion search."""

    def __init__(self, path: Path, *, fingerprint: str, rows: Iterable[tuple[str, str]]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._ensure(rows, fingerprint)

    @staticmethod
    def fingerprint(source_path: Path, entry_count: int) -> str:
        stat = source_path.stat()
        value = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{entry_count}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _ensure(self, rows: Iterable[tuple[str, str]], fingerprint: str) -> None:
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        current = self.connection.execute("SELECT value FROM metadata WHERE key = 'fingerprint'").fetchone()
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(object_id UNINDEXED, content, tokenize='unicode61')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite was built without FTS5 support.") from exc
        if current and current[0] == fingerprint:
            return
        self.connection.execute("DELETE FROM documents")
        self.connection.executemany("INSERT INTO documents(object_id, content) VALUES (?, ?)", rows)
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES('fingerprint', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (fingerprint,),
        )
        self.connection.commit()

    def candidates(self, tokens: Iterable[str]) -> set[str]:
        terms = [token.replace('"', '""') for token in tokens if token]
        if not terms:
            return set()
        match = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute("SELECT object_id FROM documents WHERE documents MATCH ?", (match,))
        return {str(row[0]) for row in rows}

    def close(self) -> None:
        self.connection.close()

