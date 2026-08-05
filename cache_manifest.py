from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


class CacheManifest:
    """SQLite metadata store for generated GLB files and LRU maintenance."""

    def __init__(self, cache_root: Path) -> None:
        self.root = cache_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ".cache_manifest.sqlite3"
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS cache_entries (
                relative_path TEXT PRIMARY KEY,
                object_id TEXT,
                dataset TEXT NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT,
                created_at REAL NOT NULL,
                last_access_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        try:
            self.connection.execute("ALTER TABLE cache_entries ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        self.connection.commit()

    def record(self, path: Path, *, object_id: str, dataset: str, checksum: bool = False, hit: bool = False) -> None:
        if not path.is_file():
            return
        now = time.time()
        size = path.stat().st_size
        digest = self._sha256(path) if checksum else None
        self.connection.execute(
            """INSERT INTO cache_entries
               (relative_path, object_id, dataset, size, sha256, created_at, last_access_at, access_count, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(relative_path) DO UPDATE SET
                 object_id=excluded.object_id, dataset=excluded.dataset, size=excluded.size,
                 sha256=COALESCE(excluded.sha256, cache_entries.sha256),
                 last_access_at=excluded.last_access_at, access_count=cache_entries.access_count + 1, hit_count=cache_entries.hit_count + excluded.hit_count""",
            (str(path.resolve().relative_to(self.root)), object_id, dataset, size, digest, now, now, 1 if hit else 0),
        )
        self.connection.commit()

    def reconcile(self) -> int:
        known: set[str] = set()
        for path in self.root.rglob("*.glb"):
            relative = str(path.relative_to(self.root))
            known.add(relative)
            row = self.connection.execute(
                "SELECT object_id, dataset FROM cache_entries WHERE relative_path = ?", (relative,)
            ).fetchone()
            self.record(path, object_id=str(row[0]) if row else "", dataset=str(row[1]) if row else path.parts[0])
        self.connection.executemany(
            "DELETE FROM cache_entries WHERE relative_path = ?",
            ((path,) for (path,) in self.connection.execute("SELECT relative_path FROM cache_entries") if path not in known),
        )
        self.connection.commit()
        return len(known)

    def stats(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(SUM(access_count), 0), COALESCE(SUM(hit_count), 0) FROM cache_entries"
        ).fetchone()
        return {"file_count": int(row[0]), "total_bytes": int(row[1]), "access_count": int(row[2]), "hit_count": int(row[3]), "hit_rate": round(int(row[3]) / max(int(row[2]), 1), 4)}

    def prune(self, *, max_bytes: int, dataset: str | None = None, dry_run: bool = False) -> list[Path]:
        query = "SELECT relative_path, size FROM cache_entries"
        args: tuple[str, ...] = ()
        if dataset:
            query += " WHERE dataset = ?"
            args = (dataset,)
        rows = self.connection.execute(query + " ORDER BY last_access_at ASC", args).fetchall()
        total = sum(int(row[1]) for row in rows)
        removed: list[Path] = []
        for relative, size in rows:
            if total <= max_bytes:
                break
            path = self.root / str(relative)
            removed.append(path)
            total -= int(size)
            if not dry_run:
                path.unlink(missing_ok=True)
                self.connection.execute("DELETE FROM cache_entries WHERE relative_path = ?", (relative,))
        if not dry_run:
            self.connection.commit()
        return removed

    def cleanup_temporary(self) -> int:
        removed = 0
        for path in self.root.rglob("*.tmp.*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def close(self) -> None:
        self.connection.close()

