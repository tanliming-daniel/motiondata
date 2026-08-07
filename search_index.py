from __future__ import annotations

import gc
import hashlib
import json
import re
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLAUSE_RE = re.compile(r"[，。！？；,!?;\n]+|\b(?:and|then|while|after|before)\b", re.I)


def split_query_clauses(text: str) -> list[str]:
    parts = [
        re.sub(
            r"^(?:然后|接着|并且|and|then)\s*",
            "",
            " ".join(part.split()),
            flags=re.I,
        )
        for part in CLAUSE_RE.split(text)
        if part.strip()
    ]
    return parts or [text.strip()]


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

    def candidates(self, tokens: Iterable[str], limit: int | None = None) -> set[str]:
        terms = [token.replace('"', '""') for token in tokens if token]
        if not terms:
            return set()
        match = " OR ".join(f'"{term}"' for term in terms)
        if limit is None:
            rows = self.connection.execute("SELECT object_id FROM documents WHERE documents MATCH ?", (match,))
        else:
            rows = self.connection.execute(
                "SELECT object_id FROM documents WHERE documents MATCH ? ORDER BY bm25(documents) LIMIT ?",
                (match, max(int(limit), 1)),
            )
        return {str(row[0]) for row in rows}

    def close(self) -> None:
        self.connection.close()


class SemanticRetriever:
    """Optional local dense retrieval for motion captions."""

    def __init__(
        self,
        index_dir: Path,
        model_name: str | None,
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.model_name = model_name
        self.device_name = device
        self.max_length = max(16, int(max_length))
        self.embeddings: np.ndarray | None = None
        self.rows: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._embedding_tensor = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._search_cache: OrderedDict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = OrderedDict()
        self._search_cache_lock = threading.Lock()
        self._search_cache_size = 128
        self._load_index()

    @property
    def available(self) -> bool:
        return bool(self.model_name and self.embeddings is not None and self.rows)

    def _load_index(self) -> None:
        manifest_path = self.index_dir / "manifest.json"
        embedding_path = self.index_dir / "embeddings.npy"
        rows_path = self.index_dir / "captions.jsonl"
        if not (manifest_path.is_file() and embedding_path.is_file() and rows_path.is_file()):
            return
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.embeddings = np.load(embedding_path, mmap_mode="r")
            with rows_path.open("r", encoding="utf-8") as handle:
                self.rows = [json.loads(line) for line in handle if line.strip()]
            if self.embeddings.ndim != 2 or len(self.rows) != self.embeddings.shape[0]:
                self.embeddings = None
                self.rows = []
        except (OSError, ValueError, json.JSONDecodeError):
            self.embeddings = None
            self.rows = []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_name:
            raise RuntimeError("semantic model is not configured")
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("semantic retrieval requires torch and transformers") from exc
            device = self.device_name
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if isinstance(device, str) and device.startswith("cuda:"):
                resolved_device = device
            elif str(device).isdigit():
                resolved_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = device
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_options = {"low_cpu_mem_usage": True}
            if str(resolved_device).startswith("cuda"):
                model_options["torch_dtype"] = torch.float16
            self._model = AutoModel.from_pretrained(self.model_name, **model_options)
            self._model.to(resolved_device).eval()
            self._torch = torch
            self._device = resolved_device
            if str(resolved_device).startswith("cuda"):
                embedding_array = np.array(self.embeddings, copy=True)
                self._embedding_tensor = torch.from_numpy(
                    embedding_array
                ).to(resolved_device)
                del embedding_array
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (ImportError, OSError, AttributeError):
                pass

    def _encode_tensor(self, texts: list[str]):
        self._ensure_model()
        torch = self._torch
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self._model(**encoded)
            pooled = output.last_hidden_state[:, 0]
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._inference_lock:
            pooled = self._encode_tensor(texts)
            return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    def search(self, query: str, top_k: int = 300) -> list[tuple[float, dict[str, Any]]]:
        if not self.available:
            raise RuntimeError("semantic index or model is unavailable")
        normalized_query = " ".join(query.strip().split())
        count = min(max(int(top_k), 1), len(self.rows))
        cache_key = (normalized_query.casefold(), count)
        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                self._search_cache.move_to_end(cache_key)
                return cached

        clauses = split_query_clauses(normalized_query)
        texts = [normalized_query, *clauses[1:]] if len(clauses) > 1 else [normalized_query]
        with self._inference_lock:
            vectors = self._encode_tensor(texts)
            if self._embedding_tensor is not None:
                matrix = self._embedding_tensor
                vectors = vectors.to(dtype=matrix.dtype)
                scores_tensor = matrix @ vectors[0]
                if len(vectors) > 1:
                    clause_scores = matrix @ vectors[1:].T
                    scores_tensor = 0.6 * scores_tensor + 0.4 * clause_scores.max(dim=1).values
                values, indices_tensor = self._torch.topk(scores_tensor, k=count)
                values_array = values.float().cpu().numpy()
                indices = indices_tensor.cpu().numpy()
                result = [
                    (float(score), self.rows[int(index)])
                    for score, index in zip(values_array, indices)
                ]
            else:
                vectors_array = vectors.float().cpu().numpy()
                embeddings = np.asarray(self.embeddings, dtype=np.float32)
                scores = embeddings @ vectors_array[0]
                if len(vectors_array) > 1:
                    scores = 0.6 * scores + 0.4 * np.max(embeddings @ vectors_array[1:].T, axis=1)
                indices = np.argpartition(-scores, count - 1)[:count]
                indices = indices[np.argsort(-scores[indices], kind="stable")]
                result = [(float(scores[index]), self.rows[int(index)]) for index in indices]

        with self._search_cache_lock:
            self._search_cache[cache_key] = result
            self._search_cache.move_to_end(cache_key)
            while len(self._search_cache) > self._search_cache_size:
                self._search_cache.popitem(last=False)
        return result
