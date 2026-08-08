from __future__ import annotations

import gc
import hashlib
import json
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


CLAUSE_RE = re.compile(r"[，。！？；,!?;\n]+|\b(?:and|then|while|after|before)\b", re.I)
MOTION_CLIP_SUFFIX_RE = re.compile(r"(?:_clip\d+)+$", re.I)
MOTION_TAKE_SUFFIX_RE = re.compile(r"_\d+$")
CAPTION_PREFIX_RE = re.compile(r"^(?:output|caption|description)\s*:\s*", re.I)
INVALID_CAPTION_PATTERNS = (
    re.compile(r"^sorry[,\s]", re.I),
    re.compile(r"(?:can't|cannot) provide (?:that|this) information", re.I),
    re.compile(r"no (?:visible )?(?:motion|action) (?:can be|is) (?:identified|determined)", re.I),
)

QWEN3_MOTION_QUERY_INSTRUCTION = (
    "Given a Chinese or English action query, retrieve English motion captions of a person doing the "
    "queried activity, not captions merely related to its topic or objects."
)
QWEN3_ENCODER_FAMILY = "qwen3-embedding"
QWEN3_INDEX_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SemanticEncoderProfile:
    family: str
    pooling: Literal["last_token"]
    query_instruction: str
    short_cjk_template: str
    padding_side: Literal["left"]
    torch_dtype: Literal["bfloat16"]


def qwen3_encoder_profile(model_name: str | None) -> SemanticEncoderProfile:
    normalized = str(model_name or "").casefold().replace("_", "-")
    if "qwen3-embedding" not in normalized:
        raise ValueError(
            "semantic retrieval requires Qwen3-Embedding; "
            f"unsupported model: {model_name or '<empty>'}"
        )
    return SemanticEncoderProfile(
        family=QWEN3_ENCODER_FAMILY,
        pooling="last_token",
        query_instruction=QWEN3_MOTION_QUERY_INSTRUCTION,
        short_cjk_template="一个人正在从事以下活动：{query}",
        padding_side="left",
        torch_dtype="bfloat16",
    )


def format_semantic_query(query: str, profile: SemanticEncoderProfile) -> str:
    normalized_query = " ".join(query.strip().split())
    if (
        profile.short_cjk_template
        and len(normalized_query) <= 8
        and any("\u4e00" <= character <= "\u9fff" for character in normalized_query)
    ):
        normalized_query = profile.short_cjk_template.format(query=normalized_query)
    return f"Instruct: {profile.query_instruction}\nQuery:{normalized_query}"


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Return the final non-padding token for left- or right-padded batches."""
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = last_hidden_states.new_tensor(
        range(last_hidden_states.shape[0]), dtype=sequence_lengths.dtype
    )
    return last_hidden_states[batch_indices, sequence_lengths]


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


def strip_motion_variant_suffix(value: str) -> str:
    stem = value
    previous = None
    while stem != previous:
        previous = stem
        stem = MOTION_CLIP_SUFFIX_RE.sub("", stem)
        stem = MOTION_TAKE_SUFFIX_RE.sub("", stem)
    return stem


def clean_motion_title(dataset: str, motion_id: str) -> str:
    if dataset.lower() != "motionxpp":
        return ""
    stem = strip_motion_variant_suffix(motion_id.split("/", 1)[-1])
    return " ".join(stem.replace("_", " ").split())


def motion_series_key(dataset: str, motion_id: str) -> str:
    if dataset.lower() != "motionxpp":
        return ""
    stem = strip_motion_variant_suffix(motion_id)
    return re.sub(r"[^a-z0-9]+", " ", stem.casefold()).strip()


def clean_search_caption(value: str) -> str:
    return CAPTION_PREFIX_RE.sub("", " ".join(str(value).split())).strip()


def is_searchable_caption(value: str) -> bool:
    caption = clean_search_caption(value)
    return bool(caption) and not any(pattern.search(caption) for pattern in INVALID_CAPTION_PATTERNS)


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
        *,
        load_index: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.model_name = model_name
        self.device_name = device
        self.max_length = max(16, int(max_length))
        self.encoder_profile = qwen3_encoder_profile(model_name) if model_name else None
        self.embeddings: np.ndarray | None = None
        self.rows: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._embedding_tensor: Any = None
        self._device: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._search_cache: OrderedDict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = OrderedDict()
        self._search_cache_lock = threading.Lock()
        self._search_cache_size = 128
        if load_index:
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
            embeddings = np.load(embedding_path, mmap_mode="r")
            with rows_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if embeddings.ndim != 2 or len(rows) != embeddings.shape[0]:
                raise ValueError("semantic index rows do not match the embedding matrix")
            self._validate_index_contract(embeddings, rows)
            self.embeddings = embeddings
            self.rows = rows
        except (OSError, json.JSONDecodeError) as exc:
            self.embeddings = None
            self.rows = []
            raise RuntimeError(f"failed to load semantic index {self.index_dir}: {exc}") from exc

    def _validate_index_contract(self, embeddings: np.ndarray, rows: list[dict[str, Any]]) -> None:
        if self.encoder_profile is None:
            raise ValueError("semantic model is not configured")
        expected = {
            "schema_version": QWEN3_INDEX_SCHEMA_VERSION,
            "encoder_family": self.encoder_profile.family,
            "pooling": self.encoder_profile.pooling,
            "query_instruction": self.encoder_profile.query_instruction,
            "short_cjk_template": self.encoder_profile.short_cjk_template,
            "document_instruction": "",
            "padding_side": self.encoder_profile.padding_side,
            "normalized": True,
        }
        for field, value in expected.items():
            if self.manifest.get(field) != value:
                raise ValueError(
                    f"semantic index {field} mismatch: expected {value!r}, "
                    f"got {self.manifest.get(field)!r}; rebuild the Qwen3 index"
                )
        if embeddings.dtype != np.float16:
            raise ValueError(f"semantic index dtype must be float16, got {embeddings.dtype}")
        dimension = int(self.manifest.get("dimension") or 0)
        count = int(self.manifest.get("count") or 0)
        if embeddings.shape != (count, dimension):
            raise ValueError(
                f"semantic index shape mismatch: manifest {(count, dimension)}, "
                f"embeddings {embeddings.shape}"
            )
        if len(rows) != count:
            raise ValueError(f"semantic index row count mismatch: manifest {count}, rows {len(rows)}")

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
            assert self.encoder_profile is not None
            self._tokenizer.padding_side = self.encoder_profile.padding_side
            model_options: dict[str, Any] = {"low_cpu_mem_usage": True}
            if str(resolved_device).startswith("cuda"):
                model_options["torch_dtype"] = torch.bfloat16
            model = AutoModel.from_pretrained(self.model_name, **model_options)
            self._validate_model_and_index(model)
            model.to(resolved_device).eval()
            self._model = model
            self._torch = torch
            self._device = resolved_device
            if str(resolved_device).startswith("cuda") and self.embeddings is not None:
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

    def _validate_model_and_index(self, model: Any) -> None:
        assert self.encoder_profile is not None
        if self.embeddings is not None:
            hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
            if hidden_size and hidden_size != int(self.embeddings.shape[1]):
                raise RuntimeError(
                    "semantic model/index dimension mismatch: "
                    f"model produces {hidden_size}, index stores {self.embeddings.shape[1]}"
                )
        expected_config_hash = str(self.manifest.get("model_config_sha256") or "")
        config_path = Path(str(self.model_name)) / "config.json"
        if expected_config_hash and config_path.is_file():
            actual_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if actual_config_hash != expected_config_hash:
                raise RuntimeError("semantic model config does not match the index manifest")

    def _encode_tensor(self, texts: list[str], *, is_query: bool):
        self._ensure_model()
        torch = self._torch
        if is_query:
            assert self.encoder_profile is not None
            texts = [format_semantic_query(text, self.encoder_profile) for text in texts]
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
            pooled = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        with self._inference_lock:
            pooled = self._encode_tensor(texts, is_query=False)
            return pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        with self._inference_lock:
            pooled = self._encode_tensor(texts, is_query=True)
            return pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)

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
            vectors = self._encode_tensor(texts, is_query=True)
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


class LocalReranker:
    """Lazy FP16 cross-encoder used only for the shortlisted retrieval candidates."""

    def __init__(self, model_name: str | None, device: str = "auto", max_length: int = 256, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device_name = device
        self.max_length = max(32, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = 8192

    @property
    def configured(self) -> bool:
        return bool(self.model_name)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_name:
            raise RuntimeError("reranker model is not configured")
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("reranking requires torch and transformers") from exc
            device = self.device_name
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif str(device).isdigit():
                device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            options: dict[str, Any] = {"low_cpu_mem_usage": True}
            if str(device).startswith("cuda"):
                options["torch_dtype"] = torch.float16
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **options)
            self._model.to(device).eval()
            self._torch = torch
            self._device = device
            gc.collect()

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        keys = [(" ".join(query.split()).casefold(), " ".join(document.split()).casefold()) for query, document in pairs]
        results: list[float | None] = [None] * len(keys)
        missing_positions: list[int] = []
        with self._cache_lock:
            for position, key in enumerate(keys):
                value = self._cache.get(key)
                if value is None:
                    missing_positions.append(position)
                else:
                    self._cache.move_to_end(key)
                    results[position] = value
        if missing_positions:
            self._ensure_model()
            with self._inference_lock, self._torch.inference_mode():
                for start in range(0, len(missing_positions), self.batch_size):
                    positions = missing_positions[start : start + self.batch_size]
                    batch = [pairs[position] for position in positions]
                    encoded = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self._device) for key, value in encoded.items()}
                    logits = self._model(**encoded, return_dict=True).logits.view(-1).float().cpu().tolist()
                    with self._cache_lock:
                        for position, value in zip(positions, logits):
                            score = float(value)
                            results[position] = score
                            self._cache[keys[position]] = score
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
        return [float(value) for value in results if value is not None]
