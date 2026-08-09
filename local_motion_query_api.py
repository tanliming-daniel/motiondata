#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
from pathlib import Path
import random
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
from typing import Any, cast


from urllib.parse import parse_qs, unquote, urlparse

import motion_taxonomy
from search_index import (
    LocalReranker,
    SearchIndex,
    SemanticRetriever,
    clean_motion_title,
    clean_search_caption,
    is_searchable_caption,
    motion_series_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MULTIMOTION_DIR = SCRIPT_DIR.parent
WEB_DIR = SCRIPT_DIR / "web"
WEB_TEMPLATE_DIR = WEB_DIR / "templates"
WEB_STATIC_DIR = WEB_DIR / "static"
DEFAULT_INDEX_PATH = SCRIPT_DIR / "dataset" / "motion_index.jsonl"
DEFAULT_CACHE_ROOT = SCRIPT_DIR / "cache" / "models"
DEFAULT_PREVIEW_ROOT = Path("/mnt/nas/cy/humanmotion/multimotion_previews")
DEFAULT_TEMP_MODEL_ROOT = Path(tempfile.gettempdir()) / "multimotion_glb"
DEFAULT_MODEL_DIR = MULTIMOTION_DIR / "models" / "smplh"
DEFAULT_SMPLX_MODEL_DIR = MULTIMOTION_DIR / "models" / "smplx"
DEFAULT_CONVERTER_ENV = "sam2dam2"
DEFAULT_CONDA_EXE = "conda"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7091
DEFAULT_SEMANTIC_MODEL = "/data5/cy/models/qwen3-embedding-0.6b"
DEFAULT_SEMANTIC_INDEX = "/data5/cy/multimotion/server_bundle_lazy/dataset/semantic_index_qwen3_06b"
DEFAULT_SEMANTIC_DEVICE = "cuda:3"
DEFAULT_RERANKER_MODEL = "/data5/cy/models/bge-reranker-v2-m3"
DEFAULT_RERANKER_DEVICE = "cuda:3"
DEFAULT_FRAME_STEP = 1
DEFAULT_INTERX_FPS = 30.0
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
FACET_CACHE_SIZE = 128
API_PREFIX = "/api/v1"
OPENAPI_PATH = "/openapi.json"
LOCAL_ENCODER_NAME = "local_motion_lexical_bm25_v1"
HYBRID_ENCODER_NAME = "local_motion_hybrid_rerank_v3"
DEFAULT_SEARCH_RANDOMNESS = 0.0
DEFAULT_RERANK_CANDIDATES = 20
MODEL_CONTENT_TYPE = "model/gltf-binary"


@lru_cache(maxsize=32)
def static_file_payload(path: str, mtime_ns: int, compressed: bool) -> tuple[bytes, str]:
    raw = Path(path).read_bytes()
    data = gzip.compress(raw, compresslevel=5) if compressed else raw
    etag = hashlib.sha256(data).hexdigest()
    return data, f'"{etag}"'

TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "in",
    "individual",
    "is",
    "it",
    "its",
    "people",
    "person",
    "quickly",
    "of",
    "on",
    "or",
    "she",
    "slowly",
    "someone",
    "the",
    "their",
    "them",
    "then",
    "to",
    "while",
    "with",
}

TOKEN_ALIASES = {
    "arms": "arm",
    "backs": "back",
    "claps": "clap",
    "clapping": "clap",
    "embrace": "hug",
    "embraces": "hug",
    "embracing": "hug",
    "greet": "greeting",
    "greets": "greeting",
    "greeting": "greeting",
    "handshake": "shake",
    "hands": "hand",
    "highfive": "high",
    "highfives": "high",
    "hugs": "hug",
    "hugging": "hug",
    "jumps": "jump",
    "jumping": "jump",
    "kicks": "kick",
    "kicking": "kick",
    "leans": "lean",
    "leaning": "lean",
    "pats": "pat",
    "patting": "pat",
    "pulls": "pull",
    "pulling": "pull",
    "pushes": "push",
    "pushing": "push",
    "raises": "raise",
    "raising": "raise",
    "runs": "run",
    "running": "run",
    "shakes": "shake",
    "shaking": "shake",
    "slapped": "slap",
    "slapping": "slap",
    "slaps": "slap",
    "strikes": "slap",
    "striking": "slap",
    "stood": "stand",
    "standing": "stand",
    "walks": "walk",
    "walking": "walk",
    "握手": "shake hand",
    "拥抱": "hug",
    "抱": "hug",
    "拍": "pat",
    "拍背": "pat back",
    "打脸": "slap face",
    "打": "slap",
    "扇耳光": "slap face",
    "扇巴掌": "slap face",
    "打耳光": "slap face",
    "抽耳光": "slap face",
    "巴掌": "slap face",
    "拍肩膀": "pat shoulder",
    "拍肩": "pat shoulder",
    "推": "push",
    "拉": "pull",
    "踢": "kick",
    "走": "walk",
    "走路": "walk",
    "跑": "run",
    "跑步": "run",
    "跳": "jump",
    "跳跃": "jump",
    "击掌": "high five",
    "高五": "high five",
    "站在": "stand",
    "站着": "stand",
    "站": "stand",
    "家庭成员": "person",
    "家人": "person",
    "厨房岛台": "kitchen platform table counter",
    "岛台": "platform table counter",
    "厨房": "kitchen",
    "整理杯子": "organize cup",
    "整理": "organize arrange",
    "收拾": "organize clean",
    "杯子": "cup",
    "杯": "cup",
    "两人": "two person",
    "两个人": "two person",
}


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def read_npy_header(handle: Any) -> dict[str, Any] | None:
    if handle.read(6) != b"\x93NUMPY":
        return None
    version = handle.read(2)
    if len(version) != 2:
        return None
    length_size = 2 if version[0] == 1 else 4
    length_bytes = handle.read(length_size)
    if len(length_bytes) != length_size:
        return None
    header_length = struct.unpack("<H" if length_size == 2 else "<I", length_bytes)[0]
    if header_length > 1_048_576:
        return None
    header = ast.literal_eval(handle.read(header_length).decode("latin1"))
    return header if isinstance(header, dict) else None


def read_npy_frame_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            header = read_npy_header(handle)
        shape = header.get("shape") if isinstance(header, dict) else None
        return int(shape[0]) if isinstance(shape, tuple) and shape and int(shape[0]) > 0 else None
    except (OSError, UnicodeDecodeError, ValueError, SyntaxError, struct.error):
        return None


def read_npy_preview_rotation(path: Path) -> float:
    try:
        with path.open("rb") as handle:
            header = read_npy_header(handle)
            if not header or header.get("fortran_order"):
                return 0.0
            descr = str(header.get("descr") or "")
            if descr not in {"<f4", ">f4", "=f4", "|f4"}:
                return 0.0
            root_orient = struct.unpack(">3f" if descr == ">f4" else "<3f", handle.read(12))
        angle = math.sqrt(sum(value * value for value in root_orient))
        if angle < 1e-8:
            source_up_x, source_up_z = 0.0, 0.0
        else:
            axis_x, axis_y, axis_z = (value / angle for value in root_orient)
            sine, cosine = math.sin(angle), math.cos(angle)
            source_up_x = -axis_z * sine + axis_x * axis_y * (1.0 - cosine)
            source_up_z = axis_x * sine + axis_z * axis_y * (1.0 - cosine)
        rotation = -math.degrees(math.atan2(source_up_x, source_up_z))
        return 0.0 if abs(rotation) < 8.0 else round(rotation, 2)
    except (OSError, UnicodeDecodeError, ValueError, SyntaxError, struct.error):
        return 0.0


def normalize_route_path(path: str) -> str:
    if not path:
        return "/"
    if path == "/":
        return path
    return path.rstrip("/") or "/"


def is_usable_preview_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < 16:
            return False
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    if path.suffix.lower() == ".webp":
        return header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if path.suffix.lower() == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    return False


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def canonicalize_token(token: str) -> str:
    lowered = token.strip().lower()
    if not lowered:
        return ""
    mapped = TOKEN_ALIASES.get(lowered, lowered)
    if mapped != lowered:
        return mapped
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 4 and lowered.endswith("ing"):
        return lowered[:-3]
    if len(lowered) > 3 and lowered.endswith("ed"):
        return lowered[:-2]
    if len(lowered) > 3 and lowered.endswith("s"):
        return lowered[:-1]
    return lowered


def extract_cjk_alias_tokens(raw_token: str) -> list[str]:
    tokens: list[str] = []
    for alias, mapped in sorted(TOKEN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if not contains_cjk(alias) or alias not in raw_token:
            continue
        for token in tokenize(mapped):
            if token not in tokens:
                tokens.append(token)
    return tokens


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        token = canonicalize_token(raw_token)
        if token == raw_token and contains_cjk(raw_token):
            cjk_tokens = extract_cjk_alias_tokens(raw_token)
            if cjk_tokens:
                tokens.extend(cjk_tokens)
                continue
        for part in str(token).split():
            if part and part not in STOPWORDS:
                tokens.append(part)
    return tokens


def safe_relative_path(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


def clamp_int(value: Any, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        number = int(str(value).strip())
    except (AttributeError, TypeError, ValueError):
        number = default
    if number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def clamp_float(value: Any, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        number = float(str(value).strip())
    except (AttributeError, TypeError, ValueError):
        number = default
    if number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    value = normalize_whitespace(values[0])
    return value or None


def alias_tokens_to_search_text(alias_tokens: list[str]) -> str:
    return normalize_whitespace(" ".join(token for token in alias_tokens if token.isascii()))


@dataclass(frozen=True)
class MotionEntry:
    object_id: str
    dataset: str
    motion_id: str
    source_motion: Path
    caption_file: Path
    captions: tuple[str, ...]
    description: str
    source_title: str
    series_key: str
    cache_glb: Path
    search_text: str
    token_counts: Counter[str]
    token_length: int
    frame_count: int | None = None
    classification: dict[str, Any] | None = None


class MotionIndex:
    def __init__(self, *, source_path: Path, cache_root: Path, entries: list[MotionEntry], search_index_path: Path | None = None) -> None:
        self.source_path = source_path
        self.cache_root = cache_root
        self.entries = entries
        self.entries_by_id = {entry.object_id: entry for entry in entries}
        self._frame_count_cache: dict[str, int | None] = {}
        self._frame_count_lock = threading.Lock()
        self._preview_rotation_cache: dict[str, float] = {}
        self._idf = self._build_idf(entries)
        self._average_length = sum(entry.token_length for entry in entries) / max(len(entries), 1)
        search_path = search_index_path or source_path.with_suffix(".fts5.sqlite3")
        self._search_index = SearchIndex(
            search_path,
            fingerprint=SearchIndex.fingerprint(source_path, len(entries)),
            rows=((entry.object_id, " ".join(tokenize(entry.search_text))) for entry in entries),
        )
        self._sorted_entries = sorted(entries, key=lambda entry: (entry.dataset, entry.motion_id))
        self.entry_positions = {entry.object_id: index for index, entry in enumerate(entries)}
        self.all_entry_mask = (1 << len(entries)) - 1
        self.all_entry_ids = {entry.object_id for entry in entries}
        self.dataset_entry_ids: dict[str, set[str]] = {}
        self.axis_entry_ids: dict[str, dict[str, set[str]]] = {
            axis: {} for axis in motion_taxonomy.AXIS_ORDER
        }
        self.node_entry_ids: dict[str, set[str]] = {node_id: set() for node_id in motion_taxonomy.NODES_BY_ID}
        self.node_direct_entry_ids: dict[str, set[str]] = {node_id: set() for node_id in motion_taxonomy.NODES_BY_ID}
        self.node_direct_counts: Counter[str] = Counter()
        default_facets: dict[str, Counter[str]] = {axis: Counter() for axis in motion_taxonomy.AXIS_ORDER}
        default_facets["dataset"] = Counter()
        default_facets["action_node"] = Counter()
        default_facets["action_node_direct"] = Counter()
        self.library_classifications: dict[str, dict[str, Any]] = {}
        for entry in entries:
            classification = motion_taxonomy.classify_motion(entry)
            self.library_classifications[entry.object_id] = classification
            self.dataset_entry_ids.setdefault(entry.dataset, set()).add(entry.object_id)
            default_facets["dataset"][entry.dataset] += 1
            for axis, value in classification["taxonomy"].items():
                self.axis_entry_ids[axis].setdefault(value, set()).add(entry.object_id)
                default_facets[axis][value] += 1
            direct_ids = {str(node["id"]) for node in classification.get("action_nodes") or ()}
            aggregate_ids: set[str] = set()
            for node_id in direct_ids:
                self.node_direct_counts[node_id] += 1
                self.node_direct_entry_ids[node_id].add(entry.object_id)
                default_facets["action_node_direct"][node_id] += 1
                current_id: str | None = node_id
                while current_id:
                    self.node_entry_ids[current_id].add(entry.object_id)
                    aggregate_ids.add(current_id)
                    current_id = motion_taxonomy.NODES_BY_ID[current_id].get("parent_id")
            for node_id in aggregate_ids:
                default_facets["action_node"][node_id] += 1
        self.default_library_facets = {axis: dict(counter) for axis, counter in default_facets.items()}
        self.default_library_hierarchy = motion_taxonomy.taxonomy_payload(
            counts=self.default_library_facets["action_node"],
            direct_counts=self.default_library_facets["action_node_direct"],
        ).get("hierarchy", [])
        self.dataset_entry_masks = {
            key: self.ids_to_mask(entry_ids) for key, entry_ids in self.dataset_entry_ids.items()
        }
        self.axis_entry_masks = {
            axis: {key: self.ids_to_mask(entry_ids) for key, entry_ids in values.items()}
            for axis, values in self.axis_entry_ids.items()
        }
        self.node_entry_masks = {
            node_id: self.ids_to_mask(entry_ids) for node_id, entry_ids in self.node_entry_ids.items()
        }
        self.node_direct_entry_masks = {
            node_id: self.ids_to_mask(entry_ids) for node_id, entry_ids in self.node_direct_entry_ids.items()
        }
        self.visible_node_ids = tuple(
            node["id"] for node in motion_taxonomy.NODES if node.get("asset_taxonomy") and not node.get("legacy_only")
        )
        self._facet_cache: OrderedDict[tuple[Any, ...], dict[str, dict[str, int]]] = OrderedDict()
        self._facet_cache_lock = threading.Lock()
        self._hybrid_cache: OrderedDict[tuple[str, str, int], list[tuple[float, MotionEntry, dict[str, Any]]]] = OrderedDict()
        self._hybrid_cache_lock = threading.Lock()

    def ids_to_mask(self, entry_ids: set[str]) -> int:
        mask = 0
        for object_id in entry_ids:
            position = self.entry_positions.get(object_id)
            if position is not None:
                mask |= 1 << position
        return mask

    def entries_to_mask(self, entries: list[MotionEntry]) -> int:
        mask = 0
        for entry in entries:
            position = self.entry_positions.get(entry.object_id)
            if position is not None:
                mask |= 1 << position
        return mask

    def mask_contains(self, mask: int, entry: MotionEntry) -> bool:
        position = self.entry_positions.get(entry.object_id)
        return position is not None and bool(mask & (1 << position))

    def entries_from_mask(self, mask: int) -> list[MotionEntry]:
        result: list[MotionEntry] = []
        while mask:
            least_significant_bit = mask & -mask
            position = least_significant_bit.bit_length() - 1
            result.append(self.entries[position])
            mask ^= least_significant_bit
        return result

    def get_cached_facets(self, key: tuple[Any, ...]) -> dict[str, dict[str, int]] | None:
        with self._facet_cache_lock:
            value = self._facet_cache.get(key)
            if value is not None:
                self._facet_cache.move_to_end(key)
            return value

    def cache_facets(self, key: tuple[Any, ...], value: dict[str, dict[str, int]]) -> None:
        with self._facet_cache_lock:
            self._facet_cache[key] = value
            self._facet_cache.move_to_end(key)
            while len(self._facet_cache) > FACET_CACHE_SIZE:
                self._facet_cache.popitem(last=False)

    @classmethod
    def from_jsonl(
        cls,
        index_path: Path,
        cache_root: Path | None = None,
        taxonomy_assignments_path: Path | None = None,
        search_index_path: Path | None = None,
    ) -> "MotionIndex":
        if not index_path.is_file():
            raise FileNotFoundError(f"Motion index does not exist: {index_path}")
        resolved_cache_root = (cache_root or DEFAULT_CACHE_ROOT).expanduser().resolve()
        entries: list[MotionEntry] = []
        assignment_path = taxonomy_assignments_path or index_path.with_name("motion_taxonomy_assignments.jsonl")
        assignments = cls._load_taxonomy_assignments(assignment_path) if assignment_path.is_file() else {}
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{index_path}:{line_number}: invalid JSON: {exc}") from exc
                entry = cls._entry_from_row(row, resolved_cache_root, assignments)
                if entry:
                    entries.append(entry)
        if not entries:
            raise RuntimeError(f"No usable motion entries found in: {index_path}")
        return cls(source_path=index_path, cache_root=resolved_cache_root, entries=entries, search_index_path=search_index_path)

    @staticmethod
    def _load_taxonomy_assignments(path: Path) -> dict[str, dict[str, Any]]:
        assignments: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                object_id = normalize_whitespace(str(row.get("object_id") or ""))
                classification = row.get("classification")
                if object_id and isinstance(classification, dict):
                    assignments[object_id] = classification
        return assignments

    @staticmethod
    def _entry_from_row(row: Any, cache_root: Path, assignments: dict[str, dict[str, Any]] | None = None) -> MotionEntry | None:
        if not isinstance(row, dict):
            return None
        dataset = normalize_whitespace(str(row.get("dataset") or "")).lower()
        motion_id = normalize_whitespace(str(row.get("motion_id") or ""))
        source_motion = Path(str(row.get("source_motion") or ""))
        caption_file = Path(str(row.get("caption_file") or ""))
        captions = tuple(clean_search_caption(str(caption)) for caption in row.get("captions") or ())
        captions = tuple(caption for caption in captions if is_searchable_caption(caption))
        description = normalize_whitespace(str(row.get("description") or (captions[0] if captions else "")))
        object_id = normalize_whitespace(str(row.get("object_id") or ""))
        frame_count = clamp_int(row.get("frame_count"), 0, minimum=0) or None
        if not object_id and dataset and motion_id:
            object_id = hashlib.sha256(f"{dataset}:{motion_id}:{source_motion}".encode("utf-8")).hexdigest()
        if not dataset or not motion_id or not str(source_motion) or not object_id:
            return None

        cache_glb = cache_root / dataset / f"{motion_id}.glb"
        source_title = clean_motion_title(dataset, motion_id)
        search_text = normalize_whitespace(" ".join([dataset, motion_id, source_title, description, *captions]))
        token_counts = Counter(tokenize(search_text))
        return MotionEntry(
            object_id=object_id,
            dataset=dataset,
            motion_id=motion_id,
            source_motion=source_motion,
            caption_file=caption_file,
            captions=captions,
            description=description,
            source_title=source_title,
            series_key=motion_series_key(dataset, motion_id),
            cache_glb=cache_glb,
            search_text=search_text,
            token_counts=token_counts,
            token_length=sum(token_counts.values()),
            frame_count=frame_count,
            classification=(assignments or {}).get(object_id),
        )

    def taxonomy_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        aggregate = {node_id: len(entry_ids) for node_id, entry_ids in self.node_entry_ids.items()}
        return aggregate, dict(self.node_direct_counts)

    def frame_count_for(self, entry: MotionEntry) -> int | None:
        if entry.frame_count is not None:
            return entry.frame_count
        if entry.dataset != "motionxpp":
            return None
        with self._frame_count_lock:
            if entry.object_id in self._frame_count_cache:
                return self._frame_count_cache[entry.object_id]
        value = read_npy_frame_count(entry.source_motion)
        with self._frame_count_lock:
            self._frame_count_cache[entry.object_id] = value
        return value

    def preview_rotation_for(self, entry: MotionEntry) -> float:
        if entry.dataset != "motionxpp":
            return 0.0
        with self._frame_count_lock:
            if entry.object_id in self._preview_rotation_cache:
                return self._preview_rotation_cache[entry.object_id]
        value = read_npy_preview_rotation(entry.source_motion)
        with self._frame_count_lock:
            self._preview_rotation_cache[entry.object_id] = value
        return value

    @staticmethod
    def _build_idf(entries: list[MotionEntry]) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for entry in entries:
            document_frequency.update(entry.token_counts.keys())
        total_documents = max(len(entries), 1)
        return {
            token: math.log(1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_summary(self) -> dict[str, Any]:
        dataset_counts = Counter(entry.dataset for entry in self.entries)
        return {
            "entry_count": len(self.entries),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "index_path": str(self.source_path),
            "preview_root": str(DEFAULT_PREVIEW_ROOT),
        }

    def get_entry(self, object_id: str) -> MotionEntry | None:
        return self.entries_by_id.get(object_id)

    def list_entries(
        self,
        *,
        q: str | None = None,
        dataset: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[list[tuple[MotionEntry, float | None]], int, float | None]:
        matches: list[tuple[MotionEntry, float | None]]
        if q:
            scored = self.search_entries(q, top_k=None)
            max_score = scored[0][0] if scored else None
            matches = [(entry, score) for score, entry in scored if self._matches_dataset(entry, dataset)]
        else:
            max_score = None
            matches = [(entry, None) for entry in self._sorted_entries if self._matches_dataset(entry, dataset)]
        total_count = len(matches)
        return matches[offset : offset + limit], total_count, max_score

    @staticmethod
    def _matches_dataset(entry: MotionEntry, dataset: str | None) -> bool:
        if not dataset:
            return True
        return entry.dataset == dataset.strip().lower()

    def search_entries(self, query_text: str, top_k: int | None = None) -> list[tuple[float, MotionEntry]]:
        normalized_query = normalize_whitespace(query_text)
        query_tokens = tokenize(normalized_query)
        if not query_tokens:
            raise ValueError("Query text does not contain searchable tokens.")
        query_counts = Counter(query_tokens)
        candidate_limit = max(top_k * 8, 1000) if top_k is not None else None
        candidate_ids = self._search_index.candidates(query_tokens, limit=candidate_limit)
        scored = [
            (score, entry)
            for object_id in candidate_ids
            if (entry := self.entries_by_id.get(object_id)) is not None
            if (score := self._score_entry(normalized_query, query_counts, entry)) > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[1].dataset, item[1].motion_id))
        if top_k is None:
            return scored
        return scored[: max(top_k, 1)]

    def search_hybrid(
        self,
        query_text: str,
        semantic: SemanticRetriever,
        top_k: int | None = None,
        candidate_k: int = 300,
        semantic_query: str | None = None,
    ) -> list[tuple[float, MotionEntry, dict[str, Any]]]:
        cache_key = (
            normalize_whitespace(query_text).casefold(),
            normalize_whitespace(semantic_query or query_text).casefold(),
            max(int(candidate_k), 1),
        )
        with self._hybrid_cache_lock:
            cached = self._hybrid_cache.get(cache_key)
            if cached is not None:
                self._hybrid_cache.move_to_end(cache_key)
                return cached[: max(top_k, 1)] if top_k is not None else cached
        lexical = self.search_entries(query_text, top_k=candidate_k)
        dense = semantic.search(semantic_query or query_text, top_k=min(candidate_k * 4, len(semantic.rows)))
        lexical_rank = {entry.object_id: rank for rank, (_score, entry) in enumerate(lexical, 1)}
        lexical_score = {entry.object_id: score for score, entry in lexical}
        dense_by_id: dict[str, tuple[int, float, str, str]] = {}
        for rank, (score, row) in enumerate(dense, 1):
            object_id = str(row.get("object_id") or "")
            if object_id and object_id not in dense_by_id:
                dense_by_id[object_id] = (
                    rank,
                    score,
                    str(row.get("caption") or ""),
                    str(row.get("source_type") or "caption"),
                )
        rows: list[tuple[float, MotionEntry, dict[str, Any]]] = []
        for object_id in set(lexical_rank) | set(dense_by_id):
            entry = self.entries_by_id.get(object_id)
            if entry is None:
                continue
            lrank = lexical_rank.get(object_id)
            drank, dscore, matched, source_type = dense_by_id.get(object_id, (None, 0.0, "", ""))
            fused = (0.45 / (60 + lrank) if lrank else 0.0) + (0.55 / (60 + drank) if drank else 0.0)
            detail = {
                "lexical_score": lexical_score.get(object_id, 0.0),
                "semantic_score": dscore,
                "matched_caption": matched,
                "matched_source_type": source_type,
            }
            rows.append((fused, entry, detail))
        rows.sort(key=lambda item: (-item[0], item[1].dataset, item[1].motion_id))
        with self._hybrid_cache_lock:
            self._hybrid_cache[cache_key] = rows
            self._hybrid_cache.move_to_end(cache_key)
            while len(self._hybrid_cache) > 128:
                self._hybrid_cache.popitem(last=False)
        return rows[: max(top_k, 1)] if top_k is not None else rows

    def _score_entry(self, normalized_query: str, query_counts: Counter[str], entry: MotionEntry) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        matched_terms = 0
        length_norm = k1 * (1.0 - b + b * entry.token_length / max(self._average_length, 1.0))
        for token, query_count in query_counts.items():
            term_frequency = entry.token_counts.get(token, 0)
            if not term_frequency:
                continue
            matched_terms += 1
            idf = self._idf.get(token, 0.0)
            term_score = idf * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
            score += term_score * min(query_count, 2)
        if score <= 0:
            return 0.0

        coverage_ratio = matched_terms / max(len(query_counts), 1)
        query_lower = normalized_query.lower()
        entry_lower = entry.search_text.lower()
        bonus = 0.0
        if query_lower in entry_lower:
            bonus += 3.0
        if query_lower == entry.motion_id.lower():
            bonus += 4.0
        if query_lower in {entry.motion_id.lower(), entry.dataset.lower()}:
            bonus += 1.0
        return round(score * (0.65 + coverage_ratio) + bonus, 6)


def convert_entry_to_glb(
    entry: MotionEntry,
    output_path: Path,
    *,
    model_dir: Path,
    frame_step: int,
    interx_fps: float,
    converter_env: str | None,
    conda_exe: str,
) -> None:
    if not entry.source_motion.exists():
        raise FileNotFoundError(f"Motion source does not exist: {entry.source_motion}")
    converter_python = "python" if converter_env else sys.executable
    if entry.dataset == "interhuman":
        converter_script = MULTIMOTION_DIR / "interhuman_to_glb.py"
        command = [
            converter_python,
            str(converter_script),
            "--input",
            str(entry.source_motion),
            "--output",
            str(output_path),
            "--model-dir",
            str(model_dir),
            "--frame-step",
            str(frame_step),
            "--overwrite",
        ]
    elif entry.dataset == "interx":
        converter_script = MULTIMOTION_DIR / "interx_to_glb.py"
        command = [
            converter_python,
            str(converter_script),
            "--input",
            str(entry.source_motion),
            "--output",
            str(output_path),
            "--model-dir",
            str(model_dir),
            "--frame-step",
            str(frame_step),
            "--fps",
            str(interx_fps),
            "--overwrite",
        ]
    elif entry.dataset == "motionxpp":
        converter_script = MULTIMOTION_DIR / "motionxpp_to_glb.py"
        command = [
            converter_python,
            str(converter_script),
            "--input",
            str(entry.source_motion),
            "--output",
            str(output_path),
            "--model-dir",
            str(DEFAULT_SMPLX_MODEL_DIR),
            "--frame-step",
            str(frame_step),
            "--fps",
            "30",
            "--overwrite",
        ]
    else:
        raise ValueError(f"Unsupported dataset for GLB conversion: {entry.dataset}")

    if converter_env:
        command = [conda_exe, "run", "-n", converter_env, *command]

    completed = subprocess.run(
        command,
        cwd=str(MULTIMOTION_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = normalize_whitespace(completed.stderr)
        stdout = normalize_whitespace(completed.stdout)
        details = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(details)


def create_temporary_glb(
    entry: MotionEntry,
    *,
    model_dir: Path,
    frame_step: int,
    interx_fps: float,
    converter_env: str | None,
    conda_exe: str,
    temp_root: Path,
    converter=convert_entry_to_glb,
) -> tuple[Path, Path]:
    """Convert one motion into a request-scoped GLB directory."""
    temp_root.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix=f"{entry.object_id[:12]}-", dir=temp_root))
    output_path = job_dir / f"{entry.object_id}.glb"
    try:
        converter(
            entry,
            output_path,
            model_dir=model_dir,
            frame_step=frame_step,
            interx_fps=interx_fps,
            converter_env=converter_env,
            conda_exe=conda_exe,
        )
        return output_path, job_dir
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


class MotionQueryServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        motion_index: MotionIndex,
        *,
        model_dir: Path,
        frame_step: int,
        interx_fps: float,
        converter_env: str | None,
        conda_exe: str,
        semantic_model: str | None = None,
        semantic_index: Path | None = None,
        semantic_device: str = "auto",
        semantic_max_length: int = 256,
        reranker_model: str | None = None,
        reranker_device: str = "auto",
        reranker_max_length: int = 128,
        reranker_batch_size: int = 32,
        search_prewarm: bool = False,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.motion_index = motion_index
        self.temp_root = DEFAULT_TEMP_MODEL_ROOT / str(os.getpid())
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir
        self.frame_step = frame_step
        self.interx_fps = interx_fps
        self.converter_env = converter_env
        self.conda_exe = conda_exe
        self.semantic_model = normalize_whitespace(semantic_model or "") or None
        self.semantic_index = (
            semantic_index or SCRIPT_DIR / "dataset" / "semantic_index_qwen3_06b"
        ).expanduser().resolve()
        self.semantic_device = semantic_device
        self.semantic_max_length = semantic_max_length
        self._semantic_retriever: SemanticRetriever | None = None
        self._semantic_lock = threading.Lock()
        self.semantic_error: str | None = None
        self.reranker_model = normalize_whitespace(reranker_model or "") or None
        self.reranker_device = reranker_device
        self.reranker_max_length = reranker_max_length
        self.reranker_batch_size = reranker_batch_size
        self.search_prewarm = search_prewarm
        self._reranker: LocalReranker | None = None
        self._reranker_lock = threading.Lock()
        self.reranker_error: str | None = None
        self._lock_guard = threading.Lock()
        self._conversion_locks: dict[str, threading.Lock] = {}

    def conversion_lock_for(self, object_id: str) -> threading.Lock:
        with self._lock_guard:
            lock = self._conversion_locks.get(object_id)
            if lock is None:
                lock = threading.Lock()
                self._conversion_locks[object_id] = lock
            return lock

    def semantic_retriever(self) -> SemanticRetriever | None:
        if not self.semantic_model:
            return None
        with self._semantic_lock:
            if self._semantic_retriever is not None:
                return self._semantic_retriever
            if self.semantic_error is not None:
                return None
            try:
                retriever = SemanticRetriever(
                    self.semantic_index, self.semantic_model, self.semantic_device, self.semantic_max_length
                )
                if not retriever.available:
                    self.semantic_error = f"Qwen3 semantic index is unavailable: {self.semantic_index}"
                    return None
                self._semantic_retriever = retriever
                return retriever
            except (OSError, ValueError, RuntimeError) as exc:
                self.semantic_error = str(exc)
                return None

    def semantic_encoder_name(self) -> str:
        if not self.semantic_model:
            return LOCAL_ENCODER_NAME
        return Path(self.semantic_model).name or self.semantic_model

    def local_reranker(self) -> LocalReranker | None:
        if not self.reranker_model:
            return None
        with self._reranker_lock:
            if self._reranker is None:
                self._reranker = LocalReranker(
                    self.reranker_model,
                    self.reranker_device,
                    self.reranker_max_length,
                    self.reranker_batch_size,
                )
            return self._reranker

    def prewarm_search(self) -> None:
        try:
            semantic = self.semantic_retriever()
            if semantic is not None:
                semantic.search("a person walks", top_k=8)
                self.semantic_error = None
        except Exception as exc:
            self.semantic_error = str(exc)
        try:
            reranker = self.local_reranker()
            if reranker is not None:
                reranker.score([("a person walks", "a person is walking forward")])
            self.reranker_error = None
        except Exception as exc:
            self.reranker_error = str(exc)


class MotionQueryRequestHandler(BaseHTTPRequestHandler):
    server_version = "MultimotionLazyQueryAPI/1.0"

    @property
    def app_server(self) -> MotionQueryServer:
        return cast(MotionQueryServer, self.server)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = normalize_route_path(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=False)

        if path in {"/", API_PREFIX}:
            self._send_json(HTTPStatus.OK, self._build_root_payload())
            return
        if path in {"/health", f"{API_PREFIX}/health"}:
            self._send_json(HTTPStatus.OK, self._build_health_payload())
            return
        if path == "/ui":
            self._send_ui_index()
            return
        if path in {"/ui/action-taxonomy", "/动作资产分类树.html"}:
            self._send_action_taxonomy_page()
            return
        if path.startswith("/ui/static/"):
            relative_path = unquote(path[len("/ui/static/") :])
            self._send_static_file(relative_path)
            return
        if path == f"{API_PREFIX}/stats":
            self._send_json(HTTPStatus.OK, {"status": "ok", "data": self.app_server.motion_index.get_summary()})
            return
        if path == f"{API_PREFIX}/taxonomy":
            started = time.perf_counter()
            aggregate, direct = self.app_server.motion_index.taxonomy_counts()
            compact = first_query_value(query, "view") == "compact"
            payload = motion_taxonomy.taxonomy_payload(
                counts=aggregate,
                direct_counts=direct,
                include_nodes=not compact,
            )
            payload["default_facets"] = {
                axis: self.app_server.motion_index.default_library_facets.get(axis, {})
                for axis in ("dataset", "participants")
            }
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "data": payload},
                cache_control="public, max-age=300, stale-while-revalidate=3600" if compact else "no-cache",
                use_etag=True,
                compact=compact,
                server_timing={"taxonomy": (time.perf_counter() - started) * 1000},
            )
            return
        if path == f"{API_PREFIX}/library":
            self._handle_library(query)
            return
        if path == f"{API_PREFIX}/entries":
            self._handle_entries_list(query)
            return
        if path.startswith(f"{API_PREFIX}/entries/"):
            object_id = unquote(path[len(f"{API_PREFIX}/entries/") :])
            self._handle_entry_detail(object_id)
            return
        if path.startswith("/models/"):
            object_id = unquote(path[len("/models/") :])
            self._handle_model_download(object_id)
            return
        if path.startswith(f"{API_PREFIX}/models/"):
            object_id = unquote(path[len(f"{API_PREFIX}/models/") :])
            self._handle_model_download(object_id)
            return
        if path.startswith(f"{API_PREFIX}/previews/"):
            preview_id = unquote(path[len(f"{API_PREFIX}/previews/") :])
            self._handle_preview_download(preview_id)
            return
        if path in {OPENAPI_PATH, f"{API_PREFIX}/openapi.json"}:
            self._send_json(HTTPStatus.OK, self._build_openapi_payload())
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown GET path: {path}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = normalize_route_path(parsed.path)
        if path not in {"/query", f"{API_PREFIX}/searches"}:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown POST path: {path}")
            return
        try:
            payload = self._read_json_payload()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._handle_search_payload(payload, legacy_mode=path == "/query")

    def _handle_entries_list(self, query: dict[str, list[str]]) -> None:
        limit = clamp_int(first_query_value(query, "limit"), DEFAULT_PAGE_SIZE, minimum=1, maximum=MAX_PAGE_SIZE)
        offset = clamp_int(first_query_value(query, "offset"), 0, minimum=0)
        include_model_info = parse_bool(first_query_value(query, "include_model_info"), True)
        search_query = first_query_value(query, "q")
        dataset_filter = first_query_value(query, "dataset")
        try:
            page, total_count, max_score = self.app_server.motion_index.list_entries(
                q=search_query,
                dataset=dataset_filter,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        data = [
            self._serialize_entry(entry, include_model_info=include_model_info, score=score, max_score=max_score)
            for entry, score in page
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "data": data,
                "meta": {
                    "count": len(data),
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "filters": {"q": search_query, "dataset": dataset_filter},
                },
            },
        )


    def _handle_library(self, query: dict[str, list[str]]) -> None:
        request_started = time.perf_counter()
        timings: dict[str, float] = {}
        limit = clamp_int(first_query_value(query, "limit"), 9, minimum=1, maximum=MAX_PAGE_SIZE)
        offset = clamp_int(first_query_value(query, "offset"), 0, minimum=0)
        dataset_filter = first_query_value(query, "dataset")
        sort_key = first_query_value(query, "sort") or "relevance"
        search_query = first_query_value(query, "q") or first_query_value(query, "search")
        retrieval_mode = (first_query_value(query, "retrieval_mode") or "hybrid").lower()
        use_reranker = parse_bool(first_query_value(query, "rerank"), True)
        use_diversity = parse_bool(first_query_value(query, "diversity"), True)
        if search_query and contains_cjk(search_query):
            use_reranker = False
            use_diversity = False
        if retrieval_mode not in {"lexical", "semantic", "hybrid"}:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "retrieval_mode must be lexical, semantic, or hybrid")
            return
        node_filter = first_query_value(query, "node_id") or first_query_value(query, "taxonomy_node")
        include_descendants = parse_bool(first_query_value(query, "include_descendants"), True)
        compact = first_query_value(query, "view") == "compact"
        include_facets = parse_bool(first_query_value(query, "include_facets"), True)
        if node_filter and node_filter not in motion_taxonomy.NODES_BY_ID:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Unknown taxonomy node: {node_filter}")
            return
        axis_filters = {axis: first_query_value(query, axis) for axis in motion_taxonomy.AXIS_ORDER}
        scored_entries: list[tuple[MotionEntry, float | None]]
        rows: list[tuple[MotionEntry, float | None, dict[str, Any]]]
        page: list[tuple[MotionEntry, float | None, dict[str, Any]]]

        use_default_snapshot = (
            not search_query
            and not dataset_filter
            and not node_filter
            and not any(axis_filters.values())
            and sort_key in {"relevance", "motion_id"}
        )

        if use_default_snapshot:
            total_count = len(self.app_server.motion_index._sorted_entries)
            entries = self.app_server.motion_index._sorted_entries[offset : offset + limit]
            page = [
                (entry, None, self.app_server.motion_index.library_classifications[entry.object_id])
                for entry in entries
            ]
            facets = self.app_server.motion_index.default_library_facets if include_facets else {}
            hierarchy = None if compact else self.app_server.motion_index.default_library_hierarchy
            max_score = None
        else:
            filter_started = time.perf_counter()
            try:
                if search_query:
                    semantic = self.app_server.semantic_retriever() if retrieval_mode in {"semantic", "hybrid"} else None
                    if retrieval_mode in {"semantic", "hybrid"} and semantic is not None:
                        hybrid = self.app_server.motion_index.search_hybrid(search_query, semantic, top_k=None)
                        scored = [(score, entry) for score, entry, _detail in hybrid]
                    else:
                        scored = self.app_server.motion_index.search_entries(search_query, top_k=None)
                        if retrieval_mode in {"semantic", "hybrid"}:
                            retrieval_mode = "lexical"
                    scored_entries = [(entry, score) for score, entry in scored]
                    base_ids = {entry.object_id for entry, _score in scored_entries}
                    max_score = scored[0][0] if scored else None
                else:
                    scored_entries = [] if compact else [(entry, None) for entry in self.app_server.motion_index._sorted_entries]
                    base_ids = self.app_server.motion_index.all_entry_ids
                    max_score = None
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if compact:
                index = self.app_server.motion_index
                base_mask = index.entries_to_mask([entry for entry, _score in scored_entries]) if search_query else index.all_entry_mask
                result_mask = self._filter_library_mask(
                    base_mask,
                    dataset_filter=dataset_filter,
                    node_filter=node_filter,
                    include_descendants=include_descendants,
                    axis_filters=axis_filters,
                )
                if search_query:
                    rows = [
                        (entry, score, index.library_classifications[entry.object_id])
                        for entry, score in scored_entries
                        if index.mask_contains(result_mask, entry)
                    ]
                else:
                    rows = [
                        (entry, None, index.library_classifications[entry.object_id])
                        for entry in index.entries_from_mask(result_mask)
                    ]
                timings["filter"] = (time.perf_counter() - filter_started) * 1000
                facets = {}
                if include_facets:
                    facet_started = time.perf_counter()
                    facet_key = None if search_query else (
                        dataset_filter or "", node_filter or "", include_descendants,
                        tuple((axis, axis_filters.get(axis) or "") for axis in motion_taxonomy.AXIS_ORDER),
                    )
                    facets = index.get_cached_facets(facet_key) if facet_key is not None else None
                    if facets is None:
                        facets = self._build_self_excluding_library_facets_bitmap(
                            base_mask,
                            dataset_filter=dataset_filter,
                            node_filter=node_filter,
                            include_descendants=include_descendants,
                            axis_filters=axis_filters,
                        )
                        if facet_key is not None:
                            index.cache_facets(facet_key, facets)
                    timings["facets"] = (time.perf_counter() - facet_started) * 1000
            else:
                result_ids = self._filter_library_ids(
                    base_ids,
                    dataset_filter=dataset_filter,
                    node_filter=node_filter,
                    include_descendants=include_descendants,
                    axis_filters=axis_filters,
                )
                rows = [
                    (entry, score, self.app_server.motion_index.library_classifications[entry.object_id])
                    for entry, score in scored_entries
                    if entry.object_id in result_ids
                ]
                facets = self._build_self_excluding_library_facets(
                    base_ids,
                    dataset_filter=dataset_filter,
                    node_filter=node_filter,
                    include_descendants=include_descendants,
                    axis_filters=axis_filters,
                ) if include_facets else {}
                timings["filter"] = (time.perf_counter() - filter_started) * 1000
            total_count = len(rows)
            rows = self._sort_library_rows(rows, sort_key)
            if (
                search_query
                and sort_key == "relevance"
                and retrieval_mode in {"semantic", "hybrid"}
                and (use_reranker or use_diversity)
            ):
                ranked = [
                    (score or 0.0, entry)
                    for entry, score, _classification in rows[:DEFAULT_RERANK_CANDIDATES]
                ]
                rerank_succeeded = not use_reranker
                if use_reranker:
                    ranked, _details, rerank_warning = self._rerank_semantic_candidates(search_query, ranked)
                    rerank_succeeded = rerank_warning is None
                if use_diversity and rerank_succeeded:
                    ranked = self._diversify_candidates(ranked)
                ranked_ids = {
                    entry.object_id: (position, score)
                    for position, (score, entry) in enumerate(ranked)
                }
                rows = sorted(
                    rows,
                    key=lambda item: (
                        ranked_ids.get(item[0].object_id, (DEFAULT_RERANK_CANDIDATES, 0.0))[0],
                        -(item[1] or 0.0),
                    ),
                )
                rows = [
                    (entry, ranked_ids.get(entry.object_id, (0, score or 0.0))[1], classification)
                    for entry, score, classification in rows
                ]
            page = rows[offset : offset + limit]
            hierarchy = None if compact else motion_taxonomy.taxonomy_payload(
                    counts=facets.get("action_node", {}),
                    direct_counts=facets.get("action_node_direct", {}),
                ).get("hierarchy", [])
        serialize_started = time.perf_counter()
        data = [
            self._serialize_entry(
                entry,
                include_model_info=True,
                score=score,
                max_score=max_score,
                classification=classification,
                compact=compact,
            )
            for entry, score, classification in page
        ]
        timings["serialize"] = (time.perf_counter() - serialize_started) * 1000
        response: dict[str, Any] = {
            "status": "ok",
            "items": data,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "facets": self._sparse_facets(facets) if compact and include_facets else facets,
            "taxonomy_version": motion_taxonomy.TAXONOMY_VERSION,
            "filters": {
                "q": search_query,
                "retrieval_mode": retrieval_mode,
                "dataset": dataset_filter,
                "node_id": node_filter,
                "include_descendants": include_descendants,
                **{axis: value for axis, value in axis_filters.items() if value},
            },
            "sort": sort_key,
            "retrieval_mode": retrieval_mode,
            "selected_encoder": (
                self.app_server.semantic_encoder_name()
                if retrieval_mode in {"semantic", "hybrid"}
                else LOCAL_ENCODER_NAME
            ),
            "ranking_version": HYBRID_ENCODER_NAME if retrieval_mode in {"semantic", "hybrid"} else LOCAL_ENCODER_NAME,
            "rerank": use_reranker,
            "diversity": use_diversity,
        }
        if compact:
            response["global_total"] = len(self.app_server.motion_index.entries)
        else:
            response["data"] = data
            response["hierarchy"] = hierarchy
            response["summary"] = self.app_server.motion_index.get_summary()
        timings["total"] = (time.perf_counter() - request_started) * 1000
        self._send_json(
            HTTPStatus.OK,
            response,
            cache_control=("private, max-age=60, stale-while-revalidate=300" if compact else "no-cache" if use_default_snapshot else None),
            use_etag=compact or use_default_snapshot,
            compact=compact,
            server_timing=timings,
        )

    def _sparse_facets(self, facets: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        visible = set(self.app_server.motion_index.visible_node_ids)
        return {
            axis: {
                key: count for key, count in values.items()
                if count and (axis not in {"action_node", "action_node_direct"} or key in visible)
            }
            for axis, values in facets.items()
        }

    def _filter_library_ids(
        self,
        base_ids: set[str],
        *,
        dataset_filter: str | None,
        node_filter: str | None,
        include_descendants: bool,
        axis_filters: dict[str, str | None],
        excluded_facet: str | None = None,
    ) -> set[str]:
        result = set(base_ids)
        index = self.app_server.motion_index
        if dataset_filter and excluded_facet != "dataset":
            result.intersection_update(index.dataset_entry_ids.get(dataset_filter.strip().lower(), set()))
        if node_filter and excluded_facet != "action_node":
            node_ids = index.node_entry_ids if include_descendants else index.node_direct_entry_ids
            result.intersection_update(node_ids.get(node_filter, set()))
        for axis, value in axis_filters.items():
            if value and excluded_facet != axis:
                result.intersection_update(index.axis_entry_ids[axis].get(value, set()))
        return result

    def _filter_library_mask(
        self,
        base_mask: int,
        *,
        dataset_filter: str | None,
        node_filter: str | None,
        include_descendants: bool,
        axis_filters: dict[str, str | None],
        excluded_facet: str | None = None,
    ) -> int:
        result = base_mask
        index = self.app_server.motion_index
        if dataset_filter and excluded_facet != "dataset":
            result &= index.dataset_entry_masks.get(dataset_filter.strip().lower(), 0)
        if node_filter and excluded_facet != "action_node":
            node_masks = index.node_entry_masks if include_descendants else index.node_direct_entry_masks
            result &= node_masks.get(node_filter, 0)
        for axis, value in axis_filters.items():
            if value and excluded_facet != axis:
                result &= index.axis_entry_masks[axis].get(value, 0)
        return result

    def _build_self_excluding_library_facets_bitmap(
        self,
        base_mask: int,
        *,
        dataset_filter: str | None,
        node_filter: str | None,
        include_descendants: bool,
        axis_filters: dict[str, str | None],
    ) -> dict[str, dict[str, int]]:
        index = self.app_server.motion_index
        dataset_scope = self._filter_library_mask(
            base_mask,
            dataset_filter=dataset_filter,
            node_filter=node_filter,
            include_descendants=include_descendants,
            axis_filters=axis_filters,
            excluded_facet="dataset",
        )
        facets: dict[str, dict[str, int]] = {
            "dataset": {
                dataset: (dataset_scope & mask).bit_count()
                for dataset, mask in index.dataset_entry_masks.items()
            }
        }
        for axis in motion_taxonomy.AXIS_ORDER:
            axis_scope = self._filter_library_mask(
                base_mask,
                dataset_filter=dataset_filter,
                node_filter=node_filter,
                include_descendants=include_descendants,
                axis_filters=axis_filters,
                excluded_facet=axis,
            )
            facets[axis] = {
                value: (axis_scope & mask).bit_count()
                for value, mask in index.axis_entry_masks[axis].items()
            }
        node_scope = self._filter_library_mask(
            base_mask,
            dataset_filter=dataset_filter,
            node_filter=node_filter,
            include_descendants=include_descendants,
            axis_filters=axis_filters,
            excluded_facet="action_node",
        )
        facets["action_node"] = {
            node_id: (node_scope & index.node_entry_masks.get(node_id, 0)).bit_count()
            for node_id in index.visible_node_ids
        }
        facets["action_node_direct"] = {
            node_id: (node_scope & index.node_direct_entry_masks.get(node_id, 0)).bit_count()
            for node_id in index.visible_node_ids
        }
        return facets

    def _build_self_excluding_library_facets(
        self,
        base_ids: set[str],
        *,
        dataset_filter: str | None,
        node_filter: str | None,
        include_descendants: bool,
        axis_filters: dict[str, str | None],
    ) -> dict[str, dict[str, int]]:
        index = self.app_server.motion_index
        dataset_scope = self._filter_library_ids(
            base_ids,
            dataset_filter=dataset_filter,
            node_filter=node_filter,
            include_descendants=include_descendants,
            axis_filters=axis_filters,
            excluded_facet="dataset",
        )
        facets: dict[str, dict[str, int]] = {
            "dataset": {
                dataset: len(dataset_scope.intersection(entry_ids))
                for dataset, entry_ids in index.dataset_entry_ids.items()
            }
        }
        for axis in motion_taxonomy.AXIS_ORDER:
            axis_scope = self._filter_library_ids(
                base_ids,
                dataset_filter=dataset_filter,
                node_filter=node_filter,
                include_descendants=include_descendants,
                axis_filters=axis_filters,
                excluded_facet=axis,
            )
            facets[axis] = {
                value: len(axis_scope.intersection(entry_ids))
                for value, entry_ids in index.axis_entry_ids[axis].items()
            }
        node_scope = self._filter_library_ids(
            base_ids,
            dataset_filter=dataset_filter,
            node_filter=node_filter,
            include_descendants=include_descendants,
            axis_filters=axis_filters,
            excluded_facet="action_node",
        )
        facets["action_node"] = {
            node_id: len(node_scope.intersection(entry_ids))
            for node_id, entry_ids in index.node_entry_ids.items()
        }
        facets["action_node_direct"] = {
            node_id: len(node_scope.intersection(entry_ids))
            for node_id, entry_ids in index.node_direct_entry_ids.items()
        }
        return facets

    def _sort_library_rows(
        self,
        rows: list[tuple[MotionEntry, float | None, dict[str, Any]]],
        sort_key: str,
    ) -> list[tuple[MotionEntry, float | None, dict[str, Any]]]:
        if sort_key == "motion_id":
            return sorted(rows, key=lambda item: (item[0].dataset, item[0].motion_id))
        if sort_key == "frame_count_desc":
            return sorted(rows, key=lambda item: (-(self.app_server.motion_index.frame_count_for(item[0]) or 0), item[0].dataset, item[0].motion_id))
        if any(score is not None for _entry, score, _classification in rows):
            return sorted(rows, key=lambda item: (-(item[1] or 0.0), item[0].dataset, item[0].motion_id))
        return sorted(rows, key=lambda item: (item[0].dataset, item[0].motion_id))

    def _handle_entry_detail(self, object_id: str) -> None:
        entry = self.app_server.motion_index.get_entry(object_id)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown object_id: {object_id}")
            return
        self._send_json(HTTPStatus.OK, {"status": "ok", "data": self._serialize_entry(entry, include_model_info=True)})

    def _handle_search_payload(self, payload: dict[str, Any], *, legacy_mode: bool) -> None:
        text = normalize_whitespace(str(payload.get("text") or ""))
        image_payload = payload.get("image_base64")
        warnings: list[str] = []
        if text and image_payload:
            warnings.append("Both text and image were provided; text search took priority.")
        if not text and image_payload:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "This lazy motion API supports text queries only.")
            return
        if not text:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Provide a text query in the 'text' field.")
            return

        search_text = self._prepare_search_text(text)
        top_k = clamp_int(payload.get("top_k", 5), 5, minimum=1, maximum=MAX_PAGE_SIZE)
        candidate_k = clamp_int(payload.get("candidate_k", 200), 200, minimum=1)
        candidate_k = max(candidate_k, top_k)
        include_model_info = parse_bool(payload.get("include_model_info"), True)
        randomness = clamp_float(payload.get("randomness", DEFAULT_SEARCH_RANDOMNESS), DEFAULT_SEARCH_RANDOMNESS, maximum=1.0)
        random_seed = payload.get("random_seed")
        retrieval_mode = str(payload.get("retrieval_mode") or "hybrid").lower()
        use_reranker = parse_bool(payload.get("rerank"), True)
        use_diversity = parse_bool(payload.get("diversity"), True)
        if use_reranker and contains_cjk(text):
            use_reranker = False
            use_diversity = False
            warnings.append(
                "Reranking and diversity are bypassed for CJK queries because the indexed captions are English."
            )
        if retrieval_mode not in {"lexical", "semantic", "hybrid"}:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "retrieval_mode must be lexical, semantic, or hybrid")
            return

        try:
            explanations: dict[str, dict[str, Any]] = {}
            semantic = self.app_server.semantic_retriever() if retrieval_mode in {"semantic", "hybrid"} else None
            if retrieval_mode in {"semantic", "hybrid"} and semantic is None:
                detail = f": {self.app_server.semantic_error}" if self.app_server.semantic_error else ""
                warnings.append(f"Semantic retrieval is unavailable{detail}; fell back to lexical retrieval.")
                retrieval_mode = "lexical"
            if retrieval_mode == "hybrid":
                assert semantic is not None
                hybrid = self.app_server.motion_index.search_hybrid(
                    search_text,
                    semantic,
                    top_k=candidate_k,
                    candidate_k=candidate_k,
                    semantic_query=text,
                )
                candidates = [(score, entry) for score, entry, detail in hybrid]
                explanations = {entry.object_id: detail for _score, entry, detail in hybrid}
            elif retrieval_mode == "semantic":
                assert semantic is not None
                dense = semantic.search(text, top_k=candidate_k)
                candidates = []
                seen: set[str] = set()
                for score, row in dense:
                    entry = self.app_server.motion_index.get_entry(str(row.get("object_id") or ""))
                    if entry is not None and entry.object_id not in seen:
                        seen.add(entry.object_id)
                        candidates.append((score, entry))
                        explanations[entry.object_id] = {
                            "semantic_score": score,
                            "matched_caption": row.get("caption", ""),
                            "lexical_score": 0.0,
                        }
            else:
                candidates = self.app_server.motion_index.search_entries(search_text, top_k=candidate_k)
        except (ValueError, RuntimeError) as exc:
            if retrieval_mode in {"semantic", "hybrid"}:
                warnings.append(f"Semantic retrieval failed; fell back to lexical retrieval: {exc}")
                retrieval_mode = "lexical"
                candidates = self.app_server.motion_index.search_entries(search_text, top_k=candidate_k)
                explanations = {}
            else:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
        if retrieval_mode in {"semantic", "hybrid"} and (use_reranker or use_diversity):
            ranked = candidates[:DEFAULT_RERANK_CANDIDATES]
            remaining = candidates[DEFAULT_RERANK_CANDIDATES:]
            rerank_succeeded = not use_reranker
            if use_reranker:
                ranked, rerank_details, rerank_warning = self._rerank_semantic_candidates(text, ranked)
                rerank_succeeded = rerank_warning is None
                for object_id, detail in rerank_details.items():
                    explanations.setdefault(object_id, {}).update(detail)
                if rerank_warning:
                    warnings.append(rerank_warning)
            if use_diversity and rerank_succeeded:
                ranked = self._diversify_candidates(ranked)
            candidates = [*ranked, *remaining]
        max_score = max((score for score, _entry in candidates), default=None)
        results = self._rerank_results(candidates, top_k=top_k, randomness=randomness, random_seed=random_seed)
        data = [
            self._serialize_entry(entry, include_model_info=include_model_info, score=score, max_score=max_score, retrieval=explanations.get(entry.object_id))
            for score, entry in results
        ]
        selected_encoder = (
            self.app_server.semantic_encoder_name()
            if retrieval_mode in {"semantic", "hybrid"}
            else LOCAL_ENCODER_NAME
        )

        common = {
            "query_mode": "text",
            "requested_encoder": payload.get("encoder"),
            "selected_encoder": selected_encoder,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "candidate_count": len(candidates),
            "randomness": randomness,
            "random_seed": random_seed,
            "result_count": len(data),
            "results": data,
            "warnings": warnings,
            "query_text": text,
            "search_text": search_text,
            "retrieval_mode": retrieval_mode,
            "ranking_version": HYBRID_ENCODER_NAME if retrieval_mode in {"semantic", "hybrid"} else LOCAL_ENCODER_NAME,
            "rerank": use_reranker,
            "diversity": use_diversity,
            "data": data,
        }
        if legacy_mode:
            response = {
                "status": "completed",
                **common,
                "index": {
                    "encoder_name": LOCAL_ENCODER_NAME,
                    "db_path": str(self.app_server.motion_index.source_path),
                    "source_type": "motion_index_jsonl",
                    "dataset_root": str(self.app_server.motion_index.cache_root),
                    "feature_dim": None,
                    "supports_text": True,
                    "lazy_conversion": True,
                    "ranking_version": HYBRID_ENCODER_NAME if retrieval_mode in {"semantic", "hybrid"} else LOCAL_ENCODER_NAME,
                    "rerank": use_reranker,
                    "diversity": use_diversity,
                },
                "meta": {"result_count": len(data), "entry_count": len(self.app_server.motion_index.entries)},
                "links": {
                    "self": self._absolute_url("/query"),
                    "rest_search": self._absolute_url(f"{API_PREFIX}/searches"),
                },
            }
        else:
            response = {
                "status": "ok",
                "retrieval_mode": retrieval_mode,
                "ranking_version": HYBRID_ENCODER_NAME if retrieval_mode in {"semantic", "hybrid"} else LOCAL_ENCODER_NAME,
                "rerank": use_reranker,
                "diversity": use_diversity,
                "query": {
                    "mode": "text",
                    "text": text,
                    "search_text": search_text,
                    "top_k": top_k,
                    "candidate_k": candidate_k,
                    "requested_encoder": payload.get("encoder"),
                    "selected_encoder": selected_encoder,
                    "candidate_count": len(candidates),
                    "retrieval_mode": retrieval_mode,
                    "randomness": randomness,
                    "random_seed": random_seed,
                },
                "data": data,
                "results": data,
                "meta": {
                    "result_count": len(data),
                    "entry_count": len(self.app_server.motion_index.entries),
                    "supports_text": True,
                    "lazy_conversion": True,
                },
                "warnings": warnings,
                "links": {"self": self._absolute_url(f"{API_PREFIX}/searches")},
            }
        self._send_json(HTTPStatus.OK, response)

    def _prepare_search_text(self, text: str) -> str:
        if not contains_cjk(text):
            return text
        return alias_tokens_to_search_text(tokenize(text)) or text

    def _rerank_results(
        self,
        candidates: list[tuple[float, MotionEntry]],
        *,
        top_k: int,
        randomness: float,
        random_seed: Any,
    ) -> list[tuple[float, MotionEntry]]:
        if not candidates or randomness <= 0:
            return candidates[:top_k]
        rng: random.Random | random.SystemRandom
        rng = random.Random(str(random_seed)) if random_seed is not None else random.SystemRandom()
        top_score = max(candidates[0][0], 0.0)
        if top_score <= 0:
            return candidates[:top_k]
        remaining = list(candidates)
        selected: list[tuple[float, MotionEntry]] = []
        while remaining and len(selected) < top_k:
            chosen = self._weighted_random_candidate(remaining, top_score=top_score, randomness=randomness, rng=rng)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _rerank_semantic_candidates(
        self,
        query: str,
        candidates: list[tuple[float, MotionEntry]],
    ) -> tuple[list[tuple[float, MotionEntry]], dict[str, dict[str, Any]], str | None]:
        reranker = self.app_server.local_reranker()
        if not candidates or reranker is None:
            warning = None if not candidates else "Reranker is unavailable; kept hybrid ranking."
            return candidates, {}, warning

        pairs: list[tuple[str, str]] = []
        pair_ranges: list[tuple[int, int]] = []
        for _score, entry in candidates:
            start = len(pairs)
            if entry.source_title:
                pairs.append((query, entry.source_title))
            documents = [entry.description, *entry.captions]
            document = " ".join(dict.fromkeys(text for text in documents if is_searchable_caption(text)))
            pairs.append((query, document[:3000]))
            pair_ranges.append((start, len(pairs)))
        try:
            scores = reranker.score(pairs)
            self.app_server.reranker_error = None
        except Exception as exc:
            self.app_server.reranker_error = str(exc)
            return candidates, {}, f"Reranking failed; kept hybrid ranking: {exc}"

        normalized_query = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", query.casefold()).strip()
        reranked: list[tuple[float, MotionEntry]] = []
        details: dict[str, dict[str, Any]] = {}
        for (base_score, entry), (start, end) in zip(candidates, pair_ranges):
            item_scores = scores[start:end]
            title_score = item_scores[0] if entry.source_title and len(item_scores) > 1 else None
            caption_score = item_scores[-1]
            if title_score is None:
                score = caption_score
            else:
                # A noisy generated caption may help, but cannot outweigh a contradictory source title.
                score = max(title_score, min(caption_score, title_score + 1.0))
            normalized_id = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", entry.motion_id.casefold()).strip()
            normalized_title = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", entry.source_title.casefold()).strip()
            exact_match = normalized_query in {normalized_id, normalized_title}
            if exact_match:
                score += 100.0
            score += min(base_score, 1.0) * 0.01
            reranked.append((score, entry))
            details[entry.object_id] = {
                "reranker_score": round(score, 6),
                "title_score": round(title_score, 6) if title_score is not None else None,
                "caption_score": round(caption_score, 6),
                "source_title": entry.source_title or None,
                "exact_source_match": exact_match,
            }
        reranked.sort(key=lambda item: (-item[0], item[1].dataset, item[1].motion_id))
        return reranked, details, None

    def _diversify_candidates(
        self,
        candidates: list[tuple[float, MotionEntry]],
    ) -> list[tuple[float, MotionEntry]]:
        if len(candidates) < 2:
            return candidates
        selected: list[tuple[float, MotionEntry]] = []
        remaining = list(candidates)
        minimum = min(score for score, _entry in candidates)
        maximum = max(score for score, _entry in candidates)
        scale = max(maximum - minimum, 1e-6)
        while remaining:
            best = max(
                remaining,
                key=lambda item: (
                    (item[0] - minimum) / scale
                    - min(
                        0.75,
                        0.15 * sum(self._entry_similarity(item[1], chosen[1]) for chosen in selected),
                    ),
                    item[0],
                    item[1].motion_id,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _entry_similarity(left: MotionEntry, right: MotionEntry) -> float:
        if left.series_key and left.series_key == right.series_key:
            return 1.0
        left_tokens = set(tokenize(left.source_title or left.description))
        right_tokens = set(tokenize(right.source_title or right.description))
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    @staticmethod
    def _weighted_random_candidate(
        candidates: list[tuple[float, MotionEntry]],
        *,
        top_score: float,
        randomness: float,
        rng: random.Random | random.SystemRandom,
    ) -> tuple[float, MotionEntry]:
        pool_top_score = max(score for score, _ in candidates)
        temperature = max(top_score * randomness, 1e-6)
        weighted: list[tuple[float, tuple[float, MotionEntry]]] = []
        total = 0.0
        for candidate in candidates:
            score, _ = candidate
            weight = math.exp((score - pool_top_score) / temperature)
            total += weight
            weighted.append((weight, candidate))
        pick = rng.random() * total
        cumulative = 0.0
        for weight, candidate in weighted:
            cumulative += weight
            if cumulative >= pick:
                return candidate
        return weighted[-1][1]

    def _serialize_entry(
        self,
        entry: MotionEntry,
        *,
        include_model_info: bool,
        score: float | None = None,
        max_score: float | None = None,
        classification: dict[str, Any] | None = None,
        compact: bool = False,
        retrieval: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        legacy_download_path = f"/models/{entry.object_id}"
        rest_download_path = f"{API_PREFIX}/models/{entry.object_id}"
        classification = classification or motion_taxonomy.classify_motion(entry)
        if compact:
            preview_download_path = f"{API_PREFIX}/previews/{entry.object_id}.webp"
            payload: dict[str, Any] = {
                "object_id": entry.object_id,
                "dataset": entry.dataset,
                "motion_id": entry.motion_id,
                "description": entry.description,
                "captions": list(entry.captions),
                "frame_count": self.app_server.motion_index.frame_count_for(entry),
                "preview": {
                    "available": True,
                    "download_path": preview_download_path,
                    "rotation_degrees": 0.0,
                },
                **classification,
            }
            if score is not None:
                payload["score"] = round(score, 6)
                payload["pooled_score"] = round(score / max_score, 6) if max_score else 0.0
            if include_model_info:
                payload["model"] = {
                    "rest_download_path": rest_download_path,
                    "filename": f"{Path(entry.motion_id).name or entry.object_id}.glb",
                    "available": True,
                    "lazy": True,
                }
            return payload
        preview_path = self._resolve_preview_path(entry)
        preview_available = preview_path is not None
        preview_is_canonical = bool(preview_path and preview_path.with_suffix(preview_path.suffix + ".canonical-v2").is_file())
        preview_download_path = f"{API_PREFIX}/previews/{entry.object_id}.webp"
        payload: dict[str, Any] = {
            "object_id": entry.object_id,
            "dataset": entry.dataset,
            "motion_id": entry.motion_id,
            "description": entry.description,
            "captions": list(entry.captions),
            "frame_count": self.app_server.motion_index.frame_count_for(entry),
            "source_motion": str(entry.source_motion),
            "caption_file": str(entry.caption_file),
            "preview": {
                "available": preview_available,
                "download_path": preview_download_path,
                "download_url": self._absolute_url(preview_download_path),
                "rotation_degrees": 0.0 if preview_is_canonical else self.app_server.motion_index.preview_rotation_for(entry),
            },
            "animal": entry.dataset,
            "track_name": entry.motion_id,
            "action_name": entry.description,
            "source_glb": str(entry.source_motion),
            "frame_start": None,
            "frame_end": None,
            "has_icon": False,
            "thumbnail": {
                "available": False,
                "download_path": None,
                "rest_download_path": None,
                "download_url": None,
            },
            "_links": {
                "self": self._absolute_url(f"{API_PREFIX}/entries/{entry.object_id}"),
                "download": self._absolute_url(rest_download_path),
                "icon": None,
                "thumbnail": None,
            },
        }
        payload.update(classification)
        if score is not None:
            payload["score"] = round(score, 6)
            payload["pooled_score"] = round(score / max_score, 6) if max_score else 0.0
            payload["best_view_id"] = None
            payload["best_view_path"] = None
        if retrieval:
            payload["retrieval"] = retrieval
        if include_model_info:
            payload["model"] = {
                "sha256": entry.object_id,
                "download_path": legacy_download_path,
                "rest_download_path": rest_download_path,
                "download_url": self._absolute_url(rest_download_path),
                "filename": f"{Path(entry.motion_id).name or entry.object_id}.glb",
                "available": True,
                "lazy": True,
            }
        return payload

    def _preview_path_for(self, entry: MotionEntry) -> Path:
        return DEFAULT_PREVIEW_ROOT / entry.dataset / f"{entry.motion_id}.webp"

    def _resolve_preview_path(self, entry: MotionEntry) -> Path | None:
        preview_path = self._preview_path_for(entry)
        if is_usable_preview_file(preview_path):
            return preview_path
        legacy_path = preview_path.with_suffix(".png")
        return legacy_path if is_usable_preview_file(legacy_path) else None

    def _handle_preview_download(self, preview_id: str) -> None:
        suffix = ".webp" if preview_id.endswith(".webp") else ".png" if preview_id.endswith(".png") else ""
        object_id = preview_id[: -len(suffix)] if suffix else preview_id
        entry = self.app_server.motion_index.get_entry(object_id)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown object_id: {object_id}")
            return
        preview_path = self._resolve_preview_path(entry)
        if preview_path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Preview is not available: {object_id}")
            return
        data = preview_path.read_bytes()
        etag = f'"{hashlib.sha256(data).hexdigest()}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self._send_cors_headers()
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "image/webp" if preview_path.suffix == ".webp" else "image/png")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write_response_body(data)

    def _handle_model_download(self, object_id: str) -> None:
        entry = self.app_server.motion_index.get_entry(object_id)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown object_id: {object_id}")
            return
        with self.app_server.conversion_lock_for(entry.object_id):
            job_dir: Path | None = None
            response_started = False
            try:
                model_path, job_dir = create_temporary_glb(
                    entry,
                    model_dir=self.app_server.model_dir,
                    frame_step=self.app_server.frame_step,
                    interx_fps=self.app_server.interx_fps,
                    converter_env=self.app_server.converter_env,
                    conda_exe=self.app_server.conda_exe,
                    temp_root=self.app_server.temp_root,
                )
                file_size = model_path.stat().st_size
                self.send_response(HTTPStatus.OK)
                self._send_cors_headers()
                self.send_header("Content-Type", MODEL_CONTENT_TYPE)
                self.send_header("Content-Length", str(file_size))
                filename = f"{Path(entry.motion_id).name or entry.object_id}.glb"
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("X-Multimotion-Generated", "true")
                self.end_headers()
                response_started = True
                with model_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except KeyboardInterrupt:
                raise
            except SystemExit as exc:
                if not response_started:
                    self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to convert motion to GLB: {exc}")
            except Exception as exc:
                if not response_started:
                    self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to convert motion to GLB: {exc}")
            finally:
                if job_dir is not None:
                    shutil.rmtree(job_dir, ignore_errors=True)

    def _build_root_payload(self) -> dict[str, Any]:
        summary = self.app_server.motion_index.get_summary()
        return {
            "service": "multimotion-lazy-query-api",
            "version": "1.0",
            "status": "ok",
            "entry_count": summary["entry_count"],
            "api_prefix": API_PREFIX,
            "openapi_url": self._absolute_url(OPENAPI_PATH),
            "lazy_conversion": True,
            "routes": {
                "health": f"{API_PREFIX}/health",
                "stats": f"{API_PREFIX}/stats",
                "entries": f"{API_PREFIX}/entries",
                "library": f"{API_PREFIX}/library",
                "taxonomy": f"{API_PREFIX}/taxonomy",
                "ui": "/ui",
                "entry_detail": f"{API_PREFIX}/entries/{{object_id}}",
                "searches": f"{API_PREFIX}/searches",
                "model_download": f"{API_PREFIX}/models/{{object_id}}",
                "preview_download": f"{API_PREFIX}/previews/{{object_id}}.webp",
            },
            "legacy_routes": {"query": "/query", "model_download": "/models/{object_id}"},
        }

    def _build_health_payload(self) -> dict[str, Any]:
        summary = self.app_server.motion_index.get_summary()
        return {
            "status": "ok",
            "entry_count": summary["entry_count"],
            "dataset_counts": summary["dataset_counts"],
            "index_path": summary["index_path"],
            "preview_root": summary["preview_root"],
            "lazy_conversion": True,
            "semantic_retrieval": {
                "configured": bool(self.app_server.semantic_model),
                "index": str(self.app_server.semantic_index),
                "model": self.app_server.semantic_model,
                "loaded": bool(self.app_server._semantic_retriever and self.app_server._semantic_retriever._model is not None),
                "encoder": self.app_server.semantic_encoder_name(),
                "error": self.app_server.semantic_error,
                "manifest": (
                    self.app_server._semantic_retriever.manifest
                    if self.app_server._semantic_retriever is not None
                    else None
                ),
            },
            "reranker": {
                "configured": bool(self.app_server.reranker_model),
                "model": self.app_server.reranker_model,
                "device": self.app_server.reranker_device,
                "loaded": bool(self.app_server._reranker and self.app_server._reranker.loaded),
                "error": self.app_server.reranker_error,
            },
        }

    def _build_openapi_payload(self) -> dict[str, Any]:
        return {
            "openapi": "3.1.0",
            "info": {
                "title": "Multimotion Lazy Query API",
                "version": "1.0.0",
                "description": "Search captioned human interaction motions and convert GLB files on demand.",
            },
            "servers": [{"url": self._absolute_url("")}],
            "paths": {
                f"{API_PREFIX}/health": {"get": {"summary": "Health check"}},
                f"{API_PREFIX}/entries": {"get": {"summary": "List or filter motion entries"}},
                f"{API_PREFIX}/library": {"get": {"summary": "List classified motion entries for the web UI"}},
                f"{API_PREFIX}/taxonomy": {"get": {"summary": "Fetch motion taxonomy axes"}},
                f"{API_PREFIX}/entries/{{object_id}}": {"get": {"summary": "Fetch one motion entry"}},
                f"{API_PREFIX}/searches": {"post": {"summary": "Run text search without GLB conversion"}},
                f"{API_PREFIX}/models/{{object_id}}": {
                    "get": {"summary": "Convert the selected motion to a temporary GLB, then download it"}
                },
                f"{API_PREFIX}/previews/{{object_id}}.webp": {
                    "get": {"summary": "Download the static WebP motion thumbnail"}
                },
            },
        }


    def _send_ui_index(self) -> None:
        template_path = WEB_TEMPLATE_DIR / "motion_library.html"
        if not template_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Motion library UI is not installed.")
            return
        data = template_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write_response_body(data)

    def _send_action_taxonomy_page(self) -> None:
        self._send_ui_index()

    def _send_static_file(self, relative_path: str) -> None:
        if not relative_path or relative_path.startswith("/"):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid static path.")
            return
        target = (WEB_STATIC_DIR / relative_path).resolve()
        try:
            target.relative_to(WEB_STATIC_DIR.resolve())
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid static path.")
            return
        if not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Static file not found: {relative_path}")
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        compress = accepts_gzip and target.suffix in {".js", ".css", ".html"}
        data, etag = static_file_payload(str(target), target.stat().st_mtime_ns, compress)
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self._send_cors_headers()
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("ETag", etag)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("ETag", etag)
        if compress:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write_response_body(data)

    def _read_json_payload(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise ValueError("Missing Content-Length header.")
        try:
            content_length = int(length_header)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Request body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request JSON payload must be an object.")
        return payload

    def _base_url(self) -> str:
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host")
        if not host:
            bind_host = self.app_server.server_address[0]
            bind_port = self.app_server.server_address[1]
            host = f"{bind_host}:{bind_port}"
        scheme = self.headers.get("X-Forwarded-Proto")
        if not scheme:
            scheme = "https" if str(host).endswith(":443") else "http"
        return f"{scheme}://{host}"

    def _absolute_url(self, path: str) -> str:
        base_url = self._base_url()
        if not path:
            return base_url
        if path.startswith(("http://", "https://")):
            return path
        normalized = path if path.startswith("/") else f"/{path}"
        return f"{base_url}{normalized}"

    def _send_json(
        self,
        status_code: HTTPStatus,
        payload: dict[str, Any],
        *,
        cache_control: str | None = None,
        use_etag: bool = False,
        compact: bool = False,
        server_timing: dict[str, float] | None = None,
    ) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        ).encode("utf-8")
        content_encoding = None
        if len(data) >= 1024 and "gzip" in self.headers.get("Accept-Encoding", "").lower():
            data = gzip.compress(data, compresslevel=5)
            content_encoding = "gzip"
        etag = f'"{hashlib.sha256(data).hexdigest()}"' if use_etag else None
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self._send_cors_headers()
            if cache_control:
                self.send_header("Cache-Control", cache_control)
            self.send_header("ETag", etag)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        if etag:
            self.send_header("ETag", etag)
        if content_encoding:
            self.send_header("Content-Encoding", content_encoding)
        self.send_header("Vary", "Accept-Encoding")
        if server_timing:
            timing_value = ", ".join(f"{name};dur={duration:.2f}" for name, duration in server_timing.items())
            self.send_header("Server-Timing", timing_value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write_response_body(data)

    def _write_response_body(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_error_json(self, status_code: HTTPStatus, message: str) -> None:
        self._send_json(status_code, {"status": "failed", "message": message})

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the multimotion lazy retrieval API.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help=f"Motion index JSONL. Default: {DEFAULT_INDEX_PATH}")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT), help="Legacy index path mapping only; the API never writes GLB files here.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help=f"SMPL-H model directory. Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument(
        "--converter-env",
        default=DEFAULT_CONVERTER_ENV,
        help=(
            "Conda environment used for lazy GLB conversion. "
            "Use an empty string to run converters in the API process environment. "
            f"Default: {DEFAULT_CONVERTER_ENV}"
        ),
    )
    parser.add_argument("--conda-exe", default=DEFAULT_CONDA_EXE, help=f"Conda executable for lazy conversion. Default: {DEFAULT_CONDA_EXE}")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP, help="Keep every Nth frame during lazy GLB export.")
    parser.add_argument("--interx-fps", type=float, default=DEFAULT_INTERX_FPS, help="Frame rate used for InterX GLB export.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port. Default: {DEFAULT_PORT}")
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL, help="Local Qwen3-Embedding model directory.")
    parser.add_argument(
        "--semantic-index",
        default=DEFAULT_SEMANTIC_INDEX,
        help="Qwen3 caption index directory.",
    )
    parser.add_argument("--semantic-device", default=DEFAULT_SEMANTIC_DEVICE, help="Dense retrieval device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL, help="Local cross-encoder model directory used for result reranking.")
    parser.add_argument("--reranker-device", default=DEFAULT_RERANKER_DEVICE, help="Reranker device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--reranker-max-length", type=int, default=128)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument(
        "--search-prewarm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load semantic and reranker models in the background after startup.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    if args.interx_fps <= 0:
        raise SystemExit("--interx-fps must be > 0")

    index_path = Path(args.index).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    converter_env = normalize_whitespace(args.converter_env) or None
    motion_index = MotionIndex.from_jsonl(index_path, cache_root=cache_root)
    server = MotionQueryServer(
        (args.host, args.port),
        MotionQueryRequestHandler,
        motion_index,
        model_dir=model_dir,
        frame_step=args.frame_step,
        interx_fps=args.interx_fps,
        converter_env=converter_env,
        conda_exe=args.conda_exe,
        semantic_model=args.semantic_model,
        semantic_index=Path(args.semantic_index),
        semantic_device=args.semantic_device,
        semantic_max_length=args.semantic_max_length,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_max_length=args.reranker_max_length,
        reranker_batch_size=args.reranker_batch_size,
        search_prewarm=args.search_prewarm,
    )
    example_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(
        json.dumps(
            {
                "status": "starting",
                "host": args.host,
                "port": args.port,
                "index_path": str(index_path),
                "preview_root": str(DEFAULT_PREVIEW_ROOT),
                "temporary_model_root": str(server.temp_root),
                "model_dir": str(model_dir),
                "converter_env": converter_env,
                "conda_exe": args.conda_exe,
                "frame_step": args.frame_step,
                "interx_fps": args.interx_fps,
                "entry_count": len(motion_index.entries),
                "lazy_conversion": True,
                "api_root": f"http://{example_host}:{args.port}{API_PREFIX}",
                "example_search": {
                    "method": "POST",
                    "url": f"http://{example_host}:{args.port}{API_PREFIX}/searches",
                    "json": {"text": "shake hands", "top_k": 5},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if server.search_prewarm:
        threading.Thread(target=server.prewarm_search, name="search-prewarm", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
