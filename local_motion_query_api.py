#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
import sys
import tempfile
import threading
from typing import Any


DEFAULT_LOCAL_TRANSLATION_MODEL = Path("/data5/cy/animodata/server_bundle_full/opus-mt-zh-en")
DEFAULT_LOCAL_TRANSLATION_DEVICE = -1
DEFAULT_LOCAL_TRANSLATION_MAX_LENGTH = 128
DEFAULT_ALIAS_ONLY_CJK_TEXT_LENGTH = 8
DISABLED_TRANSLATION_MODEL_VALUES = {"0", "false", "none", "null", "off", "disabled"}
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import motion_taxonomy
from search_index import SearchIndex


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
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7081
DEFAULT_FRAME_STEP = 1
DEFAULT_INTERX_FPS = 30.0
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
API_PREFIX = "/api/v1"
OPENAPI_PATH = "/openapi.json"
LOCAL_ENCODER_NAME = "local_motion_lexical_bm25_v1"
DEFAULT_SEARCH_RANDOMNESS = 0.12
DEFAULT_TRANSLATION_TIMEOUT_SECONDS = 8.0
MODEL_CONTENT_TYPE = "model/gltf-binary"

TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
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


def normalize_route_path(path: str) -> str:
    if not path:
        return "/"
    if path == "/":
        return path
    return path.rstrip("/") or "/"


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


class LocalTranslator:
    def __init__(self, *, model_name: str, device: int, max_length: int) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing local translation dependencies. Install transformers, torch, and sentencepiece."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.device >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA translation device requested but torch.cuda.is_available() is false.")
            model = model.to(f"cuda:{self.device}")
        else:
            model = model.to("cpu")

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def translate(self, text: str) -> str:
        normalized_text = normalize_whitespace(text)
        if not normalized_text:
            raise RuntimeError("Text is empty.")
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded = self._tokenizer(normalized_text, return_tensors="pt")
        if self.device >= 0:
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}

        with self._torch.no_grad():
            generated = self._model.generate(**encoded, max_length=self.max_length)

        translated_text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        if not translated_text:
            raise RuntimeError("Translation model returned an empty translation.")
        normalized_translation = normalize_whitespace(str(translated_text[0]))
        if not normalized_translation:
            raise RuntimeError("Translation model returned an empty translation.")
        return normalized_translation


def merge_search_text(translated_text: str, alias_tokens: list[str]) -> str:
    merged_parts: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokenize(translated_text):
        if token not in seen_tokens:
            merged_parts.append(token)
            seen_tokens.add(token)
    for token in alias_tokens:
        if token not in seen_tokens:
            merged_parts.append(token)
            seen_tokens.add(token)
    return normalize_whitespace(" ".join(merged_parts))


def alias_tokens_to_search_text(alias_tokens: list[str]) -> str:
    return normalize_whitespace(" ".join(token for token in alias_tokens if token.isascii()))


def should_use_alias_only_search(text: str, alias_tokens: list[str]) -> bool:
    normalized_text = normalize_whitespace(text)
    if not normalized_text or len(normalized_text) > DEFAULT_ALIAS_ONLY_CJK_TEXT_LENGTH:
        return False
    if not alias_tokens:
        return False
    return all(token.isascii() for token in alias_tokens)


@dataclass(frozen=True)
class MotionEntry:
    object_id: str
    dataset: str
    motion_id: str
    source_motion: Path
    caption_file: Path
    captions: tuple[str, ...]
    description: str
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
        self._idf = self._build_idf(entries)
        self._average_length = sum(entry.token_length for entry in entries) / max(len(entries), 1)
        search_path = search_index_path or source_path.with_suffix(".fts5.sqlite3")
        self._search_index = SearchIndex(
            search_path,
            fingerprint=SearchIndex.fingerprint(source_path, len(entries)),
            rows=((entry.object_id, " ".join(tokenize(entry.search_text))) for entry in entries),
        )
        self._sorted_entries = sorted(entries, key=lambda entry: (entry.dataset, entry.motion_id))
        self.node_entry_ids: dict[str, set[str]] = {node_id: set() for node_id in motion_taxonomy.NODES_BY_ID}
        self.node_direct_entry_ids: dict[str, set[str]] = {node_id: set() for node_id in motion_taxonomy.NODES_BY_ID}
        self.node_direct_counts: Counter[str] = Counter()
        for entry in entries:
            classification = motion_taxonomy.classify_motion(entry)
            direct_ids = {str(node["id"]) for node in classification.get("action_nodes") or ()}
            for node_id in direct_ids:
                self.node_direct_counts[node_id] += 1
                self.node_direct_entry_ids[node_id].add(entry.object_id)
                current_id: str | None = node_id
                while current_id:
                    self.node_entry_ids[current_id].add(entry.object_id)
                    current_id = motion_taxonomy.NODES_BY_ID[current_id].get("parent_id")

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
        captions = tuple(normalize_whitespace(str(caption)) for caption in row.get("captions") or ())
        captions = tuple(caption for caption in captions if caption)
        description = normalize_whitespace(str(row.get("description") or (captions[0] if captions else "")))
        object_id = normalize_whitespace(str(row.get("object_id") or ""))
        frame_count = clamp_int(row.get("frame_count"), 0, minimum=0) or None
        if not object_id and dataset and motion_id:
            object_id = hashlib.sha256(f"{dataset}:{motion_id}:{source_motion}".encode("utf-8")).hexdigest()
        if not dataset or not motion_id or not str(source_motion) or not captions or not object_id:
            return None

        cache_glb = cache_root / dataset / f"{motion_id}.glb"
        search_text = normalize_whitespace(" ".join([dataset, motion_id, description, *captions]))
        token_counts = Counter(tokenize(search_text))
        return MotionEntry(
            object_id=object_id,
            dataset=dataset,
            motion_id=motion_id,
            source_motion=source_motion,
            caption_file=caption_file,
            captions=captions,
            description=description,
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
        candidate_ids = self._search_index.candidates(query_tokens)
        scored = [
            (score, entry)
            for entry in self.entries
            if entry.object_id in candidate_ids
            if (score := self._score_entry(normalized_query, query_counts, entry)) > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[1].dataset, item[1].motion_id))
        if top_k is None:
            return scored
        return scored[: max(top_k, 1)]

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
        translation_url: str | None = None,
        translation_timeout: float = DEFAULT_TRANSLATION_TIMEOUT_SECONDS,
        translation_model: str | None = None,
        translation_device: int = DEFAULT_LOCAL_TRANSLATION_DEVICE,
        translation_max_length: int = DEFAULT_LOCAL_TRANSLATION_MAX_LENGTH,
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
        self.translation_url = normalize_whitespace(translation_url or "") or None
        normalized_translation_model = normalize_whitespace(translation_model or "")
        if translation_model is not None and not normalized_translation_model:
            self.translation_model = None
        elif normalized_translation_model.lower() in DISABLED_TRANSLATION_MODEL_VALUES:
            self.translation_model = None
        elif normalized_translation_model:
            self.translation_model = normalized_translation_model
        elif DEFAULT_LOCAL_TRANSLATION_MODEL.exists():
            self.translation_model = str(DEFAULT_LOCAL_TRANSLATION_MODEL)
        else:
            self.translation_model = None
        self.translation_device = translation_device
        self.translation_max_length = max(1, translation_max_length)
        self.translation_timeout = max(0.1, translation_timeout)
        self._translator: LocalTranslator | None = None
        self._translator_lock = threading.Lock()
        self._lock_guard = threading.Lock()
        self._conversion_locks: dict[str, threading.Lock] = {}

    def conversion_lock_for(self, object_id: str) -> threading.Lock:
        with self._lock_guard:
            lock = self._conversion_locks.get(object_id)
            if lock is None:
                lock = threading.Lock()
                self._conversion_locks[object_id] = lock
            return lock

    def local_translator(self) -> LocalTranslator | None:
        if not self.translation_model:
            return None
        with self._translator_lock:
            if self._translator is None:
                self._translator = LocalTranslator(
                    model_name=self.translation_model,
                    device=self.translation_device,
                    max_length=self.translation_max_length,
                )
            return self._translator


class MotionQueryRequestHandler(BaseHTTPRequestHandler):
    server: MotionQueryServer
    server_version = "MultimotionLazyQueryAPI/1.0"

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
        if path.startswith("/ui/static/"):
            relative_path = unquote(path[len("/ui/static/") :])
            self._send_static_file(relative_path)
            return
        if path == f"{API_PREFIX}/stats":
            self._send_json(HTTPStatus.OK, {"status": "ok", "data": self.server.motion_index.get_summary()})
            return
        if path == f"{API_PREFIX}/taxonomy":
            aggregate, direct = self.server.motion_index.taxonomy_counts()
            self._send_json(HTTPStatus.OK, {"status": "ok", "data": motion_taxonomy.taxonomy_payload(counts=aggregate, direct_counts=direct)})
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
            page, total_count, max_score = self.server.motion_index.list_entries(
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
        limit = clamp_int(first_query_value(query, "limit"), 9, minimum=1, maximum=MAX_PAGE_SIZE)
        offset = clamp_int(first_query_value(query, "offset"), 0, minimum=0)
        dataset_filter = first_query_value(query, "dataset")
        sort_key = first_query_value(query, "sort") or "relevance"
        search_query = first_query_value(query, "q") or first_query_value(query, "search")
        node_filter = first_query_value(query, "node_id") or first_query_value(query, "taxonomy_node")
        include_descendants = parse_bool(first_query_value(query, "include_descendants"), True)
        allowed_nodes: set[str] | None = None
        if node_filter:
            if node_filter not in motion_taxonomy.NODES_BY_ID:
                self._send_error_json(HTTPStatus.BAD_REQUEST, f"Unknown taxonomy node: {node_filter}")
                return
            allowed_nodes = motion_taxonomy.node_descendants(node_filter) if include_descendants else {node_filter}
        axis_filters = {axis: first_query_value(query, axis) for axis in motion_taxonomy.AXIS_ORDER}

        try:
            if search_query:
                scored = self.server.motion_index.search_entries(search_query, top_k=None)
                scored_entries = [(entry, score) for score, entry in scored]
                max_score = scored[0][0] if scored else None
            elif node_filter:
                candidate_ids = self.server.motion_index.node_entry_ids[node_filter] if include_descendants else self.server.motion_index.node_direct_entry_ids[node_filter]
                scored_entries = [(self.server.motion_index.entries_by_id[object_id], None) for object_id in candidate_ids]
                max_score = None
            else:
                scored_entries = [(entry, None) for entry in self.server.motion_index.entries]
                max_score = None
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        rows: list[tuple[MotionEntry, float | None, dict[str, Any]]] = []
        for entry, score in scored_entries:
            if dataset_filter and entry.dataset != dataset_filter.strip().lower():
                continue
            classification = motion_taxonomy.classify_motion(entry)
            taxonomy = classification["taxonomy"]
            assigned_nodes = {str(node["id"]) for node in classification.get("action_nodes") or ()}
            if allowed_nodes is not None and not assigned_nodes.intersection(allowed_nodes):
                continue
            if any(value and taxonomy.get(axis) != value for axis, value in axis_filters.items()):
                continue
            rows.append((entry, score, classification))

        facets = self._build_library_facets(rows)
        total_count = len(rows)
        rows = self._sort_library_rows(rows, sort_key)
        page = rows[offset : offset + limit]
        data = [
            self._serialize_entry(
                entry,
                include_model_info=True,
                score=score,
                max_score=max_score,
                classification=classification,
            )
            for entry, score, classification in page
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "items": data,
                "data": data,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "facets": facets,
                "hierarchy": motion_taxonomy.taxonomy_payload(
                    counts=facets.get("action_node", {}),
                    direct_counts=facets.get("action_node_direct", {}),
                ).get("hierarchy", []),
                "taxonomy_version": motion_taxonomy.taxonomy_payload().get("version"),
                "filters": {
                    "q": search_query,
                    "dataset": dataset_filter,
                    "node_id": node_filter,
                    "include_descendants": include_descendants,
                    **{axis: value for axis, value in axis_filters.items() if value},
                },
                "sort": sort_key,
                "summary": self.server.motion_index.get_summary(),
            },
        )

    def _build_library_facets(self, rows: list[tuple[MotionEntry, float | None, dict[str, Any]]]) -> dict[str, dict[str, int]]:
        facets: dict[str, Counter[str]] = {axis: Counter() for axis in motion_taxonomy.AXIS_ORDER}
        facets["dataset"] = Counter()
        facets["action_node"] = Counter()
        facets["action_node_direct"] = Counter()
        for entry, _score, classification in rows:
            facets["dataset"][entry.dataset] += 1
            for axis, value in classification["taxonomy"].items():
                facets[axis][value] += 1
            direct_nodes = {str(node["id"]) for node in classification.get("action_nodes") or ()}
            aggregate_nodes: set[str] = set()
            for node_id in direct_nodes:
                facets["action_node_direct"][node_id] += 1
                current_id: str | None = node_id
                while current_id:
                    aggregate_nodes.add(current_id)
                    current_id = motion_taxonomy.NODES_BY_ID[current_id].get("parent_id")
            for node_id in aggregate_nodes:
                facets["action_node"][node_id] += 1
        return {axis: dict(counter) for axis, counter in facets.items()}

    @staticmethod
    def _sort_library_rows(
        rows: list[tuple[MotionEntry, float | None, dict[str, Any]]],
        sort_key: str,
    ) -> list[tuple[MotionEntry, float | None, dict[str, Any]]]:
        if sort_key == "motion_id":
            return sorted(rows, key=lambda item: (item[0].dataset, item[0].motion_id))
        if sort_key == "frame_count_desc":
            return sorted(rows, key=lambda item: (-(item[0].frame_count or 0), item[0].dataset, item[0].motion_id))
        if any(score is not None for _entry, score, _classification in rows):
            return sorted(rows, key=lambda item: (-(item[1] or 0.0), item[0].dataset, item[0].motion_id))
        return sorted(rows, key=lambda item: (item[0].dataset, item[0].motion_id))

    def _handle_entry_detail(self, object_id: str) -> None:
        entry = self.server.motion_index.get_entry(object_id)
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

        search_text, translation_info = self._prepare_search_text(text, warnings)
        top_k = clamp_int(payload.get("top_k", 5), 5, minimum=1, maximum=MAX_PAGE_SIZE)
        candidate_k = clamp_int(payload.get("candidate_k", 200), 200, minimum=1)
        candidate_k = max(candidate_k, top_k)
        include_model_info = parse_bool(payload.get("include_model_info"), True)
        randomness = clamp_float(payload.get("randomness", DEFAULT_SEARCH_RANDOMNESS), DEFAULT_SEARCH_RANDOMNESS, maximum=1.0)
        random_seed = payload.get("random_seed")

        try:
            candidates = self.server.motion_index.search_entries(search_text, top_k=candidate_k)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        max_score = candidates[0][0] if candidates else None
        results = self._rerank_results(candidates, top_k=top_k, randomness=randomness, random_seed=random_seed)
        data = [
            self._serialize_entry(entry, include_model_info=include_model_info, score=score, max_score=max_score)
            for score, entry in results
        ]

        common = {
            "query_mode": "text",
            "requested_encoder": payload.get("encoder"),
            "selected_encoder": LOCAL_ENCODER_NAME,
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
            "translation": translation_info,
            "data": data,
        }
        if legacy_mode:
            response = {
                "status": "completed",
                **common,
                "index": {
                    "encoder_name": LOCAL_ENCODER_NAME,
                    "db_path": str(self.server.motion_index.source_path),
                    "source_type": "motion_index_jsonl",
                    "dataset_root": str(self.server.motion_index.cache_root),
                    "feature_dim": None,
                    "supports_text": True,
                    "lazy_conversion": True,
                },
                "meta": {"result_count": len(data), "entry_count": len(self.server.motion_index.entries)},
                "links": {
                    "self": self._absolute_url("/query"),
                    "rest_search": self._absolute_url(f"{API_PREFIX}/searches"),
                },
            }
        else:
            response = {
                "status": "ok",
                "query": {
                    "mode": "text",
                    "text": text,
                    "search_text": search_text,
                    "translation": translation_info,
                    "top_k": top_k,
                    "candidate_k": candidate_k,
                    "requested_encoder": payload.get("encoder"),
                    "selected_encoder": LOCAL_ENCODER_NAME,
                    "candidate_count": len(candidates),
                    "randomness": randomness,
                    "random_seed": random_seed,
                },
                "data": data,
                "results": data,
                "meta": {
                    "result_count": len(data),
                    "entry_count": len(self.server.motion_index.entries),
                    "supports_text": True,
                    "lazy_conversion": True,
                },
                "warnings": warnings,
                "links": {"self": self._absolute_url(f"{API_PREFIX}/searches")},
            }
        self._send_json(HTTPStatus.OK, response)

    def _prepare_search_text(self, text: str, warnings: list[str]) -> tuple[str, dict[str, Any] | None]:
        if not contains_cjk(text):
            return text, None
        alias_tokens = tokenize(text)
        translation_info: dict[str, Any] = {
            "status": "skipped",
            "provider": None,
            "translation_url": self.server.translation_url,
            "translation_model": self.server.translation_model,
            "original_text": text,
            "translated_text": None,
            "used_for_search": False,
            "source_alias_tokens": alias_tokens,
        }
        alias_search_text = alias_tokens_to_search_text(alias_tokens)
        if should_use_alias_only_search(text, alias_tokens):
            translation_info["status"] = "alias_only"
            translation_info["provider"] = "builtin_alias"
            translation_info["used_for_search"] = True
            translation_info["augmented_search_text"] = alias_search_text
            return alias_search_text, translation_info
        try:
            translated_text, provider = self._translate_query_text(text)
        except Exception as exc:
            translation_info["status"] = "failed"
            translation_info["message"] = str(exc)
            if alias_search_text:
                warnings.append(f"Translation failed: {exc}; using local aliases only.")
                translation_info["provider"] = "builtin_alias"
                translation_info["used_for_search"] = True
                translation_info["augmented_search_text"] = alias_search_text
                translation_info["fallback_provider"] = "builtin_alias"
                return alias_search_text, translation_info
            warnings.append(f"Translation failed: {exc}; using original query text.")
            return text, translation_info
        if not translated_text:
            if alias_search_text:
                warnings.append("Translation is unavailable; using local aliases only.")
                translation_info["status"] = "alias_only"
                translation_info["provider"] = "builtin_alias"
                translation_info["used_for_search"] = True
                translation_info["augmented_search_text"] = alias_search_text
                return alias_search_text, translation_info
            warnings.append("Translation is unavailable; using original query text.")
            return text, translation_info
        augmented = merge_search_text(translated_text, alias_tokens)
        translation_info["status"] = "ok"
        translation_info["provider"] = provider
        translation_info["translated_text"] = translated_text
        translation_info["used_for_search"] = True
        translation_info["augmented_search_text"] = augmented
        return augmented, translation_info

    def _translate_query_text(self, text: str) -> tuple[str | None, str | None]:
        errors: list[str] = []
        translator = self.server.local_translator()
        if translator is not None:
            try:
                translated_text = translator.translate(text)
                return translated_text, "local_transformers"
            except Exception as exc:
                errors.append(f"local_transformers: {exc}")
        if self.server.translation_url:
            try:
                translated_text = self._translate_query_text_external(text)
                return translated_text, "external_http"
            except Exception as exc:
                errors.append(f"external_http: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return None, None

    def _translate_query_text_external(self, text: str) -> str:
        request_payload = {"text": text, "source": "zh", "target": "en"}
        request = Request(
            self.server.translation_url,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.server.translation_timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("Translation service response must be a JSON object.")
        translated_text = response_payload.get("translated_text")
        if not isinstance(translated_text, str):
            data = response_payload.get("data")
            if isinstance(data, dict):
                translated_text = data.get("translated_text")
        if not isinstance(translated_text, str):
            raise RuntimeError("Translation service response does not contain 'translated_text'.")
        return translated_text

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
    ) -> dict[str, Any]:
        legacy_download_path = f"/models/{entry.object_id}"
        rest_download_path = f"{API_PREFIX}/models/{entry.object_id}"
        preview_path = self._preview_path_for(entry)
        preview_available = preview_path.is_file() or preview_path.with_suffix(".png").is_file()
        preview_download_path = f"{API_PREFIX}/previews/{entry.object_id}.webp"
        payload: dict[str, Any] = {
            "object_id": entry.object_id,
            "dataset": entry.dataset,
            "motion_id": entry.motion_id,
            "description": entry.description,
            "captions": list(entry.captions),
            "frame_count": entry.frame_count,
            "source_motion": str(entry.source_motion),
            "caption_file": str(entry.caption_file),
            "preview": {
                "available": preview_available,
                "download_path": preview_download_path,
                "download_url": self._absolute_url(preview_download_path),
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
        classification = classification or motion_taxonomy.classify_motion(entry)
        payload.update(classification)
        if score is not None:
            payload["score"] = round(score, 6)
            payload["pooled_score"] = round(score / max_score, 6) if max_score else 0.0
            payload["best_view_id"] = None
            payload["best_view_path"] = None
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

    def _handle_preview_download(self, preview_id: str) -> None:
        suffix = ".webp" if preview_id.endswith(".webp") else ".png" if preview_id.endswith(".png") else ""
        object_id = preview_id[: -len(suffix)] if suffix else preview_id
        entry = self.server.motion_index.get_entry(object_id)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown object_id: {object_id}")
            return
        preview_path = self._preview_path_for(entry)
        if not preview_path.is_file():
            legacy_path = preview_path.with_suffix(".png")
            if legacy_path.is_file():
                preview_path = legacy_path
        if not preview_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Preview is not available: {object_id}")
            return
        data = preview_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "image/webp" if preview_path.suffix == ".webp" else "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_model_download(self, object_id: str) -> None:
        entry = self.server.motion_index.get_entry(object_id)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown object_id: {object_id}")
            return
        with self.server.conversion_lock_for(entry.object_id):
            job_dir: Path | None = None
            response_started = False
            try:
                model_path, job_dir = create_temporary_glb(
                    entry,
                    model_dir=self.server.model_dir,
                    frame_step=self.server.frame_step,
                    interx_fps=self.server.interx_fps,
                    converter_env=self.server.converter_env,
                    conda_exe=self.server.conda_exe,
                    temp_root=self.server.temp_root,
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
        summary = self.server.motion_index.get_summary()
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
        summary = self.server.motion_index.get_summary()
        return {
            "status": "ok",
            "entry_count": summary["entry_count"],
            "dataset_counts": summary["dataset_counts"],
            "index_path": summary["index_path"],
            "preview_root": summary["preview_root"],
            "lazy_conversion": True,
            "translation": {
                "enabled": bool(self.server.translation_model or self.server.translation_url),
                "url": self.server.translation_url,
                "model": self.server.translation_model,
                "device": self.server.translation_device,
                "timeout_seconds": self.server.translation_timeout,
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
        self.wfile.write(data)

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
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
            bind_host, bind_port = self.server.server_address
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

    def _send_json(self, status_code: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    parser.add_argument("--translation-url", default=None, help="Optional external Chinese-to-English translation endpoint.")
    parser.add_argument(
        "--translation-model",
        default=str(DEFAULT_LOCAL_TRANSLATION_MODEL) if DEFAULT_LOCAL_TRANSLATION_MODEL.exists() else None,
        help="Optional local Chinese-to-English model path or Hugging Face model name used directly inside the API process. Use none/off to disable.",
    )
    parser.add_argument(
        "--translation-device",
        type=int,
        default=DEFAULT_LOCAL_TRANSLATION_DEVICE,
        help="Transformers device for in-process translation. Use -1 for CPU, 0 for the first CUDA GPU.",
    )
    parser.add_argument(
        "--translation-max-length",
        type=int,
        default=DEFAULT_LOCAL_TRANSLATION_MAX_LENGTH,
        help=f"Maximum generated translation length. Default: {DEFAULT_LOCAL_TRANSLATION_MAX_LENGTH}.",
    )
    parser.add_argument(
        "--translation-timeout",
        type=float,
        default=DEFAULT_TRANSLATION_TIMEOUT_SECONDS,
        help=f"Translation request timeout in seconds. Default: {DEFAULT_TRANSLATION_TIMEOUT_SECONDS}.",
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
        translation_url=args.translation_url,
        translation_timeout=args.translation_timeout,
        translation_model=args.translation_model,
        translation_device=args.translation_device,
        translation_max_length=args.translation_max_length,
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
                "translation": {
                    "external_url": args.translation_url,
                    "local_model": args.translation_model,
                    "device": args.translation_device,
                },
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
