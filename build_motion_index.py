#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import struct
import tempfile
from typing import Iterable

import numpy as np

from search_index import (
    QWEN3_INDEX_SCHEMA_VERSION,
    SemanticRetriever,
    clean_motion_title,
    clean_search_caption,
    is_searchable_caption,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset" / "motion_index.jsonl"
DEFAULT_SEMANTIC_OUTPUT = SCRIPT_DIR / "dataset" / "semantic_index_qwen3_06b"
DEFAULT_CACHE_ROOT = Path("cache") / "models"
DEFAULT_INTERHUMAN_ROOT = Path("/mnt/nas/cy/humanmotion/interhuman")
DEFAULT_INTERX_ROOT = Path("/mnt/nas/cy/humanmotion/interx")
DEFAULT_MOTIONXPP_ROOT = Path("/mnt/nas/cy/humanmotion/motionx++")


@dataclass(frozen=True)
class MotionIndexRecord:
    dataset: str
    motion_id: str
    source_motion: Path
    caption_file: Path
    captions: tuple[str, ...]
    description: str
    object_id: str
    cache_glb: Path
    frame_count: int | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "motion_id": self.motion_id,
            "source_motion": str(self.source_motion),
            "caption_file": str(self.caption_file),
            "captions": list(self.captions),
            "description": self.description,
            "object_id": self.object_id,
            "cache_glb": str(self.cache_glb),
        }
        if self.frame_count is not None:
            payload["frame_count"] = self.frame_count
        return payload


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def read_captions(path: Path) -> tuple[str, ...]:
    captions: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            caption = normalize_whitespace(line)
            if caption:
                captions.append(caption)
    return tuple(captions)


def build_object_id(dataset: str, motion_id: str, source_motion: Path) -> str:
    payload = f"{dataset}:{motion_id}:{source_motion}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_npy_frame_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
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
        shape = header.get("shape") if isinstance(header, dict) else None
        return int(shape[0]) if isinstance(shape, tuple) and shape and int(shape[0]) > 0 else None
    except (OSError, UnicodeDecodeError, ValueError, SyntaxError, struct.error):
        return None


def read_interhuman_frame_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        count = int(payload.get("frames") or 0) if isinstance(payload, dict) else 0
        return count or None
    except (OSError, ValueError, TypeError, pickle.UnpicklingError):
        return None


def read_interx_frame_count(sequence_dir: Path) -> int | None:
    try:
        with np.load(sequence_dir / "P1.npz", allow_pickle=False) as payload:
            shape = payload["root_orient"].shape
        return int(shape[0]) if shape and int(shape[0]) > 0 else None
    except (OSError, ValueError, KeyError):
        return None


def interhuman_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def build_interhuman_records(root: Path, cache_root: Path) -> tuple[list[MotionIndexRecord], list[str]]:
    motion_dir = root / "motions"
    caption_dir = root / "annots"
    records: list[MotionIndexRecord] = []
    warnings: list[str] = []

    for motion_path in sorted(motion_dir.glob("*.pkl"), key=interhuman_sort_key):
        motion_id = motion_path.stem
        caption_file = caption_dir / f"{motion_id}.txt"
        if not caption_file.is_file():
            warnings.append(f"interhuman/{motion_id}: missing caption file {caption_file}")
            continue
        captions = read_captions(caption_file)
        if not captions:
            warnings.append(f"interhuman/{motion_id}: empty caption file {caption_file}")
            continue
        object_id = build_object_id("interhuman", motion_id, motion_path)
        records.append(
            MotionIndexRecord(
                dataset="interhuman",
                motion_id=motion_id,
                source_motion=motion_path.resolve(),
                caption_file=caption_file.resolve(),
                captions=captions,
                description=captions[0],
                object_id=object_id,
                cache_glb=cache_root / "interhuman" / f"{motion_id}.glb",
                frame_count=read_interhuman_frame_count(motion_path),
            )
        )
    return records, warnings


def build_interx_records(root: Path, cache_root: Path) -> tuple[list[MotionIndexRecord], list[str]]:
    motion_dir = root / "motions"
    caption_dir = root / "texts"
    records: list[MotionIndexRecord] = []
    warnings: list[str] = []

    for sequence_dir in sorted(path for path in motion_dir.iterdir() if path.is_dir()):
        motion_id = sequence_dir.name
        if not (sequence_dir / "P1.npz").is_file() or not (sequence_dir / "P2.npz").is_file():
            warnings.append(f"interx/{motion_id}: missing P1.npz or P2.npz")
            continue
        caption_file = caption_dir / f"{motion_id}.txt"
        if not caption_file.is_file():
            warnings.append(f"interx/{motion_id}: missing caption file {caption_file}")
            continue
        captions = read_captions(caption_file)
        if not captions:
            warnings.append(f"interx/{motion_id}: empty caption file {caption_file}")
            continue
        object_id = build_object_id("interx", motion_id, sequence_dir)
        records.append(
            MotionIndexRecord(
                dataset="interx",
                motion_id=motion_id,
                source_motion=sequence_dir.resolve(),
                caption_file=caption_file.resolve(),
                captions=captions,
                description=captions[0],
                object_id=object_id,
                cache_glb=cache_root / "interx" / f"{motion_id}.glb",
                frame_count=read_interx_frame_count(sequence_dir),
            )
        )
    return records, warnings


def build_motionxpp_records(root: Path, cache_root: Path) -> tuple[list[MotionIndexRecord], list[str]]:
    motion_dir = root / "smplx322"
    caption_dir = root / "semantic_label"
    records: list[MotionIndexRecord] = []
    warnings: list[str] = []

    def build_record(motion_path: Path) -> tuple[MotionIndexRecord | None, str | None]:
        subset = motion_path.parent.name
        motion_id = f"{subset}/{motion_path.stem}"
        caption_file = caption_dir / subset / f"{motion_path.stem}.txt"
        if not caption_file.is_file():
            return None, f"motionxpp/{motion_id}: missing caption file {caption_file}"
        captions = read_captions(caption_file)
        if not captions:
            return None, f"motionxpp/{motion_id}: empty caption file {caption_file}"
        return MotionIndexRecord(
            dataset="motionxpp",
            motion_id=motion_id,
            source_motion=motion_path.absolute(),
            caption_file=caption_file.absolute(),
            captions=captions,
            description=captions[0],
            object_id=build_object_id("motionxpp", motion_id, motion_path),
            cache_glb=cache_root / "motionxpp" / f"{motion_id}.glb",
            frame_count=read_npy_frame_count(motion_path),
        ), None

    motion_paths = sorted(motion_dir.glob("*/*.npy"))
    with ThreadPoolExecutor(max_workers=16) as executor:
        for record, warning in executor.map(build_record, motion_paths):
            if record is not None:
                records.append(record)
            if warning is not None:
                warnings.append(warning)
    return records, warnings


def semantic_source_rows(index_path: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            object_id = str(item.get("object_id") or "").strip()
            dataset = str(item.get("dataset") or "")
            motion_id = str(item.get("motion_id") or "")
            source_title = clean_motion_title(dataset, motion_id)
            if object_id and source_title:
                rows.append(
                    {
                        "object_id": object_id,
                        "caption": source_title,
                        "caption_rank": -1,
                        "dataset": dataset,
                        "source_type": "source_title",
                    }
                )
            captions = item.get("captions") or [item.get("description") or ""]
            seen: set[str] = set()
            for rank, caption in enumerate(captions):
                text = clean_search_caption(str(caption))
                if not is_searchable_caption(text):
                    continue
                if object_id and text and text not in seen:
                    rows.append(
                        {
                            "object_id": object_id,
                            "caption": text,
                            "caption_rank": rank,
                            "dataset": dataset,
                            "source_type": "caption",
                        }
                    )
                    seen.add(text)
    return rows


def build_semantic_index(
    index_path: Path,
    output_path: Path,
    model: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, object]:
    rows = semantic_source_rows(index_path)
    if not rows:
        raise RuntimeError(f"motion index has no captions: {index_path}")
    retriever = SemanticRetriever(output_path, model, device, max_length, load_index=False)
    vectors: list[np.ndarray] = []
    for start in range(0, len(rows), max(batch_size, 1)):
        batch = rows[start : start + max(batch_size, 1)]
        vectors.append(retriever.encode_documents([str(row["caption"]) for row in batch]))
        print(f"encoded {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    output = output_path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    embeddings = np.concatenate(vectors).astype(np.float16)
    model_config_path = Path(model).expanduser() / "config.json"
    model_config_sha256 = (
        hashlib.sha256(model_config_path.read_bytes()).hexdigest()
        if model_config_path.is_file()
        else None
    )
    model_revision = getattr(retriever._model.config, "_commit_hash", None)
    manifest: dict[str, object] = {
        "model": model,
        "model_revision": model_revision,
        "model_config_sha256": model_config_sha256,
        "encoder_family": retriever.encoder_profile.family,
        "dimension": int(embeddings.shape[1]),
        "count": len(rows),
        "dtype": "float16",
        "normalized": True,
        "pooling": retriever.encoder_profile.pooling,
        "query_instruction": retriever.encoder_profile.query_instruction,
        "short_cjk_template": retriever.encoder_profile.short_cjk_template,
        "document_instruction": "",
        "padding_side": retriever.encoder_profile.padding_side,
        "max_length": max_length,
        "schema_version": QWEN3_INDEX_SCHEMA_VERSION,
        "source_title_count": sum(row.get("source_type") == "source_title" for row in rows),
        "source_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    with tempfile.TemporaryDirectory(dir=output) as temp_dir:
        temp = Path(temp_dir)
        np.save(temp / "embeddings.npy", embeddings)
        (temp / "captions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for name in ("embeddings.npy", "captions.jsonl", "manifest.json"):
            os.replace(temp / name, output / name)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a caption-only lazy retrieval index for multimotion.")
    parser.add_argument("--interhuman-root", type=Path, default=DEFAULT_INTERHUMAN_ROOT)
    parser.add_argument("--interx-root", type=Path, default=DEFAULT_INTERX_ROOT)
    parser.add_argument("--motionxpp-root", type=Path, default=DEFAULT_MOTIONXPP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Cache root written into index records, relative to bundle root by default.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "interhuman", "interx", "motionxpp"),
        default="all",
        help="Dataset subset to index.",
    )
    parser.add_argument("--max-warnings", type=int, default=20)
    parser.add_argument("--semantic-model", help="Also build a dense caption index with this local model.")
    parser.add_argument("--semantic-output", type=Path, default=DEFAULT_SEMANTIC_OUTPUT)
    parser.add_argument("--semantic-device", default="auto")
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Build only the dense index from the existing --output motion index.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.semantic_only:
        if not args.semantic_model:
            raise SystemExit("--semantic-only requires --semantic-model")
        manifest = build_semantic_index(
            args.output.expanduser().resolve(),
            args.semantic_output,
            args.semantic_model,
            args.semantic_device,
            args.semantic_batch_size,
            args.semantic_max_length,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    records: list[MotionIndexRecord] = []
    warnings: list[str] = []

    if args.dataset in {"all", "interhuman"}:
        if args.interhuman_root.is_dir():
            dataset_records, dataset_warnings = build_interhuman_records(args.interhuman_root, args.cache_root)
            records.extend(dataset_records)
            warnings.extend(dataset_warnings)
        else:
            warnings.append(f"interhuman root does not exist: {args.interhuman_root}")

    if args.dataset in {"all", "interx"}:
        if args.interx_root.is_dir():
            dataset_records, dataset_warnings = build_interx_records(args.interx_root, args.cache_root)
            records.extend(dataset_records)
            warnings.extend(dataset_warnings)
        else:
            warnings.append(f"interx root does not exist: {args.interx_root}")

    if args.dataset in {"all", "motionxpp"}:
        if args.motionxpp_root.is_dir():
            dataset_records, dataset_warnings = build_motionxpp_records(args.motionxpp_root, args.cache_root)
            records.extend(dataset_records)
            warnings.extend(dataset_warnings)
        else:
            warnings.append(f"motionxpp root does not exist: {args.motionxpp_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "status": "ok" if records else "empty",
        "output": str(args.output),
        "entry_count": len(records),
        "datasets": {
            "interhuman": sum(1 for record in records if record.dataset == "interhuman"),
            "interx": sum(1 for record in records if record.dataset == "interx"),
            "motionxpp": sum(1 for record in records if record.dataset == "motionxpp"),
        },
        "warning_count": len(warnings),
        "warnings": warnings[: max(args.max_warnings, 0)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not records:
        return 1
    if args.semantic_model:
        manifest = build_semantic_index(
            args.output.expanduser().resolve(),
            args.semantic_output,
            args.semantic_model,
            args.semantic_device,
            args.semantic_batch_size,
            args.semantic_max_length,
        )
        print(json.dumps({"semantic_index": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
