#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset" / "motion_index.jsonl"
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

    def to_json(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "motion_id": self.motion_id,
            "source_motion": str(self.source_motion),
            "caption_file": str(self.caption_file),
            "captions": list(self.captions),
            "description": self.description,
            "object_id": self.object_id,
            "cache_glb": str(self.cache_glb),
        }


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
        ), None

    motion_paths = sorted(motion_dir.glob("*/*.npy"))
    with ThreadPoolExecutor(max_workers=16) as executor:
        for record, warning in executor.map(build_record, motion_paths):
            if record is not None:
                records.append(record)
            if warning is not None:
                warnings.append(warning)
    return records, warnings


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
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
