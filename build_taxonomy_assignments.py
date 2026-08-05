#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Iterable

import motion_taxonomy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = SCRIPT_DIR / "dataset" / "motion_index.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset" / "motion_taxonomy_assignments.jsonl"
DEFAULT_REPORT = SCRIPT_DIR / "dataset" / "motion_taxonomy_report.json"


@dataclass
class EntryView:
    object_id: str
    dataset: str
    motion_id: str
    description: str
    captions: tuple[str, ...]


def iter_entries(path: Path) -> Iterable[EntryView]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            object_id = str(row.get("object_id") or "").strip()
            if not object_id:
                continue
            yield EntryView(
                object_id=object_id,
                dataset=str(row.get("dataset") or "").strip(),
                motion_id=str(row.get("motion_id") or "").strip(),
                description=str(row.get("description") or "").strip(),
                captions=tuple(str(value).strip() for value in row.get("captions") or () if str(value).strip()),
            )


def atomic_jsonl_write(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build(index_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    node_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    unmatched: list[dict[str, str]] = []
    total = 0

    def rows() -> Iterable[dict[str, object]]:
        nonlocal total
        for entry in iter_entries(index_path):
            total += 1
            dataset_counts[entry.dataset] += 1
            classification = motion_taxonomy.classify_motion(entry, use_cached=False)
            primary_id = classification.get("primary_action_node_id")
            if primary_id:
                primary_counts[str(primary_id)] += 1
            else:
                unmatched.append({"object_id": entry.object_id, "dataset": entry.dataset, "motion_id": entry.motion_id, "description": entry.description})
            for action_node in classification.get("action_nodes") or ():
                node_counts[str(action_node["id"])] += 1
            legacy_counts[str(classification["taxonomy"]["action_tag"])] += 1
            yield {"object_id": entry.object_id, "classification": classification}

    atomic_jsonl_write(output_path, rows())
    report = {
        "status": "ok",
        "taxonomy_version": motion_taxonomy.taxonomy_payload()["version"],
        "index_path": str(index_path),
        "output_path": str(output_path),
        "entry_count": total,
        "classified_count": total - len(unmatched),
        "unclassified_count": len(unmatched),
        "unclassified_rate": round(len(unmatched) / max(total, 1), 6),
        "datasets": dict(dataset_counts),
        "primary_node_counts": dict(primary_counts.most_common()),
        "all_node_counts": dict(node_counts.most_common()),
        "legacy_tag_counts": dict(legacy_counts.most_common()),
        "unclassified_examples": unmatched[:200],
        "catalog_validation_errors": motion_taxonomy.validate_catalog(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic taxonomy assignments for the motion index.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build(args.index.expanduser().resolve(), args.output.expanduser().resolve(), args.report.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["catalog_validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
