#!/usr/bin/env python3
"""Pre-cache a bounded, high-value set of Motion-X++ GLBs through the API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
DEFAULT_INDEX = HERE / "dataset" / "motion_index.jsonl"
DEFAULT_ASSIGNMENTS = HERE / "dataset" / "motion_taxonomy_assignments.jsonl"
DEFAULT_CACHE = HERE / "cache" / "models" / "motionxpp"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Motion-X++ GLB pre-cache job.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7091")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=100, help="Target total cached Motion-X++ count.")
    parser.add_argument("--reserve-gib", type=float, default=20.0, help="Stop before this much free space remains.")
    return parser.parse_args()


def load_assignments(path: Path) -> dict[str, dict]:
    assignments = {}
    if not path.is_file():
        return assignments
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("object_id") and isinstance(row.get("classification"), dict):
                assignments[row["object_id"]] = row["classification"]
    return assignments


def main() -> int:
    args = arguments()
    assignments = load_assignments(args.assignments)
    entries = []
    for line in args.index.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("dataset") != "motionxpp":
            continue
        output = args.cache_dir / f"{row['motion_id']}.glb"
        if output.is_file():
            continue
        classification = assignments.get(row.get("object_id"), {})
        nodes = classification.get("action_nodes") or []
        score = sum(1.0 + float(node.get("role") == "primary") for node in nodes)
        entries.append((score, row))
    entries.sort(key=lambda item: (-item[0], item[1].get("motion_id", "")))
    cached_before = len(list(args.cache_dir.rglob("*.glb"))) if args.cache_dir.is_dir() else 0
    needed = max(0, args.limit - cached_before)
    selected = [row for _score, row in entries[:needed]]
    reserve = args.reserve_gib * 1024**3
    result = {"target": args.limit, "cached_before": cached_before, "selected": len(selected), "generated": 0, "failed": 0, "failures": []}
    for index, row in enumerate(selected, 1):
        free = shutil.disk_usage(args.cache_dir.parent if args.cache_dir.parent.exists() else HERE).free
        if free < reserve + 10 * 1024**2:
            result["stopped_for_disk"] = True
            break
        started = time.monotonic()
        try:
            with urlopen(f"{args.base_url.rstrip('/')}/api/v1/models/{row['object_id']}", timeout=300) as response:
                while response.read(1024 * 1024):
                    pass
            result["generated"] += 1
            print(f"{index}/{len(selected)} ok {row['motion_id']} ({time.monotonic() - started:.1f}s)", flush=True)
        except Exception as exc:
            result["failed"] += 1
            if len(result["failures"]) < 20:
                result["failures"].append({"motion_id": row["motion_id"], "error": str(exc)})
            print(f"{index}/{len(selected)} failed {row['motion_id']}: {exc}", flush=True)
    result["cached_after"] = len(list(args.cache_dir.rglob("*.glb"))) if args.cache_dir.is_dir() else cached_before
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
