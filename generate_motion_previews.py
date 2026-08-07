#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
from urllib.parse import quote

from PIL import Image

import local_motion_query_api

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("/mnt/nas/cy/humanmotion/multimotion_previews")
PREVIEW_SAMPLE_SIZE = (32, 20)
MIN_PREVIEW_CHANNEL_RANGE = 12
MIN_PREVIEW_FOREGROUND_DELTA = 18
MIN_PREVIEW_FOREGROUND_RATIO = 0.02
MAX_AUDIT_SAMPLES = 50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate low-res static WebP previews without retaining GLB files.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7091")
    parser.add_argument("--index", type=Path, default=local_motion_query_api.DEFAULT_INDEX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=local_motion_query_api.DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=local_motion_query_api.DEFAULT_MODEL_DIR)
    parser.add_argument("--converter-env", default=local_motion_query_api.DEFAULT_CONVERTER_ENV)
    parser.add_argument("--conda-exe", default=local_motion_query_api.DEFAULT_CONDA_EXE)
    parser.add_argument("--limit", type=int, default=0, help="Maximum previews to generate; 0 means no limit.")
    parser.add_argument("--dataset", choices=("all", "interhuman", "interx", "motionxpp"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--quality", type=int, default=80, help="WebP quality from 1 to 100.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent PNG migration workers.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split generation into this many deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index handled by this process.")
    parser.add_argument("--migrate-only", action="store_true", help="Convert existing PNG files to WebP and stop.")
    parser.add_argument("--skip-migration", action="store_true", help="Skip the legacy PNG migration pass.")
    parser.add_argument("--object-id", dest="object_ids", action="append", default=[], help="Generate only this object ID. Repeat for multiple IDs.")
    parser.add_argument("--audit-existing", action="store_true", help="Decode and validate existing previews without generating files.")
    parser.add_argument("--repair-invalid", action="store_true", help="Audit previews and regenerate only missing or invalid files.")
    parser.add_argument("--port", type=int, default=7192, help="Temporary local renderer API port when --base-url is not reachable.")
    return parser


@contextmanager
def temporary_server(index: local_motion_query_api.MotionIndex, args: argparse.Namespace):
    server = local_motion_query_api.MotionQueryServer(
        ("127.0.0.1", args.port),
        local_motion_query_api.MotionQueryRequestHandler,
        index,
        model_dir=args.model_dir.resolve(),
        frame_step=local_motion_query_api.DEFAULT_FRAME_STEP,
        interx_fps=local_motion_query_api.DEFAULT_INTERX_FPS,
        converter_env=local_motion_query_api.normalize_whitespace(args.converter_env) or None,
        conda_exe=args.conda_exe,
        translation_model="none",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def preview_path(output_root: Path, entry: local_motion_query_api.MotionEntry) -> Path:
    return output_root / entry.dataset / f"{entry.motion_id}.webp"


def validate_preview_file(
    path: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "WEBP":
                return "invalid_format"
            if expected_width is not None and image.width != expected_width:
                return "wrong_dimensions"
            if expected_height is not None and image.height != expected_height:
                return "wrong_dimensions"
            sample = image.convert("RGB").resize(PREVIEW_SAMPLE_SIZE)
            channel_range = max(high - low for low, high in sample.getextrema())
            pixels = list(sample.getdata())
            corner_pixels = [pixels[0], pixels[PREVIEW_SAMPLE_SIZE[0] - 1], pixels[-PREVIEW_SAMPLE_SIZE[0]], pixels[-1]]
            background = tuple(sorted(pixel[channel] for pixel in corner_pixels)[len(corner_pixels) // 2] for channel in range(3))
            foreground_ratio = sum(
                1 for pixel in pixels
                if max(abs(pixel[channel] - background[channel]) for channel in range(3)) >= MIN_PREVIEW_FOREGROUND_DELTA
            ) / len(pixels)
            if channel_range < MIN_PREVIEW_CHANNEL_RANGE or foreground_ratio < MIN_PREVIEW_FOREGROUND_RATIO:
                return "blank"
    except Exception:
        return "corrupt"
    return None


def quick_preview_issue(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        if path.stat().st_size < 16:
            return "empty"
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return "unreadable"
    if header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        return "invalid_format"
    return None


def save_webp_atomically(
    image: Image.Image,
    output_path: Path,
    *,
    quality: int,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".webp", delete=False) as handle:
        temporary_webp = Path(handle.name)
    try:
        image.convert("RGB").save(temporary_webp, "WEBP", quality=quality, method=6)
        issue = validate_preview_file(
            temporary_webp,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        if issue:
            raise RuntimeError(f"Generated preview failed validation: {issue}")
        os.replace(temporary_webp, output_path)
    finally:
        temporary_webp.unlink(missing_ok=True)


def audit_previews(
    output_root: Path,
    entries: list[local_motion_query_api.MotionEntry],
    *,
    width: int,
    height: int,
    workers: int,
) -> tuple[dict[str, Any], list[local_motion_query_api.MotionEntry]]:
    def inspect(entry: local_motion_query_api.MotionEntry) -> tuple[local_motion_query_api.MotionEntry, str | None]:
        return entry, validate_preview_file(preview_path(output_root, entry), expected_width=width, expected_height=height)

    counts: dict[str, int] = {"valid": 0, "missing": 0, "empty": 0, "invalid_format": 0, "wrong_dimensions": 0, "blank": 0, "corrupt": 0}
    invalid_entries: list[local_motion_query_api.MotionEntry] = []
    samples: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for entry, issue in pool.map(inspect, entries, chunksize=64):
            if issue is None:
                counts["valid"] += 1
                continue
            counts[issue] = counts.get(issue, 0) + 1
            invalid_entries.append(entry)
            if len(samples) < MAX_AUDIT_SAMPLES:
                samples.append({"object_id": entry.object_id, "dataset": entry.dataset, "motion_id": entry.motion_id, "issue": issue})
    return {
        "checked": len(entries),
        "valid": counts.pop("valid"),
        "invalid": len(invalid_entries),
        "issues": counts,
        "samples": samples,
    }, invalid_entries


def migrate_legacy_pngs(
    output_root: Path,
    entries: list[local_motion_query_api.MotionEntry],
    *,
    dataset: str = "all",
    workers: int = 8,
    quality: int = 80,
) -> dict[str, int]:
    candidates = [
        output_root / entry.dataset / f"{entry.motion_id}.png"
        for entry in entries
        if dataset == "all" or entry.dataset == dataset
    ]

    def migrate_one(png_path: Path) -> tuple[int, int, str | None]:
        webp_path = png_path.with_suffix(".webp")
        try:
            if not png_path.is_file():
                return 0, 0, None
            if quick_preview_issue(webp_path):
                with Image.open(png_path) as image:
                    save_webp_atomically(image, webp_path, quality=quality)
                converted = 1
            else:
                converted = 0
            png_path.unlink()
            return converted, 1, None
        except Exception as exc:
            return 0, 0, str(exc)

    summary = {"converted": 0, "removed": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in as_completed([pool.submit(migrate_one, path) for path in candidates]):
            converted, removed, error = future.result()
            summary["converted"] += converted
            summary["removed"] += removed
            if error:
                summary["failed"] += 1
    return summary


def candidate_entries(index: local_motion_query_api.MotionIndex, args: argparse.Namespace) -> list[local_motion_query_api.MotionEntry]:
    eligible = [entry for entry in index.entries if args.dataset == "all" or entry.dataset == args.dataset]
    if args.object_ids:
        selected = set(args.object_ids)
        eligible = [entry for entry in eligible if entry.object_id in selected]
    eligible = eligible[args.shard_index :: args.shard_count]
    if args.overwrite:
        return eligible[: args.limit or None]

    def is_missing(entry: local_motion_query_api.MotionEntry) -> bool:
        path = preview_path(args.output_root, entry)
        if args.object_ids:
            return validate_preview_file(path, expected_width=args.width, expected_height=args.height) is not None
        return quick_preview_issue(path) is not None

    entries = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for entry, missing in zip(eligible, pool.map(is_missing, eligible)):
            if missing:
                entries.append(entry)
                if args.limit and len(entries) >= args.limit:
                    break
    return entries


def render_preview(page: Any, base_url: str, entry: local_motion_query_api.MotionEntry, output_path: Path, width: int, height: int, quality: int) -> None:
    model_url = f"{base_url}/api/v1/models/{quote(entry.object_id)}"
    render_url = f"{base_url}/ui/static/preview_renderer.html?model={quote(model_url, safe=':/?&=%')}&width={width}&height={height}"
    page.goto(render_url, wait_until="networkidle", timeout=300000)
    page.wait_for_function("window.__PREVIEW_STATUS__ === 'ready' || window.__PREVIEW_STATUS__ === 'failed'", timeout=300000)
    status = page.evaluate("window.__PREVIEW_STATUS__")
    if status != "ready":
        message = page.evaluate("window.__PREVIEW_ERROR__")
        raise RuntimeError(message or "renderer failed")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temporary_png = Path(handle.name)
    try:
        page.locator("canvas").screenshot(path=str(temporary_png))
        with Image.open(temporary_png) as image:
            save_webp_atomically(
                image,
                output_path,
                quality=quality,
                expected_width=width,
                expected_height=height,
            )
        output_path.with_suffix(output_path.suffix + ".canonical-v2").write_text("canonical-v2\n", encoding="ascii")
    finally:
        temporary_png.unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be between 0 and --shard-count - 1")
    if args.migrate_only and (args.audit_existing or args.repair_invalid):
        raise SystemExit("--migrate-only cannot be combined with --audit-existing or --repair-invalid")
    index = local_motion_query_api.MotionIndex.from_jsonl(args.index.resolve(), cache_root=args.cache_dir.resolve())
    if args.object_ids:
        known_ids = set(index.entries_by_id)
        unknown_ids = sorted(set(args.object_ids).difference(known_ids))
        if unknown_ids:
            raise SystemExit(f"Unknown object IDs: {', '.join(unknown_ids)}")
    selected_entries = [entry for entry in index.entries if args.dataset == "all" or entry.dataset == args.dataset]
    if args.object_ids:
        selected_ids = set(args.object_ids)
        selected_entries = [entry for entry in selected_entries if entry.object_id in selected_ids]
    selected_entries = selected_entries[args.shard_index :: args.shard_count]

    migration = {"converted": 0, "removed": 0, "failed": 0}
    if not args.skip_migration and not (args.audit_existing or args.repair_invalid):
        migration = migrate_legacy_pngs(args.output_root, index.entries, dataset=args.dataset, workers=args.workers, quality=args.quality)
    if args.migrate_only:
        print(json.dumps({"migration": migration}, ensure_ascii=False, indent=2))
        return 1 if migration["failed"] else 0

    audit = None
    if args.audit_existing or args.repair_invalid:
        audit, invalid_entries = audit_previews(
            args.output_root,
            selected_entries,
            width=args.width,
            height=args.height,
            workers=args.workers,
        )
        if args.audit_existing and not args.repair_invalid:
            print(json.dumps({"audit": audit}, ensure_ascii=False, indent=2))
            return 1 if audit["invalid"] else 0
        entries = invalid_entries[: args.limit or None]
    else:
        entries = candidate_entries(index, args)

    summary = {"migration": migration, "audit": audit, "candidate_count": len(entries), "generated": 0, "failed": 0, "failures": []}
    if not entries:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    from playwright.sync_api import sync_playwright

    with temporary_server(index, args) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            for entry in entries:
                target = preview_path(args.output_root, entry)
                try:
                    render_preview(page, base_url, entry, target, args.width, args.height, args.quality)
                    summary["generated"] += 1
                    print(f"[ok] {entry.dataset}/{entry.motion_id} -> {target}", flush=True)
                except Exception as exc:
                    summary["failed"] += 1
                    if len(summary["failures"]) < 20:
                        summary["failures"].append({"motion_id": entry.motion_id, "dataset": entry.dataset, "error": str(exc)})
                    print(f"[error] {entry.dataset}/{entry.motion_id}: {exc}", flush=True)
            browser.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
