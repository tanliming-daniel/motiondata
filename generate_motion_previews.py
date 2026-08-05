#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
from typing import Any
from urllib.parse import quote

from PIL import Image
from playwright.sync_api import sync_playwright

import local_motion_query_api

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("/mnt/nas/cy/humanmotion/multimotion_previews")


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
            if not webp_path.is_file():
                webp_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(png_path) as image:
                    image.convert("RGB").save(webp_path, "WEBP", quality=quality, method=6)
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
    eligible = eligible[args.shard_index :: args.shard_count]
    if args.overwrite:
        return eligible[: args.limit or None]

    def is_missing(entry: local_motion_query_api.MotionEntry) -> bool:
        return not preview_path(args.output_root, entry).is_file()

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temporary_png = Path(handle.name)
    try:
        page.locator("canvas").screenshot(path=str(temporary_png))
        with Image.open(temporary_png) as image:
            image.convert("RGB").save(output_path, "WEBP", quality=quality, method=6)
    finally:
        temporary_png.unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be between 0 and --shard-count - 1")
    index = local_motion_query_api.MotionIndex.from_jsonl(args.index.resolve(), cache_root=args.cache_dir.resolve())
    migration = {"converted": 0, "removed": 0, "failed": 0}
    if not args.skip_migration:
        migration = migrate_legacy_pngs(args.output_root, index.entries, dataset=args.dataset, workers=args.workers, quality=args.quality)
    if args.migrate_only:
        print(json.dumps({"migration": migration}, ensure_ascii=False, indent=2))
        return 1 if migration["failed"] else 0
    entries = candidate_entries(index, args)
    summary = {"migration": migration, "candidate_count": len(entries), "generated": 0, "failed": 0, "failures": []}
    if not entries:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

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
