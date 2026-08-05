#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import quote

from playwright.sync_api import sync_playwright

import local_motion_query_api

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("/mnt/nas/cy/humanmotion/multimotion_previews")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate low-res middle-frame PNG previews for cached multimotion GLBs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7091")
    parser.add_argument("--index", type=Path, default=local_motion_query_api.DEFAULT_INDEX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=local_motion_query_api.DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=local_motion_query_api.DEFAULT_MODEL_DIR)
    parser.add_argument("--converter-env", default=local_motion_query_api.DEFAULT_CONVERTER_ENV)
    parser.add_argument("--conda-exe", default=local_motion_query_api.DEFAULT_CONDA_EXE)
    parser.add_argument("--limit", type=int, default=0, help="Maximum previews to generate; 0 means no limit.")
    parser.add_argument("--dataset", choices=("all", "interhuman", "interx"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-uncached", action="store_true", help="Allow GLB lazy conversion for uncached motions. Heavy.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
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
    return output_root / entry.dataset / f"{entry.motion_id}.png"


def candidate_entries(index: local_motion_query_api.MotionIndex, args: argparse.Namespace) -> list[local_motion_query_api.MotionEntry]:
    entries = []
    for entry in index.entries:
        if args.dataset != "all" and entry.dataset != args.dataset:
            continue
        if not args.include_uncached and not entry.cache_glb.is_file():
            continue
        if preview_path(args.output_root, entry).is_file() and not args.overwrite:
            continue
        entries.append(entry)
        if args.limit and len(entries) >= args.limit:
            break
    return entries


def render_preview(page: Any, base_url: str, entry: local_motion_query_api.MotionEntry, output_path: Path, width: int, height: int) -> None:
    model_url = f"{base_url}/api/v1/models/{quote(entry.object_id)}"
    render_url = f"{base_url}/ui/static/preview_renderer.html?model={quote(model_url, safe=':/?&=%')}&width={width}&height={height}"
    page.goto(render_url, wait_until="networkidle", timeout=300000)
    page.wait_for_function("window.__PREVIEW_STATUS__ === 'ready' || window.__PREVIEW_STATUS__ === 'failed'", timeout=300000)
    status = page.evaluate("window.__PREVIEW_STATUS__")
    if status != "ready":
        message = page.evaluate("window.__PREVIEW_ERROR__")
        raise RuntimeError(message or "renderer failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.locator("canvas").screenshot(path=str(output_path))


def main() -> int:
    args = build_parser().parse_args()
    index = local_motion_query_api.MotionIndex.from_jsonl(args.index.resolve(), cache_root=args.cache_dir.resolve())
    entries = candidate_entries(index, args)
    summary = {"candidate_count": len(entries), "generated": 0, "failed": 0, "failures": []}
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
                    render_preview(page, base_url, entry, target, args.width, args.height)
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
