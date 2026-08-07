#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:7091"


class ApiClientError(RuntimeError):
    pass


def request_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiClientError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ApiClientError(f"Network error: {exc}") from exc


def download_binary(url: str, timeout: int) -> bytes:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiClientError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ApiClientError(f"Network error: {exc}") from exc


def extract_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    results = response.get("results")
    if isinstance(results, list):
        return results
    data = response.get("data")
    if isinstance(data, list):
        return data
    raise ApiClientError("Response does not contain 'results' or 'data'.")


def resolve_url(candidate: str | None, base_url: str) -> str | None:
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate
    return urljoin(base_url.rstrip("/") + "/", candidate.lstrip("/"))


def model_download_url(result: dict[str, Any], base_url: str) -> str | None:
    model = result.get("model")
    if isinstance(model, dict):
        for key in ("download_url", "rest_download_path", "download_path"):
            resolved = resolve_url(model.get(key), base_url)
            if resolved:
                return resolved
    links = result.get("_links")
    if isinstance(links, dict):
        return resolve_url(links.get("download"), base_url)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the multimotion lazy retrieval API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--randomness", type=float, default=None)
    parser.add_argument("--random-seed", default=None)
    parser.add_argument("--output-dir", default="client_output")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--skip-download-models", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    payload = {
        "text": args.text,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "include_model_info": True,
    }
    if args.randomness is not None:
        payload["randomness"] = args.randomness
    if args.random_seed is not None:
        payload["random_seed"] = args.random_seed
    response = request_json(urljoin(base_url + "/", "api/v1/searches"), payload, args.timeout)
    results = extract_results(response)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "query_status.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    saved_models: list[str] = []
    if not args.skip_download_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for rank, result in enumerate(results, start=1):
            url = model_download_url(result, base_url)
            if not url:
                continue
            model = result.get("model") if isinstance(result.get("model"), dict) else {}
            filename = str(model.get("filename") or f"rank_{rank:02d}_{result.get('object_id', 'model')}.glb")
            target = models_dir / f"rank_{rank:02d}_{filename}"
            target.write_bytes(download_binary(url, args.timeout))
            saved_models.append(str(target))

    print(
        json.dumps(
            {
                "status": response.get("status"),
                "result_count": len(results),
                "status_json": str(output_dir / "query_status.json"),
                "saved_models": saved_models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
