#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from search_index import motion_series_key


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
    items = response.get("items")
    if isinstance(items, list):
        return items
    results = response.get("results")
    if isinstance(results, list):
        return results
    data = response.get("data")
    if isinstance(data, list):
        return data
    raise ApiClientError("Response does not contain 'items', 'results', or 'data'.")


def resolve_url(candidate: str | None, base_url: str) -> str | None:
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate
    return urljoin(base_url.rstrip("/") + "/", candidate.lstrip("/"))


def model_download_url(result: dict[str, Any], base_url: str) -> str | None:
    glb = result.get("glb")
    if isinstance(glb, dict):
        resolved = resolve_url(glb.get("url"), base_url)
        if resolved:
            return resolved
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text")
    mode.add_argument("--eval-file", help="JSONL retrieval evaluation set.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--randomness", type=float, default=None)
    parser.add_argument("--random-seed", default=None)
    parser.add_argument("--output-dir", default="client_output")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--skip-download-models", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diversity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--report", default="runtime/retrieval_eval_report.json")
    return parser


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def result_matches(result: dict[str, Any], asset_ids: set[str], patterns: list[str]) -> bool:
    if str(result.get("object_id") or "") in asset_ids:
        return True
    searchable = f"{result.get('dataset', '')}/{result.get('motion_id', '')} {result.get('description', '')}"
    return any(re.search(pattern, searchable, re.I) for pattern in patterns)


def duplicate_rate(results: list[dict[str, Any]]) -> float:
    keys: list[str] = []
    for result in results[:10]:
        dataset = str(result.get("dataset") or "")
        motion_id = str(result.get("motion_id") or "").casefold()
        keys.append(motion_series_key(dataset, motion_id) or f"{dataset}/{motion_id}")
    return (len(keys) - len(set(keys))) / max(len(keys), 1)


def motion_matches(result: dict[str, Any], asset_ids: set[str], patterns: list[str]) -> bool:
    if str(result.get("object_id") or "") in asset_ids:
        return True
    motion_key = f"{result.get('dataset', '')}/{result.get('motion_id', '')}"
    return any(re.search(pattern, motion_key, re.I) for pattern in patterns)


def evaluation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "query_count": len(rows),
        "hit_at_5": statistics.fmean(row["hit_at_5"] for row in rows) if rows else 0.0,
        "mrr_at_10": statistics.fmean(row["mrr_at_10"] for row in rows) if rows else 0.0,
        "ndcg_at_10": statistics.fmean(row["ndcg_at_10"] for row in rows) if rows else 0.0,
        "recall_at_50": statistics.fmean(row["recall_at_50"] for row in rows) if rows else 0.0,
        "duplicate_rate_at_10": statistics.fmean(row["duplicate_rate_at_10"] for row in rows) if rows else 0.0,
        "forbidden_hits_at_10": sum(row["forbidden_hits_at_10"] for row in rows),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
    }


def run_evaluation(args: argparse.Namespace, base_url: str) -> int:
    cases: list[dict[str, Any]] = []
    with Path(args.eval_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ApiClientError(f"{args.eval_file}:{line_number}: {exc}") from exc
            if args.split == "all" or case.get("split") == args.split:
                cases.append(case)

    rows: list[dict[str, Any]] = []
    for case in cases:
        payload = {
            "text": case["query"],
            "top_k": 50,
            "candidate_k": max(args.candidate_k, 200),
            "include_model_info": False,
            "randomness": 0,
            "retrieval_mode": args.retrieval_mode,
            "rerank": args.rerank,
            "diversity": args.diversity,
        }
        started = time.perf_counter()
        response = request_json(urljoin(base_url + "/", "api/v1/searches"), payload, args.timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        results = extract_results(response)
        relevant_ids = {str(value) for value in case.get("relevant_asset_ids", [])}
        relevant_patterns = [str(value) for value in case.get("relevant_motion_patterns", [])]
        forbidden_ids = {str(value) for value in case.get("forbidden_asset_ids", [])}
        forbidden_patterns = [str(value) for value in case.get("forbidden_motion_patterns", [])]
        relevant_ranks = [
            rank for rank, result in enumerate(results, 1)
            if result_matches(result, relevant_ids, relevant_patterns)
        ]
        relevant_total = max(
            int(case.get("relevant_total") or (10 if relevant_patterns else len(relevant_ids))),
            1,
        )
        relevant_ranks = relevant_ranks[:relevant_total]
        first_rank = relevant_ranks[0] if relevant_ranks else None
        gains = [1 if rank in relevant_ranks else 0 for rank in range(1, 11)]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant_total, 10) + 1))
        forbidden_hits = sum(motion_matches(result, forbidden_ids, forbidden_patterns) for result in results[:10])
        rows.append({
            "id": case.get("id"), "query": case["query"], "split": case.get("split"),
            "language": case.get("language"), "query_type": case.get("query_type"),
            "latency_ms": round(latency_ms, 3),
            "first_relevant_rank": first_rank, "hit_at_5": bool(first_rank and first_rank <= 5),
            "mrr_at_10": 1 / first_rank if first_rank and first_rank <= 10 else 0.0,
            "ndcg_at_10": dcg / ideal if ideal else 0.0,
            "recall_at_50": min(len(relevant_ranks) / relevant_total, 1.0),
            "duplicate_rate_at_10": duplicate_rate(results), "forbidden_hits_at_10": forbidden_hits,
            "top_motion_ids": [result.get("motion_id") for result in results[:10]],
        })
    metrics = evaluation_metrics(rows)
    languages = sorted({str(row.get("language") or "unknown") for row in rows})
    query_types = sorted({str(row.get("query_type") or "unknown") for row in rows})
    report = {
        "config": {"base_url": base_url, "retrieval_mode": args.retrieval_mode, "rerank": args.rerank, "diversity": args.diversity, "split": args.split},
        "metrics": metrics,
        "metrics_by_language": {
            language: evaluation_metrics([row for row in rows if (row.get("language") or "unknown") == language])
            for language in languages
        },
        "metrics_by_query_type": {
            query_type: evaluation_metrics([row for row in rows if (row.get("query_type") or "unknown") == query_type])
            for query_type in query_types
        },
        "queries": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "metrics": metrics,
                "metrics_by_language": report["metrics_by_language"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    if args.eval_file:
        return run_evaluation(args, base_url)
    payload = {
        "text": args.text,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "include_model_info": True,
        "retrieval_mode": args.retrieval_mode,
        "rerank": args.rerank,
        "diversity": args.diversity,
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
            glb_value = result.get("glb")
            glb = glb_value if isinstance(glb_value, dict) else {}
            model_value = result.get("model")
            model = model_value if isinstance(model_value, dict) else {}
            filename = str(glb.get("filename") or model.get("filename") or f"rank_{rank:02d}_{result.get('object_id', 'model')}.glb")
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
