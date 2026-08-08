#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

import motion_taxonomy
from search_index import SemanticRetriever


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = SCRIPT_DIR / "dataset" / "motion_index.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset" / "motion_taxonomy_assignments.jsonl"
DEFAULT_REPORT = SCRIPT_DIR / "dataset" / "motion_taxonomy_report.json"
DEFAULT_REVIEW = SCRIPT_DIR / "dataset" / "motion_taxonomy_review.jsonl"
DEFAULT_TAXONOMY_MODEL = SCRIPT_DIR / "dataset" / "taxonomy_model"
DEFAULT_SEMANTIC_INDEX = SCRIPT_DIR / "dataset" / "semantic_index_qwen3_06b"
DEFAULT_MODEL = "/data5/cy/models/qwen3-embedding-0.6b"
CATALOG_PATH = SCRIPT_DIR / "taxonomy" / "catalog.json"
TAXONOMY_SOURCE_PATH = SCRIPT_DIR / "taxonomy" / "action_asset_source.json"


@dataclass
class EntryView:
    object_id: str
    dataset: str
    motion_id: str
    description: str
    captions: tuple[str, ...]


LEGACY_BY_MAJOR = {
    "体育运动类": ("sports", "fitness_training"),
    "舞蹈与表演类": ("dance_performance", "stage_performance"),
    "乐器与声乐表演类": ("dance_performance", "stage_performance"),
    "日常基础姿态与位移动作": ("daily_life_labor", "basic_locomotion"),
    "日常生活与劳动类": ("daily_life_labor", "housework_labor"),
    "办公学习与教育行业": ("daily_life_labor", "office_learning"),
    "交通出行与交通运输业": ("daily_life_labor", "transportation_travel"),
    "休闲娱乐与社交类": ("entertainment_leisure", "leisure_recreation"),
    "社交礼仪与文化礼俗": ("entertainment_leisure", "social_etiquette"),
    "表情姿态与非语言沟通类": ("special_actions", "expression_posture"),
    "生理动作与健康状态类": ("special_actions", "physiological_actions"),
    "职业服务与公共管理类": ("professional_skills", "other"),
    "建筑工程与制造业类": ("professional_skills", "construction_engineering"),
    "餐饮服务与食品加工类": ("professional_skills", "food_service"),
    "农业林牧渔类": ("professional_skills", "agriculture_fishing"),
    "军事战斗与安全防卫类": ("military_combat", "tactical_actions"),
    "武术格斗与对抗类": ("military_combat", "combat_sports"),
}

LEGACY_TARGET_LABELS = {
    "fixture_open_door": "开门", "fixture_open_window": "开窗",
    "fixture_drawer_cabinet": "开抽屉", "posture_stand": "自然站立",
    "posture_sit": "端坐", "posture_lie": "仰卧", "posture_kneel": "跪姿",
    "posture_crouch": "半蹲", "posture_lean": "靠墙", "posture_cross_legged": "盘腿坐",
    "locomotion_walk": "步行", "locomotion_run": "小跑", "locomotion_jump": "跳跃",
    "locomotion_crawl": "手膝爬行", "locomotion_climb": "爬楼梯",
    "transition_bend": "弯腰", "strength_squat": "徒手深蹲", "strength_deadlift": "传统硬拉",
    "strength_bench_press": "卧推", "conditioning_jump_rope": "跳绳单摇",
    "conditioning_jumping_jack": "开合跳", "conditioning_burpee": "波比跳",
    "mind_body_pilates": "普拉提百次拍", "vocal_singing": "拿麦唱歌",
    "vocal_choir": "合唱", "sports_ball": "篮球运球", "sports_water": "自由泳划水",
    "sports_winter": "高山滑雪转弯", "performance_dance": "Hip-Hop律动",
    "housework_sweep": "扫地", "housework_mop": "拖地", "housework_wipe": "擦桌子",
    "housework_laundry": "折衣服", "housework_organize": "整理衣柜",
    "daily_eating": "拿筷子", "eating_food": "咀嚼", "drinking": "喝水",
    "travel_hold_pole": "拉公交扶手", "travel_board_alight": "下车",
    "ride_bicycle": "骑自行车", "games_cards": "打牌", "games_billiards": "台球俯身击球",
    "etiquette_handshake": "握手", "etiquette_bow": "鞠躬", "etiquette_wave": "挥手",
    "etiquette_hug": "拥抱", "expression_smile": "微笑", "expression_frown": "皱眉",
    "physio_breathe": "正常呼吸", "physio_cough": "咳嗽", "physio_sneeze": "打喷嚏",
    "physio_yawn": "打哈欠", "attack_punch": "直拳", "attack_kick": "前踢",
    "defend_block": "格挡", "defend_dodge": "闪避", "exam_observe": "视诊",
    "exam_auscultate": "听诊", "exam_palpate": "触诊", "treatment_cpr": "心肺复苏",
    "treatment_bandage": "包扎", "nursing_support": "搀扶行走", "construction_hammer": "锤钉子",
    "construction_carry": "搬运", "construction_masonry": "砌砖", "construction_wheelbarrow": "推车",
    "construction_weld": "焊接", "food_chop": "切配", "food_knead": "揉面",
    "cook_stir_fry": "翻炒", "food_serving": "上菜", "serve_food": "上菜",
    "teach_explain": "讲解", "teach_board": "板书", "farm_plant": "播种",
    "fishery_catch": "驾船捕鱼", "military_cover": "利用掩体", "weapon_aim": "举枪瞄准",
    "weapon_fire": "立姿射击", "operate_radar": "雷达台操作",
}


def stable_taxonomy_id(prefix: str, *path: str) -> str:
    digest = hashlib.sha1("\x1f".join(path).encode("utf-8")).hexdigest()[:12]
    return f"asset_{prefix}_{digest}"


def build_catalog() -> dict[str, Any]:
    old = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    legacy_nodes = [node for node in old.get("nodes", ()) if not node.get("asset_taxonomy")]
    source = json.loads(TAXONOMY_SOURCE_PATH.read_text(encoding="utf-8"))["taxonomy"]
    nodes: list[dict[str, Any]] = []
    for major in source:
        legacy_category, legacy_tag = LEGACY_BY_MAJOR.get(
            major["major"], ("special_actions", "other")
        )
        major_id = stable_taxonomy_id("major", major["major"])
        nodes.append({
            "id": major_id, "parent_id": None, "kind": "domain",
            "zh_cn": major["major"], "en": major["major"],
            "aliases_zh": [major["major"]], "summary": major.get("summary", ""),
            "source_refs": ["dataset_captions"], "legacy_category": legacy_category,
            "legacy_tag": legacy_tag, "asset_taxonomy": True, "priority": 20,
        })
        for group in major.get("groups", ()):
            group_id = stable_taxonomy_id("group", major["major"], group["name"])
            nodes.append({
                "id": group_id, "parent_id": major_id, "kind": "category",
                "zh_cn": group["name"], "en": group["name"],
                "aliases_zh": [group["name"]], "focus": group.get("focus", ""),
                "source_refs": ["dataset_captions"], "asset_taxonomy": True, "priority": 30,
            })
            for tag in group.get("tags", ()):
                nodes.append({
                    "id": stable_taxonomy_id("tag", major["major"], group["name"], tag),
                    "parent_id": group_id, "kind": "action", "zh_cn": tag, "en": tag,
                    "aliases_zh": [tag], "source_refs": ["dataset_captions"],
                    "asset_taxonomy": True, "priority": 40,
                })
    for node in legacy_nodes:
        nodes.append({**node, "legacy_only": True, "source_refs": node.get("source_refs", ["dataset_captions"])})
    by_label = {node["zh_cn"]: node for node in nodes if node.get("asset_taxonomy")}
    legacy_by_id = {node["id"]: node for node in legacy_nodes}
    for legacy_id, target_label in LEGACY_TARGET_LABELS.items():
        target = by_label.get(target_label)
        legacy = legacy_by_id.get(legacy_id)
        if target is not None and legacy is not None:
            target["aliases_en"] = sorted(set([
                *(target.get("aliases_en") or ()), *(legacy.get("aliases_en") or ())
            ]))
    return {
        "schema_version": 2,
        "taxonomy_version": "2026.08-action-assets.2",
        "default_max_depth": 3,
        "nodes": nodes,
        "asset_taxonomy_counts": {
            "major": len(source),
            "group": sum(len(major.get("groups", ())) for major in source),
            "tag": sum(len(group.get("tags", ())) for major in source for group in major.get("groups", ())),
        },
    }


def write_catalog() -> dict[str, Any]:
    catalog = build_catalog()
    CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    importlib.reload(motion_taxonomy)
    return catalog


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


def rule_asset_nodes(classification: dict[str, Any]) -> set[str]:
    return {
        str(node["id"])
        for node in classification.get("action_nodes") or ()
        if str(node.get("id") or "").startswith("asset_tag_")
    }


def confidence_thresholds(
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    scores = [float(items[0]["score"]) for items in predictions.values() if items]
    margins = [
        float(items[0]["score"]) - float(items[1]["score"])
        for items in predictions.values() if len(items) > 1
    ]
    if not scores or not margins:
        raise RuntimeError("semantic classifier returned insufficient candidates")
    return {
        "high_score": round(float(np.quantile(scores, 0.75)), 6),
        "high_margin": round(float(np.quantile(margins, 0.75)), 6),
        "medium_score": round(float(np.quantile(scores, 0.25)), 6),
        "medium_margin": round(float(np.quantile(margins, 0.25)), 6),
    }


def confidence_level(candidates: list[dict[str, Any]], thresholds: dict[str, float]) -> str:
    score = float(candidates[0]["score"])
    margin = score - float(candidates[1]["score"])
    if score >= thresholds["high_score"] and margin >= thresholds["high_margin"]:
        return "high"
    if score >= thresholds["medium_score"] and margin >= thresholds["medium_margin"]:
        return "medium"
    return "low"


def persist_confidence_thresholds(model_dir: Path, thresholds: dict[str, float]) -> None:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["confidence_thresholds"] = thresholds
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)


def model_metadata(candidates: list[dict[str, Any]], model_name: str, level: str) -> dict[str, Any]:
    score = float(candidates[0]["score"]) if candidates else 0.0
    runner_up = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    return {
        "name": model_name,
        "score": round(score, 6),
        "leaf_score": float(candidates[0].get("leaf_score", 0.0)) if candidates else 0.0,
        "prototype_score": float(candidates[0].get("leaf_score", 0.0)) if candidates else 0.0,
        "group_score": float(candidates[0].get("group_score", 0.0)) if candidates else 0.0,
        "major_score": float(candidates[0].get("major_score", 0.0)) if candidates else 0.0,
        "margin": round(score - runner_up, 6),
        "candidate_node_ids": [str(item["node_id"]) for item in candidates],
        "matched_caption": str(candidates[0].get("matched_caption") or "") if candidates else "",
        "confidence_level": level,
        "status": f"{level}_confidence",
    }


def build(
    index_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    classifier_mode: str = "rule",
    taxonomy_model_path: Path = DEFAULT_TAXONOMY_MODEL,
    semantic_index_path: Path = DEFAULT_SEMANTIC_INDEX,
    review_path: Path = DEFAULT_REVIEW,
    device: str = "cuda:3",
) -> dict[str, object]:
    node_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    unmatched: list[dict[str, str]] = []
    review_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    entries = list(iter_entries(index_path))
    rules = {
        entry.object_id: motion_taxonomy.classify_motion(entry, use_cached=False)
        for entry in entries
    }
    predictions: dict[str, list[dict[str, Any]]] = {}
    thresholds: dict[str, float] = {}
    model_name = "none"
    if classifier_mode == "hybrid":
        classifier = motion_taxonomy.TaxonomyClassifier(
            taxonomy_model_path, semantic_index_path, device
        )
        rule_nodes = {
            object_id: rule_asset_nodes(classification)
            for object_id, classification in rules.items()
        }
        predictions = classifier.classify_all(rule_nodes=rule_nodes)
        model_name = str(classifier.manifest["model"])
        thresholds = confidence_thresholds(predictions)
        persist_confidence_thresholds(taxonomy_model_path, thresholds)

    audit_ids: dict[str, set[str]] = {"high": set(), "medium": set(), "low": set()}
    if classifier_mode == "hybrid":
        by_level: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
        for object_id, candidates in predictions.items():
            by_level[confidence_level(candidates, thresholds)].append(object_id)
        audit_ids = {
            level: set(sorted(object_ids)[:100])
            for level, object_ids in by_level.items()
        }

    conflict_count = 0

    def rows() -> Iterable[dict[str, object]]:
        nonlocal conflict_count
        for entry in entries:
            dataset_counts[entry.dataset] += 1
            classification = rules[entry.object_id]
            candidates = predictions.get(entry.object_id) or []
            expected = rule_asset_nodes(classification)
            status = "rule"
            if classifier_mode == "hybrid":
                if len(candidates) < 2:
                    raise RuntimeError(f"missing semantic leaf candidates for {entry.object_id}")
                level = confidence_level(candidates, thresholds)
                selected_id = str(candidates[0]["node_id"])
                rule_conflict = bool(expected and selected_id not in expected)
                if rule_conflict:
                    conflict_count += 1
                classification = motion_taxonomy.apply_semantic_nodes(
                    classification,
                    [candidates[0]],
                    model_name=model_name,
                    status=level,
                )
                classification = {
                    **classification,
                    "taxonomy_version": motion_taxonomy.TAXONOMY_VERSION,
                    "classification_method": "semantic_leaf_v2",
                    "confidence_level": level,
                    "rule_conflict": rule_conflict,
                    "classification_model": model_metadata(candidates, model_name, level),
                }
                reasons = []
                if level == "low":
                    reasons.append("low_confidence")
                if rule_conflict:
                    reasons.append("rule_model_conflict")
                if entry.object_id in audit_ids[level]:
                    reasons.append(f"{level}_confidence_audit_sample")
                if reasons:
                    review_rows.append({
                        "object_id": entry.object_id, "dataset": entry.dataset,
                        "motion_id": entry.motion_id, "description": entry.description,
                        "reason": reasons[0], "reasons": reasons, "rule_node_ids": sorted(expected),
                        "published_node_id": selected_id, "candidates": candidates,
                    })
                status = level
            status_counts[status] += 1
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
    if classifier_mode == "hybrid":
        atomic_jsonl_write(review_path, review_rows)
    total = len(entries)
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
        "classifier_mode": classifier_mode,
        "classification_status_counts": dict(status_counts),
        "confidence_thresholds": thresholds,
        "rule_conflict_count": conflict_count,
        "leaf_label_count": len(visible_action_nodes()),
        "used_leaf_label_count": len(primary_counts),
        "empty_leaf_label_count": len(visible_action_nodes()) - len(primary_counts),
        "leaf_coverage_rate": round((total - len(unmatched)) / max(total, 1), 6),
        "review_count": len(review_rows),
        "unclassified_examples": unmatched[:200],
        "catalog_validation_errors": motion_taxonomy.validate_catalog(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def visible_action_nodes() -> list[dict[str, Any]]:
    return [
        node for node in motion_taxonomy.NODES
        if node.get("asset_taxonomy") and not node.get("legacy_only") and node.get("kind") == "action"
    ]


def node_path(node: dict[str, Any]) -> list[dict[str, Any]]:
    path = [node]
    while path[-1].get("parent_id"):
        path.append(motion_taxonomy.NODES_BY_ID[path[-1]["parent_id"]])
    return list(reversed(path))


def prototype_zh(node: dict[str, Any]) -> str:
    labels = [str(item.get("zh_cn") or "") for item in node_path(node)]
    aliases = [*node.get("aliases_en", []), *node.get("aliases_zh", [])]
    unique_aliases = list(dict.fromkeys(str(value) for value in aliases if value))
    text = f"动作分类：{' > '.join(labels)}。动作：{node['zh_cn']}。"
    if unique_aliases:
        text += f"同义描述：{', '.join(unique_aliases)}。"
    return text


def hierarchy_prototypes(node: dict[str, Any]) -> tuple[str, str, str]:
    path = node_path(node)
    aliases_en = list(dict.fromkeys(str(value) for value in node.get("aliases_en", []) if value))
    chinese = prototype_zh(node)
    aliases = f" English aliases: {', '.join(aliases_en)}." if aliases_en else ""
    leaf = f"{chinese}{aliases}"
    group = f"动作类别：{path[-2]['zh_cn']}。所属领域：{path[0]['zh_cn']}。"
    major = f"动作领域：{path[0]['zh_cn']}。"
    return leaf, group, major


def build_taxonomy_model(args: argparse.Namespace) -> dict[str, Any]:
    nodes = visible_action_nodes()
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        leaf, group, major = hierarchy_prototypes(node)
        rows.append({
            "index": index, "node_id": node["id"], "parent_id": node.get("parent_id"),
            "label": node["zh_cn"],
            "prototype_zh": prototype_zh(node), "prototype": leaf,
            "group_prototype": group, "major_prototype": major,
        })
    encoder = SemanticRetriever(args.taxonomy_model, args.model, args.device, load_index=False)
    assert encoder.encoder_profile is not None

    def encode_field(field: str) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(rows), max(args.batch_size, 1)):
            batch = rows[start : start + max(args.batch_size, 1)]
            vectors.append(encoder.encode_documents([str(row[field]) for row in batch]))
            print(
                f"encoded {field} {min(start + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )
        return np.concatenate(vectors).astype(np.float16)

    leaf_embeddings = encode_field("prototype")
    group_embeddings = encode_field("group_prototype")
    major_embeddings = encode_field("major_prototype")
    model_config_path = Path(args.model).expanduser() / "config.json"
    manifest = {
        "model": args.model,
        "model_config_sha256": (
            hashlib.sha256(model_config_path.read_bytes()).hexdigest()
            if model_config_path.is_file()
            else None
        ),
        "taxonomy_version": motion_taxonomy.taxonomy_payload()["version"],
        "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "classification_method": "semantic_leaf_v2",
        "count": len(rows), "dimension": int(leaf_embeddings.shape[1]), "dtype": "float16",
        "normalized": True, "pooling": encoder.encoder_profile.pooling,
        "encoder_family": encoder.encoder_profile.family,
        "encoding_role": "document",
        "score_weights": {"leaf": 0.7, "group": 0.2, "major": 0.1},
        "rule_prior_bonus": 0.015,
    }
    output = args.taxonomy_model.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temp_dir:
        temp = Path(temp_dir)
        np.save(temp / "label_embeddings.npy", leaf_embeddings)
        np.save(temp / "group_embeddings.npy", group_embeddings)
        np.save(temp / "major_embeddings.npy", major_embeddings)
        (temp / "labels.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in (
            "label_embeddings.npy", "group_embeddings.npy",
            "major_embeddings.npy", "labels.jsonl", "manifest.json",
        ):
            os.replace(temp / name, output / name)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the taxonomy catalog, model, and motion assignments.")
    parser.add_argument(
        "--stage",
        choices=("catalog", "model", "assignments", "all"),
        default="assignments",
        help="Build stage; the default preserves the original assignments command.",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--classifier",
        choices=("rule", "hybrid"),
        default="hybrid",
        help="Classification mode; hybrid publishes one semantic leaf per asset.",
    )
    parser.add_argument("--taxonomy-model", type=Path, default=DEFAULT_TAXONOMY_MODEL)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results: dict[str, Any] = {}
    if args.stage in {"catalog", "all"}:
        results["catalog"] = write_catalog()["asset_taxonomy_counts"]
    if args.stage in {"model", "all"}:
        results["model"] = build_taxonomy_model(args)
    if args.stage in {"assignments", "all"}:
        results["assignments"] = build(
            args.index.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.report.expanduser().resolve(),
            classifier_mode=args.classifier,
            taxonomy_model_path=args.taxonomy_model.expanduser().resolve(),
            semantic_index_path=args.semantic_index.expanduser().resolve(),
            review_path=args.review.expanduser().resolve(),
            device=args.device,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    report = results.get("assignments") or {}
    return 1 if report.get("catalog_validation_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
