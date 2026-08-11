from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
TAXONOMY_DIR = MODULE_DIR / "taxonomy"
CATALOG_PATH = TAXONOMY_DIR / "catalog.json"
SOURCES_PATH = TAXONOMY_DIR / "sources.json"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
WORD_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))

# These compatibility axes are kept for existing API clients. New clients use
# taxonomy nodes and action_nodes instead of the single legacy tag.
CATEGORY_VALUES = {
    "sports": "体育运动类",
    "dance_performance": "舞蹈与表演类",
    "daily_life_labor": "日常生活与劳动类",
    "entertainment_leisure": "娱乐与休闲类",
    "professional_skills": "职业技能类",
    "military_combat": "军事与战斗类",
    "special_actions": "特殊动作类",
}
TAG_VALUES = {
    "ball_sports": "球类运动", "athletics_fitness": "田径与体能", "water_sports": "水上运动",
    "ice_snow_sports": "冰雪运动", "combat_sports": "格斗与对抗", "gymnastics_skills": "体操与技巧",
    "fitness_training": "健身与体能训练", "outdoor_extreme": "户外与极限", "vehicle_sports": "车类运动",
    "equestrian_riding": "马术与骑乘", "street_pop_dance": "街舞与流行舞", "ballroom_social_dance": "国标与社交舞",
    "ethnic_classical_dance": "民族与古典舞", "stage_performance": "舞台表演", "basic_locomotion": "基本位移",
    "housework_labor": "家务与劳作", "office_learning": "办公与学习", "transportation_travel": "交通与出行",
    "game_actions": "游戏动作", "leisure_recreation": "休闲娱乐", "social_etiquette": "社交礼仪",
    "medical_care": "医疗与护理", "construction_engineering": "建筑与工程", "food_service": "餐饮与服务",
    "agriculture_fishing": "农业与渔业", "tactical_actions": "战术动作", "weapons_use": "武器使用",
    "defense_guard": "防御与护卫", "expression_posture": "表情与姿态", "physiological_actions": "生理动作",
    "other": "其他动作",
}
AUXILIARY_VALUES = {
    "participants": {"single": "单人", "pair": "双人", "group_or_unknown": "多人/未知"},
    "space": {"indoor": "室内", "outdoor": "室外", "mixed_or_unspecified": "混合/未说明"},
}
AXES: dict[str, dict[str, Any]] = {
    "action_category": {"label": "动作大类", "values": CATEGORY_VALUES},
    "action_tag": {"label": "细分标签", "values": TAG_VALUES},
    "participants": {"label": "人数", "values": AUXILIARY_VALUES["participants"]},
    "space": {"label": "空间", "values": AUXILIARY_VALUES["space"]},
    "activity_domain": {"label": "兼容活动大类", "values": {
        "social": "社交礼仪", "entertainment_performance": "娱乐表演", "sport_fitness": "运动健身",
        "commute_locomotion": "通勤移动", "daily_work": "日常/工作", "conflict": "冲突对抗",
        "other": "其他", "assist_care": "协助照护",
    }},
}
AXIS_ORDER = ("participants", "space", "action_category", "action_tag", "activity_domain")
INDOOR_KEYWORDS = {"bed", "bedroom", "bench", "chair", "classroom", "couch", "cup", "desk", "door", "floor", "home", "house", "indoor", "kitchen", "office", "room", "seat", "sofa", "stair", "table", "window"}
OUTDOOR_KEYWORDS = {"outside", "outdoor", "park", "path", "road", "sidewalk", "street", "trail", "field", "grass"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_CATALOG = _load_json(CATALOG_PATH)
_SOURCES = _load_json(SOURCES_PATH)
TAXONOMY_VERSION = str(_CATALOG["taxonomy_version"])
NODES: tuple[dict[str, Any], ...] = tuple(_CATALOG["nodes"])
NODES_BY_ID = {node["id"]: node for node in NODES}
CHILDREN: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
for _node in NODES:
    CHILDREN[_node.get("parent_id")].append(_node)


def _normalize_word(word: str) -> str:
    value = word.lower().replace("-", " ").strip()
    for suffix, replacement in WORD_SUFFIXES:
        if len(value) > len(suffix) + 2 and value.endswith(suffix):
            return value[:-len(suffix)] + replacement
    return value


def _tokenize(text: str) -> list[str]:
    return [_normalize_word(token) for token in TOKEN_RE.findall(text.lower())]


def _phrase_tokens(phrase: str) -> tuple[str, ...]:
    return tuple(_tokenize(phrase))


def _node_depth(node_id: str) -> int:
    depth = 1
    seen: set[str] = set()
    current = NODES_BY_ID[node_id]
    while current.get("parent_id") is not None:
        if current["id"] in seen:
            raise ValueError(f"taxonomy cycle detected at {node_id}")
        seen.add(current["id"])
        depth += 1
        current = NODES_BY_ID[current["parent_id"]]
    return depth


def validate_catalog() -> list[str]:
    errors: list[str] = []
    if len(NODES_BY_ID) != len(NODES):
        errors.append("duplicate taxonomy node id")
    for node in NODES:
        parent_id = node.get("parent_id")
        if parent_id is not None and parent_id not in NODES_BY_ID:
            errors.append(f"{node['id']}: missing parent {parent_id}")
        if not node.get("source_refs"):
            errors.append(f"{node['id']}: missing source_refs")
        if _node_depth(node["id"]) > _CATALOG["default_max_depth"] and not node.get("allow_depth"):
            errors.append(f"{node['id']}: depth exceeds default maximum")
    return errors


def _aliases_for_node(node: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    aliases = [node.get("en", ""), *(node.get("aliases_en") or ())]
    aliases.extend(node.get("aliases_zh") or ())
    phrases = {_phrase_tokens(alias) for alias in aliases if alias and _phrase_tokens(alias)}
    return tuple(sorted(phrases, key=lambda phrase: (-len(phrase), phrase)))


MATCH_PATTERNS: tuple[tuple[dict[str, Any], tuple[tuple[str, ...], ...]], ...] = tuple(
    (node, _aliases_for_node(node))
    for node in NODES
    if node.get("kind") not in {"domain", "category"} or node.get("aliases_en")
)
def _compact_text(value: str) -> str:
    """Normalize CJK aliases without removing the CJK characters themselves."""
    return re.sub(r"[\s\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


TEXT_MATCH_PATTERNS: tuple[tuple[dict[str, Any], tuple[str, ...]], ...] = tuple(
    (node, tuple(sorted({
        _compact_text(alias)
        for alias in [node.get("zh_cn", ""), *(node.get("aliases_zh") or ())]
        if any("\u4e00" <= char <= "\u9fff" for char in str(alias)) and _compact_text(alias)
    }, key=lambda value: (-len(value), value))))
    for node in NODES
    if node.get("asset_taxonomy") and node.get("kind") == "action"
)
PHRASE_INDEX: dict[str, list[tuple[dict[str, Any], tuple[str, ...]]]] = defaultdict(list)
for _match_node, _phrases in MATCH_PATTERNS:
    for _phrase in _phrases:
        PHRASE_INDEX[_phrase[0]].append((_match_node, _phrase))


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    if not size or size > len(tokens):
        return False
    return any(tuple(tokens[index:index + size]) == phrase for index in range(len(tokens) - size + 1))


def _root_for(node_id: str) -> dict[str, Any]:
    node = NODES_BY_ID[node_id]
    while node.get("parent_id") is not None:
        node = NODES_BY_ID[node["parent_id"]]
    return node


def _ancestors(node_id: str) -> list[str]:
    result = [node_id]
    node = NODES_BY_ID[node_id]
    while node.get("parent_id") is not None:
        result.append(node["parent_id"])
        node = NODES_BY_ID[node["parent_id"]]
    return result


def _legacy_for_node(node_id: str | None) -> tuple[str, str]:
    if not node_id:
        return "special_actions", "other"
    node = NODES_BY_ID[node_id]
    root = _root_for(node_id)
    category = root.get("legacy_category", "special_actions")
    tag = root.get("legacy_tag", "other")
    if node_id.startswith(("etiquette_", "general_etiquette")):
        return "entertainment_leisure", "social_etiquette"
    if node_id.startswith(("attack_", "defend_", "combat_")):
        return "military_combat", "combat_sports" if node_id.startswith("attack_") else "defense_guard"
    if node_id.startswith(("weapon_", "military_")):
        return "military_combat", "weapons_use" if node_id.startswith("weapon_") else "tactical_actions"
    if node_id.startswith(("locomotion_", "travel_", "drive_", "ride_")):
        return "daily_life_labor", "transportation_travel" if node_id.startswith(("travel_", "drive_", "ride_")) else "basic_locomotion"
    if node_id.startswith(("posture_", "transition_", "fundamental_")):
        return "daily_life_labor", "basic_locomotion"
    if node_id.startswith(("strength_", "conditioning_", "mind_body_", "sports_", "gym_")):
        return "sports", "gymnastics_skills" if node_id.startswith("gym_") else "fitness_training"
    if node_id.startswith(("performance_", "vocal_", "instrument_")):
        return "dance_performance", "stage_performance"
    if node_id.startswith(("expression_", "physio_", "facial_")):
        return "special_actions", "physiological_actions" if node_id.startswith("physio_") else "expression_posture"
    if node_id.startswith("medical") or node_id.startswith(("exam_", "treatment_", "nursing_")):
        return "professional_skills", "medical_care"
    if node_id.startswith(("construction_", "electronics_", "pharma_")):
        return "professional_skills", "construction_engineering"
    if node_id.startswith(("food_", "cook_", "serve_", "prepare_")):
        return "professional_skills", "food_service"
    if node_id.startswith(("farm_", "crop_", "livestock_", "fishery_")):
        return "professional_skills", "agriculture_fishing"
    if node_id.startswith(("teach_", "education_", "classroom_", "guide_")):
        return "daily_life_labor", "office_learning"
    if node_id.startswith(("games_", "leisure_", "social_")):
        return "entertainment_leisure", "leisure_recreation"
    return category, tag


def _legacy_domain_for_tag(category: str, tag: str) -> str:
    if tag in {"combat_sports", "defense_guard", "weapons_use", "tactical_actions"}:
        return "conflict"
    if tag == "medical_care":
        return "assist_care"
    if tag in {"basic_locomotion", "transportation_travel"}:
        return "commute_locomotion"
    if category == "daily_life_labor":
        return "daily_work"
    if category == "entertainment_leisure" and tag == "social_etiquette":
        return "social"
    return {
        "sports": "sport_fitness", "dance_performance": "entertainment_performance",
        "daily_life_labor": "daily_work", "entertainment_leisure": "social",
        "professional_skills": "daily_work", "military_combat": "conflict",
        "special_actions": "other",
    }.get(category, "other")


def taxonomy_payload(
    *,
    counts: dict[str, int] | None = None,
    direct_counts: dict[str, int] | None = None,
    include_nodes: bool = True,
) -> dict[str, Any]:
    counts = counts or {}
    direct_counts = direct_counts or {}
    nodes = []
    if include_nodes:
        for node in NODES:
            item = {key: value for key, value in node.items() if key not in {"aliases_en", "aliases_zh", "priority"}}
            item["depth"] = _node_depth(node["id"])
            item["label"] = node["zh_cn"]
            item["count"] = counts.get(node["id"], 0)
            item["direct_count"] = direct_counts.get(node["id"], 0)
            nodes.append(item)
    hierarchy = []
    for root in CHILDREN[None]:
        if root.get("legacy_only"):
            continue
        hierarchy.append(_tree_node(root, counts, direct_counts))
    axes = []
    for axis in AXIS_ORDER:
        definition = AXES[axis]
        axes.append({"key": axis, "label": definition["label"], "values": [{"key": key, "label": label} for key, label in definition["values"].items()]})
    payload = {
        "version": TAXONOMY_VERSION,
        "hierarchy": hierarchy,
        "sources": _SOURCES["sources"],
        "axes": axes,
        "asset_taxonomy_counts": _CATALOG.get("asset_taxonomy_counts", {}),
    }
    if include_nodes:
        payload["nodes"] = nodes
    return payload


def _tree_node(node: dict[str, Any], counts: dict[str, int], direct_counts: dict[str, int]) -> dict[str, Any]:
    result = {
        "key": node["id"], "id": node["id"], "label": node["zh_cn"], "en": node["en"],
        "kind": node["kind"], "count": counts.get(node["id"], 0), "direct_count": direct_counts.get(node["id"], 0),
        "source_refs": node.get("source_refs", []),
        "tags": [_tree_node(child, counts, direct_counts) for child in CHILDREN.get(node["id"], []) if not child.get("legacy_only")],
    }
    for field in ("summary", "focus"):
        if node.get(field):
            result[field] = node[field]
    return result


def _entry_text(entry: Any) -> str:
    parts = [str(getattr(entry, "dataset", "") or ""), str(getattr(entry, "motion_id", "") or ""), str(getattr(entry, "description", "") or "")]
    parts.extend(str(caption) for caption in (getattr(entry, "captions", ()) or ()))
    return " ".join(parts)


def _participants_for_dataset(dataset: str) -> str:
    if dataset in {"interhuman", "interx"}:
        return "pair"
    if dataset in {"motionxpp", "motionlab"}:
        return "single"
    return "group_or_unknown"


def _normalize_cached_participants(classification: dict[str, Any], dataset: str) -> dict[str, Any]:
    participants = _participants_for_dataset(dataset)
    taxonomy = {**(classification.get("taxonomy") or {}), "participants": participants}
    labels = {
        **(classification.get("taxonomy_labels") or {}),
        "participants": AUXILIARY_VALUES["participants"][participants],
    }
    matched_keywords = {
        **(classification.get("matched_keywords") or {}),
        "participants": [dataset] if participants in {"single", "pair"} and dataset else [],
    }
    return {
        **classification,
        "taxonomy": taxonomy,
        "taxonomy_labels": labels,
        "matched_keywords": matched_keywords,
    }


def classify_motion(entry: Any, *, use_cached: bool = True) -> dict[str, Any]:
    dataset = str(getattr(entry, "dataset", "") or "").lower()
    cached = getattr(entry, "classification", None)
    if use_cached and isinstance(cached, dict) and cached.get("taxonomy_version") == _CATALOG["taxonomy_version"]:
        return _normalize_cached_participants(cached, dataset)
    text = _entry_text(entry).lower()
    compact_text = _compact_text(text)
    tokens = _tokenize(text)
    matched_by_node: dict[str, dict[str, Any]] = {}
    for index, token in enumerate(tokens):
        for node, phrase in PHRASE_INDEX.get(token, ()):
            if tuple(tokens[index:index + len(phrase)]) != phrase:
                continue
            match = matched_by_node.setdefault(node["id"], {"node": node, "matched": set(), "length": 0})
            match["matched"].add(" ".join(phrase))
            match["length"] = max(match["length"], len(phrase))
    matches: list[dict[str, Any]] = []
    for match in matched_by_node.values():
        match["matched"] = sorted(match["matched"])
        matches.append(match)

    # The business taxonomy is authored in Chinese while most source captions
    # are English. Keep a separate substring matcher for Chinese descriptions
    # and aliases; token matching above remains the strict English matcher.
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        for node, aliases in TEXT_MATCH_PATTERNS:
            matched = [alias for alias in aliases if alias in compact_text]
            if not matched:
                continue
            match = matched_by_node.setdefault(node["id"], {"node": node, "matched": set(), "length": 0})
            match["matched"].update(matched)
            match["length"] = max(match["length"], max(map(len, matched)))
        matches = []
        for match in matched_by_node.values():
            match["matched"] = sorted(match["matched"])
            matches.append(match)

    # Keep the most specific action for each branch. A posture can coexist with
    # a task, but parent/category hits are represented through the node path.
    selected: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: (-int(item["node"].get("priority", 50)), -item["length"], item["node"]["id"])):
        node_id = match["node"]["id"]
        if any(node_id in _ancestors(existing["node"]["id"]) or existing["node"]["id"] in _ancestors(node_id) for existing in selected):
            continue
        selected.append(match)
    selected = selected[:8]
    primary = selected[0] if selected else None
    primary_id = primary["node"]["id"] if primary else None
    legacy_category, legacy_tag = _legacy_for_node(primary_id)
    legacy_domain = _legacy_domain_for_tag(legacy_category, legacy_tag)

    participants = _participants_for_dataset(dataset)
    indoor = sorted(token for token in INDOOR_KEYWORDS if token in tokens)
    outdoor = sorted(token for token in OUTDOOR_KEYWORDS if token in tokens)
    if indoor and not outdoor:
        space, space_matches = "indoor", indoor
    elif outdoor and not indoor:
        space, space_matches = "outdoor", outdoor
    else:
        space, space_matches = "mixed_or_unspecified", indoor + outdoor

    action_nodes = []
    for match in selected:
        node = match["node"]
        action_nodes.append({
            "id": node["id"], "label": node["zh_cn"], "en": node["en"], "kind": node["kind"],
            "path": [_node_label(node_id) for node_id in reversed(_ancestors(node["id"]))],
            "role": "primary" if match is primary else "secondary",
            "matched_phrases": match["matched"], "source_refs": node.get("source_refs", []),
        })
    confidence = round(min(1.0, 0.45 + (0.12 * len(action_nodes)) + (0.12 if primary and primary["length"] > 1 else 0.0)), 3)
    taxonomy = {"action_category": legacy_category, "action_tag": legacy_tag, "participants": participants, "space": space, "activity_domain": legacy_domain}
    labels = {"action_category": CATEGORY_VALUES[legacy_category], "action_tag": TAG_VALUES[legacy_tag], "participants": AUXILIARY_VALUES["participants"][participants], "space": AUXILIARY_VALUES["space"][space], "activity_domain": AXES["activity_domain"]["values"][legacy_domain]}
    return {
        "taxonomy_version": _CATALOG["taxonomy_version"], "taxonomy": taxonomy, "taxonomy_labels": labels,
        "primary_action_node_id": primary_id, "action_nodes": action_nodes,
        "matched_keywords": {"action_category": [legacy_category] if primary_id else [], "action_tag": primary["matched"] if primary else [], "participants": [dataset] if participants in {"single", "pair"} and dataset else [], "space": space_matches},
        "classification_confidence": confidence,
    }


def apply_semantic_nodes(
    classification: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    status: str,
) -> dict[str, Any]:
    """Attach accepted semantic action nodes while preserving legacy axes."""
    accepted = [
        item for item in candidates
        if item.get("node_id") in NODES_BY_ID
        and str(item.get("node_id")).startswith("asset_tag_")
    ][:1]
    if not accepted:
        return classification
    primary_id = str(accepted[0]["node_id"])
    category, tag = _legacy_for_node(primary_id)
    domain = _legacy_domain_for_tag(category, tag)
    taxonomy = {
        **(classification.get("taxonomy") or {}),
        "action_category": category,
        "action_tag": tag,
        "activity_domain": domain,
    }
    labels = {
        **(classification.get("taxonomy_labels") or {}),
        "action_category": CATEGORY_VALUES[category],
        "action_tag": TAG_VALUES[tag],
        "activity_domain": AXES["activity_domain"]["values"][domain],
    }
    action_nodes = []
    for index, item in enumerate(accepted):
        node = NODES_BY_ID[str(item["node_id"])]
        action_nodes.append(
            {
                "id": node["id"],
                "label": node["zh_cn"],
                "en": node["en"],
                "kind": node["kind"],
                "path": [_node_label(value) for value in reversed(_ancestors(node["id"]))],
                "role": "primary" if index == 0 else "secondary",
                "matched_phrases": [],
                "source_refs": node.get("source_refs", []),
            }
        )
    score = float(accepted[0]["score"])
    runner_up = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    return {
        **classification,
        "taxonomy": taxonomy,
        "taxonomy_labels": labels,
        "primary_action_node_id": primary_id,
        "action_nodes": action_nodes,
        "classification_confidence": round(score, 3),
        "classification_method": "semantic_leaf_v2",
        "confidence_level": status,
        "classification_model": {
            "name": model_name,
            "score": round(score, 6),
            "margin": round(score - runner_up, 6),
            "candidate_node_ids": [str(item["node_id"]) for item in candidates],
            "matched_caption": str(accepted[0].get("matched_caption") or ""),
            "status": status,
        },
    }


class TaxonomyClassifier:
    """Batch classifier backed by the generated taxonomy and caption embeddings."""

    def __init__(self, model_dir: Path, semantic_index_dir: Path, device: str = "cuda:3") -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.semantic_index_dir = semantic_index_dir.expanduser().resolve()
        self.device = device
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.semantic_manifest = json.loads(
            (self.semantic_index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        with (self.model_dir / "labels.jsonl").open("r", encoding="utf-8") as handle:
            self.labels = [json.loads(line) for line in handle if line.strip()]
        self.label_embeddings = np.load(self.model_dir / "label_embeddings.npy", mmap_mode="r")
        self.group_embeddings = np.load(self.model_dir / "group_embeddings.npy", mmap_mode="r")
        self.major_embeddings = np.load(self.model_dir / "major_embeddings.npy", mmap_mode="r")
        self.caption_embeddings = np.load(self.semantic_index_dir / "embeddings.npy", mmap_mode="r")
        with (self.semantic_index_dir / "captions.jsonl").open("r", encoding="utf-8") as handle:
            self.caption_rows = [json.loads(line) for line in handle if line.strip()]
        self._validate_model()

    def _validate_model(self) -> None:
        expected_contract = {
            "encoder_family": "qwen3-embedding",
            "pooling": "last_token",
            "normalized": True,
        }
        for field, value in expected_contract.items():
            if self.manifest.get(field) != value:
                raise ValueError(
                    f"taxonomy model {field} mismatch: expected {value!r}, "
                    f"got {self.manifest.get(field)!r}; rebuild the Qwen3 taxonomy model"
                )
            if self.semantic_manifest.get(field) != value:
                raise ValueError(
                    f"semantic index {field} mismatch: expected {value!r}, "
                    f"got {self.semantic_manifest.get(field)!r}; rebuild the Qwen3 semantic index"
                )
        if self.manifest.get("encoding_role") != "document":
            raise ValueError("taxonomy model must contain Qwen3 document embeddings")
        if self.manifest.get("model_config_sha256") != self.semantic_manifest.get("model_config_sha256"):
            raise ValueError("taxonomy model and semantic index use different Qwen3 model configs")
        if self.label_embeddings.shape != (len(self.labels), int(self.manifest["dimension"])):
            raise ValueError("taxonomy label index shape does not match its manifest")
        if len(self.caption_rows) != self.caption_embeddings.shape[0]:
            raise ValueError("semantic caption rows do not match caption embeddings")
        if self.caption_embeddings.shape[1] != self.label_embeddings.shape[1]:
            raise ValueError("caption and taxonomy embedding dimensions differ")
        if self.group_embeddings.shape != self.label_embeddings.shape:
            raise ValueError("group and leaf taxonomy embeddings differ")
        if self.major_embeddings.shape != self.label_embeddings.shape:
            raise ValueError("major and leaf taxonomy embeddings differ")

    def classify_all(
        self,
        *,
        rule_nodes: dict[str, set[str]] | None = None,
        top_k: int = 5,
        batch_size: int = 512,
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("taxonomy classification requires torch") from exc
        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        leaf_embeddings = torch.from_numpy(
            np.asarray(self.label_embeddings, dtype=np.float32)
        ).to(resolved_device)
        group_embeddings = torch.from_numpy(
            np.asarray(self.group_embeddings, dtype=np.float32)
        ).to(resolved_device)
        major_embeddings = torch.from_numpy(
            np.asarray(self.major_embeddings, dtype=np.float32)
        ).to(resolved_device)
        weights = self.manifest.get("score_weights") or {"leaf": 0.7, "group": 0.2, "major": 0.1}
        rule_bonus = float(self.manifest.get("rule_prior_bonus", 0.015))
        label_positions = {str(row["node_id"]): index for index, row in enumerate(self.labels)}
        rule_nodes = rule_nodes or {}
        candidates: dict[str, dict[int, tuple[float, float, float, float, str, bool]]] = defaultdict(dict)
        count = len(self.caption_rows)
        with torch.inference_mode():
            for start in range(0, count, max(batch_size, 1)):
                stop = min(start + batch_size, count)
                batch = np.asarray(self.caption_embeddings[start:stop], dtype=np.float32)
                batch_tensor = torch.from_numpy(batch).to(resolved_device)
                leaf_scores = batch_tensor @ leaf_embeddings.T
                group_scores = batch_tensor @ group_embeddings.T
                major_scores = batch_tensor @ major_embeddings.T
                scores = (
                    float(weights["leaf"]) * leaf_scores
                    + float(weights["group"]) * group_scores
                    + float(weights["major"]) * major_scores
                )
                for offset, row in enumerate(self.caption_rows[start:stop]):
                    for node_id in rule_nodes.get(str(row["object_id"]), ()):
                        position = label_positions.get(node_id)
                        if position is not None:
                            scores[offset, position] += rule_bonus
                values, indices = torch.topk(scores, k=min(top_k, len(self.labels)), dim=1)
                leaf_values = torch.gather(leaf_scores, 1, indices)
                group_values = torch.gather(group_scores, 1, indices)
                major_values = torch.gather(major_scores, 1, indices)
                values_np = values.cpu().numpy()
                indices_np = indices.cpu().numpy()
                leaf_np = leaf_values.cpu().numpy()
                group_np = group_values.cpu().numpy()
                major_np = major_values.cpu().numpy()
                for offset, row in enumerate(self.caption_rows[start:stop]):
                    object_id = str(row["object_id"])
                    caption = str(row.get("caption") or "")
                    object_candidates = candidates[object_id]
                    expected = rule_nodes.get(object_id, set())
                    for score, leaf_score, group_score, major_score, label_index in zip(
                        values_np[offset], leaf_np[offset], group_np[offset],
                        major_np[offset], indices_np[offset]
                    ):
                        previous = object_candidates.get(int(label_index))
                        if previous is None or float(score) > previous[0]:
                            object_candidates[int(label_index)] = (
                                float(score),
                                float(leaf_score),
                                float(group_score),
                                float(major_score),
                                caption,
                                str(self.labels[int(label_index)]["node_id"]) in expected,
                            )
                print(f"classified captions {stop}/{count}", flush=True)

        result: dict[str, list[dict[str, Any]]] = {}
        for object_id, values in candidates.items():
            ordered = sorted(values.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
            result[object_id] = [
                {
                    "node_id": self.labels[label_index]["node_id"],
                    "label": self.labels[label_index]["label"],
                    "score": round(score, 6),
                    "leaf_score": round(leaf_score, 6),
                    "group_score": round(group_score, 6),
                    "major_score": round(major_score, 6),
                    "rule_prior_applied": rule_prior_applied,
                    "matched_caption": caption,
                }
                for label_index, (
                    score, leaf_score, group_score, major_score, caption, rule_prior_applied
                ) in ordered
            ]
        return result


def _node_label(node_id: str) -> str:
    return NODES_BY_ID[node_id]["zh_cn"]


def labels_for_taxonomy(taxonomy: dict[str, str]) -> dict[str, str]:
    return {axis: str(AXES.get(axis, {}).get("values", {}).get(value, value)) for axis, value in taxonomy.items()}


def node_descendants(node_id: str) -> set[str]:
    result = {node_id}
    for child in CHILDREN.get(node_id, []):
        result.update(node_descendants(child["id"]))
    return result


def catalog_sources() -> list[dict[str, Any]]:
    return list(_SOURCES["sources"])
