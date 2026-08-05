from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


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


def taxonomy_payload(*, counts: dict[str, int] | None = None, direct_counts: dict[str, int] | None = None) -> dict[str, Any]:
    counts = counts or {}
    direct_counts = direct_counts or {}
    nodes = []
    for node in NODES:
        item = {key: value for key, value in node.items() if key not in {"aliases_en", "aliases_zh", "priority"}}
        item["depth"] = _node_depth(node["id"])
        item["label"] = node["zh_cn"]
        item["count"] = counts.get(node["id"], 0)
        item["direct_count"] = direct_counts.get(node["id"], 0)
        nodes.append(item)
    hierarchy = []
    for root in CHILDREN[None]:
        hierarchy.append(_tree_node(root, counts, direct_counts))
    axes = []
    for axis in AXIS_ORDER:
        definition = AXES[axis]
        axes.append({"key": axis, "label": definition["label"], "values": [{"key": key, "label": label} for key, label in definition["values"].items()]})
    return {
        "version": _CATALOG["taxonomy_version"],
        "hierarchy": hierarchy,
        "nodes": nodes,
        "sources": _SOURCES["sources"],
        "axes": axes,
    }


def _tree_node(node: dict[str, Any], counts: dict[str, int], direct_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "key": node["id"], "id": node["id"], "label": node["zh_cn"], "en": node["en"],
        "kind": node["kind"], "count": counts.get(node["id"], 0), "direct_count": direct_counts.get(node["id"], 0),
        "source_refs": node.get("source_refs", []),
        "tags": [_tree_node(child, counts, direct_counts) for child in CHILDREN.get(node["id"], [])],
    }


def _entry_text(entry: Any) -> str:
    parts = [str(getattr(entry, "dataset", "") or ""), str(getattr(entry, "motion_id", "") or ""), str(getattr(entry, "description", "") or "")]
    parts.extend(str(caption) for caption in (getattr(entry, "captions", ()) or ()))
    return " ".join(parts)


def classify_motion(entry: Any, *, use_cached: bool = True) -> dict[str, Any]:
    cached = getattr(entry, "classification", None)
    if use_cached and isinstance(cached, dict) and cached.get("taxonomy_version") == _CATALOG["taxonomy_version"]:
        return cached
    text = _entry_text(entry).lower()
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

    dataset = str(getattr(entry, "dataset", "") or "").lower()
    participants = "pair" if dataset in {"interhuman", "interx"} else "group_or_unknown"
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
        "matched_keywords": {"action_category": [legacy_category] if primary_id else [], "action_tag": primary["matched"] if primary else [], "participants": [dataset] if participants == "pair" and dataset else [], "space": space_matches},
        "classification_confidence": confidence,
    }


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
