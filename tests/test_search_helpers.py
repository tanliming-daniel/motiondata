from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_motion_index import build_motionlab_records, glb_stem_to_description, read_glb_frame_count  # noqa: E402
from local_motion_query_api import MotionIndex, existing_glb_path_for_entry, semantic_query_text, tokenize  # noqa: E402
from query_api_client import duplicate_rate  # noqa: E402
from search_index import (  # noqa: E402
    clean_motion_title,
    clean_search_caption,
    is_searchable_caption,
    motion_series_key,
)


class SearchHelperTest(unittest.TestCase):
    MOTIONLAB_ROOT = Path(
        "/data5/cy/WorldModel/story2world_bundle_20260623_014225/PersonGen/assets/motionlab_glb_v3"
    )

    def test_glb_stem_to_description_creates_readable_caption(self) -> None:
        self.assertEqual(glb_stem_to_description("Air_Squat_Bent_Arms"), "Air Squat Bent Arms")
        self.assertEqual(glb_stem_to_description("Release_Hostage_-_Hostage"), "Release Hostage - Hostage")
        self.assertEqual(
            glb_stem_to_description("Standing_Melee_Combo_Attack_Ver._3"),
            "Standing Melee Combo Attack Ver. 3",
        )

    def test_motionlab_records_use_existing_glb_as_source_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            glb_path = root / "Air_Squat.glb"
            glb_path.write_bytes(b"glTF")

            records, warnings = build_motionlab_records(root, root / "cache")

            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].dataset, "motionlab")
            self.assertEqual(records[0].motion_id, "Air_Squat")
            self.assertEqual(records[0].description, "Air Squat")
            self.assertEqual(records[0].source_motion, glb_path.resolve())
            self.assertEqual(records[0].cache_glb, glb_path.resolve())

    def test_motionlab_real_glb_frame_counts_are_parsed(self) -> None:
        if not self.MOTIONLAB_ROOT.is_dir():
            self.skipTest(f"MotionLab root not available: {self.MOTIONLAB_ROOT}")

        self.assertEqual(read_glb_frame_count(self.MOTIONLAB_ROOT / "Aiming.glb"), 61)
        self.assertEqual(read_glb_frame_count(self.MOTIONLAB_ROOT / "Baseball_Hit.glb"), 118)

    def test_motionlab_real_records_include_frame_count(self) -> None:
        if not self.MOTIONLAB_ROOT.is_dir():
            self.skipTest(f"MotionLab root not available: {self.MOTIONLAB_ROOT}")

        records, warnings = build_motionlab_records(self.MOTIONLAB_ROOT, self.MOTIONLAB_ROOT / "cache")
        by_id = {record.motion_id: record for record in records}

        self.assertEqual(warnings, [])
        self.assertEqual(by_id["Aiming"].frame_count, 61)
        self.assertEqual(by_id["Baseball_Hit"].frame_count, 118)

    def test_motionlab_runtime_preserves_existing_glb_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            glb_path = root / "Air_Squat.glb"
            glb_path.write_bytes(b"glTF")
            index_path = root / "motion_index.jsonl"
            index_path.write_text(
                json.dumps(
                    {
                        "dataset": "motionlab",
                        "motion_id": "Air_Squat",
                        "source_motion": str(glb_path),
                        "caption_file": str(glb_path),
                        "captions": ["Air Squat"],
                        "description": "Air Squat",
                        "object_id": "motionlab-air-squat",
                        "cache_glb": str(glb_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            index = MotionIndex.from_jsonl(index_path, cache_root=root / "cache")
            entry = index.get_entry("motionlab-air-squat")

            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.cache_glb, glb_path)
            self.assertEqual(existing_glb_path_for_entry(entry), glb_path)

    def test_cjk_ball_sport_aliases_are_distinct(self) -> None:
        self.assertIn("baseball", tokenize("棒球"))
        self.assertNotIn("basketball", tokenize("棒球"))
        self.assertIn("basketball", tokenize("篮球"))

    def test_cjk_quality_aliases_cover_known_eval_failures(self) -> None:
        self.assertIn("high", tokenize("两个人面对面抬手击掌"))
        self.assertIn("five", tokenize("两个人面对面抬手击掌"))
        self.assertIn("clap", tokenize("两个人高兴地拍手鼓掌"))
        self.assertNotIn("pat", tokenize("两个人高兴地拍手鼓掌"))
        self.assertIn("dance", tokenize("一个人随着节奏扭动身体跳舞"))
        self.assertNotIn("jump", tokenize("跳舞"))
        self.assertIn("violin", tokenize("先举起小提琴，然后持续拉奏"))
        self.assertIn("strum", tokenize("先摆好持琴姿势，然后快速扫弦"))
        self.assertIn("billiard", tokenize("台球"))
        self.assertNotIn("tennis", tokenize("台球"))
        self.assertIn("busines", tokenize("两个人握手之后互相交换名片"))
        self.assertIn("three", tokenize("先把腿绑在一起，然后同步向前走"))

    def test_cjk_semantic_query_keeps_original_text_and_adds_aliases(self) -> None:
        enriched = semantic_query_text("先摆好持琴姿势，然后快速扫弦")
        self.assertIn("先摆好持琴姿势", enriched)
        self.assertIn("strum", enriched)
        self.assertIn("instrument", enriched)
        self.assertEqual(semantic_query_text("walking forward"), "walking forward")

    def test_motion_title_removes_nested_clip_and_take_suffixes(self) -> None:
        self.assertEqual(
            clean_motion_title("motionxpp", "idea400/shake_hands_clip1_2"),
            "shake hands",
        )
        self.assertEqual(
            motion_series_key("motionxpp", "idea400/Shake_Hands_2_clip1"),
            "idea400 shake hands",
        )

    def test_non_motionx_title_is_not_inferred(self) -> None:
        self.assertEqual(clean_motion_title("interhuman", "G001T000A000R000"), "")

    def test_generated_caption_prefix_and_refusal_are_filtered(self) -> None:
        self.assertEqual(clean_search_caption(" Caption:   A person walks. "), "A person walks.")
        self.assertTrue(is_searchable_caption("A person walks forward."))
        self.assertFalse(is_searchable_caption("Sorry, I cannot provide that information."))

    def test_duplicate_rate_uses_the_runtime_series_key(self) -> None:
        results = [
            {"dataset": "motionxpp", "motion_id": "set/walk_clip1"},
            {"dataset": "motionxpp", "motion_id": "set/walk_clip2_1"},
            {"dataset": "interhuman", "motion_id": "G001T000A000R000"},
        ]

        self.assertAlmostEqual(duplicate_rate(results), 1 / 3)


if __name__ == "__main__":
    unittest.main()
