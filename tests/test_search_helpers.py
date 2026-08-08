from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from query_api_client import duplicate_rate  # noqa: E402
from search_index import (  # noqa: E402
    clean_motion_title,
    clean_search_caption,
    is_searchable_caption,
    motion_series_key,
)


class SearchHelperTest(unittest.TestCase):
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
