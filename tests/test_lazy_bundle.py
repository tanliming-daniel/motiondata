from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import urlopen

import build_motion_index
from cache_manifest import CacheManifest
import local_motion_query_api
import motion_taxonomy


class LazyBundleTests(unittest.TestCase):
    def test_cache_manifest_prunes_oldest_files_to_byte_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            manifest = CacheManifest(root)
            try:
                first = root / "interhuman" / "first.glb"
                second = root / "motionxpp" / "second.glb"
                first.parent.mkdir(parents=True)
                second.parent.mkdir(parents=True)
                first.write_bytes(b"1234")
                second.write_bytes(b"5678")
                manifest.record(first, object_id="first", dataset="interhuman")
                time.sleep(0.01)
                manifest.record(second, object_id="second", dataset="motionxpp")
                removed = manifest.prune(max_bytes=4)
                self.assertEqual(removed, [first])
                self.assertFalse(first.exists())
                self.assertTrue(second.exists())
            finally:
                manifest.close()

    def test_read_captions_keeps_non_empty_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.txt"
            path.write_text("\nshake hands\n  hug each other  \n", encoding="utf-8")
            self.assertEqual(build_motion_index.read_captions(path), ("shake hands", "hug each other"))

    def test_build_index_records_skip_missing_or_empty_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "interhuman"
            (root / "motions").mkdir(parents=True)
            (root / "annots").mkdir(parents=True)
            (root / "motions" / "1.pkl").write_bytes(b"raw")
            (root / "motions" / "2.pkl").write_bytes(b"raw")
            (root / "motions" / "3.pkl").write_bytes(b"raw")
            (root / "annots" / "1.txt").write_text("two people shake hands\n", encoding="utf-8")
            (root / "annots" / "2.txt").write_text("\n", encoding="utf-8")

            records, warnings = build_motion_index.build_interhuman_records(root, Path("cache") / "models")

            self.assertEqual([record.motion_id for record in records], ["1"])
            self.assertEqual(len(warnings), 2)

    def test_build_motionxpp_records_uses_recursive_subset_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "motionx++"
            motion = root / "smplx322" / "idea400" / "clip.npy"
            caption = root / "semantic_label" / "idea400" / "clip.txt"
            motion.parent.mkdir(parents=True)
            caption.parent.mkdir(parents=True)
            motion.write_bytes(b"npy")
            caption.write_text("a person waves\n", encoding="utf-8")

            records, warnings = build_motion_index.build_motionxpp_records(root, Path("cache") / "models")

            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].dataset, "motionxpp")
            self.assertEqual(records[0].motion_id, "idea400/clip")
            self.assertEqual(records[0].cache_glb, Path("cache/models/motionxpp/idea400/clip.glb"))

    def test_search_prefers_caption_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "1.pkl"
            caption_file = tmp_path / "1.txt"
            source.write_bytes(b"raw")
            caption_file.write_text("two people greet each other by shaking hands.\n", encoding="utf-8")
            index_path = tmp_path / "motion_index.jsonl"
            row = {
                "dataset": "interhuman",
                "motion_id": "1",
                "source_motion": str(source),
                "caption_file": str(caption_file),
                "captions": ["two people greet each other by shaking hands."],
                "description": "two people greet each other by shaking hands.",
                "object_id": "obj1",
                "cache_glb": "cache/models/interhuman/1.glb",
            }
            index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            index = local_motion_query_api.MotionIndex.from_jsonl(index_path, cache_root=tmp_path / "cache" / "models")
            result = index.search_entries("shake hands", top_k=1)

            self.assertEqual(result[0][1].dataset, "interhuman")
            self.assertEqual(result[0][1].motion_id, "1")

    def test_ensure_cached_glb_is_lazy_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "1.pkl"
            source.write_bytes(b"raw")
            entry = local_motion_query_api.MotionEntry(
                object_id="obj1",
                dataset="interhuman",
                motion_id="1",
                source_motion=source,
                caption_file=tmp_path / "1.txt",
                captions=("shake hands",),
                description="shake hands",
                cache_glb=tmp_path / "cache" / "models" / "interhuman" / "1.glb",
                search_text="interhuman 1 shake hands",
                token_counts=local_motion_query_api.Counter({"shake": 1, "hand": 1}),
                token_length=2,
            )
            calls: list[Path] = []

            def fake_converter(entry, output_path, **kwargs):
                calls.append(output_path)
                output_path.write_bytes(b"glb")

            self.assertFalse(entry.cache_glb.exists())
            path, generated = local_motion_query_api.ensure_cached_glb(
                entry,
                lock=threading.Lock(),
                model_dir=tmp_path,
                frame_step=1,
                interx_fps=30.0,
                converter_env=None,
                conda_exe="conda",
                converter=fake_converter,
            )
            self.assertTrue(generated)
            self.assertEqual(path.read_bytes(), b"glb")

            path, generated = local_motion_query_api.ensure_cached_glb(
                entry,
                lock=threading.Lock(),
                model_dir=tmp_path,
                frame_step=1,
                interx_fps=30.0,
                converter_env=None,
                conda_exe="conda",
                converter=fake_converter,
            )
            self.assertFalse(generated)
            self.assertEqual(len(calls), 1)

    def test_merge_search_text_deduplicates_alias_tokens(self) -> None:
        merged = local_motion_query_api.merge_search_text("Shake hands.", ["shake", "hand"])
        self.assertEqual(merged, "shake hand")

    def test_default_search_randomness_is_enabled(self) -> None:
        self.assertGreater(local_motion_query_api.DEFAULT_SEARCH_RANDOMNESS, 0.0)
        self.assertLessEqual(local_motion_query_api.DEFAULT_SEARCH_RANDOMNESS, 1.0)

    def test_rerank_results_is_reproducible_with_seed(self) -> None:
        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(
            local_motion_query_api.MotionQueryRequestHandler
        )

        def make_entry(index: int) -> local_motion_query_api.MotionEntry:
            return local_motion_query_api.MotionEntry(
                object_id=f"obj{index}",
                dataset="test",
                motion_id=str(index),
                source_motion=Path(f"/tmp/{index}.pkl"),
                caption_file=Path(f"/tmp/{index}.txt"),
                captions=("walk",),
                description="walk",
                cache_glb=Path(f"/tmp/{index}.glb"),
                search_text="walk",
                token_counts=local_motion_query_api.Counter({"walk": 1}),
                token_length=1,
            )

        candidates = [(10.0 - index * 0.1, make_entry(index)) for index in range(8)]

        first = handler._rerank_results(candidates, top_k=5, randomness=0.5, random_seed="fixed")
        second = handler._rerank_results(candidates, top_k=5, randomness=0.5, random_seed="fixed")

        self.assertEqual(
            [entry.object_id for _, entry in first],
            [entry.object_id for _, entry in second],
        )

    def test_prepare_search_text_uses_local_translator(self) -> None:
        class FakeTranslator:
            def translate(self, text: str) -> str:
                return "slap face"

        server = local_motion_query_api.MotionQueryServer.__new__(local_motion_query_api.MotionQueryServer)
        server.translation_url = None
        server.translation_model = "fake-model"
        server.translation_device = -1
        server.translation_timeout = 8.0
        server.local_translator = lambda: FakeTranslator()

        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(local_motion_query_api.MotionQueryRequestHandler)
        handler.server = server

        warnings: list[str] = []
        search_text, translation_info = handler._prepare_search_text("一个人扇另一个人巴掌", warnings)

        self.assertEqual(search_text, "slap face")
        self.assertEqual(warnings, [])
        self.assertIsNotNone(translation_info)
        assert translation_info is not None
        self.assertEqual(translation_info["status"], "ok")
        self.assertEqual(translation_info["provider"], "local_transformers")
        self.assertEqual(translation_info["translated_text"], "slap face")

    def test_prepare_search_text_uses_alias_only_for_short_cjk_query(self) -> None:
        server = local_motion_query_api.MotionQueryServer.__new__(local_motion_query_api.MotionQueryServer)
        server.translation_url = None
        server.translation_model = "fake-model"
        server.translation_device = -1
        server.translation_timeout = 8.0
        server.local_translator = lambda: None

        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(local_motion_query_api.MotionQueryRequestHandler)
        handler.server = server

        warnings: list[str] = []
        search_text, translation_info = handler._prepare_search_text("握手", warnings)

        self.assertEqual(search_text, "shake hand")
        self.assertEqual(warnings, [])
        self.assertIsNotNone(translation_info)
        assert translation_info is not None
        self.assertEqual(translation_info["status"], "alias_only")
        self.assertEqual(translation_info["provider"], "builtin_alias")
        self.assertEqual(translation_info["source_alias_tokens"], ["shake", "hand"])

    def test_prepare_search_text_falls_back_to_alias_tokens(self) -> None:
        server = local_motion_query_api.MotionQueryServer.__new__(local_motion_query_api.MotionQueryServer)
        server.translation_url = None
        server.translation_model = None
        server.translation_device = -1
        server.translation_timeout = 8.0
        server.local_translator = lambda: None

        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(local_motion_query_api.MotionQueryRequestHandler)
        handler.server = server

        warnings: list[str] = []
        search_text, translation_info = handler._prepare_search_text("扇巴掌", warnings)

        self.assertEqual(search_text, "slap face")
        self.assertEqual(warnings, [])
        self.assertIsNotNone(translation_info)
        assert translation_info is not None
        self.assertEqual(translation_info["status"], "alias_only")
        self.assertEqual(translation_info["provider"], "builtin_alias")
        self.assertEqual(translation_info["source_alias_tokens"], ["slap", "face"])

    def test_prepare_search_text_uses_external_translation_after_local_failure(self) -> None:
        class FailingTranslator:
            def translate(self, text: str) -> str:
                raise RuntimeError("local model failed")

        server = local_motion_query_api.MotionQueryServer.__new__(local_motion_query_api.MotionQueryServer)
        server.translation_url = "http://127.0.0.1:7099/translate"
        server.translation_model = "fake-model"
        server.translation_device = -1
        server.translation_timeout = 8.0
        server.local_translator = lambda: FailingTranslator()

        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(local_motion_query_api.MotionQueryRequestHandler)
        handler.server = server
        handler._translate_query_text_external = lambda text: "stand by a table and organize cups"

        warnings: list[str] = []
        search_text, translation_info = handler._prepare_search_text("另一个家庭成员站在厨房岛台旁整理杯子。", warnings)

        self.assertIn("stand", search_text.split())
        self.assertIn("cup", search_text.split())
        self.assertEqual(warnings, [])
        self.assertIsNotNone(translation_info)
        assert translation_info is not None
        self.assertEqual(translation_info["status"], "ok")
        self.assertEqual(translation_info["provider"], "external_http")
        self.assertEqual(translation_info["translated_text"], "stand by a table and organize cups")

    def test_prepare_search_text_uses_aliases_when_long_query_translation_fails(self) -> None:
        class FailingTranslator:
            def translate(self, text: str) -> str:
                raise RuntimeError("local model failed")

        server = local_motion_query_api.MotionQueryServer.__new__(local_motion_query_api.MotionQueryServer)
        server.translation_url = None
        server.translation_model = "fake-model"
        server.translation_device = -1
        server.translation_timeout = 8.0
        server.local_translator = lambda: FailingTranslator()

        handler = local_motion_query_api.MotionQueryRequestHandler.__new__(local_motion_query_api.MotionQueryRequestHandler)
        handler.server = server

        warnings: list[str] = []
        search_text, translation_info = handler._prepare_search_text("另一个家庭成员站在厨房岛台旁整理杯子。", warnings)

        tokens = set(search_text.split())
        self.assertIn("stand", tokens)
        self.assertIn("kitchen", tokens)
        self.assertIn("cup", tokens)
        self.assertNotIn("另一个家庭成员站在厨房岛台旁整理杯子。", search_text)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Translation failed", warnings[0])
        self.assertIsNotNone(translation_info)
        assert translation_info is not None
        self.assertEqual(translation_info["status"], "failed")
        self.assertEqual(translation_info["provider"], "builtin_alias")
        self.assertTrue(translation_info["used_for_search"])

    def test_motion_taxonomy_classifies_common_domains(self) -> None:
        def entry(description: str) -> local_motion_query_api.MotionEntry:
            return local_motion_query_api.MotionEntry(
                object_id="obj",
                dataset="interhuman",
                motion_id="1",
                source_motion=Path("/tmp/1.pkl"),
                caption_file=Path("/tmp/1.txt"),
                captions=(description,),
                description=description,
                cache_glb=Path("/tmp/1.glb"),
                search_text=description,
                token_counts=local_motion_query_api.Counter(),
                token_length=0,
            )

        self.assertEqual(motion_taxonomy.classify_motion(entry("two people greet and shake hands"))["taxonomy"]["activity_domain"], "social")
        self.assertEqual(motion_taxonomy.classify_motion(entry("two performers dance and clap"))["taxonomy"]["activity_domain"], "entertainment_performance")
        self.assertEqual(motion_taxonomy.classify_motion(entry("one person kicks a ball in sport"))["taxonomy"]["activity_domain"], "sport_fitness")
        self.assertEqual(motion_taxonomy.classify_motion(entry("two people walk along a street"))["taxonomy"]["activity_domain"], "commute_locomotion")
        self.assertEqual(motion_taxonomy.classify_motion(entry("one person organizes cups on a kitchen table"))["taxonomy"]["activity_domain"], "daily_work")
        self.assertEqual(motion_taxonomy.classify_motion(entry("one person punches and blocks the other"))["taxonomy"]["activity_domain"], "conflict")
        self.assertEqual(motion_taxonomy.classify_motion(entry("one person helps and supports another"))["taxonomy"]["activity_domain"], "assist_care")

    def test_motion_taxonomy_defaults_when_unspecified(self) -> None:
        entry = local_motion_query_api.MotionEntry(
            object_id="obj",
            dataset="custom",
            motion_id="x",
            source_motion=Path("/tmp/x.pkl"),
            caption_file=Path("/tmp/x.txt"),
            captions=("ambiguous motion",),
            description="ambiguous motion",
            cache_glb=Path("/tmp/x.glb"),
            search_text="ambiguous motion",
            token_counts=local_motion_query_api.Counter(),
            token_length=0,
        )
        taxonomy = motion_taxonomy.classify_motion(entry)["taxonomy"]
        self.assertEqual(taxonomy["activity_domain"], "other")
        self.assertEqual(taxonomy["space"], "mixed_or_unspecified")
        self.assertEqual(taxonomy["participants"], "group_or_unknown")

    def test_motion_taxonomy_catalog_is_valid_and_source_backed(self) -> None:
        self.assertEqual(motion_taxonomy.validate_catalog(), [])
        payload = motion_taxonomy.taxonomy_payload()
        self.assertGreaterEqual(len(payload["hierarchy"]), 15)
        self.assertIn("version", payload)
        self.assertTrue(all(node["source_refs"] for node in payload["nodes"]))
        self.assertFalse(any(node["id"].startswith(("country_", "nation_")) for node in payload["nodes"]))

    def test_motion_taxonomy_uses_word_boundaries_and_multiple_nodes(self) -> None:
        entry = local_motion_query_api.MotionEntry(
            object_id="obj", dataset="custom", motion_id="x", source_motion=Path("/tmp/x.pkl"),
            caption_file=Path("/tmp/x.txt"), captions=("a single person stands and opens the door",),
            description="a single person stands and opens the door", cache_glb=Path("/tmp/x.glb"),
            search_text="", token_counts=local_motion_query_api.Counter(), token_length=0,
        )
        classification = motion_taxonomy.classify_motion(entry)
        node_ids = {node["id"] for node in classification["action_nodes"]}
        self.assertIn("fixture_open_door", node_ids)
        self.assertIn("posture_stand", node_ids)
        self.assertNotIn("vocal_singing", node_ids)

    def test_motion_taxonomy_classifies_requested_examples(self) -> None:
        expected = {
            "the athlete performs a deadlift": "strength_deadlift",
            "a person holds the handrail on the train": "travel_hold_pole",
            "the doctor uses a stethoscope": "exam_auscultate",
            "the teacher points at the blackboard": "teach_board",
            "a worker is hammering": "construction_hammer",
        }
        for description, node_id in expected.items():
            entry = local_motion_query_api.MotionEntry(
                object_id=node_id, dataset="custom", motion_id=node_id, source_motion=Path("/tmp/x.pkl"),
                caption_file=Path("/tmp/x.txt"), captions=(description,), description=description,
                cache_glb=Path("/tmp/x.glb"), search_text="", token_counts=local_motion_query_api.Counter(), token_length=0,
            )
            node_ids = {node["id"] for node in motion_taxonomy.classify_motion(entry)["action_nodes"]}
            self.assertIn(node_id, node_ids, description)

    def test_library_api_filters_and_serves_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for motion_id, description in [("1", "two people greet and shake hands"), ("2", "one person punches and blocks the other")]:
                source = tmp_path / f"{motion_id}.pkl"
                caption = tmp_path / f"{motion_id}.txt"
                source.write_bytes(b"raw")
                caption.write_text(description + "\n", encoding="utf-8")
                rows.append(
                    {
                        "dataset": "interhuman",
                        "motion_id": motion_id,
                        "source_motion": str(source),
                        "caption_file": str(caption),
                        "captions": [description],
                        "description": description,
                        "object_id": f"obj{motion_id}",
                        "cache_glb": f"cache/models/interhuman/{motion_id}.glb",
                        "frame_count": 240 + int(motion_id),
                    }
                )
            index_path = tmp_path / "motion_index.jsonl"
            index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            motion_index = local_motion_query_api.MotionIndex.from_jsonl(index_path, cache_root=tmp_path / "cache" / "models")
            server = local_motion_query_api.MotionQueryServer(
                ("127.0.0.1", 0),
                local_motion_query_api.MotionQueryRequestHandler,
                motion_index,
                model_dir=tmp_path,
                frame_step=1,
                interx_fps=30.0,
                converter_env=None,
                conda_exe="conda",
                translation_model="none",
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                taxonomy = json.loads(urlopen(base_url + "/api/v1/taxonomy", timeout=5).read().decode("utf-8"))
                self.assertEqual(taxonomy["data"]["axes"][0]["key"], "participants")
                self.assertIn("nodes", taxonomy["data"])
                self.assertTrue(any(node["count"] > 0 for node in taxonomy["data"]["nodes"]))
                library = json.loads(urlopen(base_url + "/api/v1/library?activity_domain=social", timeout=5).read().decode("utf-8"))
                self.assertEqual(library["total"], 1)
                self.assertEqual(library["items"][0]["object_id"], "obj1")
                self.assertEqual(library["items"][0]["taxonomy"]["activity_domain"], "social")
                by_node = json.loads(urlopen(base_url + "/api/v1/library?node_id=etiquette_handshake", timeout=5).read().decode("utf-8"))
                self.assertEqual(by_node["total"], 1)
                self.assertEqual(by_node["items"][0]["primary_action_node_id"], "etiquette_handshake")
                html = urlopen(base_url + "/ui", timeout=5).read().decode("utf-8")
                self.assertIn("Multimotion 动作库", html)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
