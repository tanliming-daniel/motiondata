from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_index import (  # noqa: E402
    QWEN3_MOTION_QUERY_INSTRUCTION,
    SemanticRetriever,
    format_semantic_query,
    last_token_pool,
    qwen3_encoder_profile,
)


class SemanticEncoderTest(unittest.TestCase):
    def test_qwen3_profile_and_query_instruction(self) -> None:
        profile = qwen3_encoder_profile("/models/qwen3-embedding-0.6b")

        self.assertEqual(profile.family, "qwen3-embedding")
        self.assertEqual(profile.pooling, "last_token")
        self.assertEqual(profile.padding_side, "left")
        self.assertEqual(profile.torch_dtype, "bfloat16")
        self.assertEqual(
            format_semantic_query("台球", profile),
            f"Instruct: {QWEN3_MOTION_QUERY_INSTRUCTION}\nQuery:一个人正在从事以下活动：台球",
        )
        self.assertEqual(
            format_semantic_query("一个人俯身用球杆击打台球", profile),
            f"Instruct: {QWEN3_MOTION_QUERY_INSTRUCTION}\nQuery:一个人俯身用球杆击打台球",
        )

    def test_non_qwen_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires Qwen3-Embedding"):
            qwen3_encoder_profile("/models/bge-m3")

    def test_last_token_pool_handles_right_padding(self) -> None:
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        pooled = last_token_pool(hidden, attention_mask)

        torch.testing.assert_close(pooled, torch.stack((hidden[0, 1], hidden[1, 2])))

    def test_last_token_pool_handles_left_padding(self) -> None:
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        attention_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

        pooled = last_token_pool(hidden, attention_mask)

        torch.testing.assert_close(pooled, hidden[:, -1])

    def test_legacy_index_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = SemanticRetriever(
                Path(temp_dir),
                "/models/qwen3-embedding-0.6b",
            )
            retriever.manifest = {
                "schema_version": 2,
                "encoder_family": "generic-cls",
                "pooling": "cls",
            }

            with self.assertRaisesRegex(ValueError, "schema_version mismatch"):
                retriever._validate_index_contract(
                    np.zeros((1, 1024), dtype=np.float16),
                    [{"object_id": "one", "caption": "walk"}],
                )

    def test_model_index_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = SemanticRetriever(
                Path(temp_dir),
                "/models/qwen3-embedding-0.6b",
            )
            retriever.embeddings = np.zeros((1, 1024), dtype=np.float16)
            model = SimpleNamespace(config=SimpleNamespace(hidden_size=1024))

            retriever._validate_model_and_index(model)
            model.config.hidden_size = 768

            with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
                retriever._validate_model_and_index(model)


if __name__ == "__main__":
    unittest.main()
