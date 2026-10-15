import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import compare_hf_cache_paths
from scripts.compare_hf_cache_paths import (
    build_exact_prefix,
    build_prefix_allowed_tokens_fn,
    normalize_top_k,
    select_tokenspeed_topk,
    write_json_no_overwrite,
)


class TestHFCachePaths(unittest.TestCase):
    def test_records_transformers_version_metadata(self) -> None:
        get_version = getattr(compare_hf_cache_paths, "get_transformers_version", None)
        self.assertIsNotNone(get_version, "Transformers version metadata is missing")
        self.assertEqual(get_version(SimpleNamespace(__version__="5.17.0")), "5.17.0")

    def test_generation_must_return_the_exact_forced_prefix(self) -> None:
        validate = getattr(
            compare_hf_cache_paths, "validate_generated_target_prefix", None
        )
        self.assertIsNotNone(validate, "generated-prefix validation is missing")

        self.assertEqual(validate([10, 11, 20], [10, 11], target_index=2), 20)
        with self.assertRaisesRegex(ValueError, "forced prefix"):
            validate([10, 99, 20], [10, 11], target_index=2)
        with self.assertRaisesRegex(ValueError, "target step"):
            validate([10, 11], [10, 11], target_index=2)

    def test_builds_only_the_shared_prefix_before_first_divergence(self) -> None:
        prefix, target_index = build_exact_prefix(
            [7, 8], [10, 11, 12], [10, 11, 13], expected_target_index=2
        )

        self.assertEqual(prefix, [7, 8, 10, 11])
        self.assertEqual(target_index, 2)

    def test_rejects_a_misreported_first_divergence(self) -> None:
        with self.assertRaisesRegex(ValueError, "first mismatch"):
            build_exact_prefix([7], [10, 11], [10, 12], expected_target_index=0)

    def test_rejects_sequences_without_a_divergence(self) -> None:
        with self.assertRaisesRegex(ValueError, "no reference mismatch"):
            build_exact_prefix([7], [10, 11], [10, 11])

    def test_forces_the_recorded_output_tokens_one_at_a_time(self) -> None:
        allowed = build_prefix_allowed_tokens_fn(
            2, [10, 11], target_index=2, vocab_size=14
        )

        self.assertEqual(allowed(0, [7, 8]), [10])
        self.assertEqual(allowed(0, [7, 8, 10]), [11])
        self.assertEqual(allowed(0, [7, 8, 10, 11]), list(range(14)))

    def test_aligns_tokenspeed_top_k_by_last_token_and_absolute_position(self) -> None:
        rows = [
            {
                "capture_mode": "full_recompute",
                "forward": {"input_id_last": 271, "position_last": 1009},
                "top_token_ids": [4, 5],
                "top_logits": [2.0, 1.0],
            },
            {
                "capture_mode": "continuous",
                "forward": {"input_id_last": 271, "position_last": 1010},
                "top_token_ids": [6, 7],
                "top_logits": [3.0, 2.0],
            },
            {
                "capture_mode": "full_recompute",
                "forward": {"input_id_last": 271, "position_last": 1010},
                "top_token_ids": [8, 9],
                "top_logits": [4.0, 3.0],
            },
        ]

        aligned = select_tokenspeed_topk(rows, input_token_id=271, position=1010)

        self.assertEqual(aligned["continuous"]["top_token_ids"], [6, 7])
        self.assertEqual(aligned["full_recompute"]["top_token_ids"], [8, 9])

    def test_rejects_missing_or_ambiguous_tokenspeed_matches(self) -> None:
        row = {
            "capture_mode": "continuous",
            "forward": {"input_id_last": 271, "position_last": 1010},
            "top_token_ids": [8, 9],
            "top_logits": [4.0, 3.0],
        }
        with self.assertRaisesRegex(ValueError, "full_recompute"):
            select_tokenspeed_topk([row], input_token_id=271, position=1010)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_tokenspeed_topk([row, row], input_token_id=271, position=1010)

    def test_normalizes_only_the_requested_top_k(self) -> None:
        self.assertEqual(
            normalize_top_k([9, 8, 7], [1.5, 1.0, 0.5], k=2),
            {"top_token_ids": [9, 8], "top_logits": [1.5, 1.0]},
        )

    def test_refuses_to_overwrite_an_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_json_no_overwrite(output, {"result": True})

            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
