# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import subprocess
import sys
import unittest
from pathlib import Path

try:
    from benchmarks.course_project import correctness
except ImportError:
    correctness = None

from benchmarks.course_project.core import ChatResult

build_correctness_cases = getattr(correctness, "build_correctness_cases", None)
make_correctness_record = getattr(correctness, "make_correctness_record", None)
compare_token_ids = getattr(correctness, "compare_token_ids", None)
compare_token_ids_with_stop_alignment = getattr(
    correctness, "compare_token_ids_with_stop_alignment", None
)
collect_stop_token_ids = getattr(correctness, "collect_stop_token_ids", None)
truncate_at_stop_token = getattr(correctness, "truncate_at_stop_token", None)
build_reference_attention_mask = getattr(
    correctness, "build_reference_attention_mask", None
)


class TestCorrectnessCases(unittest.TestCase):
    def test_covers_required_prompt_categories(self) -> None:
        self.assertIsNotNone(build_correctness_cases, "correctness cases are missing")
        cases = build_correctness_cases()
        self.assertEqual(
            {case.category for case in cases},
            {"chinese", "english", "math", "code", "long_context", "multi_turn"},
        )
        multi_turn = next(case for case in cases if case.category == "multi_turn")
        self.assertEqual(len(multi_turn.turns), 3)
        long_context = next(case for case in cases if case.category == "long_context")
        self.assertGreater(len(long_context.turns[0]), 1000)

    def test_record_distinguishes_retokenized_text_from_generated_tokens(self) -> None:
        self.assertIsNotNone(make_correctness_record, "correctness recorder is missing")
        case = build_correctness_cases()[0]
        result = ChatResult(
            text="北京是中国的首都。",
            input_tokens=12,
            output_tokens=8,
            ttft_ms=15.0,
            e2e_ms=55.0,
            tpot_ms=5.0,
        )
        record = make_correctness_record(
            case=case,
            turn_index=1,
            input_token_ids=[1, 2, 3],
            retokenized_output_ids=[4, 5],
            max_new_tokens=128,
            result=result,
        )
        self.assertEqual(record["case"], case.name)
        self.assertEqual(record["category"], "chinese")
        self.assertEqual(record["client_input_token_ids"], [1, 2, 3])
        self.assertEqual(record["retokenized_output_ids"], [4, 5])
        self.assertIsNone(record["output_token_ids"])
        self.assertFalse(record["token_correctness_verified"])
        self.assertEqual(record["output_text"], result.text)
        self.assertEqual(record["ttft_ms"], 15.0)

    def test_record_preserves_engine_token_ids_without_claiming_reference_match(
        self,
    ) -> None:
        case = build_correctness_cases()[0]
        result = ChatResult(
            text="raw engine output",
            input_tokens=3,
            output_tokens=2,
            ttft_ms=1.0,
            e2e_ms=2.0,
            tpot_ms=1.0,
        )
        record = make_correctness_record(
            case=case,
            turn_index=1,
            input_token_ids=[1, 2, 3],
            retokenized_output_ids=[9],
            engine_output_token_ids=[4, 5],
            max_new_tokens=64,
            result=result,
        )
        self.assertEqual(record["output_token_ids"], [4, 5])
        self.assertEqual(
            record["output_token_ids_source"], "tokenspeed_generate_output_ids"
        )
        self.assertFalse(record["token_correctness_verified"])
        self.assertEqual(
            record["sampling_params"],
            {"temperature": 0, "max_new_tokens": 64},
        )

    def test_compares_engine_and_reference_token_sequences(self) -> None:
        self.assertIsNotNone(compare_token_ids, "token comparator is missing")
        self.assertEqual(
            compare_token_ids([1, 2, 3], [1, 2, 3]),
            {
                "exact_match": True,
                "engine_token_count": 3,
                "reference_token_count": 3,
                "first_mismatch_index": None,
            },
        )
        self.assertEqual(
            compare_token_ids([1, 2, 3], [1, 4]),
            {
                "exact_match": False,
                "engine_token_count": 3,
                "reference_token_count": 2,
                "first_mismatch_index": 1,
            },
        )
        self.assertEqual(
            compare_token_ids([1, 2], [1, 2, 3])["first_mismatch_index"], 2
        )

    def test_stop_alignment_discards_terminal_markers_but_keeps_raw_comparison(
        self,
    ) -> None:
        self.assertIsNotNone(collect_stop_token_ids, "stop-token collection is missing")
        self.assertIsNotNone(truncate_at_stop_token, "stop-token alignment is missing")
        stop_ids = collect_stop_token_ids(248044, [248046, 248044], None)
        self.assertEqual(stop_ids, [248044, 248046])
        engine_ids = [10, 11]
        reference_ids = [10, 11, 248046, 198, 248044]
        self.assertFalse(compare_token_ids(engine_ids, reference_ids)["exact_match"])
        self.assertEqual(truncate_at_stop_token(engine_ids, stop_ids), engine_ids)
        self.assertEqual(truncate_at_stop_token(reference_ids, stop_ids), [10, 11])
        self.assertTrue(
            compare_token_ids(
                truncate_at_stop_token(engine_ids, stop_ids),
                truncate_at_stop_token(reference_ids, stop_ids),
            )["exact_match"]
        )
        self.assertIsNotNone(
            compare_token_ids_with_stop_alignment,
            "stop-aligned token comparison is missing",
        )
        comparison = compare_token_ids_with_stop_alignment(
            engine_ids, reference_ids, stop_ids
        )
        self.assertFalse(comparison["exact_match"])
        self.assertTrue(comparison["stop_aligned_exact_match"])
        self.assertEqual(comparison["stop_aligned_first_mismatch_index"], None)

    def test_stop_alignment_leaves_sequences_without_stop_tokens_unchanged(
        self,
    ) -> None:
        self.assertIsNotNone(truncate_at_stop_token, "stop-token alignment is missing")
        self.assertEqual(truncate_at_stop_token([1, 2, 3], [248044, 248046]), [1, 2, 3])
        self.assertEqual(truncate_at_stop_token([248046, 2], [248046]), [])

    def test_reference_mask_preserves_eos_tokens_in_unpadded_inputs(self) -> None:
        self.assertIsNotNone(
            build_reference_attention_mask, "reference attention-mask helper is missing"
        )
        self.assertEqual(
            build_reference_attention_mask([7, 248044, 9, 248044]),
            [1, 1, 1, 1],
        )

    def test_correctness_script_help_runs_without_model_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/run_correctness.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--base-url", result.stdout)

    def test_reference_comparison_script_help_runs_without_model_dependencies(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/compare_correctness_reference.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-new-tokens", result.stdout)


if __name__ == "__main__":
    unittest.main()
