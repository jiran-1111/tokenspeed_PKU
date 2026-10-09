from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.report import (
    _series_for,
    generate_standard_reports,
    load_summary_records,
)


class TestCourseReport(unittest.TestCase):
    def test_raw_run_leaf_infers_ancestor_family_and_generates_agent_svgs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "results" / "raw" / "agent" / "qwen35_4b_run"
            raw_dir.mkdir(parents=True)
            summary = raw_dir / "sample.summary.jsonl"
            summary.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "request_count": 1,
                            "shared_prefix_ratio": ratio,
                            "ttft_p95_ms": 100 + ratio,
                            "token_throughput": 20 + ratio,
                        }
                    )
                    for ratio in (0.0, 0.5)
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_summary_records(raw_dir)
            generated = generate_standard_reports(records, raw_dir.parent / "plots")
            names = {path.name for path in generated}

            self.assertTrue((raw_dir.parent / "plots" / "summary.csv").exists())

        self.assertEqual({record["result_family"] for record in records}, {"agent"})
        self.assertIn("prefix_cache.svg", names)
        self.assertIn("prefix_throughput.svg", names)

    def test_summary_loader_labels_raw_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            summary = raw_dir / "agent" / "run" / "sample.summary.jsonl"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"request_count": 1}) + "\n", encoding="utf-8")

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "agent")

    def test_ablation_summary_loader_keeps_matrix_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            matrix = "qwen35_0.8b_formal_chunk512_20261009"
            summary = raw_dir / "ablation" / matrix / "sample.summary.jsonl"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"request_count": 12}) + "\n", encoding="utf-8")

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "ablation")
        self.assertEqual(records[0].get("experiment_matrix"), matrix)

    def test_ablation_plots_show_variants_and_keep_chunk_sweeps_separate(self) -> None:
        records = []
        for chunk_size in (512, 1024):
            matrix = f"chunk{chunk_size}"
            for index, variant in enumerate(("E0", "E1", "E2", "E3")):
                effective_chunk = 4096 if index < 2 else chunk_size
                for repetition in range(1, 4):
                    records.append(
                        {
                            "result_family": "ablation",
                            "experiment_matrix": matrix,
                            "served_model_name": "qwen35-0.8b",
                            "variant": variant,
                            "experiment_index": index,
                            "chunk_size": effective_chunk,
                            "concurrency": 4,
                            "target_input_tokens": 2048,
                            "target_output_tokens": 32,
                            "tp_size": 1,
                            "token_throughput": float(index + repetition),
                        }
                    )

        with tempfile.TemporaryDirectory() as temporary:
            generated = generate_standard_reports(records, Path(temporary))
            names = {path.name for path in generated}
            self.assertIn("ablation.svg", names)
            self.assertIn("chunk_size.svg", names)
            ablation_svg = (Path(temporary) / "ablation.svg").read_text(encoding="utf-8")
            chunk_svg = (Path(temporary) / "chunk_size.svg").read_text(encoding="utf-8")

        self.assertIn("experiment_matrix=chunk512", ablation_svg)
        self.assertIn("experiment_matrix=chunk1024", ablation_svg)
        self.assertIn("variant=E2", chunk_svg)
        self.assertIn("variant=E3", chunk_svg)
        self.assertNotIn("variant=E0", chunk_svg)
        self.assertNotIn("variant=E1", chunk_svg)

    def test_baseline_plots_separate_models(self) -> None:
        records = []
        for model, scale in (("qwen35-0.8b", 1.0), ("qwen35-9b", 10.0)):
            for concurrency in (1, 4):
                records.append(
                    {
                        "result_family": "baseline",
                        "served_model_name": model,
                        "server_metrics_available": True,
                        "prefix_cache_hit_ratio": 0.0,
                        "concurrency": concurrency,
                        "target_input_tokens": 128,
                        "target_output_tokens": 32,
                        "tp_size": 1,
                        "token_throughput": scale * concurrency,
                        "ttft_p95_ms": scale * concurrency,
                        "tpot_p95_ms": scale * concurrency,
                    }
                )

        with tempfile.TemporaryDirectory() as temporary:
            generate_standard_reports(records, Path(temporary))
            throughput_svg = (Path(temporary) / "throughput_concurrency.svg").read_text(
                encoding="utf-8"
            )

        self.assertIn("qwen35-0.8b, target_input_tokens=128", throughput_svg)
        self.assertIn("qwen35-9b, target_input_tokens=128", throughput_svg)

    def test_reports_isolate_experiments_and_skip_single_value_sweeps(self) -> None:
        records = []
        for concurrency, values in ((1, (10.0, 12.0)), (8, (20.0, 22.0))):
            for repetition, throughput in enumerate(values, start=1):
                records.append(
                    {
                        "result_family": "baseline",
                        "served_model_name": "qwen35-0.8b",
                        "server_metrics_available": True,
                        "prefix_cache_hit_ratio": 0.0,
                        "concurrency": concurrency,
                        "target_input_tokens": 128,
                        "target_output_tokens": 32,
                        "tp_size": 1,
                        "chunk_size": 4096,
                        "block_size": 128,
                        "token_throughput": throughput,
                        "ttft_p95_ms": throughput * 2,
                        "tpot_p95_ms": throughput / 2,
                        "repetition": repetition,
                    }
                )
        for ratio, ttft in ((0.0, 100.0), (0.5, 4000.0), (0.9, 120.0)):
            records.append(
                {
                    "result_family": "agent",
                    "served_model_name": "qwen35-0.8b",
                    "shared_prefix_ratio": ratio,
                    "turns": 8,
                    "concurrency": 1,
                    "token_throughput": 10.0,
                    "ttft_p95_ms": ttft,
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            generated = generate_standard_reports(records, Path(temporary))
            names = {path.name for path in generated}
            csv_text = (Path(temporary) / "summary.csv").read_text(encoding="utf-8")

        concurrency_series = _series_for(
            [record for record in records if record["result_family"] == "baseline"],
            x_key="concurrency",
            y_key="token_throughput",
            group_keys=("target_input_tokens",),
        )

        self.assertEqual(
            names,
            {
                "throughput_concurrency.svg",
                "ttft_concurrency.svg",
                "tpot_concurrency.svg",
                "prefix_cache.svg",
                "prefix_throughput.svg",
            },
        )
        self.assertIn("result_family", csv_text)
        self.assertEqual(
            concurrency_series["qwen35-0.8b, target_input_tokens=128"],
            [(1.0, 11.0), (8.0, 21.0)],
        )


if __name__ == "__main__":
    unittest.main()
