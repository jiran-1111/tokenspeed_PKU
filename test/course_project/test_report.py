from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from unittest.mock import patch

from benchmarks.report import (
    _series_for,
    generate_standard_reports,
    load_summary_records,
)


class TestCourseReport(unittest.TestCase):
    def test_kv_cache_usage_plot_uses_comparable_baselines_and_medians(self) -> None:
        records = []
        values = {
            "qwen35-0.8b": ((0.2, 0.4), (0.6, 0.8)),
            "qwen35-4b": ((0.1, 0.1), (0.9, 0.9)),
        }
        for model, concurrency_values in values.items():
            for concurrency, repeats in zip((1, 4), concurrency_values):
                for ratio in repeats:
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
                            "kv_cache_usage_ratio_peak": ratio,
                        }
                    )
        decoy = dict(records[0])
        records.extend(
            [
                {**decoy, "result_family": "agent", "kv_cache_usage_ratio_peak": 0.99},
                {
                    **decoy,
                    "target_input_tokens": 1024,
                    "kv_cache_usage_ratio_peak": 0.99,
                },
                {
                    **decoy,
                    "target_output_tokens": 64,
                    "kv_cache_usage_ratio_peak": 0.99,
                },
                {**decoy, "tp_size": 2, "kv_cache_usage_ratio_peak": 0.99},
                {
                    **decoy,
                    "server_metrics_available": False,
                    "kv_cache_usage_ratio_peak": 0.99,
                },
                {
                    **decoy,
                    "prefix_cache_hit_ratio": 0.5,
                    "kv_cache_usage_ratio_peak": 0.99,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generate_standard_reports(records, output)
            svg = (output / "kv_cache_usage.svg").read_text(encoding="utf-8")
        polylines = re.findall(r'<polyline points="([^"]+)"', svg)
        self.assertEqual(len(polylines), 2)
        self.assertEqual(
            polylines, ["90.00,365.00 860.00,165.00", "90.00,465.00 860.00,65.00"]
        )
        self.assertIn("Scheduler KV Active-Page Ratio vs Concurrency", svg)
        self.assertIn("Concurrency", svg)
        self.assertIn("Peak active pages / total pages ratio (not HBM occupancy)", svg)

    def _session_kv_fixture(self, root: Path, *, invalid: str | None = None) -> Path:
        raw_dir = root / "raw"
        raw_dir.mkdir(parents=True)
        common = {
            "status": "completed",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "fixture-revision",
            "frontier_policy": "current-frontier",
            "generic_prefix_caching": False,
            "kvstore_disabled": True,
            "measurement_scope": "serial Engine.generate request latency; not TTFT",
            "launch_configuration": {"model_config": {"tp_size": 1}},
        }
        (raw_dir / "sample.summary.jsonl").write_text(
            json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
        )

        def pair(target, initial, repetition, order, session, stateless):
            return {
                "context_length_target": target,
                "initial_context_tokens": initial,
                "repetition": repetition,
                "execution_order": order,
                "measurement": {
                    "session": {"request_latency_ms": session},
                    "stateless": {"request_latency_ms": stateless},
                    "output_ids_match": True,
                },
            }

        runs = {
            "run-a": [
                pair(512, 541, 1, ["session", "stateless"], 80.0, 100.0),
                pair(512, 542, 2, ["stateless", "session"], 70.0, 100.0),
                pair(1024, 1053, 1, ["session", "stateless"], 90.0, 100.0),
            ],
            "run-b": [
                pair(1024, 1054, 2, ["stateless", "session"], 60.0, 100.0),
                pair(1024, 1055, 3, ["session", "stateless"], 50.0, 100.0),
            ],
        }
        for run_id, measurements in runs.items():
            payload = {**common, "run_id": run_id, "measurements": measurements}
            if invalid == "output_ids_match" and run_id == "run-a":
                payload["measurements"][0]["measurement"]["output_ids_match"] = False
            elif invalid == "request_latency_ms" and run_id == "run-a":
                payload["measurements"][0]["measurement"]["session"][
                    "request_latency_ms"
                ] = float("nan")
            path = raw_dir / f"{run_id}-session-kv-perf-fixture.json"
            path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        return raw_dir

    def _run_report(self, raw_dir: Path, output_dir: Path) -> None:
        from benchmarks import report

        with patch.object(
            sys,
            "argv",
            ["report", "--raw-dir", str(raw_dir), "--output-dir", str(output_dir)],
        ):
            report.main()

    def test_session_kv_cli_generates_paired_csv_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = self._session_kv_fixture(root)
            output = root / "plots"
            self._run_report(raw_dir, output)
            csv_path = output / "session_kv_paired_reduction.csv"
            self.assertTrue(
                csv_path.exists(), "Session KV CSV sidecar was not generated"
            )
            with csv_path.open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            svg = output / "session_kv_paired_reduction.svg"
            tree = ElementTree.parse(svg)
            summary_path = output / "summary.csv"
            with summary_path.open(encoding="utf-8", newline="") as stream:
                summary_records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 5)
        self.assertEqual(
            list(records[0]),
            [
                "run_id",
                "model_id",
                "model_revision",
                "frontier_policy",
                "context_length_target",
                "initial_context_tokens",
                "repetition",
                "execution_order",
                "session_request_latency_ms",
                "stateless_request_latency_ms",
                "paired_reduction_percent",
                "measurement_scope",
            ],
        )
        expected = [
            ("run-a", 512, 541, 1, ["session", "stateless"], 80.0, 100.0),
            ("run-a", 512, 542, 2, ["stateless", "session"], 70.0, 100.0),
            ("run-a", 1024, 1053, 1, ["session", "stateless"], 90.0, 100.0),
            ("run-b", 1024, 1054, 2, ["stateless", "session"], 60.0, 100.0),
            ("run-b", 1024, 1055, 3, ["session", "stateless"], 50.0, 100.0),
        ]
        self.assertEqual(len(summary_records), 1)
        self.assertEqual(summary_records[0]["request_count"], "1")
        self.assertNotIn("session_request_latency_ms", summary_records[0])
        self.assertNotIn("paired_reduction_percent", summary_records[0])
        self.assertNotIn("initial_input_ids", records[0])
        for row, values in zip(records, expected):
            run_id, target, initial, repetition, order, session, stateless = values
            self.assertEqual(row["run_id"], run_id)
            self.assertEqual(row["model_id"], "Qwen/Qwen3.5-4B")
            self.assertEqual(row["model_revision"], "fixture-revision")
            self.assertEqual(row["frontier_policy"], "current-frontier")
            self.assertEqual(int(row["context_length_target"]), target)
            self.assertEqual(int(row["initial_context_tokens"]), initial)
            self.assertEqual(int(row["repetition"]), repetition)
            self.assertEqual(
                row["execution_order"], json.dumps(order, separators=(",", ":"))
            )
            self.assertEqual(float(row["session_request_latency_ms"]), session)
            self.assertEqual(float(row["stateless_request_latency_ms"]), stateless)
            self.assertEqual(
                float(row["paired_reduction_percent"]),
                100.0 * (1.0 - session / stateless),
            )
            self.assertEqual(
                row["measurement_scope"],
                "serial Engine.generate request latency; not TTFT",
            )
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        pairs = tree.findall(".//svg:circle[@class='pair']", namespace)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(
            [float(item.attrib["data-reduction-percent"]) for item in pairs],
            [float(row["paired_reduction_percent"]) for row in records],
        )
        self.assertEqual(
            [
                (item.attrib["data-run-id"], int(item.attrib["data-repetition"]))
                for item in pairs
            ],
            [(row["run_id"], int(row["repetition"])) for row in records],
        )
        context_centers = {
            node.text: float(node.attrib["x"])
            for node in tree.findall(".//svg:text", namespace)
            if node.text in {"512", "1024"}
        }
        group_spacing = abs(context_centers["1024"] - context_centers["512"])
        circle_x = [float(item.attrib["cx"]) for item in pairs]
        for start, count, target in ((0, 2, "512"), (2, 3, "1024")):
            offsets = [
                circle_x[index] - context_centers[target]
                for index in range(start, start + count)
            ]
            self.assertLessEqual(
                max(abs(offset) for offset in offsets), group_spacing * 0.12 + 0.01
            )
            expected_step = group_spacing * 0.24 / (count - 1)
            for left, right in zip(offsets, offsets[1:]):
                self.assertAlmostEqual(right - left, expected_step, delta=0.01)
        medians = tree.findall(".//svg:polygon[@class='median']", namespace)
        self.assertEqual(len(medians), 2)
        self.assertEqual(
            [float(item.attrib["data-median-percent"]) for item in medians],
            [25.0, 40.0],
        )
        text_nodes = [
            " ".join(node.itertext()).strip()
            for node in tree.findall(".//svg:text", namespace)
        ]
        self.assertIn("Session KV paired latency reduction", text_nodes)
        self.assertIn("Target context length", text_nodes)
        self.assertIn("Paired latency reduction (%)", text_nodes)
        self.assertIn("Paired observations (circle)", text_nodes)
        self.assertIn("Context median (diamond)", text_nodes)
        self.assertIn(
            "serial Engine.generate wall time; not TTFT",
            " ".join(text_nodes),
        )

    def test_session_kv_cli_rejects_output_mismatch_and_nan_latency(self) -> None:
        for invalid, field in (
            ("output_ids_match", "output_ids_match"),
            ("request_latency_ms", "request_latency_ms"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw_dir = self._session_kv_fixture(root, invalid=invalid)
                filename = "run-a-session-kv-perf-fixture.json"
                with self.assertRaisesRegex(ValueError, rf"{filename}.*{field}"):
                    self._run_report(raw_dir, root / "plots")

    def test_session_kv_cli_rejects_incompatible_metadata(self) -> None:
        mutations = (
            (("status",), "failed", "status"),
            (("model_id",), "other-model", "model_id"),
            (("frontier_policy",), "legacy", "frontier_policy"),
            (("generic_prefix_caching",), True, "generic_prefix_caching"),
            (("kvstore_disabled",), False, "kvstore_disabled"),
            (("measurement_scope",), "TTFT", "measurement_scope"),
            (("launch_configuration", "model_config", "tp_size"), 2, "tp_size"),
        )
        for key_path, value, field in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw_dir = self._session_kv_fixture(root)
                candidate = raw_dir / "run-a-session-kv-perf-fixture.json"
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                target = payload
                for key in key_path[:-1]:
                    target = target[key]
                target[key_path[-1]] = value
                candidate.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{re.escape(candidate.name)}.*{field}",
                ):
                    self._run_report(raw_dir, root / "plots")

    def test_no_session_kv_input_creates_only_ordinary_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw" / "other"
            raw.mkdir(parents=True)
            (raw / "sample.summary.jsonl").write_text(
                json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
            )
            output = root / "plots"
            self._run_report(root / "raw", output)
            self.assertTrue((output / "summary.csv").exists())
            self.assertFalse((output / "session_kv_paired_reduction.csv").exists())
            self.assertFalse((output / "session_kv_paired_reduction.svg").exists())

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
            summary.write_text(
                json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
            )

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "agent")

    def test_summary_loader_ignores_benchmark_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "benchmark" / "results" / "raw"
            summary = raw_dir / "other" / "sample.summary.jsonl"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
            )

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "other")

    def test_summary_loader_ignores_nested_benchmark_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "results" / "raw"
            summary = raw_dir / "other" / "benchmark" / "run" / "sample.summary.jsonl"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
            )

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "other")

    def test_ablation_summary_loader_keeps_matrix_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            matrix = "qwen35_0.8b_formal_chunk512_20261009"
            summary = raw_dir / "ablation" / matrix / "sample.summary.jsonl"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"request_count": 12}) + "\n", encoding="utf-8"
            )

            records = load_summary_records(raw_dir)

        self.assertEqual(records[0]["result_family"], "ablation")
        self.assertEqual(records[0].get("experiment_matrix"), matrix)

    def test_ablation_matrix_leaf_preserves_matrix_and_generates_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix = "qwen35_0.8b_formal_chunk512_20261009"
            raw_dir = Path(temporary) / "results" / "raw" / "ablation" / matrix
            raw_dir.mkdir(parents=True)
            summary = raw_dir / "sample.summary.jsonl"
            summary.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "experiment_index": index,
                            "variant": f"E{index}",
                            "token_throughput": float(index + 1),
                            "target_input_tokens": 2048,
                            "target_output_tokens": 32,
                            "tp_size": 1,
                        }
                    )
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_summary_records(raw_dir)
            generated = generate_standard_reports(records, raw_dir.parent / "plots")
            names = {path.name for path in generated}
            ablation_svg = (raw_dir.parent / "plots" / "ablation.svg").read_text(
                encoding="utf-8"
            )

        self.assertEqual({record["result_family"] for record in records}, {"ablation"})
        self.assertEqual({record["experiment_matrix"] for record in records}, {matrix})
        self.assertIn("ablation.svg", names)
        self.assertIn(f"experiment_matrix={matrix}", ablation_svg)

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
            ablation_svg = (Path(temporary) / "ablation.svg").read_text(
                encoding="utf-8"
            )
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

    def test_corrected_itl_benchmarks_are_baseline_and_generate_itl_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "results" / "raw"
            run_dir = raw_dir / "benchmark" / "itl_20261009_4b_corrected"
            run_dir.mkdir(parents=True)
            rows = [
                {
                    "request_count": 32,
                    "served_model_name": model,
                    "server_metrics_available": True,
                    "prefix_cache_hit_ratio": 0.0,
                    "concurrency": concurrency,
                    "target_input_tokens": 128,
                    "target_output_tokens": 32,
                    "tp_size": 1,
                    "itl_p50_ms": float(concurrency * scale),
                    "itl_p95_ms": float(concurrency**2 * scale * 2),
                }
                for model, scale in (("qwen35-4b", 1), ("qwen35-9b", 2))
                for concurrency in (1, 4)
            ]
            rows.append(
                {
                    "request_count": 32,
                    "served_model_name": "excluded-itl-model",
                    "server_metrics_available": False,
                    "prefix_cache_hit_ratio": 0.0,
                    "concurrency": 8,
                    "target_input_tokens": 128,
                    "target_output_tokens": 32,
                    "tp_size": 1,
                    "itl_p50_ms": 1000000.0,
                    "itl_p95_ms": 2000000.0,
                }
            )
            (run_dir / "sample.summary.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            records = load_summary_records(raw_dir)
            generated = generate_standard_reports(records, raw_dir.parent / "plots")
            names = {path.name for path in generated}
            p50_svg = (raw_dir.parent / "plots" / "itl_p50_concurrency.svg").read_text(
                encoding="utf-8"
            )
            p95_svg = (raw_dir.parent / "plots" / "itl_p95_concurrency.svg").read_text(
                encoding="utf-8"
            )

        self.assertEqual({record["result_family"] for record in records}, {"baseline"})
        self.assertIn("itl_p50_concurrency.svg", names)
        self.assertIn("itl_p95_concurrency.svg", names)
        for percentile, svg in (("P50", p50_svg), ("P95", p95_svg)):
            self.assertIn(f"Concurrency vs Run-level ITL {percentile}", svg)
            self.assertIn(f"Run-level ITL {percentile} (ms)", svg)
            self.assertIn("Concurrency", svg)
            for model in ("qwen35-4b", "qwen35-9b"):
                self.assertIn(
                    f"{model}, target_input_tokens=128, target_output_tokens=32, tp_size=1",
                    svg,
                )
            self.assertNotIn("excluded-itl-model", svg)

        p50_polylines = re.findall(r'<polyline points="([^"]+)"', p50_svg)
        p95_polylines = re.findall(r'<polyline points="([^"]+)"', p95_svg)
        self.assertEqual(len(p50_polylines), 2)
        self.assertEqual(len(p95_polylines), 2)
        self.assertNotEqual(p50_polylines, p95_polylines)

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
