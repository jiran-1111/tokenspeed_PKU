from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.course_project import profiler_analysis

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestProfilerAnalysis(unittest.TestCase):
    def test_capture_summary_keeps_task_gaps_and_hbm_units_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "PROF_fixture"
            output = capture / "mindstudio_profiler_output"
            output.mkdir(parents=True)
            with (output / "op_summary_fixture.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["Op Name", "OP Type", "Task Type", "aicore_time(us)"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "Op Name": "matmul",
                            "OP Type": "MatMul",
                            "Task Type": "AI_CORE",
                            "aicore_time(us)": "10",
                        },
                        {
                            "Op Name": "matmul",
                            "OP Type": "MatMul",
                            "Task Type": "AI_CORE",
                            "aicore_time(us)": "8",
                        },
                        {
                            "Op Name": "vector",
                            "OP Type": "Vector",
                            "Task Type": "AI_VECTOR_CORE",
                            "aicore_time(us)": "0",
                        },
                    ]
                )

            with (output / "task_time_fixture.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["task_start(us)", "task_stop(us)", "task_time(us)"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "task_start(us)": "0",
                            "task_stop(us)": "3",
                            "task_time(us)": "3",
                        },
                        {
                            "task_start(us)": "2",
                            "task_stop(us)": "4",
                            "task_time(us)": "2",
                        },
                        {
                            "task_start(us)": "6",
                            "task_stop(us)": "30",
                            "task_time(us)": "24",
                        },
                        {
                            "task_start(us)": "10",
                            "task_stop(us)": "11",
                            "task_time(us)": "1",
                        },
                    ]
                )

            sqlite_dir = capture / "device_0" / "sqlite"
            sqlite_dir.mkdir(parents=True)
            database = sqlite3.connect(sqlite_dir / "hbm.db")
            database.execute(
                "CREATE TABLE HBMbwData "
                "(device_id INTEGER, timestamp REAL, bandwidth REAL, hbmid INTEGER, event_type TEXT)"
            )
            database.executemany(
                "INSERT INTO HBMbwData VALUES (0, ?, ?, ?, ?)",
                [
                    (100, 1, 0, "read"),
                    (100, 2, 1, "read"),
                    (100, 3, 2, "read"),
                    (100, 4, 3, "read"),
                    (120, 2, 0, "read"),
                    (120, 2, 1, "read"),
                    (120, 2, 2, "read"),
                    (120, 2, 3, "read"),
                ],
            )
            database.commit()
            database.close()

            summary = profiler_analysis.analyze_capture(capture)

        self.assertEqual(summary["aicore"]["row_count"], 2)
        top_ops = summary["aicore"]["top_ops_by_cumulative_aicore_time"]
        self.assertEqual(top_ops[0]["op_name"], "matmul")
        self.assertEqual(top_ops[0]["total_us"], 18.0)
        self.assertEqual(summary["task_timeline"]["envelope_us"], 30.0)
        self.assertEqual(summary["task_timeline"]["recorded_union_us"], 28.0)
        self.assertEqual(summary["task_timeline"]["uncovered_within_envelope_us"], 2.0)
        self.assertEqual(
            summary["hbm_raw_bandwidth"]["event_types"]["read"][
                "mean_sum_across_channels"
            ],
            9.0,
        )
        self.assertIn(
            "not independently validated", summary["hbm_raw_bandwidth"]["unit"]
        )

    def test_window_overlap_uses_half_open_intersections(self) -> None:
        overlap = getattr(profiler_analysis, "overlap_us", None)
        self.assertTrue(callable(overlap), "window overlap function is missing")
        self.assertEqual(overlap(8.0, 15.0, 10.0, 20.0), 5.0)
        self.assertEqual(overlap(1.0, 9.0, 10.0, 20.0), 0.0)
        self.assertEqual(overlap(20.0, 22.0, 10.0, 20.0), 0.0)
        self.assertEqual(overlap(10.0, 10.0, 10.0, 20.0), 0.0)

    def test_verified_clock_writes_one_row_with_torch_and_cann_window_sums(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            raw_dir, profile_dir = self._write_window_fixture(Path(temporary))

            report = profiler_analysis.analyze_single_request_ttft_window(
                raw_dir, profile_dir
            )
            output_path = profiler_analysis.write_single_request_ttft_window_csv(
                report, Path(temporary) / "analysis"
            )

            with output_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["clock_status"], "verified")
        self.assertEqual(rows[0]["torch_event_count"], "4")
        self.assertEqual(rows[0]["torch_window_event_count"], "2")
        self.assertEqual(
            float(rows[0]["torch_window_per_event_overlap_us_sum"]), 3000.0
        )
        self.assertEqual(rows[0]["cann_interval_count"], "3")
        self.assertEqual(rows[0]["cann_window_interval_count"], "1")
        self.assertEqual(float(rows[0]["cann_window_per_event_overlap_us_sum"]), 3000.0)
        self.assertEqual(
            rows[0]["request_send_epoch_ns"], str(1_800_000_000_000_000_000)
        )
        self.assertEqual(rows[0]["output_tokens"], "1")
        self.assertEqual(rows[0]["stream_completed"], "True")

    def test_clock_drift_over_one_millisecond_keeps_native_ranges_but_no_overlap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            raw_dir, profile_dir = self._write_window_fixture(
                Path(temporary), clock_drift_ns=2_000_000
            )

            report = profiler_analysis.analyze_single_request_ttft_window(
                raw_dir, profile_dir
            )

        self.assertEqual(report["clock_status"], "unverified")
        self.assertGreater(report["max_clock_error_ns"], 1_000_000)
        self.assertIsNotNone(report["torch_native_start_us"])
        self.assertIsNotNone(report["cann_native_start_us"])
        self.assertIsNone(report["torch_window_per_event_overlap_us_sum"])
        self.assertIsNone(report["cann_window_per_event_overlap_us_sum"])

    def test_missing_profiler_monotonic_anchor_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            raw_dir, profile_dir = self._write_window_fixture(
                Path(temporary), missing_profiler_monotonic_anchor=True
            )

            report = profiler_analysis.analyze_single_request_ttft_window(
                raw_dir, profile_dir
            )

        self.assertEqual(report["clock_status"], "unverified")
        self.assertEqual(
            report["clock_status_reason"], "profiler_monotonic_anchor_missing"
        )
        self.assertIsNone(report["torch_window_per_event_overlap_us_sum"])
        self.assertIsNone(report["cann_window_per_event_overlap_us_sum"])

    def test_incomplete_torch_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            raw_dir, profile_dir = self._write_window_fixture(
                Path(temporary), incomplete_trace=True
            )

            with self.assertRaisesRegex(ValueError, "Torch trace"):
                profiler_analysis.analyze_single_request_ttft_window(
                    raw_dir, profile_dir
                )

    def test_missing_trace_or_cann_task_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            raw_dir, profile_dir = self._write_window_fixture(root, include_trace=False)
            with self.assertRaisesRegex(FileNotFoundError, "trace"):
                profiler_analysis.analyze_single_request_ttft_window(
                    raw_dir, profile_dir
                )

            raw_dir, profile_dir = self._write_window_fixture(
                root / "missing-cann", include_cann=False
            )
            with self.assertRaisesRegex(FileNotFoundError, "task_time"):
                profiler_analysis.analyze_single_request_ttft_window(
                    raw_dir, profile_dir
                )

    def test_analysis_csv_refuses_to_overwrite_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            raw_dir, profile_dir = self._write_window_fixture(Path(temporary))
            report = profiler_analysis.analyze_single_request_ttft_window(
                raw_dir, profile_dir
            )
            analysis_dir = Path(temporary) / "analysis"
            analysis_dir.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                profiler_analysis.write_single_request_ttft_window_csv(
                    report, analysis_dir
                )

    def test_profiler_window_cli_help_is_available(self) -> None:
        script = REPO_ROOT / "scripts" / "analyze_profiler_window.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--raw-run-dir", result.stdout)
        self.assertIn("--profile-dir", result.stdout)
        self.assertIn("--analysis-dir", result.stdout)
        self.assertIn("inside the TokenSpeed project", result.stdout)
        self.assertIn("never overwritten", result.stdout)

    def test_profiler_window_cli_rejects_paths_outside_the_project(self) -> None:
        script = REPO_ROOT / "scripts" / "analyze_profiler_window.py"
        outside = REPO_ROOT.parent
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--raw-run-dir",
                str(outside),
                "--profile-dir",
                str(REPO_ROOT),
                "--analysis-dir",
                str(REPO_ROOT / ".tmp" / "must-not-be-created"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside the TokenSpeed project", result.stderr)
        self.assertFalse((REPO_ROOT / ".tmp" / "must-not-be-created").exists())

    def test_profiler_window_cli_refuses_existing_analysis_directory(self) -> None:
        script = REPO_ROOT / "scripts" / "analyze_profiler_window.py"
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            profile_dir = root / "profile"
            analysis_dir = root / "analysis"
            raw_dir.mkdir()
            profile_dir.mkdir()
            analysis_dir.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--raw-run-dir",
                    str(raw_dir),
                    "--profile-dir",
                    str(profile_dir),
                    "--analysis-dir",
                    str(analysis_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def _write_window_fixture(
        self,
        root: Path,
        *,
        clock_drift_ns: int = 0,
        missing_profiler_monotonic_anchor: bool = False,
        incomplete_trace: bool = False,
        include_trace: bool = True,
        include_cann: bool = True,
    ) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        raw_dir = root / "raw"
        raw_dir.mkdir()
        profile_dir = root / "profile"
        profile_dir.mkdir()
        base_epoch_ns = 1_800_000_000_000_000_000
        base_monotonic_ns = 600_000_000_000
        first_epoch_ns = base_epoch_ns + 10_000_000 + clock_drift_ns
        first_monotonic_ns = base_monotonic_ns + 10_000_000
        record = {
            "run_id": "fixture-run",
            "request_id": "single-request-ttft",
            "status": "success",
            "request_send_epoch_ns": base_epoch_ns,
            "request_send_monotonic_ns": base_monotonic_ns,
            "first_chunk_epoch_ns": first_epoch_ns,
            "first_chunk_monotonic_ns": first_monotonic_ns,
            "profiler_start_request_epoch_ns": base_epoch_ns - 5_000_000,
            "profiler_start_response_epoch_ns": base_epoch_ns - 2_000_000,
            "profiler_stop_request_epoch_ns": first_epoch_ns + 100_000,
            "profiler_stop_response_epoch_ns": first_epoch_ns + 500_000,
            "stream_drain_epoch_ns": base_epoch_ns + 30_000_000,
            "stream_drain_monotonic_ns": base_monotonic_ns + 30_000_000,
            "first_chunk_field": "content",
            "ttft_ms": 10.0,
            "e2e_ms": 30.0,
            "input_tokens": 512,
            "output_tokens": 1,
            "stream_completed": True,
            "profiler_start_acknowledged": True,
            "profiler_stop_acknowledged": True,
            "stop_callback_acknowledged": True,
        }
        (raw_dir / "request.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

        if include_trace:
            base_epoch_us = base_epoch_ns // 1000
            first_epoch_us = first_epoch_ns // 1000
            trace_events = [
                {"ph": "X", "ts": base_epoch_us + 1000, "dur": 2000},
                {"ph": "X", "ts": first_epoch_us, "dur": 1000},
                {"ph": "X", "ts": base_epoch_us - 1000, "dur": 1000},
                {"ph": "X", "ts": base_epoch_us - 1000, "dur": 2000},
            ]
            trace_path = profile_dir / "fixture.trace.json"
            if incomplete_trace:
                trace_path.write_text('[{"ph":"X"}', encoding="utf-8")
            else:
                trace_path.write_text(json.dumps(trace_events), encoding="utf-8")

        cann_dir = profile_dir / "cann" / "mindstudio_profiler_output"
        cann_dir.mkdir(parents=True)
        if include_cann:
            base_epoch_us = base_epoch_ns // 1000
            first_epoch_us = first_epoch_ns // 1000
            task_path = cann_dir / "task_time_fixture.csv"
            with task_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["task_start(us)", "task_stop(us)"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "task_start(us)": base_epoch_us + 2000,
                            "task_stop(us)": base_epoch_us + 5000,
                        },
                        {
                            "task_start(us)": first_epoch_us,
                            "task_stop(us)": first_epoch_us + 1000,
                        },
                        {
                            "task_start(us)": base_epoch_us - 1000,
                            "task_stop(us)": base_epoch_us,
                        },
                    ]
                )

        profile_info = {
            "end_info": {
                "collectionTimeEnd": base_epoch_ns + 40_000_000,
                "MonotonicTimeEnd": base_monotonic_ns + 40_000_000,
            }
        }
        if missing_profiler_monotonic_anchor:
            profile_info["end_info"].pop("MonotonicTimeEnd")
        (profile_dir / "cann" / "profiler_info_0.json").write_text(
            json.dumps(profile_info), encoding="utf-8"
        )
        return raw_dir, profile_dir


if __name__ == "__main__":
    unittest.main()
