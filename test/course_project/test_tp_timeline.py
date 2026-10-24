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

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.course_project import tp_timeline

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "analyze_tp_timeline.py"


def _metadata() -> list[dict]:
    return [
        {"ph": "M", "name": "process_name", "pid": 10, "args": {"name": "Python"}},
        {"ph": "M", "name": "process_name", "pid": 20, "args": {"name": "CANN"}},
        {
            "ph": "M",
            "name": "process_name",
            "pid": 30,
            "args": {"name": "Ascend Hardware"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 30,
            "tid": 1,
            "args": {"name": "Stream 42"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 30,
            "tid": 2,
            "args": {"name": "Stream 45"},
        },
    ]


def _event(
    name: str, start: int | float | str, duration: int | float | str, *, stream: int
) -> dict:
    return {
        "ph": "X",
        "name": name,
        "pid": 30,
        "tid": stream,
        "ts": start,
        "dur": duration,
        "args": {"Task Type": "AI_CORE"},
    }


class TestTpTimeline(unittest.TestCase):
    def _analyze(self, events: list[dict], *, window: tuple) -> dict:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            path = Path(temporary) / "rank0.trace.json"
            path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            return tp_timeline.analyze_tp_timeline(
                {0: path},
                windows_us={0: window},
                clock_ids={0: "rank0-native"},
                collective_name_patterns=(r"^hcclAllReduce",),
            )["ranks"]["0"]

    def test_union_handles_nested_overlapping_and_duplicate_stream_work(self) -> None:
        result = self._analyze(
            [
                _event("MatMul", 0, 10, stream=1),
                _event("Nested", 2, 4, stream=1),
                _event("MatMul duplicate", 0, 10, stream=2),
                _event("Add", 8, 7, stream=2),
                _event("hcclAllReduce", 5, 15, stream=2),
                _event("hcclAllReduce nested", 7, 2, stream=2),
                *_metadata(),
            ],
            window=(0, 30),
        )
        self.assertEqual(result["device_compute_busy_us"], 15)
        self.assertEqual(result["device_collective_busy_us"], 15)
        self.assertEqual(result["compute_collective_overlap_us"], 10)
        self.assertEqual(result["compute_or_collective_busy_us"], 20)
        self.assertEqual(result["uncovered_compute_or_collective_us"], 10)
        self.assertEqual(result["compute_only_us"], 5)
        self.assertEqual(result["collective_only_us"], 5)
        self.assertEqual(result["intervals_us"]["device_compute"], [[0, 15]])
        self.assertEqual(result["intervals_us"]["uncovered"], [[20, 30]])

    def test_cpu_hccl_enqueue_and_cann_call_are_excluded_from_device_work(self) -> None:
        enqueue = _event("hcclAllReduce", 0, 29, stream=1)
        enqueue.update(pid=10, cat="enqueue")
        cann = _event("hcclAllReduce", 0, 28, stream=1)
        cann.update(pid=20)
        waits = [
            {
                "ph": "X",
                "pid": 10,
                "tid": 1,
                "name": "Event::synchronize",
                "ts": 2,
                "dur": 8,
            },
            {
                "ph": "X",
                "pid": 20,
                "tid": 1,
                "name": "AscendCL@aclrtSynchronizeStream",
                "ts": 5,
                "dur": 8,
            },
        ]
        result = self._analyze([enqueue, cann, *waits, *_metadata()], window=(0, 30))
        self.assertEqual(result["device_collective_busy_us"], 0)
        self.assertEqual(result["device_compute_busy_us"], 0)
        self.assertEqual(result["host_wait_busy_us"], 11)
        self.assertEqual(result["host_other_busy_us"], 29)
        self.assertEqual(result["uncovered_compute_or_collective_us"], 30)

    def test_window_clips_intervals_and_excludes_empty_edges(self) -> None:
        result = self._analyze(
            [
                *_metadata(),
                _event("MatMul", 0, 12, stream=1),
                _event("hcclAllReduce", 8, 30, stream=2),
                _event("Add outside", 20, 5, stream=1),
                _event("Zero", 15, 0, stream=1),
            ],
            window=(10, 20),
        )
        self.assertEqual(result["window_duration_us"], 10)
        self.assertEqual(result["device_compute_busy_us"], 2)
        self.assertEqual(result["device_collective_busy_us"], 10)
        self.assertEqual(result["compute_collective_overlap_us"], 2)
        self.assertEqual(result["uncovered_compute_or_collective_us"], 0)
        self.assertEqual(result["intervals_us"]["device_collective"], [[10, 20]])

    def test_unknown_keeps_unconfirmed_device_and_ambiguous_lane_evidence(self) -> None:
        missing_type = _event("MysteryKernel", 1, 4, stream=1)
        missing_type["args"] = {}
        absent_stream = _event("MatMul", 10, 3, stream=99)
        conflicting_host = _event("MatMul", 12, 3, stream=1)
        conflicting_host["cat"] = "cpu_op"
        unconfirmed_collective = _event("hcclAllGather", 17, 3, stream=1)
        result = self._analyze(
            [
                *_metadata(),
                missing_type,
                absent_stream,
                conflicting_host,
                unconfirmed_collective,
            ],
            window=(0, 30),
        )
        self.assertEqual(result["device_compute_busy_us"], 0)
        self.assertEqual(result["device_collective_busy_us"], 0)
        self.assertEqual(result["unknown_busy_us"], 12)
        self.assertEqual(
            [event["name"] for event in result["unknown_events"]],
            ["MysteryKernel", "MatMul", "MatMul", "hcclAllGather"],
        )
        self.assertTrue(all(event["reason"] for event in result["unknown_events"]))

    def test_two_ranks_with_different_clocks_are_never_added(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            paths = {}
            for rank, start in [(0, 0), (1, 1000)]:
                paths[rank] = Path(temporary) / f"rank{rank}.json"
                paths[rank].write_text(
                    json.dumps([*_metadata(), _event("MatMul", start, 10, stream=1)]),
                    encoding="utf-8",
                )
            report = tp_timeline.analyze_tp_timeline(
                paths,
                windows_us={0: (0, 20), 1: (1000, 1020)},
                clock_ids={0: "rank0-clock", 1: "rank1-clock"},
                collective_name_patterns=(),
            )
        self.assertEqual(set(report["ranks"]), {"0", "1"})
        self.assertNotIn("wall_time_us", report)
        for result in report["ranks"].values():
            self.assertEqual(result["device_compute_busy_us"], 10)
            self.assertEqual(result["uncovered_compute_or_collective_us"], 10)

    def test_epoch_decimal_strings_preserve_submicrosecond_durations(self) -> None:
        result = self._analyze(
            [*_metadata(), _event("MatMul", "1792051077986574.428", "1.640", stream=1)],
            window=("1792051077986574.428", "1792051077986576.068"),
        )
        self.assertEqual(result["device_compute_busy_us"], 1.64)
        self.assertEqual(result["uncovered_compute_or_collective_us"], 0)

    def test_malformed_or_missing_complete_event_times_are_rejected(self) -> None:
        invalid_fields = [
            ("ts", -1),
            ("dur", -1),
            ("ts", "NaN"),
            ("dur", float("inf")),
            ("dur", True),
            ("ts", None),
            ("dur", ""),
            ("dur", "1e400"),
        ]
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                event = _event("MatMul", 1, 2, stream=1)
                event[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._analyze([*_metadata(), event], window=(0, 30))
        for field in ("ts", "dur"):
            event = _event("MatMul", 1, 2, stream=1)
            del event[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, field):
                self._analyze([*_metadata(), event], window=(0, 30))

    def test_invalid_window_and_unsupported_duration_phases_are_rejected(self) -> None:
        for window in [(3, 3), (5, 3), (-1, 3), (0, float("nan"))]:
            with self.subTest(window=window), self.assertRaisesRegex(
                ValueError, "window"
            ):
                self._analyze(_metadata(), window=window)
        with self.assertRaisesRegex(ValueError, "phase"):
            self._analyze(
                [*_metadata(), {"ph": "B", "ts": 1, "pid": 30, "tid": 1}],
                window=(0, 30),
            )

    def test_metadata_conflicts_are_rejected_instead_of_guessing(self) -> None:
        conflicting = {
            "ph": "M",
            "name": "process_name",
            "pid": 30,
            "args": {"name": "Python"},
        }
        with self.assertRaisesRegex(ValueError, "metadata"):
            self._analyze([*_metadata(), conflicting], window=(0, 30))

    def test_disjoint_streams_and_touching_boundaries_preserve_gaps(self) -> None:
        result = self._analyze(
            [
                *_metadata(),
                _event("A", 2, 4, stream=1),
                _event("B", 6, 2, stream=1),
                _event("C", 12, 3, stream=1),
                _event("hcclAllReduce first", 4, 2, stream=2),
                _event("hcclAllReduce second", 7, 6, stream=2),
                _event("hcclAllReduce edge", 15, 2, stream=2),
            ],
            window=(0, 20),
        )
        self.assertEqual(result["compute_collective_overlap_us"], 4)
        self.assertEqual(result["intervals_us"]["overlap"], [[4, 6], [7, 8], [12, 13]])
        self.assertEqual(result["intervals_us"]["uncovered"], [[0, 2], [17, 20]])
        self.assertEqual(result["uncovered_compute_or_collective_us"], 5)

    def test_malformed_event_fields_are_rejected_with_value_error(self) -> None:
        for field, value in [
            ("ph", []),
            ("pid", None),
            ("tid", True),
            ("cat", []),
            ("args", []),
        ]:
            event = _event("MatMul", 1, 2, stream=1)
            event[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._analyze([*_metadata(), event], window=(0, 30))
        event = _event("MatMul", 1, 2, stream=1)
        event["args"]["Task Type"] = []
        with self.assertRaisesRegex(ValueError, "Task Type"):
            self._analyze([*_metadata(), event], window=(0, 30))

    def test_bad_times_outside_measurement_window_are_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dur"):
            self._analyze(
                [*_metadata(), _event("MatMul", 100, -1, stream=1)],
                window=(0, 30),
            )

    def test_rank_clock_window_and_pattern_configuration_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            path = Path(temporary) / "trace.json"
            path.write_text(json.dumps(_metadata()), encoding="utf-8")
            for clocks, windows, patterns in [
                ({}, {0: (0, 10)}, ()),
                ({0: "native"}, {}, ()),
                ({0: ""}, {0: (0, 10)}, ()),
                ({0: "native"}, {0: (0, 10)}, ("[",)),
                ({0: "native"}, {0: (0, 10)}, (b"hccl",)),
                ({0: "native"}, {0: (0, 10)}, ("",)),
            ]:
                with self.subTest(clocks=clocks, windows=windows, patterns=patterns):
                    with self.assertRaises(ValueError):
                        tp_timeline.analyze_tp_timeline(
                            {0: path},
                            windows_us=windows,
                            clock_ids=clocks,
                            collective_name_patterns=patterns,
                        )

    def test_malformed_json_and_wrong_trace_container_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            path = Path(temporary) / "bad.json"
            for content in ('[{"ph": "X"}', '{"traceEvents": {}}', '{"events": []}'):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    tp_timeline.analyze_tp_timeline(
                        {0: path},
                        windows_us={0: (0, 10)},
                        clock_ids={0: "native"},
                        collective_name_patterns=(),
                    )

    def test_cli_new_process_writes_per_rank_json_for_pattern_choice(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            directory = Path(temporary)
            options = []
            for rank, start in [(0, 0), (1, 1000)]:
                path = directory / f"rank{rank}.json"
                path.write_text(
                    json.dumps(
                        [
                            *_metadata(),
                            _event("MatMul", start, 10, stream=1),
                            _event("hcclAllReduce", start + 5, 2, stream=2),
                        ]
                    ),
                    encoding="utf-8",
                )
                options.extend(
                    [
                        "--trace",
                        f"{rank}:{path}",
                        "--window",
                        f"{rank}:{start}:{start + 20}",
                        "--clock-id",
                        f"{rank}:rank{rank}-clock",
                    ]
                )
            for label, patterns in [
                ("empty", []),
                ("known", ["^hcclAllReduce", "^other"]),
            ]:
                output = directory / f"{label}.json"
                process = subprocess.run(
                    [
                        sys.executable,
                        str(CLI),
                        *options,
                        "--output",
                        str(output),
                        "--collective-patterns",
                        *patterns,
                    ],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(set(report["ranks"]), {"0", "1"})
                for result in report["ranks"].values():
                    self.assertEqual(result["device_compute_busy_us"], 10)
                    self.assertEqual(
                        result["device_collective_busy_us"], 2 if patterns else 0
                    )

    def test_cli_rejects_missing_choices_and_bad_rank_mappings(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            directory = Path(temporary)
            trace = directory / "trace.json"
            trace.write_text(json.dumps(_metadata()), encoding="utf-8")
            output = directory / "analysis.json"
            options = [
                "--trace",
                f"0:{trace}",
                "--window",
                "0:0:20",
                "--clock-id",
                "0:native",
                "--output",
                str(output),
            ]
            for extra in [
                [],
                ["--collective-patterns", "--trace", f"0:{trace}"],
                ["--collective-patterns", "--clock-id", "1:another-clock"],
                ["--collective-patterns", "--window", "0:1:2"],
                ["--collective-patterns", "["],
            ]:
                with self.subTest(extra=extra):
                    process = subprocess.run(
                        [sys.executable, str(CLI), *options, *extra],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertFalse(output.exists())

    def test_cli_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            directory = Path(temporary)
            trace = directory / "trace.json"
            trace.write_text(json.dumps(_metadata()), encoding="utf-8")
            output = directory / "keep.json"
            output.write_text("keep existing output", encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "--trace",
                    f"0:{trace}",
                    "--window",
                    "0:0:20",
                    "--clock-id",
                    "0:native",
                    "--output",
                    str(output),
                    "--collective-patterns",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("already exists", process.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep existing output")


if __name__ == "__main__":
    unittest.main()
