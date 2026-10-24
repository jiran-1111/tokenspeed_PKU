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

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

import benchmarks.course_project.tp_comparison as tp_comparison
from benchmarks.course_project.tp_comparison import compare_runs

ROOT = Path(__file__).resolve().parents[2]
# Actual whole-card idle response from final_27b/before_processes.txt.
CARD_IDLE = (
    "\tNPU ID                         : 1\n"
    "\tChip Count                     : 2\n\n"
    "\tNo process in device.\n\tChip ID                        : 0\n\n"
    "\tNo process in device.\n\tChip ID                        : 1\n\n"
)
CARD_OTHER_BUSY = (
    "NPU ID : 1\nChip Count : 2\n"
    "No process in device.\nChip ID : 0\n"
    "Process id:1579579 Process name:tokenspeedsch     Process memory(MB):3102\n"
    "Chip ID : 1\n"
)


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_requests(path: Path, rows: list[dict]) -> None:
    path.with_name("requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(root: Path, tp_size: int) -> tuple[Path, dict, list[dict]]:
    root.mkdir()
    config = json.loads((ROOT / "configs/final/qwen35_4b.yaml").read_text())
    config["tp_size"] = tp_size
    protocol = {
        "input_lengths": [4, 8],
        "concurrency": [1, 2],
        "max_new_tokens": 3,
        "warmup_requests": 2,
        "requests_per_cell": 4,
        "repetitions": 3,
        "seed_text": "finite greedy comparison",
        "hbm_interval_seconds": 0.1,
        "engine_port": 18880,
    }
    inputs = {str(n): [1, 2] * (n // 2) for n in protocol["input_lengths"]}
    extension = "tokenspeed_scheduler_ext.cpython-311-x86_64-linux-gnu.so"
    dependencies_sha = _write_json(
        root / "runtime_dependencies.json",
        {
            "external_extensions": {extension: "e" * 64},
        },
    )
    run = {
        "status": "passed",
        "profile_mode": "off",
        "normal_performance_eligible": True,
        "config": config,
        "protocol": protocol,
        "inputs": inputs,
        "source_manifest_sha256": "a" * 64,
        "source_after_mismatches": [],
        "guard": {
            "sources_sha": "a" * 64,
            "sources_after_sha": "a" * 64,
            "after_target_idle": True,
            "sources_unchanged": True,
            "guard_started_ns": 800_000_000,
            "guard_ended_ns": 10_000_000_000,
        },
        "runtime_dependencies_sha256": dependencies_sha,
        "runtime_dependencies_after_sha256": dependencies_sha,
        "runtime_origins_unchanged": True,
        "runtime_origins": {
            name: {"path": source, "sha256": "f" * 64}
            for name, source in (
                (
                    "tokenspeed.runtime.entrypoints.engine",
                    "python/tokenspeed/runtime/entrypoints/engine.py",
                ),
                (
                    "tokenspeed_kernel",
                    "tokenspeed-kernel/python/tokenspeed_kernel/__init__.py",
                ),
                (
                    "tokenspeed_kernel_npu",
                    "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/__init__.py",
                ),
                (
                    "tokenspeed_scheduler",
                    "tokenspeed-scheduler/python/tokenspeed_scheduler/__init__.py",
                ),
            )
        },
        "foreign_processes": [],
        "process_samples": [],
        "device": {
            "npu_id": 1,
            "chip_logic_ids": {str(i): i + 2 for i in range(tp_size)},
        },
        "clock": {
            "name": "perf_counter_ns",
            "resolution_seconds": 1e-9,
            "implementation": "test clock",
        },
        "sampling_params": {"temperature": 0, "ignore_eos": True, "stream_interval": 1},
        "sampler_errors": [],
        "sampler_started_ns": 900_000_000,
        "batches": [],
        "hbm_samples": [],
    }
    run["runtime_origins"]["tokenspeed_scheduler.tokenspeed_scheduler_ext"] = {
        "basename": extension,
        "sha256": "e" * 64,
    }
    for name, value in (("config", config), ("protocol", protocol), ("inputs", inputs)):
        run[f"{name}_sha256"] = _write_json(root / f"{name}.json", value)
    rows = []
    current = 1_000_000_000
    for length in protocol["input_lengths"]:
        for concurrency in protocol["concurrency"]:
            for phase, repetition, count in [("warmup", 0, 2)] + [
                ("formal", r, 4) for r in range(1, 4)
            ]:
                duration = (
                    ((count + concurrency - 1) // concurrency) * 20_000_000 + 2_000_000
                ) // tp_size
                batch = {
                    "input_length": length,
                    "concurrency": concurrency,
                    "phase": phase,
                    "repetition": repetition,
                    "status": "passed",
                    "requests": count,
                    "output_tokens": count * 3,
                    "started_ns": current,
                    "finished_ns": current + duration,
                    "wall_ns": duration,
                    "output_tokens_per_second": count * 3e9 / duration,
                }
                run["batches"].append(batch)
                for index in range(count):
                    started = (
                        current
                        + (1_000_000 + (index // concurrency) * 20_000_000) // tp_size
                    )
                    first = started + 5_000_000 // tp_size
                    finished = started + 18_000_000 // tp_size
                    chunks = [
                        {"observed_ns": first, "output_tokens": 1},
                        {
                            "observed_ns": finished - 1_000_000 // tp_size,
                            "output_tokens": 2,
                        },
                    ]
                    rows.append(
                        {
                            "input_length": length,
                            "concurrency": concurrency,
                            "phase": phase,
                            "repetition": repetition,
                            "request_index": index,
                            "status": "passed",
                            "error": None,
                            "input_ids": inputs[str(length)],
                            "output_ids": [10, 11, 12],
                            "started_ns": started,
                            "finished_ns": finished,
                            "ttft_ms": (first - started) / 1e6,
                            "e2e_ms": (finished - started) / 1e6,
                            "chunks": chunks,
                            "inter_chunk_ms": [
                                (chunks[1]["observed_ns"] - first) / 1e6
                            ],
                        }
                    )
                chips = {
                    str(i): {
                        "used_mb": 1000 if tp_size == 1 else 600 + i * 100,
                        "capacity_mb": 65536,
                    }
                    for i in range(tp_size)
                }
                raw_name = f'npu_smi_{len(run["hbm_samples"]):05d}.txt'
                table = "| 1 Ascend910 | OK | ignored |\n" + "".join(
                    f"| {i} {i + 2} | 0000:0{i + 2}:00.0 | {chips[str(i)]['used_mb']} / 65536 |\n"
                    for i in range(tp_size)
                )
                table += (
                    "| NPU Chip | Process id | Process name | Process memory(MB) |\n"
                    "+===+===+===+===+\n"
                    + "".join(
                        f"| 1 {i} | {110 + i} | scheduler | 112 |\n"
                        for i in range(tp_size)
                    )
                    + "+===+===+===+===+\n"
                )
                (root / raw_name).write_text(table, encoding="utf-8")
                run["hbm_samples"].append(
                    {
                        "started_ns": current + 500_000,
                        "finished_ns": current + duration - 500_000,
                        "raw_table": raw_name,
                        "chips": chips,
                    }
                )
                run["process_samples"].append(
                    {
                        "observed_ns": current + duration // 2,
                        "root_pid": 100,
                        "raw_table": raw_name,
                        "chips": {
                            str(i): {
                                "pids": [110 + i],
                                "foreign_pids": [],
                                "unknown_pids": [],
                                "parents": {str(110 + i): 100},
                            }
                            for i in range(tp_size)
                        },
                    }
                )
                current += duration + 10_000_000
    run["sampler_finished_ns"] = current + 10_000_000
    path = root / "run.json"
    _write_json(path, run)
    _write_requests(path, rows)
    return path, run, rows


@pytest.fixture
def pair(tmp_path: Path):
    return _fixture(tmp_path / "tp1", 1), _fixture(tmp_path / "tp2", 2)


def _release_fixture(path: Path, run: dict, *, pending: bool) -> dict:
    root_pid = run["process_samples"][0]["root_pid"]
    chips = run["device"]["chip_logic_ids"]
    run["guard"].update(
        after_target_idle=not pending,
        parent_context_pending_exit=pending,
        engine_children_released=True,
        after_process_pids={
            chip: [root_pid] if pending and chip == "0" else [] for chip in chips
        },
        parent_pid=root_pid,
    )
    if pending:
        run["guard"]["after_npu_smi"] = "after_npu_smi.txt"
        path.with_name("after_npu_smi.txt").write_text(
            "| NPU Chip | Process id | Process name | Process memory(MB) |\n"
            "+===+===+===+===+\n"
            f"| 1 0 | {root_pid} | python | 112 |\n"
            "+===+===+===+===+\n"
        )
    _write_json(path, run)
    mapping = "driver_after_mapping.txt"
    path.with_name(mapping).write_text(
        "".join(f"1 {chip} {logic} 0 Ascend910\n" for chip, logic in chips.items())
    )
    process_files = {chip: f"driver_after_processes_chip{chip}.txt" for chip in chips}
    for chip, name in process_files.items():
        path.with_name(name).write_text(CARD_IDLE)
    release = {
        "run_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_manifest_sha256": run["source_manifest_sha256"],
        "child_exit_code": 0,
        "devices_released": True,
        "after_mapping": mapping,
        "after_processes": process_files,
        "observed_after_exit_utc": "2026-10-24T00:00:00Z",
        "npu_id": run["device"]["npu_id"],
        "chip_logic_ids": copy.deepcopy(chips),
        "pid": root_pid,
    }
    _write_json(path.with_name("device_release.json"), release)
    return release


@pytest.mark.parametrize("pending", [False, True])
def test_release_sidecar_verifies_parent_context_after_process_exit_without_rewriting_run(
    pair, pending
):
    (one, _, _), (two, run, _) = pair
    _release_fixture(two, run, pending=pending)
    original = two.read_bytes()
    result = compare_runs(one, two)
    assert result["execution_valid"] is True, result["execution_errors"]
    assert result["performance_valid"] is True
    assert two.read_bytes() == original


@pytest.mark.parametrize(
    "fault",
    [
        "missing_sidecar",
        "run_sha",
        "source_sha",
        "device_identity",
        "mapping",
        "busy",
        "ambiguous_idle",
        "wrong_chip",
        "exit_code",
        "bool_exit_code",
        "not_released",
        "pid",
        "nonroot_after_pid",
        "pending_flag",
        "children_flag",
        "empty_pending",
        "timestamp",
        "process_keys",
        "escaping_file",
        "guard_raw_pid",
    ],
)
def test_release_sidecar_and_raw_evidence_faults_suppress_performance(pair, fault):
    (one, _, _), (two, run, _) = pair
    release = _release_fixture(two, run, pending=True)
    if fault == "source_sha":
        release["source_manifest_sha256"] = "b" * 64
    elif fault == "device_identity":
        release["npu_id"] = 0
    elif fault == "mapping":
        two.with_name(release["after_mapping"]).write_text(
            "0 0 2 0 Ascend910\n0 1 3 0 Ascend910\n"
        )
    elif fault == "busy":
        two.with_name(release["after_processes"]["0"]).write_text(
            "NPU ID : 1\n| 100 | python | 112 |\nChip ID : 0\n"
        )
    elif fault == "ambiguous_idle":
        two.with_name(release["after_processes"]["0"]).write_text(
            "NPU ID : 1\nNo process in device.\nChip ID : 0\nNo process in device.\nChip ID : 0\n"
        )
    elif fault == "wrong_chip":
        two.with_name(release["after_processes"]["0"]).write_text(
            "NPU ID : 1\nNo process in device.\nChip ID : 1\n"
        )
    elif fault == "exit_code":
        release["child_exit_code"] = 1
    elif fault == "bool_exit_code":
        release["child_exit_code"] = False
    elif fault == "not_released":
        release["devices_released"] = False
    elif fault == "pid":
        release["pid"] = 999
    elif fault == "nonroot_after_pid":
        run["guard"]["after_process_pids"]["0"] = [110]
    elif fault == "pending_flag":
        run["guard"]["parent_context_pending_exit"] = False
    elif fault == "children_flag":
        run["guard"]["engine_children_released"] = False
    elif fault == "empty_pending":
        run["guard"]["after_process_pids"] = {"0": [], "1": []}
    elif fault == "timestamp":
        release["observed_after_exit_utc"] = "2026-10-24T08:00:00+08:00"
    elif fault == "process_keys":
        release["after_processes"].pop("1")
    elif fault == "escaping_file":
        release["after_processes"]["0"] = "../outside.txt"
    elif fault == "guard_raw_pid":
        raw = two.with_name("after_npu_smi.txt")
        raw.write_text(raw.read_text().replace("| 1 0 | 100 |", "| 1 0 | 110 |"))
    _write_json(two, run)
    release["run_json_sha256"] = (
        "0" * 64 if fault == "run_sha" else hashlib.sha256(two.read_bytes()).hexdigest()
    )
    _write_json(two.with_name("device_release.json"), release)
    if fault == "missing_sidecar":
        two.with_name("device_release.json").unlink()
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


def test_modern_idle_guard_requires_sidecar_and_legacy_idle_guard_validates_one_when_present(
    pair,
):
    (one, _, _), (two, run, _) = pair
    release = _release_fixture(two, run, pending=False)
    two.with_name("device_release.json").unlink()
    assert compare_runs(one, two)["execution_valid"] is False
    for name in (
        "parent_context_pending_exit",
        "engine_children_released",
        "after_process_pids",
        "parent_pid",
    ):
        run["guard"].pop(name)
    _write_json(two, run)
    release["run_json_sha256"] = "0" * 64
    _write_json(two.with_name("device_release.json"), release)
    assert compare_runs(one, two)["execution_valid"] is False


@pytest.mark.parametrize("format", ["whole_card", "legacy_single_chip"])
def test_release_accepts_real_whole_card_idle_and_legacy_single_chip_evidence(
    pair, format
):
    (one, _, _), (two, run, _) = pair
    release = _release_fixture(two, run, pending=True)
    if format == "whole_card":
        name = "driver_after_processes_card.txt"
        two.with_name(name).write_text(CARD_IDLE)
        release["after_processes"] = {chip: name for chip in release["after_processes"]}
        _write_json(two.with_name("device_release.json"), release)
    else:
        for chip, name in release["after_processes"].items():
            two.with_name(name).write_text(
                f"NPU ID : 1\nNo process in device.\nChip ID : {chip}\n"
            )
    result = compare_runs(one, two)
    assert result["execution_valid"] is True, result["execution_errors"]


def test_release_ignores_known_busy_processes_on_an_unselected_chip(pair):
    (one, run, _), (two, _, _) = pair
    release = _release_fixture(one, run, pending=False)
    one.with_name(release["after_processes"]["0"]).write_text(CARD_OTHER_BUSY)
    result = compare_runs(one, two)
    assert result["execution_valid"] is True, result["execution_errors"]


def test_release_rejects_unknown_process_lines_on_an_unselected_chip(pair):
    (one, run, _), (two, _, _) = pair
    release = _release_fixture(one, run, pending=False)
    one.with_name(release["after_processes"]["0"]).write_text(
        CARD_OTHER_BUSY.replace("Process memory(MB):3102", "Process memory(MB):unknown")
    )
    assert compare_runs(one, two)["execution_valid"] is False


@pytest.mark.parametrize(
    "table",
    [
        CARD_IDLE.replace(
            "Chip ID                        : 1", "Chip ID                        : 9"
        ),
        CARD_IDLE.replace(
            "Chip Count                     : 2", "Chip Count                     : 3"
        )
        + "No process in device.\nChip ID : 2\n",
    ],
)
def test_release_rejects_impossible_whole_card_chip_identity_for_tp1(pair, table):
    (one, run, _), (two, _, _) = pair
    release = _release_fixture(one, run, pending=False)
    one.with_name(release["after_processes"]["0"]).write_text(table)
    assert compare_runs(one, two)["execution_valid"] is False


@pytest.mark.parametrize(
    "table",
    [
        CARD_OTHER_BUSY,
        CARD_IDLE.replace(
            "NPU ID                         : 1", "NPU ID                         : 0"
        ),
        "NPU ID : 1\n" + CARD_IDLE,
        CARD_IDLE.replace(
            "Chip Count                     : 2", "Chip Count                     : 3"
        ),
        CARD_IDLE.replace(
            "Chip Count                     : 2", "Chip Count                     : 0"
        ),
        CARD_IDLE.replace(
            "Chip ID                        : 1", "Chip ID                        : 0"
        ),
        "NPU ID : 1\nChip Count : 2\nNo process in device.\nChip ID : 0\n",
        CARD_IDLE + "unknown tail\n",
        CARD_IDLE.replace("No process in device.", "unrecognized process status", 1),
        CARD_IDLE.replace(
            "No process in device.",
            "No process in device.\nProcess id:123 Process name:python Process memory(MB):112",
            1,
        ),
        "No process in device.\nChip ID : 0\n",
    ],
)
def test_release_rejects_busy_selected_chip_and_ambiguous_whole_card_evidence(
    pair, table
):
    (one, _, _), (two, run, _) = pair
    release = _release_fixture(two, run, pending=True)
    for name in release["after_processes"].values():
        two.with_name(name).write_text(table)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


def test_recomputes_formal_throughput_ttft_and_observed_hbm(pair):
    (one, _, _), (two, _, _) = pair
    result = compare_runs(one, two)
    assert result["execution_valid"] is True
    assert result["sequence_agreement"] is True
    assert result["hbm_coverage_complete"] is True
    assert result["sequence_comparison"]["compared_requests"] == 56
    assert len(result["performance"]) == 4
    first = result["performance"][0]
    assert first["tp1"]["throughput"]["median"] == pytest.approx(12e9 / 82_000_000)
    assert first["tp2"]["throughput"]["median"] == pytest.approx(12e9 / 41_000_000)
    assert first["tp1"]["ttft_ms"]["p50"] == 5
    assert first["tp2"]["ttft_ms"]["p50"] == 2.5
    assert first["throughput_ratio_tp2_over_tp1"] == 2
    assert result["runs"]["tp2"]["hbm_observed_max_mb_per_chip"] == {"0": 600, "1": 700}
    assert "instantaneous" in " ".join(result["limitations"])


def test_all_output_differences_are_retained_without_invalidating_execution(pair):
    (one, _, _), (two, _, rows) = pair
    rows[0]["output_ids"][1] = 99
    rows[-1]["output_ids"][2] = 98
    _write_requests(two, rows)
    result = compare_runs(one, two)
    assert result["execution_valid"] is True
    assert result["sequence_agreement"] is False
    differences = result["sequence_comparison"]["mismatches"]
    assert len(differences) == 2
    assert {row["first_difference_index"] for row in differences} == {1, 2}
    assert differences[0]["tp1_output_ids"] == [10, 11, 12]
    assert result["performance"][0]["throughput_ratio_tp2_over_tp1"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_manifest_sha256", "b" * 64),
        ("source_after_mismatches", ["changed.py"]),
        ("sampler_errors", [{"error": "sampling failed"}]),
        ("status", "failed"),
    ],
)
def test_rejects_invalid_provenance_and_failed_collection(pair, field, value):
    (one, _, _), (two, run, _) = pair
    run[field] = value
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["execution_errors"]


def test_config_change_besides_tp_is_not_a_matched_comparison(pair):
    (one, _, _), (two, run, _) = pair
    run["config"]["max_num_seqs"] = 4
    run["config_sha256"] = _write_json(two.with_name("config.json"), run["config"])
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert any("config" in error for error in result["execution_errors"])


@pytest.mark.parametrize("name", ["config", "protocol", "inputs"])
def test_hash_cannot_hide_a_changed_saved_json(pair, name):
    (one, _, _), (two, _, _) = pair
    two.with_name(f"{name}.json").write_text("{}\n")
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert any(name in error for error in result["execution_errors"])


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "input", "short", "bool_token", "time", "chunk", "failed"],
)
def test_request_faults_do_not_pass_finite_sequence_validation(pair, fault):
    (one, _, _), (two, _, rows) = pair
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif fault == "input":
        rows[-1]["input_ids"] = [99] * 8
    elif fault == "short":
        rows[-1]["output_ids"].pop()
    elif fault == "bool_token":
        rows[-1]["output_ids"][0] = True
    elif fault == "time":
        rows[-1]["finished_ns"] = rows[-1]["started_ns"] - 1
    elif fault == "chunk":
        rows[-1]["chunks"][0]["observed_ns"] = rows[-1]["finished_ns"] + 1
    else:
        rows[-1].update(status="failed", error="partial stream")
    _write_requests(two, rows)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["execution_errors"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_batch",
        "wall",
        "throughput",
        "output_count",
        "sampler_time",
        "missing_chip",
        "bad_raw",
    ],
)
def test_batch_or_chip_evidence_faults_are_rejected(pair, fault):
    (one, _, _), (two, run, _) = pair
    if fault == "missing_batch":
        run["batches"].pop()
    elif fault == "wall":
        run["batches"][-1]["wall_ns"] = 1
    elif fault == "throughput":
        run["batches"][-1]["output_tokens_per_second"] = 1e30
    elif fault == "output_count":
        run["batches"][-1]["output_tokens"] = 1
    elif fault == "sampler_time":
        run["sampler_finished_ns"] = run["batches"][-1]["started_ns"]
    elif fault == "missing_chip":
        run["hbm_samples"][-1]["chips"].pop("1")
    else:
        two.with_name(run["hbm_samples"][-1]["raw_table"]).write_text("no chip data")
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["execution_errors"]


def test_boundary_crossing_hbm_query_is_not_assigned_as_batch_peak(pair):
    (one, _, _), (two, run, _) = pair
    sample = run["hbm_samples"][-1]
    sample["started_ns"] = run["batches"][-1]["started_ns"] - 1
    sample["finished_ns"] = run["batches"][-1]["finished_ns"] + 1
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is True
    assert result["hbm_coverage_complete"] is False
    assert result["runs"]["tp2"]["batches"][-1]["hbm_observed_max_mb_per_chip"] is None


def test_no_formal_metric_uses_warmup_observations(pair):
    (one, _, rows), (two, _, _) = pair
    for row in rows:
        if row["phase"] == "warmup":
            row["chunks"][0]["observed_ns"] = row["started_ns"] + 1
            row["ttft_ms"] = 1e-6
            row["inter_chunk_ms"] = [
                (row["chunks"][1]["observed_ns"] - row["chunks"][0]["observed_ns"])
                / 1e6
            ]
    _write_requests(one, rows)
    result = compare_runs(one, two)
    assert result["execution_valid"] is True
    assert all(cell["tp1"]["ttft_ms"]["p50"] == 5 for cell in result["performance"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("after_target_idle", False),
        ("sources_unchanged", False),
        ("sources_sha", "c" * 64),
        ("guard_ended_ns", 1),
    ],
)
def test_guard_failures_are_not_successful_execution(pair, field, value):
    (one, _, _), (two, run, _) = pair
    run["guard"][field] = value
    _write_json(two, run)
    assert compare_runs(one, two)["execution_valid"] is False


def test_greedy_sampling_is_required_even_when_outputs_happen_to_match(pair):
    (one, _, _), (two, run, _) = pair
    run["sampling_params"]["temperature"] = 1
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert any("sampling_params" in error for error in result["execution_errors"])


def test_malformed_output_is_reported_without_crashing_or_claiming_agreement(pair):
    (one, _, _), (two, _, rows) = pair
    rows[-1]["output_ids"] = True
    _write_requests(two, rows)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["sequence_agreement"] is None
    assert result["sequence_comparison"]["mismatches"][-1]["tp2_output_ids"] is True


def test_profiled_collection_does_not_enter_ordinary_performance_comparison(pair):
    (one, _, _), (two, run, _) = pair
    run.update(profile_mode="all-batches", normal_performance_eligible=False)
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance_valid"] is False
    assert result["performance"] == []


def test_invalid_provenance_suppresses_performance_ratios(pair):
    (one, _, _), (two, run, _) = pair
    run["source_after_mismatches"] = ["python/changed.py"]
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance_valid"] is False
    assert result["performance"] == []


def test_real_cpu_matrix_artifact_matches_the_comparison_schema(tmp_path: Path):
    from benchmarks.course_project.tp_matrix import run_matrix

    class Engine:
        def generate(self, **kwargs):
            yield {"output_ids": [10]}
            time.sleep(0.003)
            yield {"output_ids": [11, 12]}

    paths = []
    for tp_size in (1, 2):
        template, run, _ = _fixture(tmp_path / f"template{tp_size}", tp_size)
        output = tmp_path / f"actual{tp_size}"
        output.mkdir()
        run["protocol"]["hbm_interval_seconds"] = 0.001
        for name in ("config", "protocol", "inputs"):
            run[f"{name}_sha256"] = _write_json(output / f"{name}.json", run[name])
        (output / "runtime_dependencies.json").write_bytes(
            template.with_name("runtime_dependencies.json").read_bytes()
        )
        table = template.with_name(run["hbm_samples"][0]["raw_table"]).read_text()
        process_chips = copy.deepcopy(run["process_samples"][0]["chips"])
        run["process_samples"] = []

        def query_hbm():
            run["process_samples"].append(
                {
                    "observed_ns": time.perf_counter_ns(),
                    "root_pid": 100,
                    "raw_table": f'npu_smi_{len(run["hbm_samples"]):05d}.txt',
                    "chips": copy.deepcopy(process_chips),
                }
            )
            return table

        run["guard"]["guard_started_ns"] = time.perf_counter_ns()
        run_matrix(
            Engine(),
            protocol=run["protocol"],
            inputs=run["inputs"],
            output_dir=output,
            npu_id=1,
            chip_logic_ids={i: i + 2 for i in range(tp_size)},
            query_hbm=query_hbm,
            record=run,
        )
        run["guard"]["guard_ended_ns"] = time.perf_counter_ns()
        path = output / "run.json"
        _write_json(path, run)
        paths.append(path)
    result = compare_runs(*paths)
    assert result["execution_valid"] is True, result["execution_errors"]
    assert result["sequence_agreement"] is True
    assert len(result["performance"]) == 4


def test_tp1_baseline_must_use_one_of_the_tp2_target_chips(pair):
    (one, run, _), (two, _, _) = pair
    run["device"]["npu_id"] = 0
    _write_json(one, run)
    assert compare_runs(one, two)["execution_valid"] is False


@pytest.mark.parametrize("payload", ["[]", "invalid JSON", "null"])
def test_malformed_run_artifact_is_reported_as_invalid(pair, payload):
    (one, _, _), (two, _, _) = pair
    two.write_text(payload)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["execution_errors"]


@pytest.mark.parametrize("field", ["config", "device", "guard"])
def test_malformed_embedded_object_is_rejected_without_crashing(pair, field):
    (one, _, _), (two, run, _) = pair
    run[field] = []
    _write_json(two, run)
    assert compare_runs(one, two)["execution_valid"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_dependencies_after_sha256", "b" * 64),
        ("runtime_dependencies_sha256", "not-a-sha"),
        ("runtime_origins_unchanged", False),
        ("foreign_processes", [{"pid": 999, "status": "foreign"}]),
    ],
)
def test_runtime_provenance_and_foreign_process_faults_suppress_ratios(
    pair, field, value
):
    (one, _, _), (two, run, _) = pair
    run[field] = value
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


@pytest.mark.parametrize(
    "fault",
    [
        "missing_module",
        "absolute_path",
        "escaping_path",
        "bad_sha",
        "wrong_external_module",
        "extension_sha",
        "ambiguous_origin",
        "dependencies_bytes",
        "dependencies_missing",
    ],
)
def test_runtime_origin_and_dependency_evidence_is_checked(pair, fault):
    (one, _, _), (two, run, _) = pair
    origins = run["runtime_origins"]
    if fault == "missing_module":
        origins.pop("tokenspeed_kernel")
    elif fault == "absolute_path":
        origins["tokenspeed_kernel"]["path"] = "/old-editable/kernel.py"
    elif fault == "escaping_path":
        origins["tokenspeed_kernel"]["path"] = "../old-editable/kernel.py"
    elif fault == "bad_sha":
        origins["tokenspeed_kernel"]["sha256"] = True
    elif fault == "wrong_external_module":
        origins["tokenspeed_kernel"] = copy.deepcopy(
            origins["tokenspeed_scheduler.tokenspeed_scheduler_ext"]
        )
    elif fault == "extension_sha":
        origins["tokenspeed_scheduler.tokenspeed_scheduler_ext"]["sha256"] = "0" * 64
    elif fault == "ambiguous_origin":
        origins["tokenspeed_kernel"]["basename"] = "kernel.py"
    elif fault == "dependencies_bytes":
        two.with_name("runtime_dependencies.json").write_text("{}\n")
    else:
        two.with_name("runtime_dependencies.json").unlink()
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


@pytest.mark.parametrize(
    "fault",
    [
        "missing_sample",
        "extra_sample",
        "raw_name",
        "time",
        "root",
        "pid",
        "foreign",
        "unknown",
        "parents_missing",
        "parents_cycle",
        "parents_foreign",
        "raw_extra_pid",
        "sample_error",
    ],
)
def test_process_samples_must_match_raw_pids_and_ancestry(pair, fault):
    (one, _, _), (two, run, _) = pair
    sample = run["process_samples"][-1]
    chip = sample["chips"]["0"]
    if fault == "missing_sample":
        run["process_samples"].pop()
    elif fault == "extra_sample":
        run["process_samples"].append(copy.deepcopy(sample))
    elif fault == "raw_name":
        sample["raw_table"] = "npu_smi_00000.txt"
    elif fault == "time":
        sample["observed_ns"] = run["hbm_samples"][-1]["finished_ns"] + 1
    elif fault == "root":
        sample["root_pid"] = 999
    elif fault == "pid":
        chip["pids"] = [999]
    elif fault in {"foreign", "unknown"}:
        chip[f"{fault}_pids"] = [110]
    elif fault == "parents_missing":
        chip["parents"] = {}
    elif fault == "parents_cycle":
        chip["parents"] = {"110": 120, "120": 110}
    elif fault == "parents_foreign":
        chip["parents"] = {"110": 1}
    elif fault == "raw_extra_pid":
        raw = two.with_name(sample["raw_table"])
        raw.write_text(raw.read_text().replace("| 1 0 | 110 |", "| 1 0 | 999 |"))
    else:
        sample["error"] = "unknown process"
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


def test_paired_runtime_dependencies_must_match_even_when_each_snapshot_is_valid(pair):
    (one, _, _), (two, run, _) = pair
    dependency_path = two.with_name("runtime_dependencies.json")
    dependencies = json.loads(dependency_path.read_text())
    dependencies["environment"] = "different deployment"
    sha = _write_json(dependency_path, dependencies)
    run.update(runtime_dependencies_sha256=sha, runtime_dependencies_after_sha256=sha)
    _write_json(two, run)
    result = compare_runs(one, two)
    assert result["execution_valid"] is False
    assert result["performance"] == []


def test_public_single_run_audit_supports_tp2_without_exposing_internal_records(pair):
    (_, _, _), (two, _, _) = pair
    result = tp_comparison.audit_run(two, expected_tp=2)
    assert result["execution_valid"] is True
    assert result["evidence_complete"] is True
    assert len(result["cells"]) == 4
    assert "record" not in result and "indexed" not in result
    assert "unsigned" in " ".join(result["limitations"])


def test_public_single_run_audit_suppresses_cells_when_evidence_is_invalid(pair):
    (_, _, _), (two, run, _) = pair
    run["runtime_origins_unchanged"] = False
    _write_json(two, run)
    result = tp_comparison.audit_run(two, expected_tp=2)
    assert result["execution_valid"] is False
    assert result["evidence_complete"] is False
    assert result["cells"] == []
