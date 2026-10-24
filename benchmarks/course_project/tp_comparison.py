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

"""Audit paired TP matrix artifacts without running models or trusting summaries."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from statistics import fmean, median

from benchmarks.course_project.tp_devices import (
    is_descendant,
    parse_mapping,
    target_process_pids,
    validate_target,
)

_BATCH_FIELDS = ("phase", "input_length", "concurrency", "repetition")
_RUNTIME_MODULES = {
    "tokenspeed.runtime.entrypoints.engine",
    "tokenspeed_kernel",
    "tokenspeed_kernel_npu",
    "tokenspeed_scheduler",
    "tokenspeed_scheduler.tokenspeed_scheduler_ext",
}
_PROVENANCE_LIMITATION = (
    "These unsigned artifacts establish internal consistency only; saved module origins, "
    "hashes and process-parent observations do not authenticate historical binaries or "
    "protect against coordinated rewriting of the evidence."
)


def _key(row: dict, *, request: bool) -> tuple:
    fields = _BATCH_FIELDS + (("request_index",) if request else ())
    key = tuple(row[field] for field in fields)
    if not isinstance(key[0], str) or any(type(n) is not int for n in key[1:]):
        raise ValueError("invalid batch/request key")
    return key


def _ids(value: object) -> bool:
    return isinstance(value, list) and all(type(n) is int and n >= 0 for n in value)


def _time(value: object) -> bool:
    return type(value) is int and value > 0


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _close(value: object, expected: float) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-6)
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _raw_hbm(path: Path, *, npu_id: int, chips: dict[str, int]) -> dict:
    values, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(columns) != 3:
            continue
        card = re.fullmatch(r"(\d+)\s+Ascend910", columns[0])
        if card:
            current = int(card[1])
        location = re.fullmatch(r"(\d+)\s+(\d+)", columns[0])
        if current != npu_id or location is None:
            continue
        chip, logical = location.groups()
        if chip not in chips or int(logical) != chips[chip]:
            continue
        usage = re.search(r"(\d+)\s*/\s*(\d+)\s*$", columns[2])
        if usage is None or chip in values:
            raise ValueError("missing or duplicate raw HBM value")
        values[chip] = dict(zip(("used_mb", "capacity_mb"), map(int, usage.groups())))
    return values


def _require_raw_chip_idle(table: str, *, npu_id: int, chip_id: int) -> None:
    """Read proc-mem bodies terminated by Chip ID, allowing other chips to be busy."""
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    if not lines or re.fullmatch(rf"NPU ID\s*:\s*{npu_id}", lines[0]) is None:
        raise ValueError("device release process card identity is missing or wrong")
    body_start, declared_count = 1, None
    if len(lines) > 1 and lines[1].startswith("Chip Count"):
        count = re.fullmatch(r"Chip Count\s*:\s*([1-9]\d*)", lines[1])
        if count is None or int(count[1]) not in (1, 2):
            raise ValueError("device release process chip count is invalid")
        declared_count, body_start = int(count[1]), 2
    sections, body = {}, []
    process_pattern = (
        r"Process id\s*:\s*[1-9]\d*\s+Process name\s*:\s*\S.*?\s+"
        r"Process memory\(MB\)\s*:\s*\d+"
    )
    for line in lines[body_start:]:
        marker = re.fullmatch(r"Chip ID\s*:\s*(\d+)", line)
        if marker is None:
            body.append(line)
            continue
        chip = int(marker[1])
        if (
            chip not in {0, 1}
            or chip in sections
            or not body
            or (
                body != ["No process in device."]
                and not all(re.fullmatch(process_pattern, row) for row in body)
            )
        ):
            raise ValueError("device release process chip body is duplicate or unknown")
        sections[chip], body = body, []
    if (
        body
        or not sections
        or declared_count is not None
        and set(sections) != set(range(declared_count))
    ):
        raise ValueError(
            "device release process response has missing sections or unknown tail"
        )
    if sections.get(chip_id) != ["No process in device."]:
        raise ValueError("device release selected chip is busy or missing")


def _audit_device_release(
    path: Path, run: dict, *, run_sha: str, root_pid: int | None, target: dict[str, int]
) -> None:
    """Check post-exit evidence without changing the original collection receipt."""
    guard, device = run["guard"], run["device"]
    modern = any(
        field in guard
        for field in (
            "parent_context_pending_exit",
            "engine_children_released",
            "after_process_pids",
            "parent_pid",
        )
    )

    def evidence_file(name: object) -> Path:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or any(char in name for char in "/\\:")
        ):
            raise ValueError("device release evidence must use local file basenames")
        return path.with_name(name)

    if modern:
        pending = guard.get("parent_context_pending_exit")
        if (
            guard.get("engine_children_released") is not True
            or type(pending) is not bool
            or guard["after_target_idle"] is not (not pending)
            or not _time(root_pid)
            or not _time(guard.get("parent_pid"))
            or guard["parent_pid"] != root_pid
        ):
            raise ValueError("device release guard has inconsistent parent/child state")
        after = guard.get("after_process_pids")
        if not isinstance(after, dict) or set(after) != set(target):
            raise ValueError("device release guard target PID sets are missing")
        if (
            any(
                not isinstance(pids, list)
                or any(not _time(pid) or pid != root_pid for pid in pids)
                or pids != sorted(set(pids))
                for pids in after.values()
            )
            or any(bool(pids) for pids in after.values()) is not pending
        ):
            raise ValueError("device release guard retains a worker or unknown PID")
        if pending:
            observed_pids = target_process_pids(
                evidence_file(guard["after_npu_smi"]).read_text(encoding="utf-8"),
                npu_id=device["npu_id"],
                chip_ids={int(chip) for chip in target},
            )
            if observed_pids != {int(chip): set(pids) for chip, pids in after.items()}:
                raise ValueError("device release guard PID sets differ from raw table")
    elif guard["after_target_idle"] is not True:
        raise ValueError(
            "target devices were not released and parent exit is unverified"
        )

    release_path = path.with_name("device_release.json")
    if not release_path.exists():
        if modern:
            raise ValueError("device release sidecar is missing")
        return
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if not isinstance(release, dict):
        raise ValueError("device release sidecar must contain an object")
    if (
        release["run_json_sha256"] != run_sha
        or release["source_manifest_sha256"] != run["source_manifest_sha256"]
    ):
        raise ValueError("device release sidecar SHA differs from original run/source")
    if (
        type(release["child_exit_code"]) is not int
        or release["child_exit_code"] != 0
        or release["devices_released"] is not True
        or not _time(release["pid"])
        or release["pid"] != root_pid
    ):
        raise ValueError(
            "device release sidecar has unsuccessful or inconsistent child exit"
        )
    if (
        type(release["npu_id"]) is not int
        or release["npu_id"] != device["npu_id"]
        or not isinstance(release["chip_logic_ids"], dict)
        or release["chip_logic_ids"] != target
        or any(
            type(logical) is not int for logical in release["chip_logic_ids"].values()
        )
    ):
        raise ValueError("device release sidecar device mapping differs from run")
    timestamp = release["observed_after_exit_utc"]
    if not isinstance(timestamp, str):
        raise ValueError("device release observation time must identify UTC")
    observed = datetime.fromisoformat(
        timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    )
    if observed.utcoffset() != timedelta(0):
        raise ValueError("device release observation time must identify UTC")
    if release["after_mapping"] != "driver_after_mapping.txt":
        raise ValueError("device release mapping filename differs from driver evidence")
    validate_target(
        parse_mapping(
            evidence_file(release["after_mapping"]).read_text(encoding="utf-8")
        ),
        npu_id=device["npu_id"],
        chip_logic_ids={int(chip): logical for chip, logical in target.items()},
        tp_size=len(target),
    )
    process_files = release["after_processes"]
    if not isinstance(process_files, dict) or set(process_files) != set(target):
        raise ValueError("device release process evidence lacks target chips")
    for chip, name in process_files.items():
        _require_raw_chip_idle(
            evidence_file(name).read_text(encoding="utf-8"),
            npu_id=device["npu_id"],
            chip_id=int(chip),
        )


def _audit_run(path: Path, *, expected_tp: int) -> dict:
    errors, rows, indexed, batches, cells = [], [], {}, [], []
    result = {
        "path": str(path),
        "errors": errors,
        "batches": batches,
        "hbm_coverage_complete": False,
        "cells": cells,
        "indexed": indexed,
        "sequence_eligible": False,
    }

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        run_raw = path.read_bytes()
        run = json.loads(run_raw)
        if not isinstance(run, dict):
            raise ValueError("run.json must contain an object")
        config, protocol, inputs = (
            run[name] for name in ("config", "protocol", "inputs")
        )
        if not all(isinstance(value, dict) for value in (config, protocol, inputs)):
            raise ValueError("config, protocol, and inputs must contain objects")
        if not all(
            isinstance(run[name], dict) for name in ("guard", "device", "clock")
        ) or not isinstance(run["device"].get("chip_logic_ids"), dict):
            raise ValueError("guard, device mapping, and clock must contain objects")
        result["record"] = run
        check(run["status"] == "passed", "run status is not passed")
        check(config["tp_size"] == expected_tp, "unexpected tp_size")
        check(
            isinstance(config["model_id"], str)
            and bool(config["model_id"])
            and re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]) is not None,
            "model identity or revision is missing/invalid",
        )
        check(
            run["profile_mode"] == "off" and run["normal_performance_eligible"] is True,
            "profiled collection is not eligible for normal performance comparison",
        )
        check(
            re.fullmatch(r"[0-9a-f]{64}", run["source_manifest_sha256"]) is not None,
            "invalid source_manifest_sha256",
        )
        check(
            run["source_after_mismatches"] == [], "source_after_mismatches are present"
        )
        check(run["sampler_errors"] == [], "sampler_errors are present")
        check(
            _sha(run["runtime_dependencies_sha256"])
            and run["runtime_dependencies_after_sha256"]
            == run["runtime_dependencies_sha256"],
            "runtime dependency SHA is invalid or changed",
        )
        dependencies_raw = path.with_name("runtime_dependencies.json").read_bytes()
        check(
            hashlib.sha256(dependencies_raw).hexdigest()
            == run["runtime_dependencies_sha256"],
            "runtime dependency SHA differs from saved JSON",
        )
        dependencies = json.loads(dependencies_raw)
        if not isinstance(dependencies, dict) or not isinstance(
            dependencies.get("external_extensions"), dict
        ):
            raise ValueError("runtime dependencies require an extension allowlist")
        check(
            run["runtime_origins_unchanged"] is True,
            "runtime origins changed or were not verified",
        )
        origins = run["runtime_origins"]
        if not isinstance(origins, dict):
            raise ValueError("runtime origins must contain an object")
        check(
            set(origins) == _RUNTIME_MODULES,
            "runtime origins lack required modules or contain unexpected modules",
        )
        for module, origin in origins.items():
            if not isinstance(origin, dict) or not _sha(origin.get("sha256")):
                raise ValueError(f"invalid runtime origin SHA: {module}")
            if set(origin) == {"path", "sha256"} and isinstance(origin["path"], str):
                source = PurePosixPath(origin["path"])
                check(
                    bool(origin["path"])
                    and not source.is_absolute()
                    and ".." not in source.parts
                    and source.as_posix() == origin["path"]
                    and "\\" not in origin["path"]
                    and ":" not in origin["path"],
                    f"runtime origin path escapes source workspace: {module}",
                )
            elif set(origin) == {"basename", "sha256"} and isinstance(
                origin["basename"], str
            ):
                name = origin["basename"]
                check(
                    module == "tokenspeed_scheduler.tokenspeed_scheduler_ext"
                    and name.startswith("tokenspeed_scheduler_ext.")
                    and Path(name).name == name
                    and "\\" not in name
                    and ":" not in name
                    and Path(name).suffix in {".so", ".pyd"}
                    and dependencies["external_extensions"].get(name)
                    == origin["sha256"],
                    f"runtime external extension is not pinned: {module}",
                )
            else:
                errors.append(f"invalid or ambiguous runtime origin: {module}")
        check(
            run["foreign_processes"] == [],
            "foreign or unknown target processes are present",
        )
        clock = run["clock"]
        check(
            clock["name"] == "perf_counter_ns"
            and type(clock["resolution_seconds"]) in (int, float)
            and math.isfinite(clock["resolution_seconds"])
            and clock["resolution_seconds"] > 0,
            "clock does not identify the monotonic perf_counter_ns",
        )
        check(
            run["sampling_params"]
            == {"temperature": 0, "ignore_eos": True, "stream_interval": 1},
            "sampling_params differ from fixed greedy collection",
        )
        guard = run["guard"]
        check(guard["sources_unchanged"] is True, "guard sources changed")
        check(
            guard["sources_sha"] == run["source_manifest_sha256"],
            "guard source SHA differs",
        )
        check(
            guard["sources_after_sha"] == guard["sources_sha"],
            "guard source manifest changed",
        )
        for name, value in (
            ("config", config),
            ("protocol", protocol),
            ("inputs", inputs),
        ):
            raw = path.with_name(f"{name}.json").read_bytes()
            check(
                hashlib.sha256(raw).hexdigest() == run[f"{name}_sha256"],
                f"{name} SHA differs from saved JSON",
            )
            check(json.loads(raw) == value, f"embedded {name} differs from saved JSON")
        for name in ("input_lengths", "concurrency"):
            values = protocol[name]
            check(
                isinstance(values, list)
                and len(values) >= 1
                and all(type(n) is int and n > 0 for n in values)
                and len(set(values)) == len(values),
                f"invalid protocol {name}",
            )
        for name in (
            "warmup_requests",
            "requests_per_cell",
            "repetitions",
            "max_new_tokens",
        ):
            check(
                type(protocol[name]) is int and protocol[name] > 0,
                f"invalid protocol {name}",
            )
        if errors and any("invalid protocol" in error for error in errors):
            return result
        check(
            config["max_num_seqs"] >= max(protocol["concurrency"]),
            "configuration cannot cover requested concurrency",
        )
        check(
            max(protocol["input_lengths"]) + protocol["max_new_tokens"]
            <= config["max_model_len"],
            "protocol exceeds model context capacity",
        )
        check(
            set(inputs) == {str(n) for n in protocol["input_lengths"]},
            "input length keys differ",
        )
        for length in protocol["input_lengths"]:
            check(
                _ids(inputs[str(length)]) and len(inputs[str(length)]) == length,
                f"invalid input IDs for length {length}",
            )
        expected = {}
        for length in protocol["input_lengths"]:
            for concurrency in protocol["concurrency"]:
                expected[("warmup", length, concurrency, 0)] = protocol[
                    "warmup_requests"
                ]
                for repetition in range(1, protocol["repetitions"] + 1):
                    expected[("formal", length, concurrency, repetition)] = protocol[
                        "requests_per_cell"
                    ]
        expected_requests = {
            key + (index,) for key, count in expected.items() for index in range(count)
        }
        rows = [
            json.loads(line)
            for line in path.with_name("requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("requests.jsonl must contain objects")
        for number, row in enumerate(rows):
            try:
                key = _key(row, request=True)
                check(key not in indexed, f"duplicate request key {key}")
                indexed.setdefault(key, row)
                check(
                    row["status"] == "passed" and row["error"] is None,
                    f"failed request {key}: {row.get('error')}",
                )
                check(
                    _ids(row["input_ids"])
                    and row["input_ids"] == inputs[str(row["input_length"])],
                    f"request input IDs differ {key}",
                )
                check(
                    _ids(row["output_ids"])
                    and len(row["output_ids"]) == protocol["max_new_tokens"],
                    f"request output IDs/count invalid {key}",
                )
                started, finished = row["started_ns"], row["finished_ns"]
                check(
                    _time(started) and _time(finished) and started < finished,
                    f"request monotonic time invalid {key}",
                )
                chunks = row["chunks"]
                check(bool(chunks), f"request chunks missing {key}")
                observed = [chunk["observed_ns"] for chunk in chunks]
                check(
                    all(_time(n) and started <= n <= finished for n in observed)
                    and observed == sorted(observed),
                    f"chunk time invalid {key}",
                )
                check(
                    all(
                        type(chunk["output_tokens"]) is int
                        and chunk["output_tokens"] > 0
                        for chunk in chunks
                    )
                    and sum(chunk["output_tokens"] for chunk in chunks)
                    == len(row["output_ids"]),
                    f"chunk output count invalid {key}",
                )
                if observed:
                    check(
                        _close(row["ttft_ms"], (observed[0] - started) / 1e6),
                        f"TTFT differs {key}",
                    )
                check(
                    _close(row["e2e_ms"], (finished - started) / 1e6),
                    f"E2E differs {key}",
                )
                intervals = [
                    (right - left) / 1e6 for left, right in zip(observed, observed[1:])
                ]
                check(
                    row["inter_chunk_ms"] == intervals,
                    f"inter-chunk times differ {key}",
                )
            except (KeyError, TypeError, ValueError, IndexError) as error:
                errors.append(f"malformed request {number}: {error}")
        check(
            set(indexed) == expected_requests and len(rows) == len(expected_requests),
            "missing, extra, or duplicate requests",
        )
        result["sequence_eligible"] = (
            set(indexed) == expected_requests
            and len(rows) == len(expected_requests)
            and all(
                _ids(row.get("output_ids"))
                and len(row["output_ids"]) == protocol["max_new_tokens"]
                and row.get("status") == "passed"
                and row.get("error") is None
                and row.get("input_ids") == inputs[str(row["input_length"])]
                for row in rows
            )
        )
        device = run["device"]
        target = {
            str(chip): logical for chip, logical in device["chip_logic_ids"].items()
        }
        check(
            type(device["npu_id"]) is int
            and device["npu_id"] >= 0
            and len(target) == expected_tp
            and all(chip in {"0", "1"} for chip in target)
            and all(type(n) is int and n >= 0 for n in target.values())
            and len(set(target.values())) == expected_tp,
            "target device/chip count invalid",
        )
        sampler_start, sampler_end = (
            run["sampler_started_ns"],
            run["sampler_finished_ns"],
        )
        check(
            _time(sampler_start) and _time(sampler_end) and sampler_start < sampler_end,
            "sampler monotonic interval invalid",
        )
        check(
            _time(guard["guard_started_ns"])
            and _time(guard["guard_ended_ns"])
            and guard["guard_started_ns"]
            <= sampler_start
            <= sampler_end
            <= guard["guard_ended_ns"],
            "guard does not cover sampler interval",
        )
        samples, previous = run["hbm_samples"], sampler_start
        check(bool(samples), "HBM samples missing")
        processes = run["process_samples"]
        if not isinstance(processes, list):
            raise ValueError("process samples must contain a list")
        check(
            len(processes) == len(samples),
            "process sample count differs from HBM samples",
        )
        root_pid = None
        for number, sample in enumerate(samples):
            try:
                start, end = sample["started_ns"], sample["finished_ns"]
                check(
                    _time(start)
                    and _time(end)
                    and previous <= start <= end <= sampler_end,
                    f"HBM sample monotonic interval invalid {number}",
                )
                previous = end
                chips = {str(chip): value for chip, value in sample["chips"].items()}
                check(set(chips) == set(target), f"HBM target chips missing {number}")
                for value in chips.values():
                    check(
                        type(value["used_mb"]) is int
                        and type(value["capacity_mb"]) is int
                        and 0 <= value["used_mb"] <= value["capacity_mb"]
                        and value["capacity_mb"] > 0,
                        f"HBM value invalid {number}",
                    )
                name = sample["raw_table"]
                check(
                    isinstance(name, str) and Path(name).name == name,
                    f"HBM raw_table filename invalid {number}",
                )
                if isinstance(name, str) and Path(name).name == name:
                    check(
                        _raw_hbm(
                            path.with_name(name), npu_id=device["npu_id"], chips=target
                        )
                        == chips,
                        f"HBM raw table differs or lacks target chips {number}",
                    )
                    raw_pids = target_process_pids(
                        path.with_name(name).read_text(encoding="utf-8"),
                        npu_id=device["npu_id"],
                        chip_ids={int(chip) for chip in target},
                    )
                    process = processes[number]
                    if not isinstance(process, dict) or not isinstance(
                        process["chips"], dict
                    ):
                        raise ValueError("invalid process sample")
                    if number == 0:
                        root_pid = process["root_pid"]
                    check(
                        _time(process["root_pid"]) and process["root_pid"] == root_pid,
                        f"process sample root PID differs {number}",
                    )
                    check(
                        process["raw_table"] == name
                        and "error" not in process
                        and _time(process["observed_ns"])
                        and start <= process["observed_ns"] <= end,
                        f"process sample does not match raw HBM query {number}",
                    )
                    check(
                        set(process["chips"]) == set(target),
                        f"process sample target chips differ {number}",
                    )
                    for chip in target:
                        row = process["chips"][chip]
                        pids, parents = row["pids"], row["parents"]
                        check(
                            isinstance(pids, list)
                            and all(_time(pid) for pid in pids)
                            and pids == sorted(set(pids))
                            and set(pids) == raw_pids[int(chip)],
                            f"process sample PID set differs from raw table {number}/{chip}",
                        )
                        check(
                            row["foreign_pids"] == [] and row["unknown_pids"] == [],
                            f"foreign or unknown process sample PID {number}/{chip}",
                        )
                        if not isinstance(parents, dict) or not all(
                            isinstance(pid, str)
                            and re.fullmatch(r"[1-9]\d*", pid)
                            and type(parent) is int
                            and parent >= 0
                            for pid, parent in parents.items()
                        ):
                            raise ValueError("invalid saved process-parent edges")
                        parent_ids = {
                            int(pid): parent for pid, parent in parents.items()
                        }
                        check(
                            all(
                                is_descendant(
                                    pid,
                                    root_pid=process["root_pid"],
                                    read_parent=parent_ids.get,
                                )
                                for pid in raw_pids[int(chip)]
                            ),
                            f"raw process PID lacks known ancestry to this run {number}/{chip}",
                        )
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                errors.append(f"malformed HBM sample {number}: {error}")
        _audit_device_release(
            path,
            run,
            run_sha=hashlib.sha256(run_raw).hexdigest(),
            root_pid=root_pid,
            target=target,
        )
        actual_batches, previous = set(), 0
        for number, batch in enumerate(run["batches"]):
            try:
                key = _key(batch, request=False)
                check(key not in actual_batches, f"duplicate batch {key}")
                actual_batches.add(key)
                group = [row for row in rows if _key(row, request=False) == key]
                start, end = batch["started_ns"], batch["finished_ns"]
                check(
                    _time(start) and _time(end) and previous <= start < end,
                    f"batch monotonic interval invalid {key}",
                )
                previous = end
                wall = end - start
                check(batch["status"] == "passed", f"failed batch {key}")
                check(
                    len(group) == expected.get(key) == batch["requests"],
                    f"batch request count differs {key}",
                )
                check(
                    all(
                        start <= row["started_ns"] < row["finished_ns"] <= end
                        for row in group
                    ),
                    f"batch does not contain actual request interval {key}",
                )
                check(
                    sampler_start <= start <= end <= sampler_end,
                    f"sampler does not cover actual batch interval {key}",
                )
                tokens = sum(len(row["output_ids"]) for row in group)
                check(
                    tokens == batch["output_tokens"],
                    f"batch output count differs {key}",
                )
                check(
                    wall > 0 and batch["wall_ns"] == wall, f"batch wall differs {key}"
                )
                throughput = tokens * 1e9 / wall if wall > 0 else None
                check(
                    throughput is not None
                    and _close(batch["output_tokens_per_second"], throughput),
                    f"stored throughput differs {key}",
                )
                contained = [
                    sample
                    for sample in samples
                    if start <= sample["started_ns"] <= sample["finished_ns"] <= end
                ]
                peak = (
                    {
                        chip: max(
                            sample["chips"][chip]["used_mb"] for sample in contained
                        )
                        for chip in target
                    }
                    if contained
                    and all(set(sample["chips"]) == set(target) for sample in contained)
                    else None
                )
                batches.append(
                    {
                        **dict(zip(_BATCH_FIELDS, key)),
                        "wall_ns": wall,
                        "output_tokens": tokens,
                        "output_tokens_per_second": throughput,
                        "ttft_ms": [
                            (row["chunks"][0]["observed_ns"] - row["started_ns"]) / 1e6
                            for row in group
                            if row["chunks"]
                        ],
                        "hbm_contained_samples": len(contained),
                        "hbm_observed_max_mb_per_chip": peak,
                    }
                )
            except (KeyError, TypeError, ValueError, IndexError) as error:
                errors.append(f"malformed batch {number}: {error}")
        check(
            actual_batches == set(expected) and len(run["batches"]) == len(expected),
            "missing, extra, or duplicate batches",
        )
        result["hbm_coverage_complete"] = (
            bool(batches)
            and len(batches) == len(expected)
            and all(
                batch["hbm_observed_max_mb_per_chip"] is not None for batch in batches
            )
        )
        result["hbm_observed_max_mb_per_chip"] = {
            chip: max(
                sample["chips"][chip]["used_mb"]
                for sample in samples
                if chip in sample["chips"]
            )
            for chip in target
            if any(chip in sample["chips"] for sample in samples)
        }
        for length in protocol["input_lengths"]:
            for concurrency in protocol["concurrency"]:
                formal = [
                    batch
                    for batch in batches
                    if batch["phase"] == "formal"
                    and batch["input_length"] == length
                    and batch["concurrency"] == concurrency
                ]
                speeds = [
                    batch["output_tokens_per_second"]
                    for batch in formal
                    if batch["output_tokens_per_second"] is not None
                ]
                latencies = [value for batch in formal for value in batch["ttft_ms"]]
                if speeds and latencies:
                    cells.append(
                        {
                            "input_length": length,
                            "concurrency": concurrency,
                            "repetitions": len(formal),
                            "throughput": {
                                "min": min(speeds),
                                "median": median(speeds),
                                "max": max(speeds),
                            },
                            "ttft_ms": {
                                "mean": fmean(latencies),
                                "p50": median(latencies),
                                "p95": _percentile(latencies, 0.95),
                            },
                            "formal_request_count": len(latencies),
                        }
                    )
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"incomplete run artifact: {error}")
    return result


def audit_run(path: Path, *, expected_tp: int) -> dict:
    """Audit one saved matrix run, including a TP2-only model or strategy arm.

    This checks the same execution contract as paired comparison, without
    inferring a baseline, speedup or cross-TP sequence agreement.
    """
    if type(expected_tp) is not int or expected_tp not in (1, 2):
        raise ValueError("expected_tp must be 1 or 2")
    audit = _audit_run(Path(path), expected_tp=expected_tp)
    valid = not audit["errors"]
    return {
        **{
            key: value
            for key, value in audit.items()
            if key not in {"record", "indexed", "errors", "cells"}
        },
        "execution_valid": valid,
        "execution_errors": audit["errors"],
        "performance_valid": valid,
        "evidence_complete": valid and audit["hbm_coverage_complete"],
        "cells": audit["cells"] if valid else [],
        "limitations": [_PROVENANCE_LIMITATION],
    }


def compare_runs(tp1_path: Path, tp2_path: Path) -> dict:
    """Compare two saved run.json paths; retain failures and every ID disagreement.

    execution_valid validates execution/provenance independently of finite greedy
    sequence_agreement. evidence_complete additionally requires contained HBM
    observations for every batch. No logits accuracy or statistical significance
    is inferred from these artifacts.
    """
    audited = {
        "tp1": _audit_run(Path(tp1_path), expected_tp=1),
        "tp2": _audit_run(Path(tp2_path), expected_tp=2),
    }
    errors = [
        f"{name}: {error}"
        for name, audit in audited.items()
        for error in audit["errors"]
    ]
    first, second = (audited[name].get("record", {}) for name in ("tp1", "tp2"))
    devices = [run.get("device", {}) for run in (first, second)]
    if devices[0].get("npu_id") != devices[1].get("npu_id") or any(
        devices[1].get("chip_logic_ids", {}).get(chip) != logical
        for chip, logical in devices[0].get("chip_logic_ids", {}).items()
    ):
        errors.append("paired devices do not select the same physical card/chips")
    for field in (
        "source_manifest_sha256",
        "protocol_sha256",
        "inputs_sha256",
        "protocol",
        "inputs",
        "runtime_dependencies_sha256",
        "runtime_origins",
    ):
        if field not in first or first.get(field) != second.get(field):
            errors.append(f"paired {field} differs or is missing")
    configs = [
        {key: value for key, value in run.get("config", {}).items() if key != "tp_size"}
        for run in (first, second)
    ]
    if not configs[0] or configs[0] != configs[1]:
        errors.append("paired config differs beyond tp_size")
    one, two = (audited[name]["indexed"] for name in ("tp1", "tp2"))
    mismatches, comparable, matched = (
        [],
        bool(one)
        and set(one) == set(two)
        and all(audit["sequence_eligible"] for audit in audited.values()),
        0,
    )
    for key in sorted(set(one) | set(two)):
        left, right = one.get(key), two.get(key)
        left_ids = left.get("output_ids") if left else None
        right_ids = right.get("output_ids") if right else None
        comparable &= bool(
            left
            and right
            and _ids(left_ids)
            and _ids(right_ids)
            and left.get("input_ids") == right.get("input_ids")
            and left.get("status") == right.get("status") == "passed"
        )
        if left_ids != right_ids:
            left_array = left_ids if isinstance(left_ids, list) else []
            right_array = right_ids if isinstance(right_ids, list) else []
            index = next(
                (
                    i
                    for i, values in enumerate(zip(left_array, right_array))
                    if values[0] != values[1]
                ),
                min(len(left_array), len(right_array)),
            )
            mismatches.append(
                {
                    **dict(zip(_BATCH_FIELDS + ("request_index",), key)),
                    "first_difference_index": index,
                    "tp1_output_ids": left_ids,
                    "tp2_output_ids": right_ids,
                }
            )
        else:
            matched += 1
    execution_valid = not errors
    performance = []
    normal_performance = all(
        run.get("profile_mode") == "off"
        and run.get("normal_performance_eligible") is True
        for run in (first, second)
    )
    for left in (
        audited["tp1"]["cells"] if execution_valid and normal_performance else []
    ):
        right = next(
            (
                cell
                for cell in audited["tp2"]["cells"]
                if cell["input_length"] == left["input_length"]
                and cell["concurrency"] == left["concurrency"]
            ),
            None,
        )
        if right:
            performance.append(
                {
                    "input_length": left["input_length"],
                    "concurrency": left["concurrency"],
                    "tp1": left,
                    "tp2": right,
                    "throughput_ratio_tp2_over_tp1": right["throughput"]["median"]
                    / left["throughput"]["median"],
                }
            )
    hbm_complete = all(audit["hbm_coverage_complete"] for audit in audited.values())
    return {
        "execution_valid": execution_valid,
        "execution_errors": errors,
        "performance_valid": execution_valid and normal_performance,
        "hbm_coverage_complete": hbm_complete,
        "evidence_complete": execution_valid and hbm_complete,
        "sequence_agreement": (not mismatches) if comparable else None,
        "sequence_comparison": {
            "compared_requests": len(set(one) & set(two)),
            "matched_requests": matched,
            "mismatches": mismatches,
        },
        "runs": {
            name: {
                key: value
                for key, value in audit.items()
                if key not in {"record", "indexed", "cells"}
            }
            for name, audit in audited.items()
        },
        "performance": performance,
        "limitations": [
            _PROVENANCE_LIMITATION,
            "Sequence agreement covers only the recorded finite greedy token sequences; it does not validate full logits.",
            "HBM maxima are discrete observed samples, not instantaneous peaks; boundary-crossing queries are not assigned as batch peaks.",
            "Contained HBM observations are required in every warmup/formal batch for evidence_complete; sampler lifetime must bracket execution.",
            "Repetitions provide descriptive min/median/max only, without significance or confidence claims.",
            "Performance requires execution_valid; output disagreements remain visible and do not invalidate measured execution timings.",
        ],
    }
