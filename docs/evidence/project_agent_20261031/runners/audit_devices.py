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

"""Offline checks of raw device tables, source inventory and outer exit receipt.

Process identities and model-file stat checks are recorded observations, not a
live /proc or model-byte check. Shared observations never establish exclusivity.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.course_project.native_performance import parse_target_hbm
from benchmarks.course_project.tp_devices import (
    check_sources,
    parse_mapping,
    target_process_pids,
    validate_target,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_file(run_path: Path, filename: str) -> Path:
    root = run_path.parent.resolve()
    path = (root / filename).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("raw evidence path escapes run directory or is missing")
    return path


def pids(value: list) -> set[int]:
    if any(type(pid) is not int or pid < 1 for pid in value) or len(set(value)) != len(
        value
    ):
        raise ValueError("invalid or duplicate process IDs")
    return set(value)


def process_map(value: dict, chips: dict) -> dict[int, set[int]]:
    if set(value) != {str(chip) for chip in chips}:
        raise ValueError("process chip set differs from selected devices")
    return {int(chip): pids(values) for chip, values in value.items()}


def identities(value: dict) -> dict[int, list]:
    result = {}
    for pid, identity in value.items():
        if (
            str(int(pid)) != pid
            or int(pid) < 1
            or not isinstance(identity, list)
            or len(identity) != 2
            or type(identity[0]) is not int
            or identity[0] < 0
            or not isinstance(identity[1], str)
            or not identity[1].isdigit()
        ):
            raise ValueError("invalid UID/starttime process identity")
        result[int(pid)] = identity
    return result


def selected_devices(record: dict) -> tuple[int, dict[int, int]]:
    devices = record["devices"]
    card = devices["npu_id"]
    chips = {int(chip): logical for chip, logical in devices["chip_logic_ids"].items()}
    if (
        type(card) is not int
        or card < 0
        or any(type(v) is not int or v < 0 for v in chips.values())
    ):
        raise ValueError("invalid explicit device identity")
    return card, chips


def verify(
    run_path: Path, record: dict, source_root: Path, source_manifest: Path
) -> dict:
    """Recompute the selected-device and own-process-release evidence offline."""
    digest = check_sources(source_root, source_manifest)
    if digest != record["source_manifest_sha256"]:
        raise ValueError(
            "run source manifest SHA differs from complete verified manifest"
        )
    card, chips = selected_devices(record)
    guard = record["device_guard"]
    receipt_path = raw_file(run_path, "post_exit_verification.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["run_sha256"] != sha(run_path):
        raise ValueError("post-exit receipt run SHA mismatch")
    if receipt["devices"] != record["devices"]:
        raise ValueError("post-exit selected device mismatch")
    if receipt["exit_code"] != 0 or receipt["timed_out"] is not False:
        raise ValueError("collector did not exit successfully within its budget")
    for field in (
        "source_unchanged",
        "model_files_unchanged",
        "own_processes_released",
    ):
        if receipt[field] is not True:
            raise ValueError(f"post-exit failed observation: {field}")
    if receipt["unknown_pids"]:
        raise ValueError("post-exit receipt reports unknown process IDs")
    for field in ("before_mapping", "after_mapping"):
        validate_target(
            parse_mapping(raw_file(run_path, guard[field]).read_text()),
            npu_id=card,
            chip_logic_ids=chips,
            tp_size=record["config"]["tp_size"],
        )
    validate_target(
        parse_mapping(raw_file(run_path, "post_exit_mapping.txt").read_text()),
        npu_id=card,
        chip_logic_ids=chips,
        tp_size=record["config"]["tp_size"],
    )
    if (
        guard["violations"]
        or guard["after_errors"]
        or guard.get("error")
        or guard["engine_children_released"] is not True
        or guard["sources_unchanged"] is not True
        or guard["sources_sha"] != digest
        or guard["sources_after_sha"] != digest
    ):
        raise ValueError("device guard or source cleanup failed")
    if guard["resource_mode"] != record["resource_mode"] or record[
        "resource_mode"
    ] not in {"shared", "idle"}:
        raise ValueError("inconsistent resource mode")
    shared = record["resource_mode"] == "shared"
    if shared and (
        type(guard["min_free_mb"]) is not int or guard["min_free_mb"] < 8192
    ):
        raise ValueError("shared memory reserve is missing or invalid")
    parent = receipt["parent_pid"]
    if parent != guard["parent_pid"] or pids([parent]) != {parent}:
        raise ValueError("collector parent identity mismatch")
    own_identities = identities({str(parent): receipt["parent_identity"]})
    baseline = {
        int(chip): identities(values) for chip, values in guard["baseline"].items()
    }
    if set(baseline) != set(chips) or any(
        parent in values for values in baseline.values()
    ):
        raise ValueError("invalid background baseline chip/process set")
    if not shared and any(baseline.values()):
        raise ValueError("idle mode has background processes")
    samples = guard["samples"]
    if (
        not samples
        or samples[0]["label"] != "before"
        or samples[-1]["label"] != "after_engine_exit"
    ):
        raise ValueError("missing before/after guard observations")
    known = {parent}
    for index, sample in enumerate(samples):
        if sample.get("error"):
            raise ValueError("failed guard query")
        if (
            not guard["guard_started_ns"]
            <= sample["observed_ns"]
            <= sample["ended_ns"]
            <= guard["guard_ended_ns"]
        ):
            raise ValueError("guard sample timing outside guard")
        text = raw_file(run_path, sample["raw_table"]).read_text()
        parsed = target_process_pids(text, npu_id=card, chip_ids=set(chips))
        if parsed != process_map(sample["pids"], chips):
            raise ValueError("guard raw process table differs from saved PIDs")
        memory = parse_target_hbm(text, npu_id=card, chip_logic_ids=chips)
        if memory != {int(chip): value for chip, value in sample["memory"].items()}:
            raise ValueError("guard raw HBM table differs from saved memory")
        if shared and any(
            v["capacity_mb"] - v["used_mb"] < guard["min_free_mb"]
            for v in memory.values()
        ):
            raise ValueError("shared memory reserve violated in raw observation")
        exited = pids(sample["exited_during_query"])
        all_pids = set().union(*parsed.values())
        if not exited <= all_pids:
            raise ValueError("exited process absent from raw table")
        live = {chip: values - exited for chip, values in parsed.items()}
        observed = identities(sample["identities"])
        if (
            live != process_map(sample["live_pids"], chips)
            or set(observed) != all_pids - exited
        ):
            raise ValueError("guard live process identity set mismatch")
        if index == 0:
            expected = {
                chip: (
                    {pid: observed[pid] for pid in values if pid != parent}
                    if shared
                    else {}
                )
                for chip, values in live.items()
            }
            if baseline != expected or (not shared and any(live.values())):
                raise ValueError(
                    "background baseline differs from initial raw observation"
                )
        owned = pids(sample["owned_pids"])
        recomputed_owned = set()
        for chip, values in live.items():
            for pid in values:
                identity = observed[pid]
                if pid in baseline[chip]:
                    if baseline[chip][pid] != identity:
                        raise ValueError("background PID identity changed")
                else:
                    if pid not in owned or identity[0] != receipt["parent_identity"][0]:
                        raise ValueError("unknown process in guard raw observation")
                    if pid in own_identities and own_identities[pid] != identity:
                        raise ValueError("owned PID identity changed")
                    own_identities[pid] = identity
                    recomputed_owned.add(pid)
        if recomputed_owned != owned:
            raise ValueError("guard owned process set inconsistent with raw table")
        known |= owned
        if (
            sample["label"] == "monitor"
            and sample["ended_ns"] > guard["monitor_stopped_ns"]
        ):
            raise ValueError("device monitor ended after monitor stop")
        if sample["label"] == "after_engine_exit" and owned - {parent}:
            # Earlier grace-period observations may still contain exiting workers.
            if index == len(samples) - 1:
                raise ValueError("own Engine child remains in final guard observation")
    if (
        not guard["guard_started_ns"]
        <= guard["monitor_stopped_ns"]
        <= samples[-1]["observed_ns"]
        <= guard["guard_ended_ns"]
    ):
        raise ValueError("missing/reversed monitor teardown boundary")
    if known != pids(receipt["known_owned_pids"]):
        raise ValueError(
            "outer owned process inventory differs from guard observations"
        )
    post = receipt["samples"]
    if not post:
        raise ValueError("missing post-exit raw observation")
    for sample in post:
        text = raw_file(run_path, sample["raw_table"]).read_text()
        parsed = target_process_pids(text, npu_id=card, chip_ids=set(chips))
        parse_target_hbm(text, npu_id=card, chip_logic_ids=chips)
        if parsed != process_map(sample["pids"], chips):
            raise ValueError("post-exit raw table differs from saved PIDs")
        all_pids = set().union(*parsed.values())
        exited = pids(sample["exited_during_query"])
        observed = identities(sample["identities"])
        if not exited <= all_pids or set(observed) != all_pids - exited:
            raise ValueError("post-exit live identity set mismatch")
        remaining = (all_pids - exited) & known
        if remaining != pids(sample["remaining_owned_pids"]):
            raise ValueError("post-exit own-process remainder differs from raw table")
        for chip, values in parsed.items():
            for pid in values - exited - known:
                if baseline[chip].get(pid) != observed[pid]:
                    raise ValueError(
                        "post-exit unknown process or background identity changed"
                    )
        if sample is post[-1] and remaining:
            raise ValueError("own process remains after collector exit")
    return {
        "passed": True,
        "devices": record["devices"],
        "resource_mode": record["resource_mode"],
        "source_manifest_sha256": digest,
        "post_exit_receipt_sha256": sha(receipt_path),
        "guard_samples": len(samples),
        "post_exit_samples": len(post),
        "known_owned_pids": sorted(known),
        "scope": "selected chips observed; own processes released; shared background identities checked when still present; no exclusive or whole-card-idle claim; ancestry/model stat observations are not independently recoverable from raw npu-smi tables",
    }


def audit_devices(
    run_path: Path, record: dict, source_root: Path, source_manifest: Path
) -> dict:
    try:
        result = verify(run_path, record, source_root, source_manifest)
        result["violations"] = []
        return result
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {
            "passed": False,
            "violations": [f"device/source audit: {type(error).__name__}: {error}"],
        }
