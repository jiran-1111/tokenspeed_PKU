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

"""Conservative, rank-local analysis of torch_npu Chrome complete events.

Supported device evidence is the observed ``Ascend Hardware`` process and
``Stream <number>`` thread metadata. Compute also requires a known Task Type.
Collective names have no assumed schema: callers supply patterns verified
against their device trace. CPU enqueue/API events never become device work.
Unknown events remain visible. Only X durations are measured; B/E durations
are rejected, while metadata, flows, counters and instants are not measured.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

_Interval = tuple[Decimal, Decimal]
_COMPUTE_TASK_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV"}
_HOST_CATEGORIES = {"cpu_op", "enqueue", "dequeue", "python_function"}
_KINDS = ("device_compute", "device_collective", "host_wait", "host_other", "unknown")
_WAIT_NAME = re.compile(
    r"^(?:(?:event|stream|device)::synchronize|"
    r"(?:aclrt|rt)(?:synchronize(?:stream|event|device)|"
    r"(?:stream|event|device)synchronize))$",
    re.IGNORECASE,
)
_COLLECTIVE_HINT = re.compile(
    r"hccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|broadcast",
    re.IGNORECASE,
)


def _number(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{field} must be a finite, nonnegative number")
    try:
        number = Decimal(str(value))
        finite = number.is_finite() and math.isfinite(float(number))
    except (InvalidOperation, ValueError, OverflowError):
        finite = False
    if not finite or number < 0:
        raise ValueError(f"{field} must be a finite, nonnegative number")
    return number


def _identifier(value: object, *, field: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        raise ValueError(f"{field} must be an integer or nonempty string")
    return value


def _load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        trace = json.load(source, parse_float=Decimal)
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        raise ValueError(
            f"{path}: expected a list or an object containing traceEvents list"
        )
    return events


def _metadata(events: list[dict]) -> tuple[dict, dict]:
    processes: dict[int | str, str] = {}
    threads: dict[tuple[int | str, int | str], str] = {}
    for index, event in enumerate(events):
        if event.get("ph") != "M" or event.get("name") not in (
            "process_name",
            "thread_name",
        ):
            continue
        pid = _identifier(event.get("pid"), field=f"metadata event {index} pid")
        args = event.get("args")
        if not isinstance(args, dict) or not isinstance(args.get("name"), str):
            raise ValueError(f"metadata event {index} needs a string args.name")
        if event["name"] == "process_name":
            mapping, key = processes, pid
        else:
            tid = _identifier(event.get("tid"), field=f"metadata event {index} tid")
            mapping, key = threads, (pid, tid)
        if key in mapping and mapping[key] != args["name"]:
            raise ValueError(f"conflicting metadata at event {index}")
        mapping[key] = args["name"]
    return processes, threads


def _classify(
    event: dict,
    *,
    process_name: str,
    thread_name: str,
    collective_patterns: Sequence[re.Pattern],
) -> tuple[str, str]:
    name = event["name"]
    category = event.get("cat", "")
    args = event.get("args", {})
    task_type = args.get("Task Type")
    host_category = any(
        part.strip() in _HOST_CATEGORIES for part in category.split(",")
    )
    if process_name == "Ascend Hardware":
        if host_category:
            return "unknown", "hardware process conflicts with CPU category"
        if re.fullmatch(r"Stream \d+", thread_name) is None:
            return "unknown", "hardware process has no confirmed Stream thread metadata"
        if any(pattern.search(name) for pattern in collective_patterns):
            return (
                "device_collective",
                "hardware Stream and explicit collective name pattern",
            )
        if _COLLECTIVE_HINT.search(name):
            return "unknown", "collective-looking device name has no confirmed pattern"
        if task_type in _COMPUTE_TASK_TYPES:
            return "device_compute", "hardware Stream and known compute Task Type"
        return (
            "unknown",
            "hardware task has no supported compute/collective classification",
        )
    if process_name in {"Python", "CANN"} or host_category:
        if _WAIT_NAME.fullmatch(name.rsplit("@", 1)[-1]):
            return "host_wait", "CPU lane and explicit synchronization API name"
        return "host_other", "CPU process or CPU category; includes enqueue/API work"
    return "unknown", "no confirmed hardware Stream or CPU lane evidence"


def _union(intervals: Sequence[_Interval]) -> list[_Interval]:
    merged: list[_Interval] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _intersection(left: list[_Interval], right: list[_Interval]) -> list[_Interval]:
    intervals: list[_Interval] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            intervals.append((start, end))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return intervals


def _duration(intervals: Sequence[_Interval]) -> Decimal:
    return sum((end - start for start, end in intervals), Decimal(0))


def _analyze_rank(
    path: Path,
    *,
    rank: int,
    window: _Interval,
    clock_id: str,
    collective_patterns: Sequence[re.Pattern],
) -> dict:
    events = _load_events(path)
    processes, threads = _metadata(events)
    intervals: dict[str, list[_Interval]] = {kind: [] for kind in _KINDS}
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    unknown: list[dict] = []
    complete_count = 0
    for index, event in enumerate(events):
        phase = event.get("ph")
        if not isinstance(phase, str):
            raise ValueError(f"{path}: event {index} phase must be a string")
        if phase in {"M", "s", "t", "f", "C", "i", "I"}:
            continue
        if phase != "X":
            raise ValueError(f"{path}: event {index} has unsupported phase {phase!r}")
        complete_count += 1
        start = _number(event.get("ts"), field=f"event {index} ts")
        duration = _number(event.get("dur"), field=f"event {index} dur")
        end = _number(start + duration, field=f"event {index} ts + dur")
        pid = _identifier(event.get("pid"), field=f"event {index} pid")
        tid = _identifier(event.get("tid"), field=f"event {index} tid")
        if not isinstance(event.get("name"), str) or not event["name"]:
            raise ValueError(f"event {index} name must be a nonempty string")
        if not isinstance(event.get("args", {}), dict):
            raise ValueError(f"event {index} args must be an object")
        task_type = event.get("args", {}).get("Task Type")
        if task_type is not None and not isinstance(task_type, str):
            raise ValueError(f"event {index} Task Type must be a string")
        if not isinstance(event.get("cat", ""), str):
            raise ValueError(f"event {index} cat must be a string")
        start, end = max(start, window[0]), min(end, window[1])
        if start >= end:
            continue
        process_name = processes.get(pid, "")
        thread_name = threads.get((pid, tid), "")
        kind, reason = _classify(
            event,
            process_name=process_name,
            thread_name=thread_name,
            collective_patterns=collective_patterns,
        )
        intervals[kind].append((start, end))
        counts[kind] += 1
        reasons[reason] += 1
        if kind == "unknown":
            unknown.append(
                {
                    "event_index": index,
                    "name": event["name"],
                    "pid": pid,
                    "tid": tid,
                    "process_name": process_name,
                    "thread_name": thread_name,
                    "category": event.get("cat", ""),
                    "task_type": event.get("args", {}).get("Task Type"),
                    "reason": reason,
                    "clipped_interval_us": [float(start), float(end)],
                }
            )

    merged = {kind: _union(values) for kind, values in intervals.items()}
    compute, collective = merged["device_compute"], merged["device_collective"]
    overlap = _intersection(compute, collective)
    busy = _union([*compute, *collective])
    uncovered: list[_Interval] = []
    cursor = window[0]
    for start, end in busy:
        if cursor < start:
            uncovered.append((cursor, start))
        cursor = end
    if cursor < window[1]:
        uncovered.append((cursor, window[1]))
    compute_time, collective_time, overlap_time = map(
        _duration, (compute, collective, overlap)
    )
    return {
        "rank": rank,
        "source": str(path),
        "clock_id": clock_id,
        "window_us": [float(value) for value in window],
        "window_duration_us": float(window[1] - window[0]),
        "complete_event_count": complete_count,
        "event_counts_within_window": {kind: counts[kind] for kind in _KINDS},
        "classification_reasons": dict(reasons),
        **{f"{kind}_busy_us": float(_duration(merged[kind])) for kind in _KINDS},
        "compute_collective_overlap_us": float(overlap_time),
        "compute_only_us": float(compute_time - overlap_time),
        "collective_only_us": float(collective_time - overlap_time),
        "compute_or_collective_busy_us": float(_duration(busy)),
        "uncovered_compute_or_collective_us": float(_duration(uncovered)),
        "intervals_us": {
            kind: [[float(start), float(end)] for start, end in values]
            for kind, values in {
                **merged,
                "overlap": overlap,
                "uncovered": uncovered,
            }.items()
        },
        "unknown_events": unknown,
    }


def analyze_tp_timeline(
    trace_paths: Mapping[int, Path],
    *,
    windows_us: Mapping[int, tuple[object, object]],
    clock_ids: Mapping[int, str],
    collective_name_patterns: Sequence[str],
) -> dict:
    """Measure unions/intersections within each explicit rank measurement window.

    Args:
        trace_paths: One torch_npu Chrome JSON file per nonnegative integer rank.
        windows_us: Start/end bounds in each rank's native trace microsecond clock.
        clock_ids: Caller-confirmed common host/device clock label for each file;
            this does not establish synchronization between different ranks.
        collective_name_patterns: Regexes verified against actual device task
            names. An empty sequence intentionally leaves collective-looking
            tasks unknown. Patterns never match CPU work into device time.

    Returns:
        JSON-serializable per-rank durations, merged intervals, classification
        reasons and unknown event evidence. All interval arithmetic uses Decimal
        before conversion to JSON numbers (absolute endpoints can round at
        epoch scale). No cross-rank wall time is inferred.
        Uncovered time means no *classified compute or collective* was observed,
        not proven idleness. Host waits can overlap device work and do not prove
        a communication stall. Collective tasks may include device synchronization;
        their duration does not measure active network transfer alone. Captured
        busy time is not request end-to-end time.

    Raises:
        ValueError: Invalid time, window, metadata, regex, rank/clock mapping or
            unsupported duration phase. Unrecognized valid X events are unknown.
    """
    if not trace_paths or any(
        type(rank) is not int or rank < 0 for rank in trace_paths
    ):
        raise ValueError("trace_paths requires nonnegative integer ranks")
    if set(windows_us) != set(trace_paths) or set(clock_ids) != set(trace_paths):
        raise ValueError("window and clock mappings must exactly match trace ranks")
    if (
        isinstance(collective_name_patterns, (str, bytes))
        or not isinstance(collective_name_patterns, Sequence)
        or any(
            not isinstance(pattern, str) or not pattern
            for pattern in collective_name_patterns
        )
    ):
        raise ValueError(
            "collective_name_patterns must be a sequence of nonempty regex strings"
        )
    try:
        patterns = [re.compile(pattern) for pattern in collective_name_patterns]
    except (re.error, TypeError) as error:
        raise ValueError(f"invalid collective name pattern: {error}") from error
    ranks = {}
    for rank, path in sorted(trace_paths.items()):
        clock_id = clock_ids[rank]
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise ValueError(f"rank {rank} requires a nonempty clock_id")
        bounds = windows_us[rank]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"rank {rank} window requires start and end")
        window = tuple(_number(value, field=f"rank {rank} window") for value in bounds)
        if window[0] >= window[1]:
            raise ValueError(f"rank {rank} window end must exceed start")
        ranks[str(rank)] = _analyze_rank(
            Path(path),
            rank=rank,
            window=window,
            clock_id=clock_id,
            collective_patterns=patterns,
        )
    return {
        "time_unit": "us",
        "aggregation": "independent rank-local clock windows",
        "collective_name_patterns": list(collective_name_patterns),
        "ranks": ranks,
    }
