from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median

_MAX_CLOCK_ERROR_NS = 1_000_000
_TIMELINE_MARGIN_NS = 1_000_000_000
_WINDOW_CSV_FIELDS = (
    "run_id",
    "request_id",
    "request_send_epoch_ns",
    "request_send_monotonic_ns",
    "first_chunk_epoch_ns",
    "first_chunk_monotonic_ns",
    "profiler_start_request_epoch_ns",
    "profiler_start_response_epoch_ns",
    "profiler_stop_request_epoch_ns",
    "profiler_stop_response_epoch_ns",
    "stream_drain_epoch_ns",
    "stream_drain_monotonic_ns",
    "first_chunk_field",
    "ttft_ms",
    "e2e_ms",
    "input_tokens",
    "output_tokens",
    "stream_completed",
    "client_epoch_minus_monotonic_offset_ns",
    "profiler_epoch_minus_monotonic_offset_ns",
    "max_clock_error_ns",
    "clock_status",
    "clock_status_reason",
    "torch_trace_file",
    "torch_native_time_unit",
    "torch_event_count",
    "torch_native_start_us",
    "torch_native_stop_us",
    "torch_window_event_count",
    "torch_window_per_event_overlap_us_sum",
    "cann_task_time_file",
    "cann_native_time_unit",
    "cann_interval_count",
    "cann_native_start_us",
    "cann_native_stop_us",
    "cann_window_interval_count",
    "cann_window_per_event_overlap_us_sum",
    "overlap_sum_semantics",
)


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def _summarize_aicore(path: Path) -> dict:
    totals: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"task_count": 0, "total_us": 0.0}
    )
    row_count = 0
    total_us = 0.0
    for row in _read_csv(path):
        duration = _number(row.get("aicore_time(us)"))
        if duration is None or duration <= 0:
            continue
        row_count += 1
        total_us += duration
        key = (row.get("Op Name", ""), row.get("OP Type", ""))
        totals[key]["task_count"] += 1
        totals[key]["total_us"] += duration

    top_ops = [
        {
            "op_name": name,
            "op_type": op_type,
            "task_count": int(values["task_count"]),
            "total_us": round(float(values["total_us"]), 6),
        }
        for (name, op_type), values in sorted(
            totals.items(), key=lambda item: float(item[1]["total_us"]), reverse=True
        )[:10]
    ]
    return {
        "source": path.name,
        "row_count": row_count,
        "total_recorded_aicore_us": round(total_us, 6),
        "top_ops_by_cumulative_aicore_time": top_ops,
    }


def _summarize_task_timeline(path: Path) -> dict:
    intervals: list[tuple[float, float]] = []
    for row in _read_csv(path):
        start = _number(row.get("task_start(us)"))
        stop = _number(row.get("task_stop(us)"))
        if start is not None and stop is not None and stop > start:
            intervals.append((start, stop))
    if not intervals:
        return {
            "source": path.name,
            "positive_interval_count": 0,
            "envelope_us": 0.0,
            "recorded_union_us": 0.0,
            "uncovered_within_envelope_us": 0.0,
            "largest_uncovered_interval_us": 0.0,
        }

    intervals.sort()
    merged: list[list[float]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)

    envelope = max(stop for _start, stop in intervals) - min(
        start for start, _stop in intervals
    )
    union = sum(stop - start for start, stop in merged)
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    return {
        "source": path.name,
        "positive_interval_count": len(intervals),
        "envelope_us": round(envelope, 6),
        "recorded_union_us": round(union, 6),
        "uncovered_within_envelope_us": round(max(0.0, envelope - union), 6),
        "largest_uncovered_interval_us": round(max(gaps, default=0.0), 6),
    }


def _summarize_hbm(path: Path) -> dict:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT timestamp, bandwidth, hbmid, event_type "
            "FROM HBMbwData ORDER BY timestamp, event_type, hbmid"
        ).fetchall()
    finally:
        connection.close()

    samples: dict[tuple[float, str], dict[str, float | int]] = defaultdict(
        lambda: {"sum": 0.0, "channel_count": 0}
    )
    timestamps: set[float] = set()
    for timestamp, bandwidth, _hbmid, event_type in rows:
        timestamp_value = _number(str(timestamp))
        bandwidth_value = _number(str(bandwidth))
        if timestamp_value is None or bandwidth_value is None:
            continue
        timestamps.add(timestamp_value)
        sample = samples[(timestamp_value, str(event_type))]
        sample["sum"] += bandwidth_value
        sample["channel_count"] += 1

    by_type: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for (_timestamp, event_type), sample in samples.items():
        by_type[event_type].append(sample)

    event_types = {}
    for event_type, values in sorted(by_type.items()):
        sums = [float(value["sum"]) for value in values]
        channel_counts = [int(value["channel_count"]) for value in values]
        event_types[event_type] = {
            "timestamp_sample_count": len(values),
            "mean_sum_across_channels": round(sum(sums) / len(sums), 6),
            "max_sum_across_channels": round(max(sums), 6),
            "min_channel_count_per_sample": min(channel_counts),
            "max_channel_count_per_sample": max(channel_counts),
        }

    ordered_timestamps = sorted(timestamps)
    timestamp_steps = [
        right - left for left, right in zip(ordered_timestamps, ordered_timestamps[1:])
    ]
    return {
        "source": "/".join(path.parts[-3:]),
        "unit": "raw profiler bandwidth value; unit not independently validated",
        "timestamp_unit": "raw profiler timestamp unit",
        "median_timestamp_step_raw": (
            round(median(timestamp_steps), 6) if timestamp_steps else None
        ),
        "event_types": event_types,
    }


def analyze_capture(capture_dir: Path) -> dict:
    output_dir = capture_dir / "mindstudio_profiler_output"
    op_summaries = sorted(output_dir.glob("op_summary_*.csv"))
    task_summaries = sorted(output_dir.glob("task_time_*.csv"))
    hbm_databases = sorted(capture_dir.glob("device_*/sqlite/hbm.db"))
    if not op_summaries or not task_summaries or not hbm_databases:
        raise FileNotFoundError(
            f"Profiler capture is missing expected summary files: {capture_dir}"
        )

    return {
        "capture_id": capture_dir.name,
        "aicore": _summarize_aicore(op_summaries[0]),
        "task_timeline": _summarize_task_timeline(task_summaries[0]),
        "hbm_raw_bandwidth": _summarize_hbm(hbm_databases[0]),
        "memory_usage": {
            "status": "not reported",
            "reason": "The capture README says DDR/HBM usage fields are unsupported on this product.",
        },
    }


def analyze_root(input_root: Path) -> dict:
    captures = sorted(
        {path.parent.parent for path in input_root.rglob("op_summary_*.csv")},
        key=lambda path: path.as_posix(),
    )
    if not captures:
        raise FileNotFoundError(f"No op_summary_*.csv files found under {input_root}")
    project_root = Path(__file__).resolve().parents[2]
    try:
        input_root_label = input_root.resolve().relative_to(project_root).as_posix()
    except ValueError:
        input_root_label = input_root.as_posix()
    return {
        "input_root": input_root_label,
        "capture_count": len(captures),
        "captures": [analyze_capture(path) for path in captures],
        "interpretation_limits": [
            "AICore values are cumulative task counters, not wall time or per-request latency.",
            "Task interval gaps are time not covered by recorded task intervals; they do not prove NPU idleness.",
            "HBM values are aggregated across channel IDs but retain the profiler's raw unit.",
            "These captures do not correlate profiler records to individual requests.",
        ],
    }


def overlap_us(
    start_us: float,
    stop_us: float,
    window_start_us: float,
    window_stop_us: float,
) -> float:
    """Return the half-open intersection duration between two intervals."""
    return max(0.0, min(stop_us, window_stop_us) - max(start_us, window_start_us))


def _timestamp_ns(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _decimal_number(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError):
        return None
    return number if number.is_finite() else None


def _single_jsonl_record(path: Path) -> dict:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Profile request JSONL must contain an object")
                records.append(value)
    if len(records) != 1:
        raise ValueError(
            "Single-request profile must contain exactly one request record"
        )
    record = records[0]
    if record.get("status") != "success":
        raise ValueError("Single-request profile request record is not successful")
    if record.get("output_tokens") != 1 or record.get("stream_completed") is not True:
        raise ValueError(
            "Single-request profile must contain one completed output token"
        )
    if record.get("profiler_start_acknowledged") is not True:
        raise ValueError("Profiler start was not acknowledged")
    if record.get("profiler_stop_acknowledged") is not True:
        raise ValueError("Profiler stop was not acknowledged")
    if record.get("stop_callback_acknowledged") is not True:
        raise ValueError("Profiler stop callback was not acknowledged")
    if record.get("first_chunk_field") not in {"content", "reasoning_content"}:
        raise ValueError("Profile request has no recognized first output chunk")
    return record


def _single_file(root: Path, pattern: str, label: str) -> Path:
    paths = sorted(root.rglob(pattern), key=lambda path: path.as_posix())
    if not paths:
        raise FileNotFoundError(f"Profiler capture is missing {label}: {root}")
    if len(paths) != 1:
        raise ValueError(f"Profiler capture has multiple {label} files: {root}")
    return paths[0]


def _read_torch_intervals(
    profile_dir: Path,
) -> tuple[Path, list[tuple[Decimal, Decimal]]]:
    paths = sorted(
        (
            path
            for path in profile_dir.rglob("*.trace.json")
            if not path.name.endswith(".normalized.trace.json")
        ),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(
            f"Profiler capture is missing Torch trace: {profile_dir}"
        )
    if len(paths) != 1:
        raise ValueError(
            f"Profiler capture has multiple Torch trace files: {profile_dir}"
        )
    trace_path = paths[0]
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, json.JSONDecodeError):
        raise ValueError(
            f"Torch trace is incomplete or invalid: {trace_path.name}"
        ) from None
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("Torch trace does not contain a traceEvents list")

    intervals = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        start = _decimal_number(event.get("ts"))
        duration = _decimal_number(event.get("dur"))
        if start is None or duration is None or duration <= 0:
            continue
        stop = start + duration
        if stop.is_finite() and stop > start:
            intervals.append((start, stop))
    if not intervals:
        raise ValueError("Torch trace has no valid complete X events")
    return trace_path, intervals


def _read_cann_intervals(
    profile_dir: Path,
) -> tuple[Path, list[tuple[Decimal, Decimal]]]:
    task_path = _single_file(profile_dir, "task_time_*.csv", "CANN task_time CSV")
    intervals = []
    for row in _read_csv(task_path):
        start = _decimal_number(row.get("task_start(us)"))
        stop = _decimal_number(row.get("task_stop(us)"))
        if start is not None and stop is not None and start > 0 and stop > start:
            intervals.append((start, stop))
    if not intervals:
        raise ValueError("CANN task_time CSV has no valid positive intervals")
    return task_path, intervals


def _read_profiler_clock_anchor(
    profile_dir: Path,
) -> tuple[Path | None, int | None, int | None, str | None]:
    paths = sorted(
        profile_dir.rglob("profiler_info_0.json"),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        return None, None, None, "profiler_info_missing"
    if len(paths) != 1:
        return None, None, None, "profiler_info_ambiguous"
    info_path = paths[0]
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return info_path, None, None, "profiler_info_invalid"
    end_info = info.get("end_info") if isinstance(info, dict) else None
    if not isinstance(end_info, dict):
        return info_path, None, None, "profiler_end_info_missing"
    epoch_ns = _timestamp_ns(end_info.get("collectionTimeEnd"))
    monotonic_ns = _timestamp_ns(end_info.get("MonotonicTimeEnd"))
    if epoch_ns is None:
        return info_path, None, None, "profiler_epoch_anchor_missing"
    if monotonic_ns is None:
        return info_path, epoch_ns, None, "profiler_monotonic_anchor_missing"
    return info_path, epoch_ns, monotonic_ns, None


def _interval_range(
    intervals: list[tuple[Decimal, Decimal]],
) -> tuple[Decimal, Decimal]:
    return min(start for start, _stop in intervals), max(
        stop for _start, stop in intervals
    )


def _window_overlap(
    intervals: list[tuple[Decimal, Decimal]],
    *,
    epoch_minus_monotonic_ns: int,
    window_start_monotonic_ns: int,
    window_stop_monotonic_ns: int,
) -> tuple[int, float]:
    overlaps = []
    for start_epoch_us, stop_epoch_us in intervals:
        start_monotonic_ns = int(start_epoch_us * 1000) - epoch_minus_monotonic_ns
        stop_monotonic_ns = int(stop_epoch_us * 1000) - epoch_minus_monotonic_ns
        duration_ns = max(
            0,
            min(stop_monotonic_ns, window_stop_monotonic_ns)
            - max(start_monotonic_ns, window_start_monotonic_ns),
        )
        if duration_ns > 0:
            overlaps.append(duration_ns / 1000)
    return len(overlaps), round(sum(overlaps), 6)


def analyze_single_request_ttft_window(raw_run_dir: Path, profile_dir: Path) -> dict:
    """Analyze one successful request's TTFT window without changing its inputs."""
    request_path = raw_run_dir / "request.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError(
            f"Single-request profile is missing request.jsonl: {raw_run_dir}"
        )
    record = _single_jsonl_record(request_path)
    trace_path, torch_intervals = _read_torch_intervals(profile_dir)
    task_path, cann_intervals = _read_cann_intervals(profile_dir)
    profiler_info_path, profiler_epoch_ns, profiler_monotonic_ns, anchor_error = (
        _read_profiler_clock_anchor(profile_dir)
    )

    timestamp_fields = (
        "request_send_epoch_ns",
        "request_send_monotonic_ns",
        "first_chunk_epoch_ns",
        "first_chunk_monotonic_ns",
        "stream_drain_epoch_ns",
        "stream_drain_monotonic_ns",
        "profiler_start_request_epoch_ns",
        "profiler_start_response_epoch_ns",
        "profiler_stop_request_epoch_ns",
        "profiler_stop_response_epoch_ns",
    )
    timestamps = {field: _timestamp_ns(record.get(field)) for field in timestamp_fields}
    if any(value is None for value in timestamps.values()):
        raise ValueError("Successful profile record is missing required timestamps")
    client_clock_samples = (
        (timestamps["request_send_epoch_ns"], timestamps["request_send_monotonic_ns"]),
        (timestamps["first_chunk_epoch_ns"], timestamps["first_chunk_monotonic_ns"]),
        (timestamps["stream_drain_epoch_ns"], timestamps["stream_drain_monotonic_ns"]),
    )
    client_offsets = [epoch - monotonic for epoch, monotonic in client_clock_samples]
    client_offset_ns = sorted(client_offsets)[len(client_offsets) // 2]
    client_interval_errors = [
        abs((right_epoch - left_epoch) - (right_monotonic - left_monotonic))
        for (left_epoch, left_monotonic), (right_epoch, right_monotonic) in zip(
            client_clock_samples, client_clock_samples[1:]
        )
    ]
    max_clock_error_ns = max(client_interval_errors, default=0)
    if profiler_epoch_ns is not None and profiler_monotonic_ns is not None:
        profiler_offset_ns = profiler_epoch_ns - profiler_monotonic_ns
        max_clock_error_ns = max(
            max_clock_error_ns,
            *(abs(offset - profiler_offset_ns) for offset in client_offsets),
        )
    else:
        profiler_offset_ns = None

    torch_native_start_us, torch_native_stop_us = _interval_range(torch_intervals)
    cann_native_start_us, cann_native_stop_us = _interval_range(cann_intervals)
    clock_status_reason = anchor_error
    if clock_status_reason is None and max_clock_error_ns > _MAX_CLOCK_ERROR_NS:
        clock_status_reason = "epoch_monotonic_error_over_1ms"
    if clock_status_reason is None:
        ordered = (
            timestamps["profiler_start_request_epoch_ns"]
            <= timestamps["profiler_start_response_epoch_ns"]
            <= timestamps["request_send_epoch_ns"]
            <= timestamps["first_chunk_epoch_ns"]
            <= timestamps["profiler_stop_request_epoch_ns"]
            <= timestamps["profiler_stop_response_epoch_ns"]
            <= timestamps["stream_drain_epoch_ns"]
        )
        if not ordered:
            clock_status_reason = "client_timestamp_order_invalid"
    if clock_status_reason is None:
        capture_start_ns = timestamps["profiler_start_request_epoch_ns"]
        capture_end_ns = profiler_epoch_ns
        for name, bounds in (
            ("torch", (torch_native_start_us, torch_native_stop_us)),
            ("cann", (cann_native_start_us, cann_native_stop_us)),
        ):
            start_ns, stop_ns = (int(value * 1000) for value in bounds)
            if (
                capture_end_ns is None
                or stop_ns < capture_start_ns - _TIMELINE_MARGIN_NS
                or start_ns > capture_end_ns + _TIMELINE_MARGIN_NS
            ):
                clock_status_reason = f"{name}_time_domain_unverified"
                break

    clock_verified = clock_status_reason is None and profiler_offset_ns is not None
    torch_window_count = None
    torch_overlap_sum = None
    cann_window_count = None
    cann_overlap_sum = None
    if clock_verified:
        torch_window_count, torch_overlap_sum = _window_overlap(
            torch_intervals,
            epoch_minus_monotonic_ns=profiler_offset_ns,
            window_start_monotonic_ns=timestamps["request_send_monotonic_ns"],
            window_stop_monotonic_ns=timestamps["first_chunk_monotonic_ns"],
        )
        cann_window_count, cann_overlap_sum = _window_overlap(
            cann_intervals,
            epoch_minus_monotonic_ns=profiler_offset_ns,
            window_start_monotonic_ns=timestamps["request_send_monotonic_ns"],
            window_stop_monotonic_ns=timestamps["first_chunk_monotonic_ns"],
        )

    torch_trace_label = trace_path.relative_to(profile_dir).as_posix()
    cann_task_label = task_path.relative_to(profile_dir).as_posix()
    return {
        "run_id": record.get("run_id"),
        "request_id": record.get("request_id"),
        **timestamps,
        "first_chunk_field": record["first_chunk_field"],
        "ttft_ms": record.get("ttft_ms"),
        "e2e_ms": record.get("e2e_ms"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "stream_completed": record.get("stream_completed"),
        "client_epoch_minus_monotonic_offset_ns": client_offset_ns,
        "profiler_epoch_minus_monotonic_offset_ns": profiler_offset_ns,
        "max_clock_error_ns": max_clock_error_ns,
        "clock_status": "verified" if clock_verified else "unverified",
        "clock_status_reason": clock_status_reason,
        "torch_trace_file": torch_trace_label,
        "torch_native_time_unit": (
            "Unix epoch microseconds"
            if clock_verified
            else "microseconds (unverified time domain)"
        ),
        "torch_event_count": len(torch_intervals),
        "torch_native_start_us": torch_native_start_us,
        "torch_native_stop_us": torch_native_stop_us,
        "torch_window_event_count": torch_window_count,
        "torch_window_per_event_overlap_us_sum": torch_overlap_sum,
        "cann_task_time_file": cann_task_label,
        "cann_native_time_unit": (
            "Unix epoch microseconds"
            if clock_verified
            else "microseconds (unverified time domain)"
        ),
        "cann_interval_count": len(cann_intervals),
        "cann_native_start_us": cann_native_start_us,
        "cann_native_stop_us": cann_native_stop_us,
        "cann_window_interval_count": cann_window_count,
        "cann_window_per_event_overlap_us_sum": cann_overlap_sum,
        "overlap_sum_semantics": "per-event sum; overlapping intervals may count time more than once",
    }


def write_single_request_ttft_window_csv(report: dict, analysis_dir: Path) -> Path:
    if analysis_dir.exists():
        raise FileExistsError(f"Analysis output already exists: {analysis_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    output_path = analysis_dir / "single_request_ttft_window.csv"
    with output_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=_WINDOW_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerow(report)
    return output_path
