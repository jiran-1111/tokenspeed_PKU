#!/usr/bin/env python3
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

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

_COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2")

_SESSION_KV_CSV_FIELDS = (
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
)
_SESSION_KV_METADATA = (
    ("status", "completed"),
    ("model_id", "Qwen/Qwen3.5-4B"),
    ("frontier_policy", "current-frontier"),
    ("measurement_scope", "serial Engine.generate request latency; not TTFT"),
)


def _session_kv_value(payload: dict, field: str, path: Path) -> object:
    value: object = payload
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"{path}: {field} is missing or malformed")
        value = value[part]
    return value


def _session_kv_positive_integer(value: object, path: Path, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive integer")
    return value


def _session_kv_latency(value: object, path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {field} must be a positive finite number")
    try:
        latency = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{path}: {field} must be a positive finite number") from None
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError(f"{path}: {field} must be a positive finite number")
    return latency


def _load_session_kv_file(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON input: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object")

    for field, expected in _SESSION_KV_METADATA:
        actual = _session_kv_value(payload, field, path)
        if actual != expected or not isinstance(actual, str):
            raise ValueError(f"{path}: {field} must be {expected!r}")

    if _session_kv_value(payload, "generic_prefix_caching", path) is not False:
        raise ValueError(f"{path}: generic_prefix_caching must be false")
    if _session_kv_value(payload, "kvstore_disabled", path) is not True:
        raise ValueError(f"{path}: kvstore_disabled must be true")
    tp_size = _session_kv_value(
        payload, "launch_configuration.model_config.tp_size", path
    )
    if type(tp_size) is not int or tp_size != 1:
        raise ValueError(f"{path}: launch_configuration.model_config.tp_size must be 1")

    run_id = _session_kv_value(payload, "run_id", path)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{path}: run_id must be a nonempty string")
    model_revision = _session_kv_value(payload, "model_revision", path)
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError(f"{path}: model_revision must be a nonempty string")

    measurements = _session_kv_value(payload, "measurements", path)
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path}: measurements must be a nonempty list")

    rows = []
    for index, item in enumerate(measurements):
        prefix = f"measurements[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{path}: {prefix} must be an object")
        context_length_target = _session_kv_positive_integer(
            _session_kv_value(item, "context_length_target", path),
            path,
            f"{prefix}.context_length_target",
        )
        initial_context_tokens = _session_kv_positive_integer(
            _session_kv_value(item, "initial_context_tokens", path),
            path,
            f"{prefix}.initial_context_tokens",
        )
        repetition = _session_kv_positive_integer(
            _session_kv_value(item, "repetition", path), path, f"{prefix}.repetition"
        )
        execution_order = _session_kv_value(item, "execution_order", path)
        if (
            not isinstance(execution_order, list)
            or len(execution_order) != 2
            or not all(isinstance(part, str) for part in execution_order)
            or sorted(execution_order) != ["session", "stateless"]
        ):
            raise ValueError(
                f"{path}: {prefix}.execution_order must contain session and stateless once each"
            )

        measurement = _session_kv_value(item, "measurement", path)
        if not isinstance(measurement, dict):
            raise ValueError(f"{path}: {prefix}.measurement must be an object")
        if _session_kv_value(measurement, "output_ids_match", path) is not True:
            raise ValueError(
                f"{path}: {prefix}.measurement.output_ids_match must be true"
            )
        session_latency = _session_kv_latency(
            _session_kv_value(measurement, "session.request_latency_ms", path),
            path,
            f"{prefix}.measurement.session.request_latency_ms",
        )
        stateless_latency = _session_kv_latency(
            _session_kv_value(measurement, "stateless.request_latency_ms", path),
            path,
            f"{prefix}.measurement.stateless.request_latency_ms",
        )
        reduction = 100.0 * (1.0 - session_latency / stateless_latency)
        if not math.isfinite(reduction):
            raise ValueError(
                f"{path}: {prefix}.measurement.request_latency_ms produces a non-finite reduction"
            )

        rows.append(
            {
                "run_id": run_id,
                "model_id": payload["model_id"],
                "model_revision": model_revision,
                "frontier_policy": payload["frontier_policy"],
                "context_length_target": context_length_target,
                "initial_context_tokens": initial_context_tokens,
                "repetition": repetition,
                "execution_order": execution_order,
                "session_request_latency_ms": session_latency,
                "stateless_request_latency_ms": stateless_latency,
                "paired_reduction_percent": reduction,
                "measurement_scope": payload["measurement_scope"],
            }
        )
    return rows


def load_session_kv_pairs(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.rglob("*session-kv-perf-*.json")):
        rows.extend(_load_session_kv_file(path))
    return sorted(
        rows,
        key=lambda row: (
            row["context_length_target"],
            row["run_id"],
            row["repetition"],
        ),
    )


def _session_kv_svg(rows: list[dict]) -> str:
    width, height = 900, 560
    left, right, top, plot_bottom = 90, 40, 65, 415
    plot_width = width - left - right
    plot_height = plot_bottom - top
    targets = sorted({row["context_length_target"] for row in rows})
    group_spacing = plot_width / len(targets)
    centers = {
        target: left + group_spacing * (index + 0.5)
        for index, target in enumerate(targets)
    }
    grouped_rows = {
        target: sorted(
            (row for row in rows if row["context_length_target"] == target),
            key=lambda row: (row["run_id"], row["repetition"]),
        )
        for target in targets
    }

    reductions = [row["paired_reduction_percent"] for row in rows]
    minimum, maximum = min(reductions), max(reductions)
    padding = max((maximum - minimum) * 0.1, abs(minimum) * 0.1, 1.0)
    y_min, y_max = minimum - padding, maximum + padding
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        scale = max(abs(minimum), abs(maximum), 1.0)
        normalized_min = minimum / scale
        normalized_max = maximum / scale
        normalized_padding = max(
            (normalized_max - normalized_min) * 0.1,
            abs(normalized_min) * 0.1,
            0.01,
        )
        y_min = normalized_min - normalized_padding
        y_max = normalized_max + normalized_padding
        scaled = True
    else:
        scaled = False

    def y_position(value: float) -> float:
        plotted = value / scale if scaled else value
        low = y_min
        high = y_max
        return plot_bottom - (plotted - low) / (high - low) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">Session KV paired latency reduction</text>',
        f'<line x1="{left}" y1="{plot_bottom}" x2="{left + plot_width}" y2="{plot_bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="#111827"/>',
    ]
    for index in range(6):
        fraction = index / 5
        y_value = y_max - fraction * (y_max - y_min)
        y = top + fraction * plot_height
        if scaled:
            label_value = y_value * scale
            label = f"{label_value:.3g}"
        else:
            label = f"{y_value:.3g}"
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )

    for target in targets:
        center = centers[target]
        target_text = html.escape(str(target))
        target_attribute = html.escape(str(target), quote=True)
        elements.append(
            f'<line x1="{center:.2f}" y1="{plot_bottom}" x2="{center:.2f}" y2="{plot_bottom + 5}" stroke="#111827"/>'
        )
        elements.append(
            f'<text x="{center:.2f}" y="{plot_bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{target_text}</text>'
        )
        group_rows = grouped_rows[target]
        count = len(group_rows)
        group_reductions = [row["paired_reduction_percent"] for row in group_rows]
        group_median = median(group_reductions)
        median_y = y_position(group_median)
        radius = 7
        diamond_points = " ".join(
            (
                f"{center:.2f},{median_y - radius:.2f}",
                f"{center + radius:.2f},{median_y:.2f}",
                f"{center:.2f},{median_y + radius:.2f}",
                f"{center - radius:.2f},{median_y:.2f}",
            )
        )
        for observation_index, row in enumerate(group_rows):
            offset = (
                0.0
                if count == 1
                else group_spacing * (-0.12 + 0.24 * observation_index / (count - 1))
            )
            x = center + offset
            y = y_position(row["paired_reduction_percent"])
            run_id = html.escape(row["run_id"], quote=True)
            elements.append(
                f'<circle class="pair" cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb" data-reduction-percent="{html.escape(repr(row["paired_reduction_percent"]), quote=True)}" data-run-id="{run_id}" data-repetition="{html.escape(str(row["repetition"]), quote=True)}"/>'
            )
        elements.append(
            f'<polygon class="median" points="{diamond_points}" fill="#dc2626" stroke="#7f1d1d" data-context-length-target="{target_attribute}" data-median-percent="{html.escape(repr(group_median), quote=True)}"/>'
        )

    elements.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="470" text-anchor="middle" font-family="sans-serif" font-size="15">Target context length</text>',
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 24 {top + plot_height / 2:.2f})">Paired latency reduction (%)</text>',
            '<circle class="legend-pair" cx="300" cy="500" r="4" fill="#2563eb"/>',
            '<text x="312" y="504" font-family="sans-serif" font-size="12">Paired observations (circle)</text>',
            '<polygon class="legend-median" points="545,493 552,500 545,507 538,500" fill="#dc2626" stroke="#7f1d1d"/>',
            '<text x="560" y="504" font-family="sans-serif" font-size="12">Context median (diamond)</text>',
            '<text x="450" y="540" text-anchor="middle" font-family="sans-serif" font-size="12">serial Engine.generate wall time; not TTFT</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def generate_session_kv_paired_report(raw_dir: Path, output_dir: Path) -> list[Path]:
    rows = load_session_kv_pairs(raw_dir)
    if not rows:
        return []
    svg = _session_kv_svg(rows)

    csv_path = output_dir / "session_kv_paired_reduction.csv"
    svg_path = output_dir / "session_kv_paired_reduction.svg"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_SESSION_KV_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "execution_order": json.dumps(
                        row["execution_order"], separators=(",", ":")
                    ),
                }
            )
    svg_path.write_text(svg, encoding="utf-8")
    return [csv_path, svg_path]


def write_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    width, height = 900, 540
    left, right, top, bottom = 90, 40, 65, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in series.values() for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = left + fraction * plot_width
        y_value = y_max - fraction * (y_max - y_min)
        y = top + fraction * plot_height
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.3g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.3g}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(x_label)}</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(y_label)}</text>',
        ]
    )
    for index, (label, points) in enumerate(sorted(series.items())):
        color = _COLORS[index % len(_COLORS)]
        ordered = sorted(points)
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in ordered
        )
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y in ordered:
            elements.append(
                f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" r="4" fill="{color}"/>'
            )
        legend_y = top + 20 * index
        elements.append(
            f'<text x="{left + 12}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="{color}">{html.escape(label)}</text>'
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def _series_for(
    records: list[dict],
    *,
    x_key: str,
    y_key: str,
    group_keys: tuple[str, ...],
) -> dict[str, list[tuple[float, float]]]:
    grouped_values = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get(x_key) is None or record.get(y_key) is None:
            continue
        label_parts = [str(record.get("served_model_name", "model"))]
        label_parts.extend(
            f"{key}={record[key]}"
            for key in group_keys
            if key in record and record[key] is not None
        )
        grouped_values[", ".join(label_parts)][float(record[x_key])].append(
            float(record[y_key])
        )
    return {
        label: [(x_value, median(values)) for x_value, values in sorted(by_x.items())]
        for label, by_x in grouped_values.items()
    }


def _sweep_series(
    records: list[dict],
    *,
    x_key: str,
    y_key: str,
    group_keys: tuple[str, ...],
) -> dict[str, list[tuple[float, float]]]:
    series = _series_for(
        records,
        x_key=x_key,
        y_key=y_key,
        group_keys=group_keys,
    )
    return {
        label: points
        for label, points in series.items()
        if len({x for x, _ in points}) >= 2
    }


def _write_summary_csv(records: list[dict], path: Path) -> None:
    fieldnames = sorted({key for record in records for key in record})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def generate_standard_reports(records: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(records, output_dir / "summary.csv")
    baseline = [
        record
        for record in records
        if record.get("result_family") == "baseline"
        and record.get("server_metrics_available") is True
        and record.get("prefix_cache_hit_ratio") is not None
        and float(record["prefix_cache_hit_ratio"]) <= 0.01
    ]
    kv_cache_usage = [
        record
        for record in baseline
        if record.get("target_input_tokens") == 128
        and record.get("target_output_tokens") == 32
        and record.get("tp_size") == 1
    ]
    agent = [record for record in records if record.get("result_family") == "agent"]
    ablation = [
        record for record in records if record.get("result_family") == "ablation"
    ]
    specs = (
        (
            "throughput_concurrency.svg",
            "Concurrency vs Token Throughput",
            baseline,
            "concurrency",
            "token_throughput",
            ("target_input_tokens", "target_output_tokens", "tp_size"),
            "Concurrency",
            "Tokens/s",
        ),
        (
            "ttft_concurrency.svg",
            "Concurrency vs p95 TTFT",
            baseline,
            "concurrency",
            "ttft_p95_ms",
            ("target_input_tokens", "target_output_tokens", "tp_size"),
            "Concurrency",
            "p95 TTFT (ms)",
        ),
        (
            "tpot_concurrency.svg",
            "Concurrency vs p95 TPOT",
            baseline,
            "concurrency",
            "tpot_p95_ms",
            ("target_input_tokens", "target_output_tokens", "tp_size"),
            "Concurrency",
            "p95 TPOT (ms)",
        ),
        (
            "itl_p50_concurrency.svg",
            "Concurrency vs Run-level ITL P50",
            baseline,
            "concurrency",
            "itl_p50_ms",
            ("target_input_tokens", "target_output_tokens", "tp_size"),
            "Concurrency",
            "Run-level ITL P50 (ms)",
        ),
        (
            "itl_p95_concurrency.svg",
            "Concurrency vs Run-level ITL P95",
            baseline,
            "concurrency",
            "itl_p95_ms",
            ("target_input_tokens", "target_output_tokens", "tp_size"),
            "Concurrency",
            "Run-level ITL P95 (ms)",
        ),
        (
            "chunk_size.svg",
            "Chunk Size Trade-off",
            records,
            "chunk_size",
            "token_throughput",
            (
                "result_family",
                "variant",
                "concurrency",
                "target_input_tokens",
                "target_output_tokens",
            ),
            "Chunk size",
            "Tokens/s",
        ),
        (
            "prefix_cache.svg",
            "Prefix Reuse Ratio vs p95 TTFT",
            agent,
            "shared_prefix_ratio",
            "ttft_p95_ms",
            ("concurrency", "turns", "target_prefix_tokens"),
            "Prefix reuse ratio",
            "p95 TTFT (ms)",
        ),
        (
            "prefix_throughput.svg",
            "Prefix Reuse Ratio vs Token Throughput",
            agent,
            "shared_prefix_ratio",
            "token_throughput",
            ("concurrency", "turns", "target_prefix_tokens"),
            "Prefix reuse ratio",
            "Tokens/s",
        ),
        (
            "session_reuse.svg",
            "Conversation Turn vs p95 TTFT",
            records,
            "turn_number",
            "ttft_p95_ms",
            ("reuse_mode", "concurrency"),
            "Conversation turn",
            "p95 TTFT (ms)",
        ),
        (
            "kv_cache.svg",
            "KV Block Size vs Throughput",
            records,
            "block_size",
            "token_throughput",
            ("result_family", "concurrency", "tp_size"),
            "Block size",
            "Tokens/s",
        ),
        (
            "kv_cache_usage.svg",
            "Scheduler KV Active-Page Ratio vs Concurrency",
            kv_cache_usage,
            "concurrency",
            "kv_cache_usage_ratio_peak",
            (),
            "Concurrency",
            "Peak active pages / total pages ratio (not HBM occupancy)",
        ),
        (
            "ablation.svg",
            "Ablation vs Throughput",
            ablation,
            "experiment_index",
            "token_throughput",
            (
                "experiment_matrix",
                "target_input_tokens",
                "target_output_tokens",
                "tp_size",
            ),
            "Experiment (E0–E3)",
            "Tokens/s",
        ),
        (
            "tp_comparison.svg",
            "Tensor Parallel Size vs Throughput",
            records,
            "tp_size",
            "token_throughput",
            ("result_family", "served_model_name", "concurrency"),
            "TP size",
            "Tokens/s",
        ),
    )
    generated = []
    for (
        filename,
        title,
        source_records,
        x_key,
        y_key,
        group_keys,
        x_label,
        y_label,
    ) in specs:
        series = _sweep_series(
            source_records,
            x_key=x_key,
            y_key=y_key,
            group_keys=group_keys,
        )
        if not series:
            continue
        path = output_dir / filename
        write_line_plot(
            path,
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=series,
        )
        generated.append(path)
    return generated


def load_summary_records(raw_dir: Path) -> list[dict]:
    records = []
    result_families = {"baseline", "agent", "ablation", "tuning"}
    for path in sorted(raw_dir.rglob("*.summary.jsonl")):
        relative_path = path.relative_to(raw_dir)
        result_family = (
            raw_dir.name
            if raw_dir.name in result_families
            else next(
                (
                    part
                    for part in reversed(raw_dir.parts[:-1])
                    if part in result_families
                ),
                (
                    "baseline"
                    if relative_path.parts[0] == "benchmark"
                    or raw_dir.name == "benchmark"
                    else relative_path.parts[0]
                ),
            )
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = {**json.loads(line), "result_family": result_family}
            if result_family == "ablation":
                if raw_dir.name == result_family:
                    family_relative_path = relative_path
                else:
                    family_root = next(
                        (
                            parent
                            for parent in raw_dir.parents
                            if parent.name == result_family
                        ),
                        None,
                    )
                    family_relative_path = (
                        path.relative_to(family_root)
                        if family_root is not None
                        else Path(*relative_path.parts[1:])
                    )
                if family_relative_path.parts:
                    record["experiment_matrix"] = family_relative_path.parts[0]
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate course-project CSV and SVG reports."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    records = load_summary_records(args.raw_dir)
    if not records:
        raise SystemExit(f"No *.summary.jsonl records found under {args.raw_dir}")
    generated = generate_standard_reports(records, args.output_dir)
    generated.extend(generate_session_kv_paired_report(args.raw_dir, args.output_dir))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
