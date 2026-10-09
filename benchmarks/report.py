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
from collections import defaultdict
from pathlib import Path
from statistics import median


_COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2")


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
    agent = [record for record in records if record.get("result_family") == "agent"]
    ablation = [
        record for record in records if record.get("result_family") == "ablation"
    ]
    specs = (
        ("throughput_concurrency.svg", "Concurrency vs Token Throughput", baseline, "concurrency", "token_throughput", ("target_input_tokens", "target_output_tokens", "tp_size"), "Concurrency", "Tokens/s"),
        ("ttft_concurrency.svg", "Concurrency vs p95 TTFT", baseline, "concurrency", "ttft_p95_ms", ("target_input_tokens", "target_output_tokens", "tp_size"), "Concurrency", "p95 TTFT (ms)"),
        ("tpot_concurrency.svg", "Concurrency vs p95 TPOT", baseline, "concurrency", "tpot_p95_ms", ("target_input_tokens", "target_output_tokens", "tp_size"), "Concurrency", "p95 TPOT (ms)"),
        ("chunk_size.svg", "Chunk Size Trade-off", records, "chunk_size", "token_throughput", ("result_family", "variant", "concurrency", "target_input_tokens", "target_output_tokens"), "Chunk size", "Tokens/s"),
        ("prefix_cache.svg", "Prefix Reuse Ratio vs p95 TTFT", agent, "shared_prefix_ratio", "ttft_p95_ms", ("concurrency", "turns", "target_prefix_tokens"), "Prefix reuse ratio", "p95 TTFT (ms)"),
        ("prefix_throughput.svg", "Prefix Reuse Ratio vs Token Throughput", agent, "shared_prefix_ratio", "token_throughput", ("concurrency", "turns", "target_prefix_tokens"), "Prefix reuse ratio", "Tokens/s"),
        ("session_reuse.svg", "Conversation Turn vs p95 TTFT", records, "turn_number", "ttft_p95_ms", ("reuse_mode", "concurrency"), "Conversation turn", "p95 TTFT (ms)"),
        ("kv_cache.svg", "KV Block Size vs Throughput", records, "block_size", "token_throughput", ("result_family", "concurrency", "tp_size"), "Block size", "Tokens/s"),
        ("ablation.svg", "Ablation vs Throughput", ablation, "experiment_index", "token_throughput", ("experiment_matrix", "target_input_tokens", "target_output_tokens", "tp_size"), "Experiment (E0–E3)", "Tokens/s"),
        ("tp_comparison.svg", "Tensor Parallel Size vs Throughput", records, "tp_size", "token_throughput", ("result_family", "served_model_name", "concurrency"), "TP size", "Tokens/s"),
    )
    generated = []
    for filename, title, source_records, x_key, y_key, group_keys, x_label, y_label in specs:
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
                relative_path.parts[0],
            )
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = {**json.loads(line), "result_family": result_family}
            if result_family == "ablation":
                matrix_index = 0 if raw_dir.name == "ablation" else 1
                matrix_index = min(matrix_index, len(relative_path.parts) - 1)
                record["experiment_matrix"] = relative_path.parts[matrix_index]
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate course-project CSV and SVG reports.")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    records = load_summary_records(args.raw_dir)
    if not records:
        raise SystemExit(f"No *.summary.jsonl records found under {args.raw_dir}")
    for path in generate_standard_reports(records, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
