#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any
from xml.etree import ElementTree

CSV_NAME = "session_kv_multiturn_ttft.csv"
SVG_NAME = "session_kv_multiturn_ttft.svg"
CSV_FIELDS = (
    "repetition",
    "turn",
    "full_context_tokens",
    "session_suffix_tokens",
    "session_ttft_ms",
    "stateless_ttft_ms",
    "reused_context_tokens",
    "newly_prefilled_tokens",
    "output_ids_match",
    "ttft_reduction_pct",
    "session_turn_median_ttft_ms",
    "session_turn_min_ttft_ms",
    "session_turn_max_ttft_ms",
    "stateless_turn_median_ttft_ms",
    "stateless_turn_min_ttft_ms",
    "stateless_turn_max_ttft_ms",
)
REPETITIONS = 4
TURNS = 16
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 8
SVG_NS = "http://www.w3.org/2000/svg"


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _token_ids(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        type(token_id) is not int for token_id in value
    ):
        raise ValueError(f"{field} must be a list of integer token IDs")
    return value


def _ttft(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _validate_request(
    pair: Mapping[str, Any], *, path: str, expected_input_tokens: int
) -> tuple[list[int], list[int], float]:
    request = pair.get(path)
    if not isinstance(request, Mapping):
        raise ValueError(f"{path} request must be an object")
    if request.get("path") != path:
        raise ValueError(f"{path} request path is invalid")
    input_ids = _token_ids(request.get("input_ids"), f"{path}.input_ids")
    input_token_count = request.get("input_token_count")
    if type(input_token_count) is not int or input_token_count != expected_input_tokens:
        raise ValueError(f"{path} input token count does not match the pair")
    if len(input_ids) != expected_input_tokens:
        raise ValueError(f"{path} input IDs do not match the recorded token count")
    output_ids = _token_ids(request.get("output_ids"), f"{path}.output_ids")
    if not output_ids:
        raise ValueError(f"{path} output IDs must not be empty")
    if len(output_ids) > MAX_NEW_TOKENS:
        raise ValueError(f"{path} output exceeds the 8-token output limit")
    output_token_count = request.get("output_token_count")
    if type(output_token_count) is not int or output_token_count != len(output_ids):
        raise ValueError(f"{path} output token count does not match output IDs")
    ttft_ms = _ttft(request.get("ttft_ms"), f"{path}.ttft_ms")
    return input_ids, output_ids, ttft_ms


def _validate_pair(pair: object, index: int) -> dict[str, Any]:
    label = f"measurements[{index}]"
    if not isinstance(pair, Mapping):
        raise ValueError(f"{label} must be an object")
    repetition = pair.get("repetition")
    turn = pair.get("turn")
    if type(repetition) is not int or not 1 <= repetition <= REPETITIONS:
        raise ValueError(f"{label}.repetition must be between 1 and {REPETITIONS}")
    if type(turn) is not int or not 1 <= turn <= TURNS:
        raise ValueError(f"{label}.turn must be between 1 and {TURNS}")
    expected_order = (
        ["session", "stateless"] if repetition % 2 else ["stateless", "session"]
    )
    if pair.get("execution_order") != expected_order:
        raise ValueError(f"{label}.execution_order is invalid")

    full_context_tokens = _positive_int(
        pair.get("full_context_tokens"), f"{label}.full_context_tokens"
    )
    session_suffix_tokens = _positive_int(
        pair.get("session_suffix_tokens"), f"{label}.session_suffix_tokens"
    )
    if full_context_tokens + MAX_NEW_TOKENS > MAX_MODEL_LEN:
        raise ValueError(f"{label}.context exceeds the pinned 4096-token limit")
    if session_suffix_tokens > full_context_tokens:
        raise ValueError(f"{label}.session_suffix_tokens exceeds full context")
    if turn == 1 and session_suffix_tokens != full_context_tokens:
        raise ValueError(f"{label}.first-turn Session input must be the full context")

    session_input, session_output, session_ttft = _validate_request(
        pair, path="session", expected_input_tokens=session_suffix_tokens
    )
    stateless_input, stateless_output, stateless_ttft = _validate_request(
        pair, path="stateless", expected_input_tokens=full_context_tokens
    )
    if turn == 1 and session_input != stateless_input:
        raise ValueError(f"{label}.first-turn input IDs differ between paths")
    if pair.get("output_ids_match") is not True or session_output != stateless_output:
        raise ValueError(f"{label}.output IDs do not match between paths")
    if not session_input or not stateless_input:
        raise ValueError(f"{label}.input IDs must not be empty")

    session_request = pair["session"]
    metadata = session_request.get("session_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label}.session metadata is missing or invalid")
    prior_history_tokens = full_context_tokens - session_suffix_tokens
    reused = max(prior_history_tokens - 1, 0)
    newly_prefilled = full_context_tokens - reused
    expected_metadata = {
        "total_context_tokens": full_context_tokens,
        "newly_prefilled_tokens": newly_prefilled,
        "reused_context_tokens": reused,
    }
    actual_metadata = {key: metadata.get(key) for key in expected_metadata}
    if any(type(value) is not int for value in actual_metadata.values()) or (
        actual_metadata != expected_metadata
    ):
        raise ValueError(f"{label}.Session metadata does not match current-frontier")

    reduction = 100.0 * (1.0 - session_ttft / stateless_ttft)
    recorded_reduction = pair.get("ttft_reduction_pct")
    if isinstance(recorded_reduction, bool) or not isinstance(
        recorded_reduction, (int, float)
    ):
        raise ValueError(f"{label}.TTFT reduction is invalid")
    try:
        recorded_reduction_value = float(recorded_reduction)
    except (OverflowError, ValueError):
        raise ValueError(f"{label}.TTFT reduction is invalid") from None
    if not math.isfinite(recorded_reduction_value) or not math.isclose(
        recorded_reduction_value, reduction, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{label}.TTFT reduction is invalid")

    return {
        "repetition": repetition,
        "turn": turn,
        "full_context_tokens": full_context_tokens,
        "session_suffix_tokens": session_suffix_tokens,
        "session_ttft_ms": session_ttft,
        "stateless_ttft_ms": stateless_ttft,
        "reused_context_tokens": reused,
        "newly_prefilled_tokens": newly_prefilled,
        "output_ids_match": True,
        "ttft_reduction_pct": reduction,
        "_session_input_ids": session_input,
        "_stateless_input_ids": stateless_input,
        "_session_output_ids": session_output,
    }


def _validated_rows(raw_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid raw JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("raw run must be a JSON object")
    if payload.get("status") != "completed":
        raise ValueError("raw run must have status='completed'")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != REPETITIONS * TURNS:
        raise ValueError("completed run must contain exactly 64 measurement pairs")

    rows = [_validate_pair(pair, index) for index, pair in enumerate(measurements)]
    seen = {(row["repetition"], row["turn"]) for row in rows}
    expected = {
        (repetition, turn)
        for repetition in range(1, REPETITIONS + 1)
        for turn in range(1, TURNS + 1)
    }
    if seen != expected:
        raise ValueError(
            "completed run must contain each of the 64 repetition-turn pairs once"
        )
    rows.sort(key=lambda row: (row["repetition"], row["turn"]))
    for repetition in range(1, REPETITIONS + 1):
        repetition_rows = [row for row in rows if row["repetition"] == repetition]
        prior_history_ids: list[int] | None = None
        for row in repetition_rows:
            label = f"repetition {repetition} turn {row['turn']}"
            full_ids = row["_stateless_input_ids"]
            session_ids = row["_session_input_ids"]
            if row["turn"] == 1:
                prior_history_ids = full_ids + row["_session_output_ids"]
                continue
            assert prior_history_ids is not None
            if full_ids[: len(prior_history_ids)] != prior_history_ids:
                raise ValueError(
                    f"{label}: stateless full history does not extend the prior turn"
                )
            expected_suffix = full_ids[len(prior_history_ids) :]
            if session_ids != expected_suffix:
                raise ValueError(
                    f"{label}: Session input IDs do not match the full-history suffix"
                )
            prior_history_ids = full_ids + row["_session_output_ids"]

    for row in rows:
        row.pop("_session_input_ids")
        row.pop("_stateless_input_ids")
        row.pop("_session_output_ids")
    return rows


def _rows_with_turn_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    statistics_by_turn = {}
    for turn in range(1, TURNS + 1):
        turn_rows = [row for row in rows if row["turn"] == turn]
        if len(turn_rows) != REPETITIONS:
            raise ValueError(f"turn {turn} must have exactly four observations")
        statistics_by_turn[turn] = {}
        for path in ("session", "stateless"):
            values = [row[f"{path}_ttft_ms"] for row in turn_rows]
            statistics_by_turn[turn][path] = {
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }

    enriched = []
    for row in rows:
        output = dict(row)
        for path in ("session", "stateless"):
            for statistic, value in statistics_by_turn[row["turn"]][path].items():
                output[f"{path}_turn_{statistic}_ttft_ms"] = value
        enriched.append(output)
    return enriched


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    "True"
                    if value is True
                    else (
                        _format_number(value)
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        else value
                    )
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _svg_text(rows: list[dict[str, Any]]) -> str:
    ElementTree.register_namespace("", SVG_NS)
    width, height = 1000, 600
    left, right, top, bottom = 82, 32, 64, 88
    plot_bottom = height - bottom
    plot_width = width - left - right
    plot_height = plot_bottom - top
    max_value = max(
        row[f"{path}_ttft_ms"] for row in rows for path in ("session", "stateless")
    )
    y_max = max_value * 1.12
    if not math.isfinite(y_max):
        raise ValueError("TTFT range cannot be represented in the SVG")

    def x_position(turn: int) -> float:
        return left + plot_width * (turn - 0.5) / TURNS

    def y_position(value: float) -> float:
        return plot_bottom - (value / y_max) * plot_height

    root = ElementTree.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    ElementTree.SubElement(root, f"{{{SVG_NS}}}title").text = (
        "Native Engine TTFT by conversation turn"
    )
    ElementTree.SubElement(root, f"{{{SVG_NS}}}desc").text = (
        "All four paired-run observations per turn, with median lines and observed ranges."
    )
    ElementTree.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {"x": str(left), "y": "34", "class": "chart-title"},
    ).text = "Native Engine TTFT (not HTTP TTFT)"

    for tick in range(5):
        value = y_max * tick / 4
        y = y_position(value)
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}line",
            {
                "x1": str(left),
                "x2": str(width - right),
                "y1": _format_number(y),
                "y2": _format_number(y),
                "class": "grid-line",
            },
        )
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {
                "x": str(left - 10),
                "y": _format_number(y + 4),
                "text-anchor": "end",
                "class": "y-label",
            },
        ).text = f"{value:.1f}"

    centers = {turn: x_position(turn) for turn in range(1, TURNS + 1)}
    colors = {"session": "#2563eb", "stateless": "#dc2626"}
    for turn in range(1, TURNS + 1):
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {
                "x": _format_number(centers[turn]),
                "y": str(plot_bottom + 24),
                "text-anchor": "middle",
                "class": "turn-label",
                "data-turn": str(turn),
            },
        ).text = str(turn)

    median_points = {path: [] for path in colors}
    for path, color in colors.items():
        offset = -4 if path == "session" else 4
        for turn in range(1, TURNS + 1):
            turn_rows = [row for row in rows if row["turn"] == turn]
            values = [row[f"{path}_ttft_ms"] for row in turn_rows]
            low, high = min(values), max(values)
            median_value = median(values)
            x = centers[turn] + offset
            ElementTree.SubElement(
                root,
                f"{{{SVG_NS}}}line",
                {
                    "x1": _format_number(x),
                    "x2": _format_number(x),
                    "y1": _format_number(y_position(low)),
                    "y2": _format_number(y_position(high)),
                    "stroke": color,
                    "class": f"range-marker {path}",
                    "data-turn": str(turn),
                    "data-min-ms": _format_number(low),
                    "data-max-ms": _format_number(high),
                },
            )
            for repetition, row in enumerate(turn_rows, start=1):
                point_x = x + (repetition - (REPETITIONS + 1) / 2) * 1.8
                value = row[f"{path}_ttft_ms"]
                ElementTree.SubElement(
                    root,
                    f"{{{SVG_NS}}}circle",
                    {
                        "cx": _format_number(point_x),
                        "cy": _format_number(y_position(value)),
                        "r": "2.5",
                        "fill": color,
                        "class": f"observation {path}",
                        "data-path": path,
                        "data-turn": str(turn),
                        "data-repetition": str(row["repetition"]),
                        "data-value-ms": _format_number(value),
                    },
                )
            median_points[path].append(
                f"{_format_number(x)},{_format_number(y_position(median_value))}"
            )

    for path, color in colors.items():
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}polyline",
            {
                "id": f"{path}-median",
                "points": " ".join(median_points[path]),
                "fill": "none",
                "stroke": color,
                "stroke-width": "2.5",
                "class": f"median-series {path}",
            },
        )

    ElementTree.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {"x": str(width / 2), "y": str(height - 18), "text-anchor": "middle"},
    ).text = "Conversation turn"
    for index, (path, color) in enumerate(colors.items()):
        legend_x = left + index * 180
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}line",
            {
                "x1": str(legend_x),
                "x2": str(legend_x + 24),
                "y1": str(height - 52),
                "y2": str(height - 52),
                "stroke": color,
                "stroke-width": "2.5",
            },
        )
        ElementTree.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {"x": str(legend_x + 30), "y": str(height - 48)},
        ).text = f"{path.title()} median / range / observations"

    return ElementTree.tostring(root, encoding="unicode", xml_declaration=True)


def generate_report(raw_json_path: Path, analysis_dir: Path) -> tuple[Path, Path]:
    """Validate one completed raw run and write its CSV and SVG report."""
    rows = _rows_with_turn_statistics(_validated_rows(Path(raw_json_path)))
    analysis_dir = Path(analysis_dir)
    if analysis_dir.exists() or analysis_dir.is_symlink():
        raise FileExistsError(f"analysis output path already exists: {analysis_dir}")

    csv_text = _csv_text(rows)
    svg_text = _svg_text(rows)
    analysis_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".session_kv_ttft_report_", dir=analysis_dir.parent)
    )
    try:
        (staging_dir / CSV_NAME).write_text(csv_text, encoding="utf-8", newline="")
        (staging_dir / SVG_NAME).write_text(svg_text, encoding="utf-8", newline="")
        if analysis_dir.exists() or analysis_dir.is_symlink():
            raise FileExistsError(
                f"analysis output path already exists: {analysis_dir}"
            )
        staging_dir.rename(analysis_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    csv_path = analysis_dir / CSV_NAME
    svg_path = analysis_dir / SVG_NAME
    return csv_path, svg_path
