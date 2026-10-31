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

"""Analyze explicit rank-native torch_npu windows without running any model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.tp_timeline import analyze_tp_timeline


def _rank_values(values: Sequence[str], *, label: str) -> dict[int, str]:
    result = {}
    for value in values:
        rank_text, separator, payload = value.partition(":")
        try:
            rank = int(rank_text)
        except ValueError:
            raise ValueError(f"{label} requires rank:value") from None
        if not separator or not payload or rank < 0:
            raise ValueError(f"{label} requires a nonnegative rank and nonempty value")
        if rank in result:
            raise ValueError(f"duplicate rank {rank} in {label}")
        result[rank] = payload
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-rank compute/collective interval unions in torch_npu "
            "Chrome traces. Supply native microsecond windows and a confirmed "
            "common host/device clock per file. Different ranks are never "
            "synchronized or added into wall time. "
            "Existing output is never overwritten."
        )
    )
    parser.add_argument("--trace", action="append", required=True, metavar="RANK:PATH")
    parser.add_argument(
        "--window", action="append", required=True, metavar="RANK:START_US:STOP_US"
    )
    parser.add_argument(
        "--clock-id", action="append", required=True, metavar="RANK:LABEL"
    )
    parser.add_argument(
        "--collective-patterns",
        nargs="*",
        required=True,
        metavar="REGEX",
        help=(
            "Explicit regexes verified against device task names; provide this "
            "flag with no values to leave collective-looking device tasks unknown."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, metavar="NEW_JSON_FILE")
    args = parser.parse_args(argv)
    try:
        traces = {
            rank: Path(path).resolve()
            for rank, path in _rank_values(args.trace, label="--trace").items()
        }
        clocks = _rank_values(args.clock_id, label="--clock-id")
        windows = {}
        for rank, value in _rank_values(args.window, label="--window").items():
            bounds = value.split(":")
            if len(bounds) != 2:
                raise ValueError("--window requires rank:start_us:stop_us")
            windows[rank] = (bounds[0], bounds[1])
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        report = analyze_tp_timeline(
            traces,
            windows_us=windows,
            clock_ids=clocks,
            collective_name_patterns=args.collective_patterns,
        )
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        with output.open("x", encoding="utf-8") as destination:
            destination.write(f"{content}\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
