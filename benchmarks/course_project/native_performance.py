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

import re


def arm_order(repetition_index: int) -> tuple[str, str]:
    if repetition_index < 0:
        raise ValueError("repetition index must be nonnegative")
    return (
        ("session", "stateless")
        if repetition_index % 2 == 0
        else ("stateless", "session")
    )


def validate_trace(
    rows: list[dict],
    *,
    sessions: int,
    turns: int,
    max_new_tokens: int,
    max_model_len: int,
) -> list[list[dict]]:
    if sessions < 2 or turns < 2 or max_new_tokens < 2:
        raise ValueError("invalid trace dimensions")
    expected = {(i, t) for i in range(sessions) for t in range(1, turns + 1)}
    actual = {(r["session_index"], r["turn_index"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicate trace turns")
    grouped = []
    for index in range(sessions):
        history = []
        ordered = sorted(
            (r for r in rows if r["session_index"] == index),
            key=lambda r: r["turn_index"],
        )
        for row in ordered:
            for field in ("suffix_ids", "full_input_ids", "output_ids"):
                ids = row[field]
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(type(n) is not int or n < 0 for n in ids)
                ):
                    raise ValueError(f"invalid trace token IDs in {field}")
            if row["full_input_ids"] != history + row["suffix_ids"]:
                raise ValueError("trace history is not append-only")
            if len(row["output_ids"]) != max_new_tokens:
                raise ValueError("trace output count mismatch")
            if len(row["full_input_ids"]) + max_new_tokens > max_model_len:
                raise ValueError("trace exceeds model context limit")
            history = row["full_input_ids"] + row["output_ids"]
        grouped.append(ordered)
    return grouped


def parse_target_hbm(
    output: str, *, npu_id: int, chip_logic_ids: dict[int, int]
) -> dict[int, dict[str, int]]:
    values = {}
    current = None
    for line in output.splitlines():
        cols = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cols) != 3:
            continue
        name = re.fullmatch(r"(\d+)\s+Ascend910", cols[0])
        if name:
            current = int(name.group(1))
        if current != npu_id or not re.fullmatch(r"[0-9a-fA-F:.]+", cols[1]):
            continue
        pair = re.fullmatch(r"(\d+)\s+(\d+)", cols[0])
        if not pair:
            continue
        chip, logic = map(int, pair.groups())
        if chip not in chip_logic_ids or logic != chip_logic_ids[chip]:
            continue
        match = re.search(r"(\d+)\s*/\s*(\d+)\s*$", cols[-1])
        if not match or chip in values:
            raise ValueError("missing or duplicate target HBM statistic")
        used, capacity = map(int, match.groups())
        if not 0 <= used <= capacity or capacity == 0:
            raise ValueError("invalid HBM capacity statistic")
        values[chip] = {"used_mb": used, "capacity_mb": capacity}
    if set(values) != set(chip_logic_ids):
        raise ValueError("target chip HBM statistics missing")
    return values
