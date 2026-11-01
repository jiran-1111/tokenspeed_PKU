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

"""Adapted from course_final_20261018/runners/audit_phases.py; historical scope retained."""

import argparse
import hashlib
import json
from pathlib import Path


def audit_phases(run_path: Path, r: dict, prior_run: Path | None = None) -> dict:
    v = []
    rows = r["turns"]
    controls = [t for t in rows if t["stateless_result"] is not None]
    native_finish = r.get("native_phase_finished_ns")
    control_start = r.get("controls_phase_started_ns")
    if native_finish is None or control_start is None or control_start < native_finish:
        v.append("missing or reversed phase boundary")
    closed = [s for s in r["metric_snapshots"] if s["label"] == "all_sessions_closed"]
    if (
        not closed
        or closed[-1]["metrics"]["requests_running"] != 0
        or closed[-1]["metrics"]["kv_lcm_blocks_active"] != 0
    ):
        v.append("close gauges missing or nonzero")
    if (
        closed
        and control_start is not None
        and closed[-1]["monotonic_ns"] > control_start
    ):
        v.append("controls started before close observation")
    if r["control_schedule"] != "after-run-serial":
        v.append("unexpected schedule")
    if len(controls) != r["workload"]["sessions"] * r["workload"]["turns"]:
        v.append("incomplete controls")
    for t in controls:
        c = t["stateless_result"]
        if c["started_ns"] < control_start:
            v.append("control overlaps native phase")
    for left, right in zip(controls, controls[1:]):
        if (
            left["stateless_result"]["ended_ns"]
            > right["stateless_result"]["started_ns"]
        ):
            v.append("controls overlap each other")
    prior_comparison = None
    if prior_run:
        old = json.loads(prior_run.read_text())
        bykey = {(t["session_index"], t["turn_index"]): t for t in old["turns"]}
        prior_comparison = {
            "prior_run_sha256": hashlib.sha256(prior_run.read_bytes()).hexdigest(),
            "pairs_compared": len(rows),
            "different_input_pairs": [],
            "different_output_pairs": [],
        }
        for t in rows:
            key = (t["session_index"], t["turn_index"])
            prev = bykey[key]
            if prev["full_input_ids"] != t["full_input_ids"]:
                prior_comparison["different_input_pairs"].append(key)
            if (
                prev["session_result"]["output_ids"]
                != t["session_result"]["output_ids"]
            ):
                prior_comparison["different_output_pairs"].append(key)
    out = {
        "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "phase_and_close_checks_passed": not v,
        "violations": v,
        "controls_collected": len(controls),
        "native_phase_seconds": (
            (native_finish - r["native_phase_started_ns"]) / 1e9
            if native_finish
            else None
        ),
        "control_phase_seconds": (
            (r["controls_phase_finished_ns"] - control_start) / 1e9
            if r.get("controls_phase_finished_ns")
            else None
        ),
        "prior_trace_comparison": prior_comparison,
        "scope": "quality and phase audit; phase durations are not an OFF/ON performance estimate",
    }
    if native_finish is not None:
        for row in rows:
            result = row["session_result"]
            if (
                not r["native_phase_started_ns"]
                <= result["started_ns"]
                < result["ended_ns"]
                <= native_finish
            ):
                v.append("native request outside native phase")
    if control_start is not None:
        for row in controls:
            result = row["stateless_result"]
            if (
                not control_start
                <= result["started_ns"]
                < result["ended_ns"]
                <= r["controls_phase_finished_ns"]
            ):
                v.append("control outside control phase")
    out["phase_and_close_checks_passed"] = not v
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior-run", type=Path)
    args = parser.parse_args()
    record = json.loads(args.run.read_text())
    result = audit_phases(args.run, record, args.prior_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["phase_and_close_checks_passed"] else 1)
