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

"""Adapted from course_final_20261018/runners/audit_native.py; statistics retained."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_devices import audit_devices
from audit_phases import audit_phases


def analyze(
    run_path: Path, output_dir: Path, *, source_root: Path, source_manifest: Path
) -> dict:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing audit directory")
    record = json.loads(run_path.read_text())
    workload = record["workload"]
    rows = record["turns"]
    violations = []
    expected_pairs = {
        (i, j)
        for i in range(workload["sessions"])
        for j in range(1, workload["turns"] + 1)
    }
    actual_pairs = {(r["session_index"], r["turn_index"]) for r in rows}
    if len(actual_pairs) != len(rows) or actual_pairs != expected_pairs:
        violations.append("missing/duplicate pairs")
    for index in range(workload["sessions"]):
        history = []
        for row in sorted(
            (r for r in rows if r["session_index"] == index),
            key=lambda r: r["turn_index"],
        ):
            full = row["full_input_ids"]
            suffix = row["suffix_ids"]
            session = row["session_result"]
            control = row["stateless_result"] or {"output_ids": []}
            meta = session["meta_info"]["session"]
            expected = {
                "total_context_tokens": len(full),
                "newly_prefilled_tokens": len(full) - max(len(history) - 1, 0),
                "reused_context_tokens": max(len(history) - 1, 0),
            }
            checks = {
                "append_only_full_history": full == history + suffix,
                "stored_prior_length": row["prior_history_tokens"] == len(history),
                "context_metadata": all(meta[k] == v for k, v in expected.items()),
                "raw_output_match": session["output_ids"] == control["output_ids"],
                "output_count": len(session["output_ids"])
                == len(control["output_ids"])
                == workload["max_new_tokens"],
                "within_model_limit": len(full) + workload["max_new_tokens"]
                <= record["config"]["max_model_len"],
                "recorded_context_validated": row["context_validated"],
                "recorded_output_match": row["output_ids_match"],
            }
            for name, passed in checks.items():
                if not passed:
                    violations.append(
                        {
                            "session_index": index,
                            "turn_index": row["turn_index"],
                            "check": name,
                        }
                    )
            history = full + session["output_ids"]
    tools = []
    for tool in record["tools"]:
        start, end = tool["started_ns"], tool["ended_ns"]
        others = [r for r in rows if r["session_index"] != tool["session_index"]]
        snapshots = [
            s for s in record["metric_snapshots"] if start < s["monotonic_ns"] < end
        ]
        completed = [r for r in others if start < r["session_result"]["ended_ns"] < end]
        started = [r for r in others if start < r["session_result"]["started_ns"] < end]
        both = [r for r in completed if start < r["session_result"]["started_ns"]]
        tools.append(
            {
                "actual_delay_seconds": (end - start) / 1e9,
                "tool_result": tool["result"],
                "expected_tool_result": sum(range(workload["tool_range_stop"])),
                "resumed_reused_context_tokens": tool.get(
                    "resumed_reused_context_tokens"
                ),
                "other_session_requests_completed_inside_window": [
                    (r["session_index"], r["turn_index"]) for r in completed
                ],
                "other_session_requests_started_inside_window": [
                    (r["session_index"], r["turn_index"]) for r in started
                ],
                "other_session_requests_both_started_and_completed_inside_window": [
                    (r["session_index"], r["turn_index"]) for r in both
                ],
                "kv_samples_inside_window": [s["metrics"] for s in snapshots],
                "positive_held_block_samples_inside_window": sum(
                    s["metrics"]["kv_lcm_blocks_active"] > 0 for s in snapshots
                ),
            }
        )
    required = (
        "requests_running",
        "kv_lcm_blocks_total",
        "kv_lcm_blocks_active",
        "kv_lcm_blocks_cached",
        "kv_lcm_blocks_empty",
        "kv_lcm_blocks_allocated",
    )
    for i, s in enumerate(record["metric_snapshots"]):
        if any(
            k not in s["metrics"] or not math.isfinite(s["metrics"][k])
            for k in required
        ):
            violations.append(
                {"metric_snapshot": i, "check": "missing/nonfinite gauge"}
            )
    idle = [
        s["metrics"]
        for s in record["metric_snapshots"]
        if s["label"] == "all_sessions_idle_before_close"
    ]
    closed = [
        s["metrics"]
        for s in record["metric_snapshots"]
        if s["label"] == "all_sessions_closed"
    ]
    device_audit = audit_devices(run_path, record, source_root, source_manifest)
    violations.extend(device_audit["violations"])
    phase_audit = audit_phases(run_path, record)
    violations.extend(phase_audit["violations"])
    if record["status"] != "passed" or record.get("error"):
        violations.append("collector status/error is not passed")
    if len(record.get("closed_session_ids", [])) != workload["sessions"] or record.get(
        "close_errors"
    ):
        violations.append("native Session close count/errors")
    if (
        not idle
        or idle[-1]["requests_running"] != 0
        or idle[-1]["kv_lcm_blocks_active"] <= 0
    ):
        violations.append("idle Session page retention is missing")
    if (
        not closed
        or closed[-1]["requests_running"] != 0
        or closed[-1]["kv_lcm_blocks_active"] != 0
    ):
        violations.append("closed Session active gauges are missing/nonzero")
    if len(tools) != 1:
        violations.append("native tool count differs from one")
    for observation, tool in zip(tools, record["tools"]):
        if (
            observation["tool_result"] != observation["expected_tool_result"]
            or observation["actual_delay_seconds"] < workload["tool_delay_seconds"]
            or not observation["other_session_requests_completed_inside_window"]
            or not observation["positive_held_block_samples_inside_window"]
            or not observation["resumed_reused_context_tokens"]
            or observation["resumed_reused_context_tokens"] <= 0
        ):
            violations.append("tool result/wait/progress/page retention failed")
        resumed = [
            row
            for row in rows
            if row["session_index"] == tool["session_index"]
            and row["turn_index"] == workload["tool_before_turn"]
        ]
        if (
            len(resumed) != 1
            or tool["session_index"] != workload["tool_session_index"]
            or resumed[0]["session_result"]["started_ns"] < tool["ended_ns"]
            or observation["resumed_reused_context_tokens"]
            != resumed[0]["session_result"]["meta_info"]["session"][
                "reused_context_tokens"
            ]
        ):
            violations.append("tool resume context/timing mismatch")
    summary = {
        "run_file_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "reported_status": record["status"],
        "reported_error": record.get("error"),
        "source_manifest_sha256": record["source_manifest_sha256"],
        "workload_sha256": record["workload_sha256"],
        "pairs_recorded": len(rows),
        "pairs_expected": len(expected_pairs),
        "output_pairs_matched": sum(r["output_ids_match"] is True for r in rows),
        "context_pairs_validated": sum(r["context_validated"] for r in rows),
        "independent_history_and_pair_checks_passed": not violations,
        "violations": violations,
        "max_actual_context_tokens": max(
            (len(r["full_input_ids"]) for r in rows), default=None
        ),
        "tool_observations": tools,
        "idle_before_close_last_metrics": idle[-1] if idle else None,
        "after_close_last_metrics": closed[-1] if closed else None,
        "closed_session_count": len(record.get("closed_session_ids", [])),
        "close_errors": record.get("close_errors"),
        "device_audit": device_audit,
        "phase_audit": phase_audit,
        "scope": "finite native correctness/retention observations; explicit control schedule; no independent performance claim",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "native_agent_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "phase_audit.json").write_text(
        json.dumps(phase_audit, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.run_json,
        args.output_dir,
        source_root=args.source_root,
        source_manifest=args.source_manifest,
    )
    raise SystemExit(0 if result["independent_history_and_pair_checks_passed"] else 1)
