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

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def p95(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def analyze(run_path, trace_path, output):
    r = json.loads(run_path.read_text())
    protocol = r["protocol"]
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    violations = []
    expected = {
        (i, t)
        for i in range(protocol["sessions"])
        for t in range(1, protocol["turns"] + 1)
    }
    rows = {(x["session_index"], x["turn_index"]): x for x in trace}
    if set(rows) != expected or len(rows) != len(trace):
        violations.append("invalid reference pair set")
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != r["trace"]["sha256"]:
        violations.append("reference SHA mismatch")
    for index in range(protocol["sessions"]):
        history = []
        for turn in range(1, protocol["turns"] + 1):
            row = rows[index, turn]
            if row["full_input_ids"] != history + row["suffix_ids"]:
                violations.append(f"reference history mismatch: {index}/{turn}")
            if len(row["output_ids"]) != protocol["max_new_tokens"]:
                violations.append(f"reference output count: {index}/{turn}")
            if (
                len(row["full_input_ids"]) + len(row["output_ids"])
                > r["config"]["max_model_len"]
            ):
                violations.append(f"reference model limit: {index}/{turn}")
            history = row["full_input_ids"] + row["output_ids"]
    expected_arms = [
        (phase, rep, mode)
        for phase, count in (
            ("warmup", protocol["warmup_pairs"]),
            ("formal", protocol["repetitions"]),
        )
        for rep in range(count)
        for mode in (
            ("session", "stateless") if rep % 2 == 0 else ("stateless", "session")
        )
    ]
    arms = r.get("arms", [])
    if [(a["phase"], a["repetition"], a["mode"]) for a in arms] != expected_arms:
        violations.append("incomplete or reordered arms")
    if r["status"] != "passed":
        violations.append(f"run status: {r['status']}; {r.get('error')}")
    if r.get("source_after_mismatches"):
        violations.append("source changed")
    if r.get("sampler_errors"):
        violations.append("sampler errors")
    request_table = []
    summaries = []
    for arm_index, arm in enumerate(arms):
        start, end = arm.get("started_ns"), arm.get("finished_ns")
        requests = arm.get("requests", [])
        tag = f"{arm['phase']}/{arm['repetition']}/{arm['mode']}"
        keys = [(x["session_index"], x["turn_index"]) for x in requests]
        if set(keys) != expected or len(keys) != len(expected):
            violations.append(f"{tag}: missing/duplicate requests")
        if arm.get("status") != "passed":
            violations.append(f"{tag}: incomplete arm")
        if not start or not end or end <= start:
            violations.append(f"{tag}: invalid wall timing")
            continue
        if arm.get("close_errors"):
            violations.append(f"{tag}: close errors")
        if len(arm.get("closed_session_ids", [])) != (
            protocol["sessions"] if arm["mode"] == "session" else 0
        ):
            violations.append(f"{tag}: close count")
        idle = arm.get("idle_after_close", {})
        if idle.get("requests_running") != 0 or idle.get("kv_lcm_blocks_active") != 0:
            violations.append(f"{tag}: missing/nonzero idle active gauge")
        tools = arm.get("tools", [])
        if len(tools) != 1:
            violations.append(f"{tag}: tool count")
        else:
            tool = tools[0]
            if tool["result"] != sum(range(protocol["tool_range_stop"])):
                violations.append(f"{tag}: tool result")
            if not start <= tool["started_ns"] < tool["ended_ns"] <= end:
                violations.append(f"{tag}: tool outside arm")
            if (tool["ended_ns"] - tool["started_ns"]) / 1e9 < protocol[
                "tool_delay_seconds"
            ]:
                violations.append(f"{tag}: short tool wait")
        observed = []
        for item in requests:
            key = (item["session_index"], item["turn_index"])
            ref = rows[key]
            result = item["result"]
            ids = result["output_ids"]
            native = arm["mode"] == "session"
            checks = {
                "submitted_ids": item["submitted_ids"]
                == ref["suffix_ids" if native else "full_input_ids"],
                "actual_outputs": ids == ref["output_ids"],
                "recorded_outputs": item["output_ids_match"] is True,
                "output_count": len(ids) == protocol["max_new_tokens"],
                "timing_inside_arm": start
                <= result["started_ns"]
                < result["ended_ns"]
                <= end,
            }
            if native:
                prior = len(ref["full_input_ids"]) - len(ref["suffix_ids"])
                reused = max(prior - 1, 0)
                meta = result["meta_info"]["session"]
                checks["context_metadata"] = (
                    meta["total_context_tokens"] == len(ref["full_input_ids"])
                    and meta["reused_context_tokens"] == reused
                    and meta["newly_prefilled_tokens"]
                    == len(ref["full_input_ids"]) - reused
                    and item["context_validated"] is True
                )
            for name, passed in checks.items():
                if not passed:
                    violations.append(f"{tag}/{key}: {name}")
            e2e = (result["ended_ns"] - result["started_ns"]) / 1e6
            ttft = result["ttft_ms"]
            tpot = (e2e - ttft) / (len(ids) - 1) if len(ids) > 1 else None
            if (
                not math.isfinite(ttft)
                or ttft <= 0
                or tpot is None
                or not math.isfinite(tpot)
                or tpot < 0
            ):
                violations.append(f"{tag}/{key}: invalid client latency")
            row = {
                "phase": arm["phase"],
                "repetition": arm["repetition"],
                "mode": arm["mode"],
                "session_index": key[0],
                "turn_index": key[1],
                "context_tokens": len(ref["full_input_ids"]),
                "submitted_tokens": len(item["submitted_ids"]),
                "output_tokens": len(ids),
                "ttft_ms": ttft,
                "e2e_ms": e2e,
                "client_tpot_ms": tpot,
            }
            request_table.append(row)
            observed.append(row)
        for index in range(protocol["sessions"]):
            session = sorted(
                (x for x in requests if x["session_index"] == index),
                key=lambda x: x["turn_index"],
            )
            if any(
                a["result"]["ended_ns"] > b["result"]["started_ns"]
                for a, b in zip(session, session[1:])
            ):
                violations.append(f"{tag}: within-session overlap")
        if observed:
            duration = (end - start) / 1e9
            summary = {
                "phase": arm["phase"],
                "repetition": arm["repetition"],
                "mode": arm["mode"],
                "wall_seconds_including_tool": duration,
                "output_tokens": sum(x["output_tokens"] for x in observed),
                "output_tokens_per_second": sum(x["output_tokens"] for x in observed)
                / duration,
                "requests": len(observed),
            }
            for metric in ("ttft_ms", "e2e_ms", "client_tpot_ms"):
                vals = [x[metric] for x in observed if x[metric] is not None]
                summary[f"{metric}_p50"] = statistics.median(vals) if vals else None
                summary[f"{metric}_p95"] = p95(vals) if vals else None
            contained = [
                s
                for s in r.get("hbm_samples", [])
                if start <= s["started_ns"] and s["ended_ns"] <= end
            ]
            began = [
                s for s in r.get("hbm_samples", []) if start <= s["started_ns"] < end
            ]
            summary["hbm_queries_wholly_inside_arm"] = len(contained)
            summary["hbm_queries_started_inside_arm"] = len(began)
            summary["hbm_contained_observed_peak_mb_per_chip"] = (
                {
                    chip: max(s["chips"][chip]["used_mb"] for s in contained)
                    for chip in contained[0]["chips"]
                }
                if contained
                else None
            )
            summaries.append(summary)
    expected_chips = {str(i) for i in range(r["config"]["tp_size"])}
    hbms = r.get("hbm_samples", [])
    if not hbms:
        violations.append("no HBM samples")
    for i, s in enumerate(hbms):
        if set(s["chips"]) != expected_chips or s["ended_ns"] <= s["started_ns"]:
            violations.append(f"HBM sample {i}: chip/timing mismatch")
        table = (run_path.parent / s["raw_table"]).resolve()
        if not table.is_relative_to(run_path.parent.resolve()) or not table.is_file():
            violations.append(f"HBM sample {i}: missing raw table")
        else:
            card = None
            parsed = {}
            for line in table.read_text().splitlines():
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) != 3:
                    continue
                name = re.fullmatch(r"(\d+)\s+Ascend910", cells[0])
                if name:
                    card = int(name.group(1))
                pair = re.fullmatch(r"(\d+)\s+(\d+)", cells[0])
                hbm = re.search(r"(\d+)\s*/\s*(\d+)\s*$", cells[2])
                if (
                    card == 1
                    and pair
                    and hbm
                    and re.fullmatch(r"[0-9a-fA-F:.]+", cells[1])
                ):
                    chip, logic = map(int, pair.groups())
                    if str(chip) in expected_chips and logic == chip + 2:
                        if str(chip) in parsed:
                            violations.append(f"HBM sample {i}: duplicate raw chip row")
                        parsed[str(chip)] = {
                            "used_mb": int(hbm[1]),
                            "capacity_mb": int(hbm[2]),
                        }
            if parsed != s["chips"]:
                violations.append(
                    f"HBM sample {i}: raw table differs from stored chip values"
                )
        for chip, value in s["chips"].items():
            if (
                not 0 <= value["used_mb"] <= value["capacity_mb"]
                or value["capacity_mb"] <= 0
            ):
                violations.append(f"HBM sample {i}: invalid values")
    metric_anomalies = []
    for i, s in enumerate(r.get("metric_samples", [])):
        m = s["metrics"]
        names = (
            "requests_running",
            "kv_lcm_blocks_total",
            "kv_lcm_blocks_active",
            "kv_lcm_blocks_cached",
            "kv_lcm_blocks_empty",
            "kv_lcm_blocks_allocated",
        )
        if any(k not in m or not math.isfinite(m[k]) or m[k] < 0 for k in names):
            violations.append(f"metric sample {i}: missing/nonfinite")
        elif (
            m["kv_lcm_blocks_total"]
            != m["kv_lcm_blocks_active"]
            + m["kv_lcm_blocks_cached"]
            + m["kv_lcm_blocks_empty"]
            or m["kv_lcm_blocks_allocated"]
            != m["kv_lcm_blocks_active"] + m["kv_lcm_blocks_cached"]
        ):
            metric_anomalies.append(i)
    if not r.get("metric_samples"):
        violations.append("no metric samples")
    formal = [x for x in summaries if x["phase"] == "formal"]
    paired = []
    for repetition in range(protocol["repetitions"]):
        pair = {x["mode"]: x for x in formal if x["repetition"] == repetition}
        if set(pair) == {"session", "stateless"}:
            on, off = pair["session"], pair["stateless"]
            paired.append(
                {
                    "repetition": repetition,
                    "session_ttft_p50_ms": on["ttft_ms_p50"],
                    "stateless_ttft_p50_ms": off["ttft_ms_p50"],
                    "ttft_p50_reduction_percent": 100
                    * (1 - on["ttft_ms_p50"] / off["ttft_ms_p50"]),
                    "throughput_ratio_session_over_stateless": on[
                        "output_tokens_per_second"
                    ]
                    / off["output_tokens_per_second"],
                    "session_wall_seconds": on["wall_seconds_including_tool"],
                    "stateless_wall_seconds": off["wall_seconds_including_tool"],
                }
            )
    queries = [(s["ended_ns"] - s["started_ns"]) / 1e9 for s in hbms]
    gaps = [(b["started_ns"] - a["started_ns"]) / 1e9 for a, b in zip(hbms, hbms[1:])]
    audit = {
        "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": r["source_manifest_sha256"],
        "trace_sha256": r["trace"]["sha256"],
        "reported_status": r["status"],
        "protocol_and_quality_passed": not violations,
        "violations": violations,
        "formal_requests": sum(x["phase"] == "formal" for x in request_table),
        "warmup_requests": sum(x["phase"] == "warmup" for x in request_table),
        "arms": summaries,
        "paired_repetitions": paired,
        "hbm_samples": len(hbms),
        "hbm_observed_peak_all_phases_mb_per_chip": (
            {
                chip: max(s["chips"][chip]["used_mb"] for s in hbms)
                for chip in expected_chips
            }
            if hbms
            else None
        ),
        "max_hbm_observation_span_seconds": max(queries) if queries else None,
        "max_hbm_query_start_gap_seconds": max(gaps) if gaps else None,
        "metric_samples": len(r.get("metric_samples", [])),
        "metric_partition_anomaly_sample_indices": metric_anomalies,
        "host_cpu_affinity": r.get("host_cpu_affinity"),
        "omp_num_threads": r.get("omp_num_threads"),
        "clocks": r.get("clocks"),
        "scope": "three alternating-order repetitions within one engine; descriptive only, not independent-run inference; TTFT/E2E/client TPOT, throughput includes scripted tool wait; per-arm HBM only wholly-contained observations, null stays missing; model peak includes warmup/between phases, not instantaneous or KV-only",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if request_table:
        with (output / "requests.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(request_table[0]))
            writer.writeheader()
            writer.writerows(request_table)
    print(
        json.dumps(
            {
                "status": r["status"],
                "protocol_and_quality_passed": not violations,
                "violations": violations,
                "formal_requests": audit["formal_requests"],
                "paired_repetitions": paired,
                "hbm_samples": len(hbms),
                "observed_peak_mb_per_chip": audit[
                    "hbm_observed_peak_all_phases_mb_per_chip"
                ],
            }
        )
    )
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_json", type=Path)
    p.add_argument("trace_jsonl", type=Path)
    p.add_argument("output_dir", type=Path)
    a = p.parse_args()
    analyze(a.run_json, a.trace_jsonl, a.output_dir)
