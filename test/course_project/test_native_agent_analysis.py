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

"""CPU-only synthetic audit records; these are not hardware measurements."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / "docs/evidence/project_agent_20261031/runners"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(chips, pids):
    lines = ["| 7 Ascend910 | OK | OK |"]
    lines += [
        f"| {c} {logical} | 0000:00:01.0 | 100 / 64000 |"
        for c, logical in chips.items()
    ]
    lines += [
        "+----------------+",
        "| NPU Chip | Process id | Process name | Process memory(MB) |",
        "+----------------+",
    ]
    lines += [
        f"| 7 {c} | {pid} | python | 10 |"
        for c, values in pids.items()
        for pid in values
    ]
    if not any(pids.values()):
        lines += ["| No running processes found |"]
    return "\n".join(lines + ["+----------------+"])


def metrics(active, running=0):
    return {
        "requests_running": running,
        "kv_lcm_blocks_total": 20,
        "kv_lcm_blocks_active": active,
        "kv_lcm_blocks_cached": 0,
        "kv_lcm_blocks_empty": 20 - active,
        "kv_lcm_blocks_allocated": active,
    }


@pytest.fixture
def synthetic(tmp_path):
    source = tmp_path / "source"
    (source / "benchmarks").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "benchmarks/cpu.py").write_text("# synthetic\n")
    (source / "scripts/cpu.py").write_text("# synthetic\n")
    manifest = tmp_path / "sources.json"
    write_json(
        manifest,
        {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*.py")},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return source, manifest, run_dir


def prepare(synthetic, chips):
    source, manifest, out = synthetic
    devices = {"npu_id": 7, "chip_logic_ids": chips}
    baseline = {str(c): {str(90 + c): [1000, str(500 + c)]} for c in chips}
    background = {c: [90 + c] for c in chips}
    identities = {
        pid: value for values in baseline.values() for pid, value in values.items()
    }
    memory = {str(c): {"used_mb": 100, "capacity_mb": 64000} for c in chips}
    mapping = "\n".join(
        f"7 {c} {logical} {c} Ascend910" for c, logical in chips.items()
    )
    for filename in (
        "before_mapping.txt",
        "after_mapping.txt",
        "post_exit_mapping.txt",
    ):
        (out / filename).write_text(mapping)
    samples = []
    for index, (label, clock) in enumerate(
        (("before", 1), ("monitor", 2), ("after_engine_exit", 1000))
    ):
        pids = {
            str(c): sorted(v + ([50] if index == 1 else []))
            for c, v in background.items()
        }
        filename = f"guard_{index}.txt"
        (out / filename).write_text(table(chips, pids))
        samples.append(
            {
                "label": label,
                "observed_ns": clock,
                "ended_ns": clock + 1,
                "raw_table": filename,
                "pids": pids,
                "live_pids": pids,
                "identities": identities
                | ({"50": [1000, "600"]} if index == 1 else {}),
                "owned_pids": [50] if index == 1 else [],
                "memory": memory,
                "exited_during_query": [],
            }
        )
    guard = {
        "parent_pid": 50,
        "guard_started_ns": 0,
        "resource_mode": "shared",
        "min_free_mb": 8192,
        "baseline": baseline,
        "samples": samples,
        "before_mapping": "before_mapping.txt",
        "after_mapping": "after_mapping.txt",
        "violations": [],
        "after_errors": [],
        "engine_children_released": True,
        "sources_sha": sha(manifest),
        "sources_after_sha": sha(manifest),
        "sources_unchanged": True,
        "monitor_stopped_ns": 900,
        "guard_ended_ns": 1002,
    }
    common = {
        "status": "passed",
        "config": {"tp_size": len(chips), "max_model_len": 1000},
        "devices": devices,
        "resource_mode": "shared",
        "source_manifest_sha256": sha(manifest),
        "device_guard": guard,
    }
    receipt = {
        "devices": devices,
        "exit_code": 0,
        "timed_out": False,
        "parent_pid": 50,
        "parent_identity": [1000, "600"],
        "known_owned_pids": [50],
        "source_unchanged": True,
        "model_files_unchanged": True,
        "own_processes_released": True,
        "unknown_pids": [],
        "samples": [
            {
                "raw_table": "post_exit_devices_00.txt",
                "pids": background,
                "identities": identities,
                "exited_during_query": [],
                "remaining_owned_pids": [],
            }
        ],
    }
    (out / "post_exit_devices_00.txt").write_text(table(chips, background))
    return common, receipt


def finish(synthetic, run, receipt):
    out = synthetic[2]
    write_json(out / "run.json", run)
    receipt["run_sha256"] = sha(out / "run.json")
    write_json(out / "post_exit_verification.json", receipt)


def quality(synthetic, chips):
    r, receipt = prepare(synthetic, chips)
    r.update(
        {
            "workload": {
                "sessions": 2,
                "turns": 2,
                "max_new_tokens": 2,
                "tool_range_stop": 3,
                "tool_delay_seconds": 0.00000001,
                "tool_session_index": 0,
                "tool_before_turn": 2,
            },
            "workload_sha256": "synthetic",
            "control_schedule": "after-run-serial",
            "native_phase_started_ns": 10,
            "native_phase_finished_ns": 100,
            "controls_phase_started_ns": 200,
            "controls_phase_finished_ns": 400,
            "closed_session_ids": ["s0", "s1"],
            "close_errors": [],
            "tools": [
                {
                    "session_index": 0,
                    "started_ns": 25,
                    "ended_ns": 65,
                    "result": 3,
                    "resumed_reused_context_tokens": 2,
                }
            ],
            "metric_snapshots": [
                {"label": "tool", "monotonic_ns": 45, "metrics": metrics(2, 1)},
                {
                    "label": "all_sessions_idle_before_close",
                    "monotonic_ns": 110,
                    "metrics": metrics(2),
                },
                {
                    "label": "all_sessions_closed",
                    "monotonic_ns": 120,
                    "metrics": metrics(0),
                },
            ],
            "turns": [],
        }
    )
    for i in range(2):
        history = []
        for t in (1, 2):
            suffix = [i + t + 10]
            full = history + suffix
            start = 10 + (t - 1) * 60 if i == 0 else 30 + (t - 1) * 40
            result = {
                "output_ids": [7, 8],
                "started_ns": start,
                "ended_ns": start + 10,
                "meta_info": {
                    "session": {
                        "total_context_tokens": len(full),
                        "reused_context_tokens": max(len(history) - 1, 0),
                        "newly_prefilled_tokens": len(full) - max(len(history) - 1, 0),
                    }
                },
            }
            control_start = 200 + len(r["turns"]) * 20
            r["turns"].append(
                {
                    "session_index": i,
                    "turn_index": t,
                    "full_input_ids": full,
                    "suffix_ids": suffix,
                    "prior_history_tokens": len(history),
                    "session_result": result,
                    "stateless_result": {
                        "output_ids": [7, 8],
                        "started_ns": control_start,
                        "ended_ns": control_start + 10,
                    },
                    "context_validated": True,
                    "output_ids_match": True,
                }
            )
            history = full + [7, 8]
    finish(synthetic, r, receipt)
    return r, receipt


def performance(synthetic, chips):
    r, receipt = prepare(synthetic, chips)
    reference = []
    for i in range(2):
        history = []
        for t in (1, 2):
            full = history + [i + t + 10]
            reference.append(
                {
                    "session_index": i,
                    "turn_index": t,
                    "full_input_ids": full,
                    "suffix_ids": [i + t + 10],
                    "output_ids": [7, 8],
                }
            )
            history = full + [7, 8]
    trace = synthetic[2] / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in reference))
    r.update(
        {
            "protocol": {
                "sessions": 2,
                "turns": 2,
                "max_new_tokens": 2,
                "warmup_pairs": 0,
                "repetitions": 2,
                "tool_range_stop": 3,
                "tool_delay_seconds": 0.01,
            },
            "trace": {"sha256": sha(trace)},
            "arms": [],
            "metric_samples": [{"metrics": metrics(0)}],
            "hbm_samples": [],
        }
    )
    for rep in range(2):
        for mode in (
            ("session", "stateless") if rep == 0 else ("stateless", "session")
        ):
            start = (len(r["arms"]) + 1) * 1000000000
            arm = {
                "phase": "formal",
                "repetition": rep,
                "mode": mode,
                "status": "passed",
                "started_ns": start,
                "finished_ns": start + 900000000,
                "close_errors": [],
                "closed_session_ids": ["s0", "s1"] if mode == "session" else [],
                "idle_after_close": metrics(0),
                "tools": [
                    {
                        "started_ns": start + 100,
                        "ended_ns": start + 100000100,
                        "result": 3,
                    }
                ],
                "requests": [],
            }
            for index, row in enumerate(reference):
                prior = len(row["full_input_ids"]) - len(row["suffix_ids"])
                reused = max(prior - 1, 0)
                arm["requests"].append(
                    {
                        "session_index": row["session_index"],
                        "turn_index": row["turn_index"],
                        "submitted_ids": row[
                            "suffix_ids" if mode == "session" else "full_input_ids"
                        ],
                        "output_ids_match": True,
                        "context_validated": True,
                        "result": {
                            "output_ids": [7, 8],
                            "started_ns": start + index * 100000000 + 100,
                            "ended_ns": start + index * 100000000 + 50000100,
                            "ttft_ms": 10,
                            "meta_info": {
                                "session": {
                                    "total_context_tokens": len(row["full_input_ids"]),
                                    "reused_context_tokens": reused,
                                    "newly_prefilled_tokens": len(row["full_input_ids"])
                                    - reused,
                                }
                            },
                        },
                    }
                )
            filename = f"hbm_{len(r['arms'])}.txt"
            (synthetic[2] / filename).write_text(
                table(chips, {c: [90 + c] for c in chips})
            )
            r["hbm_samples"].append(
                {
                    "started_ns": start + 1000,
                    "ended_ns": start + 2000,
                    "raw_table": filename,
                    "devices": r["devices"],
                    "chips": {
                        str(c): {"used_mb": 100, "capacity_mb": 64000} for c in chips
                    },
                }
            )
            r["arms"].append(arm)
    finish(synthetic, r, receipt)
    return r, receipt


def invoke(synthetic, kind, output="audit"):
    source, manifest, run_dir = synthetic
    script = "audit_native.py" if kind == "quality" else "analyze.py"
    assert (RUNNERS / script).is_file(), "current independent analysis CLI is missing"
    arguments = [sys.executable, str(RUNNERS / script), str(run_dir / "run.json")]
    if kind == "performance":
        arguments += [str(run_dir / "trace.jsonl")]
    return subprocess.run(
        arguments
        + [
            str(run_dir / output),
            "--source-root",
            str(source),
            "--source-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", ["quality", "performance"])
@pytest.mark.parametrize("chips", [{1: 13}, {0: 12, 1: 13}])
def test_current_analysis_accepts_dynamic_shared_devices(synthetic, kind, chips):
    (quality if kind == "quality" else performance)(synthetic, chips)
    result = invoke(synthetic, kind)
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads(
        (
            synthetic[2]
            / "audit"
            / ("native_agent_audit.json" if kind == "quality" else "audit.json")
        ).read_text()
    )
    assert not audit["violations"]
    assert audit["device_audit"]["devices"] == {
        "npu_id": 7,
        "chip_logic_ids": {str(c): v for c, v in chips.items()},
    }
    assert "target_chip0_idle_after_process_exit" not in audit
    if kind == "performance":
        assert len(audit["paired_repetitions"]) == 2
        assert (synthetic[2] / "audit/requests.csv").is_file()
        assert audit["hbm_observed_peak_all_phases_mb_per_chip"] == {
            str(c): 100 for c in chips
        }


@pytest.mark.parametrize("kind", ["quality", "performance"])
@pytest.mark.parametrize(
    "damage",
    [
        "receipt_sha",
        "owned",
        "unknown",
        "background_identity",
        "devices",
        "before_mapping",
        "after_mapping",
        "post_mapping",
        "source_omitted",
        "source_sha",
        "guard_hbm",
        "guard_pids",
        "status",
        "bad_output",
        "missing_request",
        "cleanup_error",
        "timeout",
    ],
)
def test_current_analysis_rejects_forged_or_incomplete_evidence(
    synthetic, kind, damage
):
    r, receipt = (quality if kind == "quality" else performance)(synthetic, {1: 13})
    out = synthetic[2]
    if damage == "receipt_sha":
        receipt["run_sha256"] = "0" * 64
    elif damage in {"owned", "unknown"}:
        pid = 50 if damage == "owned" else 999
        receipt["samples"][0]["pids"][1] += [pid]
        receipt["samples"][0]["identities"][str(pid)] = [1000, "600"]
        (out / "post_exit_devices_00.txt").write_text(table({1: 13}, {1: [91, pid]}))
    elif damage == "background_identity":
        receipt["samples"][0]["identities"]["91"] = [1000, "999"]
    elif damage == "devices":
        receipt["devices"] = {"npu_id": 1, "chip_logic_ids": {"1": 3}}
    elif damage in {"before_mapping", "after_mapping", "post_mapping"}:
        filename = (
            "post_exit_mapping.txt" if damage == "post_mapping" else damage + ".txt"
        )
        (out / filename).write_text("1 1 3 1 Ascend910")
    elif damage == "source_omitted":
        manifest = json.loads(synthetic[1].read_text())
        manifest.pop("scripts/cpu.py")
        write_json(synthetic[1], manifest)
    elif damage == "source_sha":
        (synthetic[0] / "scripts/cpu.py").write_text("# changed\n")
    elif damage == "guard_hbm":
        r["device_guard"]["samples"][0]["memory"]["1"]["used_mb"] = 101
    elif damage == "guard_pids":
        r["device_guard"]["samples"][0]["pids"]["1"] = []
    elif damage == "status":
        r["status"] = "failed"
    elif damage == "bad_output":
        result = (
            r["turns"][0]["session_result"]
            if kind == "quality"
            else r["arms"][0]["requests"][0]["result"]
        )
        result["output_ids"] = [999, 8]
    elif damage == "missing_request":
        (r["turns"] if kind == "quality" else r["arms"][0]["requests"]).pop()
    elif damage == "cleanup_error":
        r["device_guard"]["after_errors"] = ["failed cleanup"]
    elif damage == "timeout":
        receipt["timed_out"] = True
    if damage != "receipt_sha":
        finish(synthetic, r, receipt)
    else:
        write_json(out / "post_exit_verification.json", receipt)
    result = invoke(synthetic, kind)
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize("kind", ["quality", "performance"])
def test_current_analysis_accepts_natural_background_exit(synthetic, kind):
    r, receipt = (quality if kind == "quality" else performance)(synthetic, {1: 13})
    sample = receipt["samples"][0]
    sample["pids"] = {1: []}
    sample["identities"] = {}
    (synthetic[2] / sample["raw_table"]).write_text(table({1: 13}, {1: []}))
    finish(synthetic, r, receipt)
    result = invoke(synthetic, kind)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["quality", "performance"])
def test_current_analysis_refuses_existing_output_directory(synthetic, kind):
    (quality if kind == "quality" else performance)(synthetic, {1: 13})
    existing = synthetic[2] / "audit"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    assert invoke(synthetic, kind).returncode != 0
    assert list(existing.iterdir()) == [existing / "keep.txt"]


def test_performance_rejects_raw_hbm_disagreement(synthetic):
    r, receipt = performance(synthetic, {0: 12, 1: 13})
    r["hbm_samples"][0]["chips"]["1"]["used_mb"] = 101
    finish(synthetic, r, receipt)
    assert invoke(synthetic, "performance").returncode != 0


def test_performance_hbm_uses_run_devices_when_sample_has_no_device_field(synthetic):
    run, receipt = performance(synthetic, {1: 13})
    for sample in run["hbm_samples"]:
        del sample["devices"]
    finish(synthetic, run, receipt)
    result = invoke(synthetic, "performance")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "damage",
    ["tool_result", "tool_progress", "active_blocks", "idle_holds", "close", "phase"],
)
def test_quality_rejects_missing_retention_and_phase_evidence(synthetic, damage):
    r, receipt = quality(synthetic, {1: 13})
    if damage == "tool_result":
        r["tools"][0]["result"] = 999
    elif damage == "tool_progress":
        r["tools"][0]["ended_ns"] = 35
    elif damage == "active_blocks":
        r["metric_snapshots"][0]["metrics"] = metrics(0)
    elif damage == "idle_holds":
        r["metric_snapshots"][1]["metrics"] = metrics(0)
    elif damage == "close":
        r["close_errors"] = ["failed"]
    elif damage == "phase":
        r["controls_phase_started_ns"] = 50
    finish(synthetic, r, receipt)
    assert invoke(synthetic, "quality").returncode != 0
