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
import asyncio
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from benchmarks.course_project.native_performance import (
    arm_order,
    parse_target_hbm,
    validate_trace,
)
from benchmarks.course_project.telemetry import fetch_server_metrics
from scripts.run_native_agent_validation import _scripted_tool, _stream_request
from scripts.run_session_kv import (
    _engine_cli_args,
    _local_model_path,
    validate_session_context_metadata,
)


async def run_replay_arm(
    engine: Any,
    *,
    trace: list[dict],
    protocol: dict,
    mode: str,
    max_model_len: int,
    record: dict,
) -> None:
    if mode not in {"session", "stateless"}:
        raise ValueError("unsupported performance mode")
    grouped = validate_trace(
        trace,
        sessions=protocol["sessions"],
        turns=protocol["turns"],
        max_new_tokens=protocol["max_new_tokens"],
        max_model_len=max_model_len,
    )
    if (
        not 0 <= protocol["tool_session_index"] < protocol["sessions"]
        or not 1 < protocol["tool_before_turn"] <= protocol["turns"]
    ):
        raise ValueError("invalid scripted tool position")
    if protocol["tool_delay_seconds"] < 0 or protocol["tool_range_stop"] != 31:
        raise ValueError("invalid scripted tool contract")
    params = {
        "temperature": 0,
        "max_new_tokens": protocol["max_new_tokens"],
        "ignore_eos": True,
    }
    sessions = []
    record.update(
        {
            "mode": mode,
            "requests": [],
            "tools": [],
            "closed_session_ids": [],
            "close_errors": [],
        }
    )
    failed = asyncio.Event()

    async def conversation(index):
        for row in grouped[index]:
            if failed.is_set():
                return
            if (
                index == protocol["tool_session_index"]
                and row["turn_index"] == protocol["tool_before_turn"]
            ):
                event = {
                    "session_index": index,
                    "before_turn": row["turn_index"],
                    "started_ns": None,
                    "ended_ns": None,
                    "result": None,
                }
                record["tools"].append(event)
                await asyncio.to_thread(
                    _scripted_tool,
                    delay_seconds=protocol["tool_delay_seconds"],
                    range_stop=protocol["tool_range_stop"],
                    event=event,
                )
                if event["result"] != 465:
                    raise ValueError("scripted tool result mismatch")
                if failed.is_set():
                    return
            submitted = (
                row["suffix_ids"] if mode == "session" else row["full_input_ids"]
            )
            result = await asyncio.to_thread(
                _stream_request,
                engine,
                submitted,
                params,
                session_id=sessions[index] if mode == "session" else None,
            )
            item = {
                "session_index": index,
                "turn_index": row["turn_index"],
                "full_context_tokens": len(row["full_input_ids"]),
                "submitted_ids": submitted,
                "result": result,
                "output_ids_match": result["output_ids"] == row["output_ids"],
                "context_validated": None,
            }
            record["requests"].append(item)
            if not item["output_ids_match"]:
                raise ValueError("performance replay output mismatch")
            if mode == "session":
                validate_session_context_metadata(
                    result["meta_info"]["session"],
                    policy=protocol["frontier_policy"],
                    prior_history_tokens=len(row["full_input_ids"])
                    - len(row["suffix_ids"]),
                    suffix_tokens=len(row["suffix_ids"]),
                )
                item["context_validated"] = True

    async def guarded(index):
        try:
            await conversation(index)
        except Exception as error:
            failed.set()
            return error
        return None

    try:
        if mode == "session":
            for _ in grouped:
                sessions.append(engine.open_session())
        record["started_ns"] = time.monotonic_ns()
        errors = await asyncio.gather(*(guarded(i) for i in range(len(grouped))))
        record["finished_ns"] = time.monotonic_ns()
        for error in errors:
            if error is not None:
                raise error
        if (
            len(record["requests"]) != protocol["sessions"] * protocol["turns"]
            or len(record["tools"]) != 1
        ):
            raise ValueError("performance replay request/tool count mismatch")
    finally:
        for session in sessions:
            try:
                engine.close_session(session)
                record["closed_session_ids"].append(session)
            except Exception as error:
                record["close_errors"].append(str(error))
    if record["close_errors"]:
        raise ValueError("performance replay close mismatch")


def _read_metrics(metrics_url: str) -> dict:
    raw = fetch_server_metrics(metrics_url, timeout_seconds=2.0)
    names = (
        "requests_running",
        "kv_lcm_blocks_total",
        "kv_lcm_blocks_active",
        "kv_lcm_blocks_cached",
        "kv_lcm_blocks_empty",
        "kv_lcm_blocks_allocated",
    )
    if any(n not in raw or not math.isfinite(raw[n]) or raw[n] < 0 for n in names):
        raise ValueError("missing or invalid performance metrics")
    return {n: raw[n] for n in names}


async def measure_performance(
    engine: Any,
    *,
    trace: list[dict],
    protocol: dict,
    max_model_len: int,
    metrics_url: str,
    output_dir: Path,
    chip_logic_ids: dict[int, int],
    record: dict,
) -> None:
    for field in ("metrics_interval_seconds", "hbm_interval_seconds"):
        if not math.isfinite(protocol[field]) or protocol[field] <= 0:
            raise ValueError("invalid sampling interval")
    if protocol["repetitions"] < 1 or protocol["warmup_pairs"] < 1:
        raise ValueError("warmup and repetition counts must be positive")
    validate_trace(
        trace,
        sessions=protocol["sessions"],
        turns=protocol["turns"],
        max_new_tokens=protocol["max_new_tokens"],
        max_model_len=max_model_len,
    )
    record.update(
        {"arms": [], "metric_samples": [], "hbm_samples": [], "sampler_errors": []}
    )
    stopped = threading.Event()
    phase = {"label": "before_warmup"}
    metrics_ready = threading.Event()
    hbm_ready = threading.Event()

    def metric_worker():
        with (output_dir / "metric_samples.jsonl").open("x") as f:
            while not stopped.is_set():
                start = time.monotonic_ns()
                label = phase["label"]
                metrics = _read_metrics(metrics_url)
                item = {
                    "started_ns": start,
                    "ended_ns": time.monotonic_ns(),
                    "phase": label,
                    "metrics": metrics,
                }
                record["metric_samples"].append(item)
                f.write(json.dumps(item) + "\n")
                f.flush()
                metrics_ready.set()
                stopped.wait(protocol["metrics_interval_seconds"])

    def hbm_worker():
        with (output_dir / "hbm_samples.jsonl").open("x") as f:
            while not stopped.is_set():
                start = time.monotonic_ns()
                label = phase["label"]
                query = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=10
                )
                if query.returncode != 0:
                    raise ValueError(f"HBM sampling query failed: {query.returncode}")
                table = output_dir / f'npu_smi_{len(record["hbm_samples"]):05d}.txt'
                table.write_text(query.stdout)
                values = parse_target_hbm(
                    query.stdout, npu_id=1, chip_logic_ids=chip_logic_ids
                )
                item = {
                    "started_ns": start,
                    "ended_ns": time.monotonic_ns(),
                    "phase": label,
                    "chips": values,
                    "raw_table": table.name,
                }
                record["hbm_samples"].append(item)
                f.write(json.dumps(item) + "\n")
                f.flush()
                hbm_ready.set()
                stopped.wait(protocol["hbm_interval_seconds"])

    def guarded(worker):
        try:
            worker()
        except Exception as error:
            record["sampler_errors"].append(f"{type(error).__name__}: {error}")
            stopped.set()

    threads = [
        threading.Thread(target=guarded, args=(worker,))
        for worker in (metric_worker, hbm_worker)
    ]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + 15
        while not (metrics_ready.is_set() and hbm_ready.is_set()):
            if record["sampler_errors"]:
                raise ValueError("performance sampling failed")
            if time.monotonic() >= deadline:
                raise ValueError("performance sampling startup timeout")
            await asyncio.sleep(0.01)
        for stage, count in (
            ("warmup", protocol["warmup_pairs"]),
            ("formal", protocol["repetitions"]),
        ):
            for repetition in range(count):
                for mode in arm_order(repetition):
                    if record["sampler_errors"]:
                        raise ValueError("performance sampling failed")
                    phase["label"] = f"{stage}_r{repetition}_{mode}"
                    arm = {
                        "phase": stage,
                        "repetition": repetition,
                        "mode": mode,
                        "status": "started",
                    }
                    record["arms"].append(arm)
                    await run_replay_arm(
                        engine,
                        trace=trace,
                        protocol=protocol,
                        mode=mode,
                        max_model_len=max_model_len,
                        record=arm,
                    )
                    phase["label"] = "between_arms"
                    await asyncio.to_thread(engine.get_server_info)
                    deadline = time.monotonic() + 5
                    while True:
                        metrics = await asyncio.to_thread(_read_metrics, metrics_url)
                        arm["idle_after_close"] = metrics
                        if (
                            metrics["requests_running"] == 0
                            and metrics["kv_lcm_blocks_active"] == 0
                        ):
                            break
                        if time.monotonic() >= deadline:
                            raise ValueError(
                                "performance arm did not release active KV"
                            )
                        await asyncio.sleep(0.05)
                    arm["status"] = "passed"
        if record["sampler_errors"]:
            raise ValueError("performance sampling failed")
    finally:
        phase["label"] = "stopping"
        stopped.set()
        for thread in threads:
            await asyncio.to_thread(thread.join, 15)
            if thread.is_alive():
                raise RuntimeError(
                    "performance sampler did not stop before Engine teardown"
                )
    if record["sampler_errors"]:
        raise ValueError("performance sampling failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent Session ON/OFF replay of fixed native Agent traces; no concurrent quality controls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine-port", type=int, required=True)
    parser.add_argument("--metrics-port", type=int, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(REPO_ROOT):
        raise ValueError("output directory must be inside the project")
    if out.exists():
        raise FileExistsError("output directory already exists")
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))

    def source_mismatches():
        return [
            n
            for n, s in sources.items()
            if not (REPO_ROOT / n).is_file()
            or hashlib.sha256((REPO_ROOT / n).read_bytes()).hexdigest() != s
        ]

    if source_mismatches():
        raise ValueError("performance source manifest mismatch")
    config = load_model_config(args.config)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assert (
        config["device"] == "npu"
        and config["dtype"] == "bfloat16"
        and config["enforce_eager"] is False
    )
    assert (
        config["tp_size"] in (1, 2)
        and config["prefix_caching"] is False
        and config["disable_kvstore"] is True
        and config["enable_metrics"] is True
    )
    entry = protocol["traces"][config["model_id"]]
    trace_path = (REPO_ROOT / entry["path"]).resolve()
    assert trace_path.is_relative_to(REPO_ROOT)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == entry["sha256"]
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    validate_trace(
        trace,
        sessions=protocol["sessions"],
        turns=protocol["turns"],
        max_new_tokens=protocol["max_new_tokens"],
        max_model_len=config["max_model_len"],
    )
    expected_visible = "2,3" if config["tp_size"] == 2 else "2"
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == expected_visible
    assert not any(
        k.startswith("TOKENSPEED_DIAG_")
        or k.startswith("TOKENSPEED_9B_")
        or k == "TOKENSPEED_STAGE_B_ARM"
        for k in os.environ
    )
    chips = {chip: chip + 2 for chip in range(config["tp_size"])}
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    for chip, logic in chips.items():
        assert re.search(
            rf"(?m)^\s*1\s+{chip}\s+{logic}\s+{logic}\s+Ascend910\s*$", mapping
        )
    processes = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", processes)
    if config["tp_size"] == 2:
        assert processes.count("No process in device.") == 2
    for port in (args.engine_port, args.engine_port + 1, args.metrics_port):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
    _local_model_path(config)
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    config = {**config, "tokenspeed_metrics_port": args.metrics_port}
    out.mkdir(parents=True, exist_ok=False)
    (out / "mapping.txt").write_text(mapping)
    (out / "before_processes.txt").write_text(processes)
    record = {
        "config": config,
        "protocol": protocol,
        "trace": entry,
        "host_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "clocks": {
            name: {
                "resolution_seconds": time.get_clock_info(name).resolution,
                "implementation": time.get_clock_info(name).implementation,
            }
            for name in ("monotonic", "perf_counter")
        },
        "source_manifest_sha256": hashlib.sha256(
            args.source_manifest.read_bytes()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "status": "started",
        "scope": "fixed real-trajectory conditional replay; scripted tool; client stream latency; observed HBM, not instantaneous peak",
    }
    try:
        with Engine(
            server_args=prepare_server_args(
                _engine_cli_args(
                    build_serve_command(config, host="127.0.0.1", port=args.engine_port)
                )
            )
        ) as engine:
            asyncio.run(
                measure_performance(
                    engine,
                    trace=trace,
                    protocol=protocol,
                    max_model_len=config["max_model_len"],
                    metrics_url=f"http://127.0.0.1:{args.metrics_port}",
                    output_dir=out,
                    chip_logic_ids=chips,
                    record=record,
                )
            )
        if source_mismatches():
            raise ValueError("performance source changed during measurement")
        record.update({"status": "passed", "source_unchanged": True})
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        record["source_after_mismatches"] = source_mismatches()
        (out / "run.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
