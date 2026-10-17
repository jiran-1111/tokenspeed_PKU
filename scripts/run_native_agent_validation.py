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
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from benchmarks.course_project.native_agent import (
    append_qwen_user_suffix,
    initial_qwen_context,
)
from benchmarks.course_project.telemetry import fetch_server_metrics
from scripts.run_session_kv import (
    _engine_cli_args,
    _local_model_path,
    validate_session_context_metadata,
)
from scripts.run_session_kv_ttft import measure_stream_ttft


def _stream_request(engine, input_ids, params, *, session_id):
    started = time.monotonic_ns()
    result = measure_stream_ttft(engine, input_ids, params, session_id=session_id)
    return {**result, "started_ns": started, "ended_ns": time.monotonic_ns()}


def _scripted_tool(*, delay_seconds: float, range_stop: int, event: dict) -> None:
    event["started_ns"] = time.monotonic_ns()
    time.sleep(delay_seconds)
    event["result"] = sum(range(range_stop))
    event["ended_ns"] = time.monotonic_ns()


async def run_native_validation(
    engine: Any,
    tokenizer: Any,
    *,
    workload: dict,
    max_model_len: int,
    metrics_url: str,
    record: dict,
) -> None:
    count, turns = workload["sessions"], workload["turns"]
    tool_index, tool_turn = workload["tool_session_index"], workload["tool_before_turn"]
    if (
        count < 2
        or turns < 2
        or not 0 <= tool_index < count
        or not 1 < tool_turn <= turns
    ):
        raise ValueError("invalid native Agent workload dimensions")
    params = {
        "temperature": 0,
        "max_new_tokens": workload["max_new_tokens"],
        "ignore_eos": True,
    }
    session_ids = [engine.open_session() for _ in range(count)]
    record["session_ids"] = session_ids
    record["closed_session_ids"] = []
    record["close_errors"] = []
    failed = asyncio.Event()

    async def snapshot(label):
        metrics = await asyncio.to_thread(
            fetch_server_metrics, metrics_url, timeout_seconds=2.0
        )
        required = (
            "requests_running",
            "kv_lcm_blocks_total",
            "kv_lcm_blocks_active",
            "kv_lcm_blocks_cached",
            "kv_lcm_blocks_empty",
            "kv_lcm_blocks_allocated",
        )
        if any(
            k not in metrics or not math.isfinite(metrics[k]) or metrics[k] < 0
            for k in required
        ):
            raise ValueError("missing or invalid native KV block metrics")
        observed = {name: metrics[name] for name in required}
        record["metric_snapshots"].append(
            {"label": label, "monotonic_ns": time.monotonic_ns(), "metrics": observed}
        )
        return observed

    async def idle_snapshot(label, *, held):
        # Wake the control plane so gauges include the latest finish/close event.
        # The empty internal-state payload is not used as KV telemetry.
        await asyncio.to_thread(engine.get_server_info)
        deadline = time.monotonic() + 3.0
        while True:
            metrics = await snapshot(label)
            active = metrics["kv_lcm_blocks_active"]
            if metrics["requests_running"] == 0 and (
                active > 0 if held else active == 0
            ):
                return
            if time.monotonic() >= deadline:
                raise ValueError(f"native KV lifecycle mismatch at {label}")
            await asyncio.sleep(0.05)

    async def conversation(index):
        history, previous = [], []
        first_text = (
            workload["initial_user_prefix"]
            + workload["initial_filler"] * workload["initial_filler_repetitions"]
            + f"\nSession {index}."
        )
        initial_ids, prefix = initial_qwen_context(
            tokenizer, system_prompt=workload["system_prompt"], user_text=first_text
        )
        for turn in range(1, turns + 1):
            if failed.is_set():
                return
            text = workload["next_user_text"]
            tool = None
            if index == tool_index and turn == tool_turn:
                await snapshot("before_tool")
                tool = {
                    "session_index": index,
                    "before_turn": turn,
                    "started_ns": None,
                    "ended_ns": None,
                    "result": None,
                }
                record["tools"].append(tool)
                await asyncio.to_thread(
                    _scripted_tool,
                    delay_seconds=workload["tool_delay_seconds"],
                    range_stop=workload["tool_range_stop"],
                    event=tool,
                )
                text += "\n" + workload["tool_result_prefix"] + str(tool["result"])
                if failed.is_set():
                    return
            suffix = (
                initial_ids
                if turn == 1
                else append_qwen_user_suffix(
                    tokenizer,
                    generation_prefix=prefix,
                    previous_output_ids=previous,
                    user_text=text,
                )
            )
            full_input = history + suffix
            if len(full_input) + workload["max_new_tokens"] > max_model_len:
                raise ValueError("native Agent context exceeds model limit")
            pair = await asyncio.gather(
                asyncio.to_thread(
                    _stream_request,
                    engine,
                    suffix,
                    params,
                    session_id=session_ids[index],
                ),
                asyncio.to_thread(
                    _stream_request, engine, full_input, params, session_id=None
                ),
                return_exceptions=True,
            )
            for result in pair:
                if isinstance(result, BaseException):
                    raise result
            session, control = pair
            item = {
                "session_index": index,
                "turn_index": turn,
                "prior_history_tokens": len(history),
                "suffix_ids": suffix,
                "full_input_ids": full_input,
                "session_result": session,
                "stateless_result": control,
                "output_ids_match": session["output_ids"] == control["output_ids"],
                "context_validated": False,
            }
            record["turns"].append(item)
            validate_session_context_metadata(
                session["meta_info"]["session"],
                policy=workload["frontier_policy"],
                prior_history_tokens=len(history),
                suffix_tokens=len(suffix),
            )
            item["context_validated"] = True
            if not item["output_ids_match"]:
                raise ValueError(
                    f"native Session/stateless output mismatch at session {index} turn {turn}"
                )
            if len(session["output_ids"]) != workload["max_new_tokens"]:
                raise ValueError("native Agent output count mismatch")
            if tool is not None:
                tool["resumed_reused_context_tokens"] = session["meta_info"]["session"][
                    "reused_context_tokens"
                ]
            previous = session["output_ids"]
            history = full_input + previous
            if any(
                e["started_ns"] is not None and e["ended_ns"] is None
                for e in record["tools"]
            ):
                await snapshot("during_tool")

    async def guarded_conversation(index):
        try:
            await conversation(index)
        except Exception as error:
            failed.set()
            return error
        return None

    try:
        # Drain already-submitted native/control streams before closing a failed run.
        errors = await asyncio.gather(*(guarded_conversation(i) for i in range(count)))
        for error in errors:
            if error is not None:
                raise error
        if len(record["turns"]) != count * turns:
            raise ValueError("native Agent pair count mismatch")
        if len(record["tools"]) != 1:
            raise ValueError("native Agent tool count mismatch")
        tool = record["tools"][0]
        progress = [
            r
            for r in record["turns"]
            if r["session_index"] != tool_index
            and tool["started_ns"] < r["session_result"]["ended_ns"] < tool["ended_ns"]
        ]
        record["other_session_progress_during_tool"] = bool(progress)
        during_tool = [
            s
            for s in record["metric_snapshots"]
            if tool["started_ns"] < s["monotonic_ns"] < tool["ended_ns"]
            and s["metrics"]["kv_lcm_blocks_active"] > 0
        ]
        if (
            not progress
            or not during_tool
            or tool["resumed_reused_context_tokens"] <= 0
        ):
            raise ValueError("native tool wait retention/progress mismatch")
        await idle_snapshot("all_sessions_idle_before_close", held=True)
    finally:
        for session_id in session_ids:
            try:
                engine.close_session(session_id)
                record["closed_session_ids"].append(session_id)
            except Exception as error:
                record["close_errors"].append(
                    {"session_id": session_id, "error": str(error)}
                )
    if record["close_errors"]:
        raise ValueError("native Session close mismatch")
    await idle_snapshot("all_sessions_closed", held=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate native multi-turn Sessions with scripted tool wait; not an independent latency benchmark."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
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

    def check_source():
        assert all(
            hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() == sha
            for name, sha in sources.items()
        )

    check_source()
    config = load_model_config(args.config)
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
    expected_visible = "2,3" if config["tp_size"] == 2 else "2"
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == expected_visible
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    for chip in range(config["tp_size"]):
        assert re.search(
            rf"(?m)^\s*1\s+{chip}\s+{chip + 2}\s+{chip + 2}\s+Ascend910\s*$", mapping
        )
    processes = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", processes)
    if config["tp_size"] == 2:
        assert processes.count("No process in device.") == 2
    for port in (args.engine_port, args.engine_port + 1, args.metrics_port):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0, port
    model_path = _local_model_path(config)
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), revision=config["model_revision"], local_files_only=True
    )
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    config = {**config, "tokenspeed_metrics_port": args.metrics_port}
    out.mkdir(parents=True, exist_ok=False)
    (out / "mapping.txt").write_text(mapping)
    (out / "before_processes.txt").write_text(processes)
    record = {
        "config": config,
        "workload": workload,
        "source_manifest_sha256": hashlib.sha256(
            args.source_manifest.read_bytes()
        ).hexdigest(),
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "turns": [],
        "tools": [],
        "metric_snapshots": [],
        "status": "started",
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
                run_native_validation(
                    engine,
                    tokenizer,
                    workload=workload,
                    max_model_len=config["max_model_len"],
                    metrics_url=f"http://127.0.0.1:{args.metrics_port}",
                    record=record,
                )
            )
        check_source()
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
        (out / "run.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
