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

"""Published-runtime two-session close/reopen protocol for an explicit model."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("0.8b", "4b", "9b", "27b"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    key = args.model
    visible = "2,3" if key == "27b" else "2"
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == visible
    assert not any(
        k.startswith("TOKENSPEED_DIAG_")
        or k.startswith("TOKENSPEED_9B_")
        or k == "TOKENSPEED_STAGE_B_ARM"
        for k in os.environ
    )
    manifest_path = repo / "docs/evidence/agent_20261015/source_manifest.json"
    sources = json.loads(manifest_path.read_text())

    def check_source():
        assert all(
            hashlib.sha256((repo / p).read_bytes()).hexdigest() == s
            for p, s in sources.items()
        )

    check_source()
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    assert re.search(r"(?m)^\s*1\s+0\s+2\s+2\s+Ascend910\s*$", mapping)
    if key == "27b":
        assert re.search(r"(?m)^\s*1\s+1\s+3\s+3\s+Ascend910\s*$", mapping)
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    processes = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", processes)
    proc = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", proc)
    if key == "27b":
        assert proc.count("No process in device.") == 2
    for port in (18590, 18591, 39701, 39702):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
    from transformers import AutoTokenizer

    from benchmarks.course_project.core import (
        build_serve_command,
        load_model_config,
        model_load_path,
    )
    from scripts.run_session_kv import _engine_cli_args, _run_sessions
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    config = load_model_config(args.config)
    assert config["model_id"] == f"Qwen/Qwen3.5-{key[:-1]}B"
    assert re.fullmatch(r"[a-zA-Z0-9_.-]+", args.run_label) and args.run_label not in (
        ".",
        "..",
    )
    config = {**config, "prometheus_port": 39701, "tokenspeed_metrics_port": 39702}
    out = repo / f"results/raw/session_{args.run_label}"
    out.mkdir(exist_ok=False)
    (out / "before_device.txt").write_text(before)
    (out / "mapping.txt").write_text(mapping)
    shutil.copyfile(__file__, out / "runner_source.py")
    record = {
        "model_key": key,
        "config": config,
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "frontier_policy": "current-frontier",
        "max_new_tokens": 32,
        "rounds": [],
        "scope": "two native sessions with concurrent stateless controls; two turns, close/reopen; not a latency benchmark or HF correctness substitute",
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config),
        revision=config["model_revision"],
        local_files_only=True,
    )
    try:
        with Engine(
            server_args=prepare_server_args(
                _engine_cli_args(
                    build_serve_command(config, host="127.0.0.1", port=18590)
                )
            )
        ) as engine:
            for index in range(2):
                phase = {
                    "round_index": index,
                    "turns": [],
                    "closed_session_rejections": [],
                }
                record["rounds"].append(phase)
                asyncio.run(
                    _run_sessions(
                        engine,
                        tokenizer,
                        frontier_policy="current-frontier",
                        max_new_tokens=32,
                        run_record=phase,
                    )
                )
                assert (
                    len(phase["turns"]) == 4
                    and len(phase["closed_session_rejections"]) == 2
                )
                assert all(
                    x["output_ids_match"] and x["context_metadata_validated"]
                    for x in phase["turns"]
                )
            a, b = record["rounds"]
            assert set(a["session_ids"]).isdisjoint(b["session_ids"])
            assert all(
                x["stateless_control_input_ids"] == y["stateless_control_input_ids"]
                and x["session_output_ids"] == y["session_output_ids"]
                for x, y in zip(a["turns"], b["turns"])
            )
        check_source()
        record["status"] = "passed"
        record["source_unchanged"] = True
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
        after = subprocess.check_output(["npu-smi", "info"], text=True)
        (out / "after_device.txt").write_text(after)
        record["target_released"] = "No running processes found in NPU 1" in after
        (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": key,
                    "status": record["status"],
                    "rounds": len(record["rounds"]),
                    "target_released": record["target_released"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
