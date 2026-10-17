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

"""Formal 9B candidate: original eight native inputs, graph enabled, no hooks."""

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
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    out = repo / "results/raw/reproduce_9b_fp32_head_graph"
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "2"
    assert not any(
        k.startswith("TOKENSPEED_DIAG_")
        or k.startswith("TOKENSPEED_9B_")
        or k == "TOKENSPEED_STAGE_B_ARM"
        for k in os.environ
    )
    manifest = repo / "docs/evidence/agent_20261015/source_manifest.json"
    sources = json.loads(manifest.read_text())

    def check():
        assert all(
            hashlib.sha256((repo / p).read_bytes()).hexdigest() == s
            for p, s in sources.items()
        )

    check()
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    assert re.search(r"(?m)^\s*1\s+0\s+2\s+2\s+Ascend910\s*$", mapping)
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    processes = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(
        r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", processes
    ), "Target physical NPU1/chip0 is occupied"
    for port in (18610, 18611, 39811, 39812):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
    from benchmarks.course_project.core import build_serve_command, load_model_config
    from scripts.run_session_kv import _engine_cli_args
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    config = load_model_config(repo / "configs/candidates/qwen35_9b_fp32_head.yaml")
    assert (
        config["hf_overrides"] == {"npu_lm_head_dtype": "float32"}
        and config["enforce_eager"] is False
    )
    config = {**config, "prometheus_port": 39811, "tokenspeed_metrics_port": 39812}
    capture = repo / "docs/evidence/agent_20261015/native_inputs.jsonl"
    rows = [json.loads(x) for x in capture.read_text().splitlines()]
    assert len(rows) == 8
    out.mkdir(exist_ok=False)
    (out / "mapping.txt").write_text(mapping)
    (out / "before_device.txt").write_text(before)
    (out / "before_processes.txt").write_text(processes)
    shutil.copyfile(__file__, out / "runner_source.py")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "config": config,
                "source_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "input_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "scope": "formal production override, graph enabled, fixed 8 native inputs; no diagnostics/forced tokens",
            },
            indent=2,
        )
        + "\n"
    )
    try:
        with Engine(
            server_args=prepare_server_args(
                _engine_cli_args(
                    build_serve_command(config, host="127.0.0.1", port=18610)
                )
            )
        ) as engine:
            info = {
                "dtype": engine.server_args.dtype,
                "hf_overrides": engine.server_args.hf_overrides,
                "enforce_eager": engine.server_args.enforce_eager,
                "device": engine.server_args.device,
                "revision": engine.server_args.revision,
            }
            selected = {
                k: info[k]
                for k in (
                    "dtype",
                    "hf_overrides",
                    "enforce_eager",
                    "device",
                    "revision",
                )
                if k in info
            }
            (out / "server_configuration.json").write_text(
                json.dumps(selected, indent=2) + "\n"
            )
            for index, row in enumerate(rows):
                response = engine.generate(
                    input_ids=row["client_input_token_ids"],
                    sampling_params=row["sampling_params"],
                )
                record = {
                    k: row[k]
                    for k in (
                        "case",
                        "category",
                        "turn_index",
                        "client_input_token_ids",
                        "sampling_params",
                        "model_id",
                        "model_revision",
                        "served_model_name",
                    )
                }
                record.update(
                    {
                        "output_token_ids": response["output_ids"],
                        "output_text": response.get("text", ""),
                        "token_correctness_verified": False,
                    }
                )
                with (out / "engine.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                (out / f"response_{index:02d}.json").write_text(
                    json.dumps(response) + "\n"
                )
                print(f"Collected {index+1}/8", flush=True)
        check()
        refs = json.loads(
            (repo / "docs/evidence/agent_20261015/native_reference.json").read_text()
        )["comparisons"]
        from benchmarks.course_project.correctness import (
            compare_token_ids_with_stop_alignment,
        )

        collected = [
            json.loads(x) for x in (out / "engine.jsonl").read_text().splitlines()
        ]
        comparisons = []
        for original, row, ref in zip(rows, collected, refs):
            assert row["client_input_token_ids"] == original["client_input_token_ids"]
            comparisons.append(
                {
                    "case": row["case"],
                    "turn_index": row["turn_index"],
                    **compare_token_ids_with_stop_alignment(
                        row["output_token_ids"],
                        ref["reference_output_token_ids"],
                        ref["stop_token_ids"],
                    ),
                }
            )
        result = {
            "record_count": len(collected),
            "exact_match_count": sum(x["exact_match"] for x in comparisons),
            "stop_aligned_exact_match_count": sum(
                x["stop_aligned_exact_match"] for x in comparisons
            ),
            "source_unchanged": True,
            "comparisons": comparisons,
            "reference_sha256": hashlib.sha256(
                (
                    repo / "docs/evidence/agent_20261015/native_reference.json"
                ).read_bytes()
            ).hexdigest(),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        (out / "after_device.txt").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )


if __name__ == "__main__":
    main()
