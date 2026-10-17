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

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def main():
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    raw = repo / "results/raw/reproduce_control_first_query_20261017"
    out = raw / "npu"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "runner_source.py")
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert not any(
        k.startswith("TOKENSPEED_DIAG_")
        or k.startswith("TOKENSPEED_9B_")
        or k == "TOKENSPEED_STAGE_B_ARM"
        for k in os.environ
    )
    manifest = repo / "docs/evidence/control_first_query_20261017/source_manifest.json"
    sources = json.loads(manifest.read_text())

    def check():
        assert all(
            hashlib.sha256((repo / n).read_bytes()).hexdigest() == s
            for n, s in sources.items()
        )

    check()
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    assert re.search(r"(?m)^\s*1\s+0\s+2\s+2\s+Ascend910\s*$", mapping)
    proc = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", proc)
    for port in (18760, 18761, 39861, 39862):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0, port
    (out / "mapping.txt").write_text(mapping)
    (out / "before_processes.txt").write_text(proc)
    from benchmarks.course_project.core import build_serve_command, load_model_config
    from scripts.run_session_kv import _engine_cli_args
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    config = {
        **load_model_config(repo / "configs/baseline/qwen35_0.8b.yaml"),
        "prometheus_port": 39861,
        "tokenspeed_metrics_port": 39862,
    }
    assert config["enforce_eager"] is False
    result = {
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "config": config,
        "status": "started",
        "queries": [],
        "scope": "control first-query liveness before any public generate request, not a performance or HF correctness benchmark",
    }

    def query(engine, label):
        started = time.perf_counter()
        info = engine.get_server_info()
        assert isinstance(info, dict) and len(info["internal_states"]) == 1
        result["queries"].append(
            {
                "label": label,
                "completed": True,
                "elapsed_seconds": time.perf_counter() - started,
                "internal_state_ranks": len(info["internal_states"]),
                "dtype": info["dtype"],
                "enforce_eager": info["enforce_eager"],
            }
        )

    try:
        with Engine(
            server_args=prepare_server_args(
                _engine_cli_args(
                    build_serve_command(config, host="127.0.0.1", port=18760)
                )
            )
        ) as engine:
            query(engine, "before_first_generate")
            query(engine, "repeated_before_first_generate")
            response = engine.generate(
                input_ids=[1000] * 128,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
            )
            assert len(response["output_ids"]) == 16
            result["output_ids"] = response["output_ids"]
            result["first_generate_completed"] = True
            query(engine, "after_first_generate")
        check()
        result.update({"status": "passed", "source_unchanged": True})
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
