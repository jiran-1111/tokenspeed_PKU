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

"""Run matched session offloading workloads and independent full-history controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from benchmarks.course_project.core import build_serve_command, load_model_config
from scripts.run_session_kv import _engine_cli_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offloading", choices=("on", "off"), required=True)
    parser.add_argument("--host-ratio", type=float, required=True)
    parser.add_argument("--pattern", choices=("high", "low"), required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 1 or not math.isfinite(args.host_ratio) or args.host_ratio <= 0:
        parser.error("cycles and host-ratio must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["TOKENSPEED_CACHE_TRACE_DIR"] = str(out / "trace")
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    root = Path(__file__).resolve().parents[1]
    import tokenspeed_scheduler

    if not Path(tokenspeed_scheduler._ext.__file__).resolve().is_relative_to(root):
        raise RuntimeError(
            "scheduler extension must be built inside this source snapshot"
        )
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in (
            "python",
            "tokenspeed-kernel",
            "tokenspeed-kernel-npu",
            "tokenspeed-scheduler",
            "scripts",
            "benchmarks",
            "configs/homework3",
        )
        for p in (root / directory).rglob("*")
        if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".so", ".yaml", ".sh")
    }
    (out / "source_manifest.json").write_text(json.dumps(sources, indent=2))

    def check_source():
        if any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != sha
            for name, sha in sources.items()
        ):
            raise RuntimeError("source changed during experiment")

    def capture_memory(name):
        (out / f"npu-{name}.txt").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )

    capture_memory("before")
    config = load_model_config(args.config)
    config["disable_kvstore"] = args.offloading == "off"
    cli = _engine_cli_args(build_serve_command(config, host="127.0.0.1", port=39401))
    cli.extend(
        ["--kvstore-io-backend", "direct", "--kvstore-ratio", str(args.host_ratio)]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["local_model_path"], local_files_only=True
    )
    params = {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True}
    record = {
        "config": dict(config),
        "offloading": args.offloading,
        "host_ratio": args.host_ratio,
        "pattern": args.pattern,
        "turns": [],
        "controls": [],
        "status": "running",
    }

    def save():
        (out / "results.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    save()
    try:
        with Engine(server_args=prepare_server_args(cli)) as engine:
            workload_started = time.perf_counter()
            histories = {}
            sessions = {}
            try:
                for cycle in range(args.cycles):
                    for index in range(3):
                        key = (
                            str(index) if args.pattern == "high" else f"{cycle}-{index}"
                        )
                        if key not in sessions:
                            sessions[key] = engine.open_session()
                            histories[key] = []
                        if histories[key]:
                            suffix = tokenizer.encode(
                                f"\nContinue session {key} round {cycle}.\n"
                            )
                        else:
                            seed = tokenizer.encode(
                                f"Session {key}: Explain memory management carefully. "
                            )
                            suffix = (seed * (768 // len(seed) + 1))[:768]
                        full_input = histories[key] + suffix
                        started = time.perf_counter()
                        result = engine.generate(
                            input_ids=suffix,
                            sampling_params=params,
                            session_id=sessions[key],
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        ids = result["output_ids"]
                        record["turns"].append(
                            {
                                "cycle": cycle,
                                "session": key,
                                "input_ids": full_input,
                                "output_ids": ids,
                                "elapsed_ms": elapsed_ms,
                                "meta_info": result["meta_info"],
                            }
                        )
                        histories[key] = full_input + ids
                        save()
                        capture_memory(f"{cycle}-{index}")
            finally:
                for session_id in sessions.values():
                    engine.close_session(session_id)
            record["workload_elapsed_ms"] = (
                time.perf_counter() - workload_started
            ) * 1000
            time.sleep(0.2)
        # A fresh engine with Host and prefix caching disabled supplies each
        # reference from the exact input tokens observed by that session turn.
        config["disable_kvstore"] = True
        config["prefix_caching"] = False
        reference_cli = _engine_cli_args(
            build_serve_command(config, host="127.0.0.1", port=39411)
        )
        os.environ["TOKENSPEED_CACHE_TRACE_DIR"] = str(out / "reference_trace")
        with Engine(server_args=prepare_server_args(reference_cli)) as engine:
            for turn in record["turns"]:
                result = engine.generate(
                    input_ids=turn["input_ids"], sampling_params=params
                )
                record["controls"].append(
                    {
                        "output_ids": result["output_ids"],
                        "match": result["output_ids"] == turn["output_ids"],
                    }
                )
                save()
        check_source()
        capture_memory("after")
        record["status"] = (
            "passed"
            if all(c["match"] for c in record["controls"])
            else "output_mismatch"
        )
    except BaseException as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        raise
    finally:
        save()
    if record["status"] != "passed":
        raise SystemExit("session output differs from its full-history reference")


if __name__ == "__main__":
    main()
