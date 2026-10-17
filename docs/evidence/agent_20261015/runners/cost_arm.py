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

"""One formal native-streaming 9B head-cost arm; no hooks or profiler."""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_target_hbm(output):
    values = []
    current = None
    for line in output.splitlines():
        cols = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cols) != 3:
            continue
        name_match = re.fullmatch(r"(\d+)\s+Ascend910", cols[0])
        if name_match:
            current = int(name_match.group(1))
        if (
            current == 1
            and re.fullmatch(r"0\s+2", cols[0])
            and re.fullmatch(r"[0-9a-fA-F:.]+", cols[1])
        ):
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*$", cols[-1])
            assert match, cols
            values.append(int(match.group(1)))
    assert len(values) == 1, values
    return values[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("baseline", "candidate"))
    p.add_argument("repetition", type=int)
    args = p.parse_args()
    assert args.repetition in (0, 1, 2)
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    out = repo / f"results/raw/reproduce_9b_head_cost/r{args.repetition}_{args.mode}"
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
    for port in (18620, 18621, 39901, 39902):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
    out.mkdir(parents=True, exist_ok=False)
    (out / "mapping.txt").write_text(mapping)
    (out / "before_processes.txt").write_text(proc)
    shutil.copyfile(__file__, out / "runner_source.py")
    from benchmarks.course_project.core import build_serve_command, load_model_config
    from scripts.run_session_kv import _engine_cli_args
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    name = (
        "configs/baseline/qwen35_9b.yaml"
        if args.mode == "baseline"
        else "configs/candidates/qwen35_9b_fp32_head.yaml"
    )
    config = {
        **load_model_config(repo / name),
        "prometheus_port": 39901,
        "tokenspeed_metrics_port": 39902,
    }
    assert config["enforce_eager"] is False
    conditions = [(128, 32), (128, 128), (1024, 32), (1024, 128)]
    if (args.repetition + (args.mode == "candidate")) % 2:
        conditions.reverse()
    record = {
        "mode": args.mode,
        "repetition": args.repetition,
        "config": config,
        "conditions": conditions,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "warmups": "3 requests of 8 output tokens per input length before formal conditions",
        "requests_per_condition": 4,
        "native_input": "token ID1000 repeated to exact input length",
        "not_a_semantic_quality_test": True,
        "hbm": [],
    }
    (out / "protocol.json").write_text(
        json.dumps({k: v for k, v in record.items() if k != "hbm"}, indent=2) + "\n"
    )
    stopped = threading.Event()
    phase = {"label": "startup"}

    def sample():
        with (out / "hbm_samples.jsonl").open("x") as f:
            while not stopped.is_set():
                started = time.perf_counter_ns()
                epoch = time.time_ns()
                label = phase["label"]
                query = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=10
                )
                assert query.returncode == 0, query.returncode
                target = parse_target_hbm(query.stdout)
                item = {
                    "epoch_ns": epoch,
                    "monotonic_ns": started,
                    "phase": label,
                    "returncode": query.returncode,
                    "query_seconds": (time.perf_counter_ns() - started) / 1e9,
                    "hbm_mb": target,
                }
                record["hbm"].append(item)
                f.write(json.dumps(item) + "\n")
                f.flush()
                stopped.wait(1)

    sample_errors = []

    def guarded_sample():
        try:
            sample()
        except BaseException as error:
            sample_errors.append(f"{type(error).__name__}: {error}")
            stopped.set()

    sampler = threading.Thread(target=guarded_sample)
    sampler.start()
    time.sleep(1.5)
    assert not sample_errors and record["hbm"], sample_errors

    def generate(engine, input_tokens, output_tokens):
        assert not sample_errors, sample_errors
        start = time.perf_counter_ns()
        ids = []
        times = []
        counts = []
        last_meta = None
        for chunk in engine.generate(
            input_ids=[1000] * input_tokens,
            stream=True,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
                "stream_interval": 1,
            },
        ):
            values = chunk["output_ids"]
            if values:
                times.append(time.perf_counter_ns())
                counts.append(len(values))
                ids.extend(values)
            last_meta = chunk.get("meta_info")
        end = time.perf_counter_ns()
        assert (
            len(ids) == output_tokens
            and all(n == 1 for n in counts)
            and len(times) == output_tokens
        )
        assert times[0] > start and all(b >= a for a, b in zip(times, times[1:]))
        return {
            "input_tokens": input_tokens,
            "output_tokens": len(ids),
            "output_ids": ids,
            "ttft_ms": (times[0] - start) / 1e6,
            "tpot_ms": (times[-1] - times[0]) / (len(ids) - 1) / 1e6,
            "e2e_ms": (end - start) / 1e6,
            "inter_token_ms": [(b - a) / 1e6 for a, b in zip(times, times[1:])],
            "stream_chunk_sizes": counts,
            "last_meta": last_meta,
        }

    try:
        with Engine(
            server_args=prepare_server_args(
                _engine_cli_args(
                    build_serve_command(config, host="127.0.0.1", port=18620)
                )
            )
        ) as engine:
            (out / "loaded_device.txt").write_text(
                subprocess.check_output(["npu-smi", "info"], text=True)
            )
            phase["label"] = "warmup"
            for input_tokens in (128, 1024):
                for _ in range(3):
                    generate(engine, input_tokens, 8)
            with (out / "requests.jsonl").open("x") as f:
                for input_tokens, output_tokens in conditions:
                    phase["label"] = f"formal_{input_tokens}_{output_tokens}"
                    for index in range(4):
                        row = generate(engine, input_tokens, output_tokens)
                        row.update(
                            {
                                "mode": args.mode,
                                "repetition": args.repetition,
                                "request_index": index,
                            }
                        )
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                    print(
                        f"Completed {args.mode} repetition{args.repetition}: {input_tokens}/{output_tokens}",
                        flush=True,
                    )
            phase["label"] = "shutdown"
            # Engine.shutdown kills its process descendants, including an active npu-smi.
            # Finish the sampling query before leaving the Engine context.
            stopped.set()
            sampler.join(timeout=15)
            assert not sampler.is_alive() and not sample_errors, sample_errors
        assert not sample_errors and len(record["hbm"]) >= 2, sample_errors
        check()
        record["status"] = "collected"
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
        stopped.set()
        sampler.join(timeout=15)
        assert not sampler.is_alive()
        record["sampler_errors"] = sample_errors
        record["hbm_sample_count"] = len(record["hbm"])
        (out / "result.json").write_text(
            json.dumps({k: v for k, v in record.items() if k != "hbm"}, indent=2) + "\n"
        )
        (out / "after_device.txt").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )


if __name__ == "__main__":
    main()
