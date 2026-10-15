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

"""Collect one approved final-source Decode window after Stage B acceptance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.course_project.core import build_serve_command, load_model_config
from scripts import collect_homework1_references as collector
from scripts.run_session_kv import _engine_cli_args


def capture_decode_window(
    engine: Any,
    *,
    input_ids: list[int],
    start_profile: Callable[[], Any],
    stop_profile: Callable[[], Any],
    record: dict,
) -> None:
    """Profile after the first output; stop at the observed 128-token frontier."""
    record.update({"started": False, "stopped": False, "chunks": []})
    output_ids: list[int] = []
    try:
        stream = engine.generate(
            input_ids=input_ids,
            stream=True,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 256,
                "ignore_eos": True,
            },
        )
        for result in stream:
            output_ids.extend(result["output_ids"])
            count = len(output_ids)
            record["chunks"].append(
                {
                    "tokens": count,
                    "chunk_tokens": len(result["output_ids"]),
                    "epoch_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            record["final"] = {**result, "output_ids": output_ids.copy()}
            if not record["started"] and count:
                if count >= 128:
                    raise ValueError(
                        "First streamed output arrived after the capture frontier"
                    )
                record["start_requested_epoch_ns"] = time.time_ns()
                start_profile()
                record["start_ack_epoch_ns"] = time.time_ns()
                record["started"] = True
            if record["started"] and not record["stopped"] and count >= 128:
                record["stop_requested_epoch_ns"] = time.time_ns()
                stop_profile()
                record["stop_ack_epoch_ns"] = time.time_ns()
                record["stopped"] = True
        if (
            not record["started"]
            or not record["stopped"]
            or len(record["final"]["output_ids"]) != 256
        ):
            raise ValueError("Incomplete Decode profile window or generation")
    finally:
        if record["started"] and not record["stopped"]:
            stop_profile()
            record["stopped"] = True
            record["cleanup_stop"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "stage-b-summary", "source-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.stage_b_summary.read_text())
    if gate["vector_count"] != 272 or not gate["all_frozen_pass"]:
        raise ValueError("Stage B numerical acceptance is required before profiling")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2":
        raise ValueError("Requires freshly checked logical device 2 exclusively")
    if any(
        name.startswith("TOKENSPEED_DIAG_") or name == "TOKENSPEED_STAGE_B_ARM"
        for name in os.environ
    ):
        raise ValueError("Numerical capture environment must be absent")
    sources = json.loads(args.source_manifest.read_text())

    def check_sources() -> None:
        for name, sha in sources.items():
            if collector.sha_file(REPO_ROOT / name) != sha:
                raise ValueError(f"Frozen source mismatch: {name}")

    check_sources()
    output = collector.new_output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_model_config(args.config)
    config = {**config, "prometheus_port": 39491, "tokenspeed_metrics_port": 39492}
    if (
        collector.sha_file(args.manifest)
        != "23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb"
    ):
        raise ValueError("Manifest identity mismatch")
    manifest = json.loads(args.manifest.read_text())
    case = next(
        case
        for case in manifest["cases"]
        if case["group"] == "validation" and case["input_length"] == 128
    )
    os.environ["TOKENSPEED_NPU_PROFILER_LEVEL"] = "Level1"
    record = {
        "config": config,
        "input_ids": case["input_ids"],
        "case_id": case["case_id"],
        "source_manifest_sha256": collector.sha_file(args.source_manifest),
        "stage_b_summary_sha256": collector.sha_file(args.stage_b_summary),
        "not_a_performance_baseline": True,
    }
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    cli = _engine_cli_args(build_serve_command(config, host="127.0.0.1", port=18490))
    try:
        with Engine(server_args=prepare_server_args(cli)) as engine:
            for _ in range(3):
                engine.generate(
                    input_ids=case["input_ids"],
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                    },
                )

            def start() -> None:
                engine.llm.run(
                    engine.tokenizer_manager.start_profile(
                        output_dir=str(output / "profile"),
                        activities=["CPU", "NPU", "MEM"],
                        with_stack=False,
                        record_shapes=True,
                        profile_id="hw1-final-decode",
                    )
                )

            capture_decode_window(
                engine,
                input_ids=case["input_ids"],
                start_profile=start,
                stop_profile=engine.stop_profile,
                record=record,
            )
        check_sources()
        record["status"] = "collected"
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        collector.write_json(output / "record.json", record)


if __name__ == "__main__":
    main()
