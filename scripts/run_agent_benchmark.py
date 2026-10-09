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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import (
    OpenAIChatClient,
    append_jsonl,
    load_model_config,
    metrics_server_url,
    model_load_path,
)
from benchmarks.course_project.telemetry import (
    ServerMetricsSampler,
    add_metric_delta,
    capture_npu_smi,
    snapshot_server_metrics,
)
from benchmarks.course_project.runner import (
    BenchmarkRequest,
    build_agent_session_contexts,
    build_agent_turn_request,
    run_concurrent_requests,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-session Agent workloads against TokenSpeed.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--shared-prefix-ratios", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--tool-wait-ms", type=int, default=100)
    parser.add_argument(
        "--tool-wait-policy",
        choices=("all", "one-session"),
        default="all",
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "raw" / "agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = args.output_dir / f"{config['served_model_name']}-{run_id}.jsonl"
    summary_path = args.output_dir / f"{config['served_model_name']}-{run_id}.summary.jsonl"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config), revision=config["model_revision"]
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    client = OpenAIChatClient(args.base_url)
    metrics_url = metrics_server_url(config)
    warmups = [
        BenchmarkRequest(
            request_id=f"warmup-{index}",
            messages=[{"role": "user", "content": "Warm up the model."}],
            workload="warmup",
        )
        for index in range(args.warmup_requests)
    ]
    if warmups:
        run_concurrent_requests(
            client=client,
            requests=warmups,
            concurrency=min(3, len(warmups)),
            model=config["served_model_name"],
            max_tokens=min(args.output_tokens, 16),
        )
    for concurrency in args.concurrency:
        for shared_ratio in args.shared_prefix_ratios:
            shared_tokens = round(args.prefix_tokens * shared_ratio)
            for repetition in range(args.runs):
                telemetry_label = (
                    f"c{concurrency}-p{shared_ratio:g}-r{repetition + 1}"
                )
                metrics_before = snapshot_server_metrics(metrics_url)
                npu_smi_before_path = (
                    args.output_dir
                    / "hardware"
                    / f"{config['served_model_name']}-{run_id}-{telemetry_label}-before.txt"
                )
                npu_smi_before = capture_npu_smi(npu_smi_before_path)
                session_contexts = build_agent_session_contexts(
                    concurrency=concurrency,
                    total_tokens=args.prefix_tokens,
                    shared_tokens=shared_tokens,
                    repetition=repetition,
                    encode=encode,
                    decode=tokenizer.decode,
                )
                histories = {
                    session_id: [{"role": "system", "content": context}]
                    for session_id, context in session_contexts.items()
                }

                all_records = []
                wall_time_seconds = 0.0
                metrics_during = {}
                for turn_index in range(1, args.turns + 1):
                    user_message = f"Turn {turn_index}: analyze the current task and propose the next concrete action."
                    requests = []
                    for session_id, history in histories.items():
                        requests.append(
                            build_agent_turn_request(
                                request_id=f"c{concurrency}-p{shared_ratio:g}-r{repetition}-s{session_id}-t{turn_index}",
                                session_id=session_id,
                                turn_index=turn_index,
                                history=history,
                                user_message=user_message,
                                tool_wait_policy=args.tool_wait_policy,
                                tool_wait_ms=args.tool_wait_ms,
                            )
                        )
                    requests_by_session = {
                        request.session_id: request for request in requests
                    }
                    started_at = time.perf_counter()
                    metrics_sampler = ServerMetricsSampler(metrics_url)
                    metrics_sampler.start()
                    try:
                        turn_records = run_concurrent_requests(
                            client=client,
                            requests=requests,
                            concurrency=concurrency,
                            model=config["served_model_name"],
                            max_tokens=args.output_tokens,
                        )
                    finally:
                        turn_metrics = metrics_sampler.stop()
                    for key, value in turn_metrics.items():
                        metrics_during[key] = max(metrics_during.get(key, value), value)
                    wall_time_seconds += time.perf_counter() - started_at
                    for record in turn_records:
                        record.update(
                            {
                                "run_id": run_id,
                                "repetition": repetition + 1,
                                "turn_number": turn_index,
                                "model_id": config["model_id"],
                                "model_revision": config["model_revision"],
                                "served_model_name": config["served_model_name"],
                                "tp_size": config["tp_size"],
                                "shared_prefix_ratio": shared_ratio,
                                "target_prefix_tokens": args.prefix_tokens,
                                "target_output_tokens": args.output_tokens,
                                "tool_wait_policy": args.tool_wait_policy,
                            }
                        )
                        session_history = histories[record["session_id"]]
                        session_history.append(
                            requests_by_session[record["session_id"]].messages[-1]
                        )
                        session_history.append({"role": "assistant", "content": record["output_text"]})
                        append_jsonl(raw_path, record)
                    all_records.extend(turn_records)

                summary = summarize_records(all_records, wall_time_seconds=wall_time_seconds)
                summary.update(
                    {
                        "run_id": run_id,
                        "repetition": repetition + 1,
                        "model_id": config["model_id"],
                        "model_revision": config["model_revision"],
                        "served_model_name": config["served_model_name"],
                        "tp_size": config["tp_size"],
                        "concurrency": concurrency,
                        "shared_prefix_ratio": shared_ratio,
                        "target_prefix_tokens": args.prefix_tokens,
                        "turns": args.turns,
                        "tool_wait_ms": args.tool_wait_ms,
                        "tool_wait_policy": args.tool_wait_policy,
                    }
                )
                metrics_after = snapshot_server_metrics(metrics_url)
                npu_smi_after_path = (
                    args.output_dir
                    / "hardware"
                    / f"{config['served_model_name']}-{run_id}-{telemetry_label}-after.txt"
                )
                npu_smi_after = capture_npu_smi(npu_smi_after_path)
                add_metric_delta(summary, metrics_before, metrics_after, metrics_during)
                summary["npu_smi_before_path"] = (
                    str(npu_smi_before_path) if npu_smi_before else None
                )
                summary["npu_smi_after_path"] = (
                    str(npu_smi_after_path) if npu_smi_after else None
                )
                append_jsonl(summary_path, summary)

    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
