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
    chat_template_token_ids,
    load_model_config,
    metrics_server_url,
    model_load_path,
)
from benchmarks.course_project.runner import (
    BenchmarkRequest,
    make_token_sized_text,
    run_concurrent_requests,
    summarize_records,
)
from benchmarks.course_project.telemetry import (
    ServerMetricsSampler,
    add_metric_delta,
    capture_npu_smi,
    snapshot_server_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible TokenSpeed latency/throughput benchmarks."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--input-tokens", nargs="+", type=int, default=[128, 1024, 2048]
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "raw" / "benchmark"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = args.output_dir / f"{config['served_model_name']}-{run_id}.jsonl"
    summary_path = (
        args.output_dir / f"{config['served_model_name']}-{run_id}.summary.jsonl"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config), revision=config["model_revision"]
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    client = OpenAIChatClient(args.base_url)
    metrics_url = metrics_server_url(config)
    seed = "TokenSpeed Ascend Agent inference benchmark context. "
    for target_input_tokens in args.input_tokens:
        prompt = make_token_sized_text(
            target_tokens=target_input_tokens,
            seed=seed,
            encode=encode,
            decode=tokenizer.decode,
        )
        chat_input_tokens = len(
            chat_template_token_ids(tokenizer, [{"role": "user", "content": prompt}])
        )
        if chat_input_tokens + args.output_tokens > config["max_model_len"]:
            raise ValueError(
                f"Input target {target_input_tokens} produces {chat_input_tokens} chat tokens; "
                f"with {args.output_tokens} output tokens this exceeds "
                f"max_model_len={config['max_model_len']}. Increase the server/config "
                "context length or explicitly request a smaller input."
            )
        for concurrency in args.concurrency:
            warmups = _make_requests(
                "warmup",
                args.warmup_requests,
                prompt,
                encode=encode,
                decode=tokenizer.decode,
            )
            if warmups:
                run_concurrent_requests(
                    client=client,
                    requests=warmups,
                    concurrency=min(concurrency, len(warmups)),
                    model=config["served_model_name"],
                    max_tokens=args.output_tokens,
                )
            for repetition in range(args.runs):
                telemetry_label = (
                    f"c{concurrency}-i{target_input_tokens}-r{repetition + 1}"
                )
                metrics_before = snapshot_server_metrics(metrics_url)
                npu_smi_before_path = (
                    args.output_dir
                    / "hardware"
                    / f"{config['served_model_name']}-{run_id}-{telemetry_label}-before.txt"
                )
                npu_smi_before = capture_npu_smi(npu_smi_before_path)
                requests = _make_requests(
                    f"c{concurrency}-i{target_input_tokens}-r{repetition}",
                    args.num_requests,
                    prompt,
                    encode=encode,
                    decode=tokenizer.decode,
                )
                started_at = time.perf_counter()
                metrics_sampler = ServerMetricsSampler(metrics_url)
                metrics_sampler.start()
                try:
                    records = run_concurrent_requests(
                        client=client,
                        requests=requests,
                        concurrency=concurrency,
                        model=config["served_model_name"],
                        max_tokens=args.output_tokens,
                    )
                finally:
                    metrics_during = metrics_sampler.stop()
                wall_time_seconds = time.perf_counter() - started_at
                metrics_after = snapshot_server_metrics(metrics_url)
                npu_smi_after_path = (
                    args.output_dir
                    / "hardware"
                    / f"{config['served_model_name']}-{run_id}-{telemetry_label}-after.txt"
                )
                npu_smi_after = capture_npu_smi(npu_smi_after_path)
                for record in records:
                    record.update(
                        {
                            "run_id": run_id,
                            "repetition": repetition + 1,
                            "model_id": config["model_id"],
                            "model_revision": config["model_revision"],
                            "served_model_name": config["served_model_name"],
                            "tp_size": config["tp_size"],
                            "chunk_size": config["chunked_prefill_size"],
                            "block_size": config["block_size"],
                            "target_input_tokens": target_input_tokens,
                            "target_output_tokens": args.output_tokens,
                        }
                    )
                    append_jsonl(raw_path, record)
                summary = summarize_records(
                    records, wall_time_seconds=wall_time_seconds
                )
                summary.update(
                    {
                        "run_id": run_id,
                        "repetition": repetition + 1,
                        "model_id": config["model_id"],
                        "model_revision": config["model_revision"],
                        "served_model_name": config["served_model_name"],
                        "tp_size": config["tp_size"],
                        "chunk_size": config["chunked_prefill_size"],
                        "block_size": config["block_size"],
                        "prefix_caching": config["prefix_caching"],
                        "mixed_batch": config["mixed_batch"],
                        "overlap_schedule": config["overlap_schedule"],
                        "concurrency": concurrency,
                        "target_input_tokens": target_input_tokens,
                        "target_output_tokens": args.output_tokens,
                    }
                )
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


def _make_requests(
    prefix: str,
    count: int,
    prompt: str,
    *,
    encode,
    decode,
) -> list[BenchmarkRequest]:
    target_tokens = len(encode(prompt))
    return [
        BenchmarkRequest(
            request_id=f"{prefix}-{index}",
            messages=[
                {
                    "role": "user",
                    "content": make_token_sized_text(
                        target_tokens=target_tokens,
                        seed=f"{prefix} request {index:06d}: {prompt}",
                        encode=encode,
                        decode=decode,
                    ),
                }
            ],
            workload="fixed_length",
        )
        for index in range(count)
    ]


if __name__ == "__main__":
    main()
