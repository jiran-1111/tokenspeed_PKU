#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
from benchmarks.course_project.profiler import (
    ProfilerControlClient,
    profiler_control_url,
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


@dataclass(frozen=True)
class ProfileScenario:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    num_requests: int


SCENARIOS = {
    "prefill-heavy": ProfileScenario("prefill-heavy", 8192, 128, 2, 4),
    "decode-heavy": ProfileScenario("decode-heavy", 256, 512, 8, 16),
    "agent-concurrency": ProfileScenario("agent-concurrency", 2048, 128, 16, 32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Ascend profiles for prefill-heavy, decode-heavy, and "
            "high-concurrency Agent workloads."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw" / "profiler",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=REPO_ROOT / "results" / "profiles",
    )
    parser.add_argument("--warmup-requests", type=int, default=3)
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
    chat_client = OpenAIChatClient(f"{args.server_url.rstrip('/')}/v1")
    control_client = ProfilerControlClient(profiler_control_url(args.server_url))
    metrics_url = metrics_server_url(config)

    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        max_input_tokens = max(
            1, int(config["max_model_len"]) - scenario.output_tokens - 64
        )
        input_tokens = min(scenario.input_tokens, max_input_tokens)
        prompt = make_token_sized_text(
            target_tokens=input_tokens,
            seed="TokenSpeed Ascend profiler workload context. ",
            encode=encode,
            decode=tokenizer.decode,
        )
        requests = _build_requests(scenario, prompt)
        profile_id = f"{run_id}-{scenario.name}"
        profile_dir = (
            args.profile_dir / config["served_model_name"] / scenario.name
        ).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)

        warmups = _build_requests(
            ProfileScenario(
                "warmup", 1, min(scenario.output_tokens, 16), 1, args.warmup_requests
            ),
            "Warm up the model before profiling.",
        )
        if warmups:
            run_concurrent_requests(
                client=chat_client,
                requests=warmups,
                concurrency=min(scenario.concurrency, len(warmups)),
                model=config["served_model_name"],
                max_tokens=min(scenario.output_tokens, 16),
            )

        metrics_before = snapshot_server_metrics(metrics_url)
        npu_smi_before_path = (
            args.output_dir
            / "hardware"
            / f"{config['served_model_name']}-{run_id}-{scenario.name}-before.txt"
        )
        npu_smi_before = capture_npu_smi(npu_smi_before_path)
        control_client.start(output_dir=profile_dir, profile_id=profile_id)
        metrics_sampler = ServerMetricsSampler(metrics_url)
        metrics_sampler.start()
        try:
            started_at = time.perf_counter()
            records = run_concurrent_requests(
                client=chat_client,
                requests=requests,
                concurrency=scenario.concurrency,
                model=config["served_model_name"],
                max_tokens=scenario.output_tokens,
            )
            wall_time_seconds = time.perf_counter() - started_at
        finally:
            metrics_during = metrics_sampler.stop()
            control_client.stop()
        metrics_after = snapshot_server_metrics(metrics_url)
        npu_smi_after_path = (
            args.output_dir
            / "hardware"
            / f"{config['served_model_name']}-{run_id}-{scenario.name}-after.txt"
        )
        npu_smi_after = capture_npu_smi(npu_smi_after_path)

        for record in records:
            record.update(
                {
                    "run_id": run_id,
                    "profile_id": profile_id,
                    "scenario": scenario.name,
                    "model_id": config["model_id"],
                    "model_revision": config["model_revision"],
                    "served_model_name": config["served_model_name"],
                    "target_input_tokens": input_tokens,
                    "target_output_tokens": scenario.output_tokens,
                    "concurrency": scenario.concurrency,
                }
            )
            append_jsonl(raw_path, record)

        summary = summarize_records(records, wall_time_seconds=wall_time_seconds)
        summary.update(
            {
                "run_id": run_id,
                "profile_id": profile_id,
                "profile_dir": str(profile_dir),
                "scenario": scenario.name,
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "served_model_name": config["served_model_name"],
                "target_input_tokens": input_tokens,
                "target_output_tokens": scenario.output_tokens,
                "concurrency": scenario.concurrency,
                "tp_size": config["tp_size"],
                "chunk_size": config["chunked_prefill_size"],
                "block_size": config["block_size"],
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


def _build_requests(scenario: ProfileScenario, prompt: str) -> list[BenchmarkRequest]:
    if scenario.name == "agent-concurrency":
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Select the next tool call and explain the decision briefly.",
            },
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    return [
        BenchmarkRequest(
            request_id=f"{scenario.name}-{index}",
            messages=[message.copy() for message in messages],
            workload=scenario.name,
        )
        for index in range(scenario.num_requests)
    ]


if __name__ == "__main__":
    main()
