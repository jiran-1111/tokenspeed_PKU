#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.ablation import (
    build_ablation_configs,
    configure_ablation_ports,
)
from benchmarks.course_project.core import (
    OpenAIChatClient,
    append_jsonl,
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
        description="Run the independently switchable upstream TokenSpeed ablations."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunked-prefill-size", type=int)
    parser.add_argument("--variants", nargs="+", default=["E0", "E1", "E2", "E3"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw" / "ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_model_config(args.config)
    variants = build_ablation_configs(
        base_config, chunked_prefill_size=args.chunked_prefill_size
    )
    selected = [item for item in variants if item["name"] in args.variants]
    if len(selected) != len(set(args.variants)):
        raise ValueError(f"Unknown ablation variant in {args.variants}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(base_config), revision=base_config["model_revision"]
    )
    prompt = make_token_sized_text(
        target_tokens=args.input_tokens,
        seed="TokenSpeed Ascend ablation benchmark context. ",
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        decode=tokenizer.decode,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = args.output_dir / f"{base_config['served_model_name']}-{run_id}.jsonl"
    summary_path = (
        args.output_dir / f"{base_config['served_model_name']}-{run_id}.summary.jsonl"
    )
    log_dir = args.output_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    for variant_index, variant in enumerate(selected):
        port = args.port + variant_index
        config = variant["config"]
        configure_ablation_ports(config, base_config, variant_index)
        command = _serve_command(config, host=args.host, port=port)
        log_path = log_dir / f"{variant['name']}.log"
        print(f"Starting {variant['name']} ({variant['label']}) on port {port}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_for_server(process, args.host, port, args.startup_timeout)
                client = OpenAIChatClient(f"http://127.0.0.1:{port}/v1")
                warmups = _requests("warmup", args.warmup_requests, prompt)
                if warmups:
                    run_concurrent_requests(
                        client=client,
                        requests=warmups,
                        concurrency=min(args.concurrency, len(warmups)),
                        model=base_config["served_model_name"],
                        max_tokens=args.output_tokens,
                    )

                for repetition in range(args.runs):
                    telemetry_label = f"{variant['name']}-r{repetition + 1}"
                    metrics_url = metrics_server_url(config)
                    metrics_before = snapshot_server_metrics(metrics_url)
                    npu_smi_before_path = (
                        args.output_dir
                        / "hardware"
                        / f"{base_config['served_model_name']}-{run_id}-{telemetry_label}-before.txt"
                    )
                    npu_smi_before = capture_npu_smi(npu_smi_before_path)
                    requests = _requests(
                        f"{variant['name']}-r{repetition}",
                        args.num_requests,
                        prompt,
                    )
                    started_at = time.perf_counter()
                    metrics_sampler = ServerMetricsSampler(metrics_url)
                    metrics_sampler.start()
                    try:
                        records = run_concurrent_requests(
                            client=client,
                            requests=requests,
                            concurrency=args.concurrency,
                            model=base_config["served_model_name"],
                            max_tokens=args.output_tokens,
                        )
                    finally:
                        metrics_during = metrics_sampler.stop()
                    wall_time_seconds = time.perf_counter() - started_at
                    metrics_after = snapshot_server_metrics(metrics_url)
                    npu_smi_after_path = (
                        args.output_dir
                        / "hardware"
                        / f"{base_config['served_model_name']}-{run_id}-{telemetry_label}-after.txt"
                    )
                    npu_smi_after = capture_npu_smi(npu_smi_after_path)
                    for record in records:
                        record.update(
                            {
                                "run_id": run_id,
                                "variant": variant["name"],
                                "variant_label": variant["label"],
                                "experiment_index": variant_index,
                                "repetition": repetition + 1,
                                "model_id": base_config["model_id"],
                                "model_revision": base_config["model_revision"],
                                "served_model_name": base_config["served_model_name"],
                                "tp_size": config["tp_size"],
                                "chunk_size": config["chunked_prefill_size"],
                                "block_size": config["block_size"],
                                "prefix_caching": config["prefix_caching"],
                                "disable_kvstore": config.get("disable_kvstore", False),
                                "mixed_batch": config["mixed_batch"],
                                "overlap_schedule": config["overlap_schedule"],
                                "concurrency": args.concurrency,
                                "target_input_tokens": args.input_tokens,
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
                            "variant": variant["name"],
                            "variant_label": variant["label"],
                            "experiment_index": variant_index,
                            "repetition": repetition + 1,
                            "model_id": base_config["model_id"],
                            "model_revision": base_config["model_revision"],
                            "served_model_name": base_config["served_model_name"],
                            "tp_size": config["tp_size"],
                            "chunk_size": config["chunked_prefill_size"],
                            "block_size": config["block_size"],
                            "prefix_caching": config["prefix_caching"],
                            "disable_kvstore": config.get("disable_kvstore", False),
                            "mixed_batch": config["mixed_batch"],
                            "overlap_schedule": config["overlap_schedule"],
                            "concurrency": args.concurrency,
                            "target_input_tokens": args.input_tokens,
                            "target_output_tokens": args.output_tokens,
                        }
                    )
                    add_metric_delta(
                        summary, metrics_before, metrics_after, metrics_during
                    )
                    summary["npu_smi_before_path"] = (
                        str(npu_smi_before_path) if npu_smi_before else None
                    )
                    summary["npu_smi_after_path"] = (
                        str(npu_smi_after_path) if npu_smi_after else None
                    )
                    append_jsonl(summary_path, summary)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    print(raw_path)
    print(summary_path)


def _serve_command(config: dict, *, host: str, port: int) -> list[str]:
    from benchmarks.course_project.core import build_serve_command

    return build_serve_command(config, host=host, port=port)


def _wait_for_server(
    process: subprocess.Popen,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/readiness"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"TokenSpeed exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"TokenSpeed did not become healthy at {url}")


def _requests(prefix: str, count: int, prompt: str) -> list[BenchmarkRequest]:
    return [
        BenchmarkRequest(
            request_id=f"{prefix}-{index}",
            messages=[{"role": "user", "content": prompt}],
            workload="ablation",
        )
        for index in range(count)
    ]


if __name__ == "__main__":
    main()
