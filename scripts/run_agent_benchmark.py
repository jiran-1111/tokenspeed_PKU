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
import hashlib
import json
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
from benchmarks.course_project.profiler import (
    ProfilerControlClient,
    profiler_control_url,
)
from benchmarks.course_project.runner import (
    BenchmarkRequest,
    build_agent_session_contexts,
    build_agent_turn_request,
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
        description="Run multi-session Agent workloads against TokenSpeed."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument(
        "--shared-prefix-ratios",
        nargs="+",
        type=float,
        default=[0, 0.25, 0.5, 0.75, 0.9],
    )
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
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "raw" / "agent"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-single-request-ttft", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=REPO_ROOT / "results" / "profiles",
    )
    return parser.parse_args()


def validate_single_request_ttft_args(args: argparse.Namespace) -> None:
    if args.profile:
        raise ValueError(
            "--profile-single-request-ttft cannot be combined with --profile"
        )
    expected = {
        "concurrency": [1],
        "shared_prefix_ratios": [0.5],
        "prefix_tokens": 512,
        "turns": 8,
        "tool_wait_ms": 100,
        "tool_wait_policy": "one-session",
        "output_tokens": 32,
        "runs": 1,
    }
    actual_args = vars(args)
    for field, value in expected.items():
        if actual_args[field] != value:
            raise ValueError(
                f"--profile-single-request-ttft requires {field}={value!r}"
            )


def validate_single_request_ttft_config(config: dict) -> None:
    expected = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "served_model_name": "qwen35-0.8b",
        "device": "npu",
        "dtype": "bfloat16",
        "tp_size": 1,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"--profile-single-request-ttft requires config {field}={value!r}"
            )


def single_request_ttft_paths(run_id: str) -> tuple[Path, Path, Path]:
    paths = (
        REPO_ROOT / "results" / "raw" / "profiler" / "single_request_ttft" / run_id,
        REPO_ROOT
        / "results"
        / "profiles"
        / "qwen35-0.8b"
        / "single_request_ttft"
        / run_id,
        REPO_ROOT / "results" / "analysis" / f"profiler_single_request_ttft_{run_id}",
    )
    resolved = tuple(path.resolve() for path in paths)
    for path in resolved:
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(
                "single-request TTFT outputs must stay inside the project"
            ) from error
    return resolved


def ensure_single_request_ttft_paths_available(paths: tuple[Path, Path, Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Single-request TTFT output already exists: {path}")


def run_single_request_ttft_profile(
    *,
    args: argparse.Namespace,
    config: dict,
    tokenizer,
    client: OpenAIChatClient,
    run_id: str,
    paths: tuple[Path, Path, Path],
) -> None:
    raw_run_dir, profile_run_dir, analysis_dir = paths
    ensure_single_request_ttft_paths_available(paths)
    profile_id = f"{run_id}-single-request-ttft"
    config_path = args.config.resolve()
    try:
        config_path_label = config_path.relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            "single-request TTFT config must be inside the project"
        ) from error
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    raw_run_dir.mkdir(parents=True, exist_ok=False)
    profile_run_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "run_id": run_id,
        "config_path": config_path_label,
        "config_sha256": config_sha256,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "served_model_name": config["served_model_name"],
        "device": config["device"],
        "dtype": config["dtype"],
        "tp_size": config["tp_size"],
        "workload": {
            "concurrency": 1,
            "prefix_tokens": 512,
            "shared_prefix_ratio": 0.5,
            "turns": 8,
            "tool_wait_ms": 100,
            "tool_wait_policy": "one-session",
            "warmup_output_tokens": 32,
            "profiled_output_tokens": 1,
        },
        "profiler_level": "Level1",
        "profiler_metrics": ["ACL_AICORE_PIPE_UTILIZATION"],
    }
    (raw_run_dir / "config_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    request_record = {
        "run_id": run_id,
        "request_id": "single-request-ttft",
        "status": "failed",
        "workload": "agent",
        "session_id": "session-0",
        "turn_index": 8,
        "concurrency": 1,
        "shared_prefix_ratio": 0.5,
        "target_prefix_tokens": 512,
        "target_output_tokens": 32,
        "max_tokens": 1,
        "tool_wait_ms": 100,
        "tool_wait_policy": "one-session",
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "served_model_name": config["served_model_name"],
        "tp_size": config["tp_size"],
        "config_sha256": config_sha256,
        "profile_id": profile_id,
        "profile_dir": str(profile_run_dir),
    }
    timestamps = {
        "request_send_epoch_ns": None,
        "request_send_monotonic_ns": None,
        "first_chunk_epoch_ns": None,
        "first_chunk_monotonic_ns": None,
        "stream_drain_epoch_ns": None,
        "stream_drain_monotonic_ns": None,
        "first_chunk_field": None,
        "profiler_start_request_epoch_ns": None,
        "profiler_start_response_epoch_ns": None,
        "profiler_stop_request_epoch_ns": None,
        "profiler_stop_response_epoch_ns": None,
        "profiler_stop_cleanup_request_epoch_ns": None,
        "profiler_stop_cleanup_response_epoch_ns": None,
    }
    profiler = ProfilerControlClient(profiler_control_url(args.base_url))
    profile_active = False
    profile_start_acknowledged = False
    profile_stop_acknowledged = False
    stop_callback_acknowledged = False
    stop_callback_attempted = False
    cleanup_error_type = None
    work_error_type = None
    npu_before_captured = False
    npu_after_captured = False

    def before_request(request: BenchmarkRequest) -> None:
        nonlocal profile_active, profile_start_acknowledged, npu_before_captured
        if request.turn_index != 8:
            raise ValueError("single-request TTFT profiler may start only on turn 8")
        npu_before_captured = capture_npu_smi(raw_run_dir / "npu0-before.txt")
        if not npu_before_captured:
            raise RuntimeError("npu0_before_snapshot_unavailable")
        timestamps["profiler_start_request_epoch_ns"] = time.time_ns()
        profile_active = True
        profiler.start(output_dir=profile_run_dir, profile_id=profile_id)
        timestamps["profiler_start_response_epoch_ns"] = time.time_ns()
        profile_start_acknowledged = True

    def on_first_chunk(
        request: BenchmarkRequest,
        field: str,
        epoch_ns: int,
        monotonic_ns: int,
    ) -> None:
        nonlocal profile_active, profile_stop_acknowledged
        nonlocal stop_callback_acknowledged, stop_callback_attempted
        if request.turn_index != 8:
            raise ValueError("single-request TTFT profiler may stop only on turn 8")
        if stop_callback_attempted:
            raise RuntimeError("single-request TTFT stop callback ran more than once")
        stop_callback_attempted = True
        timestamps["profiler_stop_request_epoch_ns"] = time.time_ns()
        timestamps["first_chunk_field"] = field
        timestamps["first_chunk_epoch_ns"] = epoch_ns
        timestamps["first_chunk_monotonic_ns"] = monotonic_ns
        profiler.stop()
        timestamps["profiler_stop_response_epoch_ns"] = time.time_ns()
        stop_callback_acknowledged = True
        profile_stop_acknowledged = True
        profile_active = False

    tokenizer_encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    context = build_agent_session_contexts(
        concurrency=1,
        total_tokens=512,
        shared_tokens=256,
        repetition=0,
        encode=tokenizer_encode,
        decode=tokenizer.decode,
    )
    histories = {
        session_id: [{"role": "system", "content": system_context}]
        for session_id, system_context in context.items()
    }
    captured_record = None
    try:
        warmups = [
            BenchmarkRequest(
                request_id=f"single-ttft-warmup-{index}",
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

        for turn_index in range(1, 9):
            requests = [
                build_agent_turn_request(
                    request_id=f"single-ttft-session-0-turn-{turn_index}",
                    session_id="session-0",
                    turn_index=turn_index,
                    history=histories["session-0"],
                    user_message=(
                        f"Turn {turn_index}: analyze the current task and propose "
                        "the next concrete action."
                    ),
                    tool_wait_policy="one-session",
                    tool_wait_ms=100,
                )
            ]
            turn_records = run_concurrent_requests(
                client=client,
                requests=requests,
                concurrency=1,
                model=config["served_model_name"],
                max_tokens=1 if turn_index == 8 else 32,
                before_request=before_request if turn_index == 8 else None,
                on_first_chunk=on_first_chunk if turn_index == 8 else None,
            )
            record = turn_records[0]
            if turn_index == 8:
                captured_record = record
            histories["session-0"].append(requests[0].messages[-1])
            histories["session-0"].append(
                {"role": "assistant", "content": record["output_text"]}
            )
    except Exception as error:
        work_error_type = type(error).__name__
    finally:
        if profile_active and not profile_stop_acknowledged:
            cleanup_request_key = (
                "profiler_stop_cleanup_request_epoch_ns"
                if timestamps["profiler_stop_request_epoch_ns"] is not None
                else "profiler_stop_request_epoch_ns"
            )
            cleanup_response_key = (
                "profiler_stop_cleanup_response_epoch_ns"
                if cleanup_request_key == "profiler_stop_cleanup_request_epoch_ns"
                else "profiler_stop_response_epoch_ns"
            )
            timestamps[cleanup_request_key] = time.time_ns()
            try:
                profiler.stop()
            except Exception as error:
                cleanup_error_type = type(error).__name__
            else:
                timestamps[cleanup_response_key] = time.time_ns()
                profile_stop_acknowledged = True
                profile_active = False
        npu_after_captured = capture_npu_smi(raw_run_dir / "npu0-after.txt")

    if captured_record is not None:
        for field in (
            "request_id",
            "workload",
            "session_id",
            "turn_index",
            "tool_wait_ms",
            "mock_tool_result_injected",
            "concurrency",
            "input_tokens",
            "output_tokens",
            "ttft_ms",
            "tpot_ms",
            "itl_p95_ms",
            "itl_observation",
            "chunk_latency_p95_ms",
            "inter_chunk_latencies_ms",
            "latency_observation",
            "e2e_ms",
            "request_send_epoch_ns",
            "request_send_monotonic_ns",
            "first_chunk_epoch_ns",
            "first_chunk_monotonic_ns",
            "stream_drain_epoch_ns",
            "stream_drain_monotonic_ns",
            "first_chunk_field",
            "stream_completed",
        ):
            if field in captured_record:
                request_record[field] = captured_record[field]
    request_record.update(
        {
            key: value
            for key, value in timestamps.items()
            if value is not None or key not in request_record
        }
    )
    request_record.update(
        {
            "profiler_start_acknowledged": profile_start_acknowledged,
            "profiler_stop_acknowledged": profile_stop_acknowledged,
            "stop_callback_attempted": stop_callback_attempted,
            "stop_callback_acknowledged": stop_callback_acknowledged,
            "npu0_before_snapshot_captured": npu_before_captured,
            "npu0_after_snapshot_captured": npu_after_captured,
        }
    )
    if cleanup_error_type is not None:
        request_record["cleanup_failure_type"] = cleanup_error_type

    invalid_reason = None
    if work_error_type is not None:
        request_record["status"] = "failed"
        request_record["failure_type"] = work_error_type
    elif captured_record is None:
        request_record["status"] = "failed"
        request_record["failure_type"] = "RequestNotCompleted"
    else:
        invalid_checks = (
            (captured_record["first_chunk_epoch_ns"] is None, "missing_first_chunk"),
            (captured_record["output_tokens"] != 1, "output_token_count_not_one"),
            (not captured_record["stream_completed"], "stream_missing_done_marker"),
            (not profile_start_acknowledged, "profiler_start_not_acknowledged"),
            (not profile_stop_acknowledged, "profiler_stop_not_acknowledged"),
            (not npu_before_captured, "npu0_before_snapshot_unavailable"),
            (not npu_after_captured, "npu0_after_snapshot_unavailable"),
        )
        invalid_reason = next(
            (reason for failed, reason in invalid_checks if failed), None
        )
        if invalid_reason is None:
            request_record["status"] = "success"
        else:
            request_record["status"] = "invalid"
            request_record["invalid_reason"] = invalid_reason

    append_jsonl(raw_run_dir / "request.jsonl", request_record)
    if request_record["status"] == "success":
        summary = {
            "run_id": run_id,
            "status": "success",
            "request_id": request_record["request_id"],
            "profile_id": profile_id,
            "profile_dir": str(profile_run_dir),
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "served_model_name": config["served_model_name"],
            "tp_size": config["tp_size"],
            "turn_index": 8,
            "concurrency": 1,
            "shared_prefix_ratio": 0.5,
            "target_prefix_tokens": 512,
            "input_tokens": request_record["input_tokens"],
            "output_tokens": request_record["output_tokens"],
            "ttft_ms": request_record["ttft_ms"],
            "e2e_ms": request_record["e2e_ms"],
            "first_chunk_field": request_record["first_chunk_field"],
            "stream_completed": request_record["stream_completed"],
        }
        append_jsonl(raw_run_dir / "summary.jsonl", summary)

    print(raw_run_dir)
    print(profile_run_dir)
    print(analysis_dir)
    if work_error_type is not None:
        raise RuntimeError(
            f"single-request TTFT profile failed ({work_error_type})"
        ) from None
    if request_record["status"] != "success":
        reason = request_record.get("invalid_reason") or request_record.get(
            "failure_type", "unknown"
        )
        raise RuntimeError(f"single-request TTFT profile invalid ({reason})")


def main() -> None:
    args = parse_args()
    single_request_ttft_enabled = args.profile_single_request_ttft
    if single_request_ttft_enabled:
        validate_single_request_ttft_args(args)
    config = load_model_config(args.config)
    run_timestamp = datetime.now(timezone.utc)
    single_request_paths = None
    model_path = model_load_path(config)
    if single_request_ttft_enabled:
        validate_single_request_ttft_config(config)
        config_path = args.config.resolve()
        try:
            config_path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(
                "single-request TTFT config must be inside the project"
            ) from error
        model_checkpoint = Path(model_path).resolve()
        try:
            model_checkpoint.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(
                "single-request TTFT model checkpoint must be inside the project"
            ) from error
        if not model_checkpoint.is_dir():
            raise FileNotFoundError(
                "single-request TTFT requires the model checkpoint inside the project"
            )
        run_id = run_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        single_request_paths = single_request_ttft_paths(run_id)
        ensure_single_request_ttft_paths_available(single_request_paths)
        model_path = str(model_checkpoint)
    else:
        run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    raw_path = args.output_dir / f"{config['served_model_name']}-{run_id}.jsonl"
    summary_path = (
        args.output_dir / f"{config['served_model_name']}-{run_id}.summary.jsonl"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=config["model_revision"]
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    client = OpenAIChatClient(args.base_url)
    if single_request_ttft_enabled:
        run_single_request_ttft_profile(
            args=args,
            config=config,
            tokenizer=tokenizer,
            client=client,
            run_id=run_id,
            paths=single_request_paths,
        )
        return
    profile_enabled = getattr(args, "profile", False)
    profiler = (
        ProfilerControlClient(profiler_control_url(args.base_url))
        if profile_enabled
        else None
    )
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
                telemetry_label = f"c{concurrency}-p{shared_ratio:g}-r{repetition + 1}"
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

                profile_metadata = {}
                if profile_enabled:
                    profile_id = (
                        f"{run_id}-c{concurrency}-p{shared_ratio}"
                        f"-t{args.prefix_tokens}-n{shared_tokens}-r{repetition + 1}"
                    )
                    profile_dir = (
                        args.profile_dir / config["served_model_name"] / profile_id
                    ).resolve()
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    profile_metadata = {
                        "profile_id": profile_id,
                        "profile_dir": str(profile_dir),
                    }

                all_records = []
                wall_time_seconds = 0.0
                metrics_during = {}
                try:
                    if profile_enabled:
                        profiler.start(
                            output_dir=profile_dir,
                            profile_id=profile_id,
                        )
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
                            metrics_during[key] = max(
                                metrics_during.get(key, value), value
                            )
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
                                    **profile_metadata,
                                }
                            )
                            session_history = histories[record["session_id"]]
                            session_history.append(
                                requests_by_session[record["session_id"]].messages[-1]
                            )
                            session_history.append(
                                {"role": "assistant", "content": record["output_text"]}
                            )
                            append_jsonl(raw_path, record)
                        all_records.extend(turn_records)
                finally:
                    if profile_enabled:
                        active_exception = sys.exc_info()[0] is not None
                        try:
                            profiler.stop()
                        except BaseException:
                            if not active_exception:
                                raise

                summary = summarize_records(
                    all_records, wall_time_seconds=wall_time_seconds
                )
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
                        **profile_metadata,
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
