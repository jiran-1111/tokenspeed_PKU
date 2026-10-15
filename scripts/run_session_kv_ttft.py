#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from benchmarks.course_project.telemetry import capture_npu_smi

CONFIG_PATH = REPO_ROOT / "configs" / "bringup" / "qwen35_4b.yaml"
DEFAULT_OUTPUT_DIR = Path("results/raw/session_kv_ttft")
DEFAULT_ANALYSIS_DIR = Path("results/analysis")
PINNED_MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


SYSTEM_PROMPT = "You are a concise assistant. Keep each reply short."
NEXT_USER_TEXT = "Please continue with a short reply."
INITIAL_CONTEXT_TARGET = 512
MAX_NEW_TOKENS = 8
MAX_MODEL_LEN = 4096
FRONTIER_POLICY = "current-frontier"
REPETITIONS = 4
TURNS = 16


def _token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_context_prompt(tokenizer: Any, context_length: int) -> tuple[str, list[int]]:
    if context_length <= 0:
        raise ValueError("context length must be positive")
    filler_count = context_length
    while True:
        user_text = "Benchmark context: " + ("x " * filler_count)
        input_ids = _token_ids(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            add_generation_prompt=True,
        )
        if len(input_ids) >= context_length:
            return user_text, input_ids
        filler_count *= 2


def assistant_content_from_output_ids(tokenizer: Any, output_ids: list[int]) -> str:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    content_ids = output_ids
    if output_ids and eos_token_id is not None and output_ids[-1] == int(eos_token_id):
        content_ids = output_ids[:-1]
    return "<think>\n" + tokenizer.decode(content_ids, skip_special_tokens=False)


def build_multiturn_inputs(
    tokenizer: Any,
    messages: list[dict[str, str]],
    history_ids: list[int],
) -> tuple[list[int], list[int]]:
    full_input_ids = _token_ids(tokenizer, messages, add_generation_prompt=True)
    if full_input_ids[: len(history_ids)] != history_ids:
        raise ValueError(
            "tokenizer chat template does not preserve the cached session history prefix"
        )
    return full_input_ids[len(history_ids) :], full_input_ids


def measure_stream_ttft(
    engine: Any,
    input_ids: list[int],
    sampling_params: dict[str, int | float],
    *,
    session_id: str | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    stream = engine.generate(
        input_ids=input_ids,
        sampling_params={**sampling_params, "stream_interval": 1},
        session_id=session_id,
        stream=True,
    )
    try:
        iterator = iter(stream)
    except TypeError as exc:
        raise ValueError("Engine.generate did not return a stream iterator") from exc

    output_ids: list[int] = []
    ttft_ms: float | None = None
    final_meta_info: dict[str, Any] | None = None
    for chunk in iterator:
        if not isinstance(chunk, Mapping):
            raise ValueError("stream chunk must be a mapping")
        chunk_ids = chunk.get("output_ids")
        if not isinstance(chunk_ids, list) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in chunk_ids
        ):
            raise ValueError("stream chunk has invalid output_ids")
        chunk_meta_info = chunk.get("meta_info")
        if not isinstance(chunk_meta_info, Mapping):
            raise ValueError("stream chunk has invalid meta_info")
        if chunk_ids and ttft_ms is None:
            ttft_ms = (time.perf_counter() - started_at) * 1000
        output_ids.extend(chunk_ids)
        final_meta_info = dict(chunk_meta_info)

    if ttft_ms is None or not output_ids:
        raise ValueError("stream ended without a non-empty output_ids chunk")
    if not math.isfinite(ttft_ms) or ttft_ms <= 0:
        raise ValueError("stream TTFT must be finite and positive")
    if final_meta_info is None:
        raise ValueError("stream ended without final meta_info")
    if session_id is not None and not isinstance(
        final_meta_info.get("session"), Mapping
    ):
        raise ValueError("session stream ended without final Session metadata")
    return {
        "output_ids": output_ids,
        "ttft_ms": ttft_ms,
        "meta_info": final_meta_info,
    }


def execution_order(repetition_index: int) -> tuple[str, str]:
    if repetition_index < 0:
        raise ValueError("repetition index must be non-negative")
    if repetition_index % 2 == 0:
        return ("session", "stateless")
    return ("stateless", "session")


def expected_session_context_counts(
    *, policy: str, prior_history_tokens: int, suffix_tokens: int
) -> dict[str, int]:
    if policy != FRONTIER_POLICY:
        raise ValueError(f"unsupported frontier policy: {policy}")
    if prior_history_tokens < 0 or suffix_tokens < 0:
        raise ValueError("token counts must be non-negative")
    reused = max(prior_history_tokens - 1, 0)
    total = prior_history_tokens + suffix_tokens
    return {
        "total_context_tokens": total,
        "newly_prefilled_tokens": total - reused,
        "reused_context_tokens": reused,
    }


def validate_session_context_metadata(
    metadata: Mapping[str, object],
    *,
    policy: str,
    prior_history_tokens: int,
    suffix_tokens: int,
) -> None:
    expected = expected_session_context_counts(
        policy=policy,
        prior_history_tokens=prior_history_tokens,
        suffix_tokens=suffix_tokens,
    )
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"session context metadata mismatch: expected={expected}, actual={actual}"
        )


def _request_record(
    *, path: str, input_ids: list[int], result: dict[str, Any]
) -> dict[str, Any]:
    record = {
        "path": path,
        "input_ids": list(input_ids),
        "input_token_count": len(input_ids),
        "output_ids": list(result["output_ids"]),
        "output_token_count": len(result["output_ids"]),
        "ttft_ms": result["ttft_ms"],
    }
    if path == "session":
        session_metadata = result["meta_info"].get("session")
        record["session_metadata"] = (
            dict(session_metadata) if isinstance(session_metadata, Mapping) else None
        )
    return record


def _write_run_record(output_path: Path, run_record: dict[str, Any]) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned 4B multi-turn Session KV TTFT experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="project-local directory for raw run records",
    )
    return parser


def resolve_output_dir(path: Path) -> Path:
    project_root = REPO_ROOT.resolve()
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("output directory must be inside the project") from exc
    return resolved


def _local_model_path(config: dict[str, Any]) -> Path:
    configured_path = config.get("local_model_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("pinned config has no local_model_path")
    path = Path(configured_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("local model path must be inside the project") from exc
    if not path.is_dir():
        raise FileNotFoundError("pinned local model directory is absent")
    return path


def _validate_pinned_config(config: dict[str, Any]) -> Path:
    expected = {
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "tp_size": 1,
        "max_model_len": MAX_MODEL_LEN,
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value or type(config.get(field)) is not type(
            expected_value
        ):
            raise ValueError(f"config does not match pinned {field}")
    if config.get("overlap_schedule") is not True:
        raise ValueError("pinned config must retain overlap scheduling")
    return _local_model_path(config)


def _engine_cli_args(serve_command: list[str]) -> list[str]:
    cli_args = serve_command[4:]
    if "--prometheus-port" in cli_args:
        index = cli_args.index("--prometheus-port")
        del cli_args[index : index + 2]
    return cli_args


def _build_engine_config_and_cli_args(
    config: dict[str, Any], local_model_path: Path
) -> tuple[dict[str, Any], list[str]]:
    if config.get("overlap_schedule") is not True:
        raise ValueError("pinned config must retain overlap scheduling")
    local_model_path = local_model_path.resolve()
    try:
        local_model_path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("local model path must be inside the project") from exc
    if not local_model_path.is_dir():
        raise FileNotFoundError("pinned local model directory is absent")
    engine_config = {
        **config,
        "local_model_path": str(local_model_path),
        "prefix_caching": False,
        "disable_kvstore": True,
        "overlap_schedule": True,
    }
    cli_args = _engine_cli_args(
        build_serve_command(engine_config, host="127.0.0.1", port=8000)
    )
    if (
        not cli_args
        or Path(cli_args[0]).resolve() != local_model_path
        or "--disable-prefix-caching" not in cli_args
        or "--enable-prefix-caching" in cli_args
        or "--disable-kvstore" not in cli_args
        or "--disable-overlap-schedule" in cli_args
    ):
        raise ValueError("engine launch args do not match the pinned experiment")
    return engine_config, cli_args


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex}"


def create_run_paths(output_dir: Path) -> tuple[str, Path, Path]:
    output_dir = resolve_output_dir(output_dir)
    analysis_root = resolve_output_dir(DEFAULT_ANALYSIS_DIR)
    while True:
        run_id = _new_run_id()
        raw_path = output_dir / f"qwen35-4b-session-kv-ttft-{run_id}.json"
        analysis_dir = analysis_root / f"session_kv_ttft_{run_id}"
        if not raw_path.exists() and not analysis_dir.exists():
            return run_id, raw_path, analysis_dir


def _write_initial_run_record(output_path: Path, run_record: dict[str, Any]) -> None:
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(
            json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )


def _capture_hardware_snapshot(
    output_dir: Path, run_id: str, phase: str
) -> dict[str, bool | str | None]:
    snapshot_path = output_dir / "hardware" / f"{run_id}-{phase}.txt"
    available = capture_npu_smi(snapshot_path)
    return {
        "available": available,
        "path": str(snapshot_path) if available else None,
    }


def _set_offline_environment() -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _generate_report_if_completed(
    run_record: dict[str, Any], raw_path: Path, analysis_dir: Path
) -> tuple[Path, Path] | None:
    if run_record.get("status") != "completed":
        return None
    from benchmarks.course_project.session_kv_ttft_report import generate_report

    return generate_report(raw_path, analysis_dir)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        run_id, output_path, analysis_dir = create_run_paths(output_dir)
        run_record: dict[str, Any] = {
            "run_id": run_id,
            "status": "initializing",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": None,
            "experiment": "session-kv-multiturn-ttft",
            "frontier_policy": FRONTIER_POLICY,
            "initial_context_target": INITIAL_CONTEXT_TARGET,
            "repetitions": REPETITIONS,
            "turns_per_repetition": TURNS,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
            },
            "stream_interval": 1,
            "concurrency": 1,
            "max_model_len": MAX_MODEL_LEN,
            "hardware_before": None,
            "hardware_after": None,
            "measurements": [],
        }
        try:
            _write_initial_run_record(output_path, run_record)
        except FileExistsError:
            continue
        break

    try:
        run_record["hardware_before"] = _capture_hardware_snapshot(
            output_dir, run_id, "before"
        )
        _set_offline_environment()
        run_record["status"] = "running"
        _write_run_record(output_path, run_record)

        config = load_model_config(CONFIG_PATH)
        local_model_path = _validate_pinned_config(config)
        engine_config, cli_args = _build_engine_config_and_cli_args(
            config, local_model_path
        )
        run_record["configuration"] = {
            "model_config_path": str(CONFIG_PATH),
            "model_config": engine_config,
            "engine_cli_args": cli_args,
            "local_model_path": str(local_model_path),
            "runtime_environment": {
                "ASCEND_RT_VISIBLE_DEVICES": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        }
        _write_run_record(output_path, run_record)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_path),
            revision=PINNED_MODEL_REVISION,
            local_files_only=True,
        )
        from tokenspeed.runtime.entrypoints.engine import Engine
        from tokenspeed.runtime.utils.server_args import prepare_server_args

        server_args = prepare_server_args(cli_args)
        with Engine(server_args=server_args) as engine:
            _run_sweep(
                engine,
                tokenizer,
                repetitions=REPETITIONS,
                initial_context_target=INITIAL_CONTEXT_TARGET,
                max_new_tokens=MAX_NEW_TOKENS,
                max_model_len=MAX_MODEL_LEN,
                run_record=run_record,
                checkpoint=lambda: _write_run_record(output_path, run_record),
            )
        run_record["status"] = "completed"
    except Exception as error:
        run_record["status"] = "failed"
        run_record["failure_type"] = type(error).__name__
        raise
    finally:
        try:
            run_record["hardware_after"] = _capture_hardware_snapshot(
                output_dir, run_id, "after"
            )
        finally:
            run_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_run_record(output_path, run_record)

    report_paths = _generate_report_if_completed(run_record, output_path, analysis_dir)
    if report_paths is not None:
        run_record["analysis"] = {
            "csv": str(report_paths[0]),
            "svg": str(report_paths[1]),
        }
        _write_run_record(output_path, run_record)
    print(output_path)


def _mark_failure(
    run_record: dict[str, Any],
    error: Exception,
    *,
    phase: str,
    repetition: int | None,
    turn: int | None,
    path: str | None,
    completed_requests: dict[str, dict[str, Any]] | None,
    checkpoint: Callable[[], None] | None,
) -> None:
    run_record["status"] = "failed"
    run_record["failure"] = {
        "type": type(error).__name__,
        "message": str(error),
        "phase": phase,
        "repetition": repetition,
        "turn": turn,
        "path": path,
    }
    if completed_requests:
        run_record["failed_pair"] = {
            "repetition": repetition,
            "turn": turn,
            "completed_requests": completed_requests,
        }
    if checkpoint is not None:
        checkpoint()


def _run_sweep(
    engine: Any,
    tokenizer: Any,
    *,
    repetitions: int,
    initial_context_target: int,
    max_new_tokens: int,
    max_model_len: int,
    run_record: dict[str, Any],
    checkpoint: Callable[[], None] | None = None,
) -> None:
    if repetitions != REPETITIONS:
        raise ValueError(f"this experiment requires exactly {REPETITIONS} repetitions")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if max_model_len != MAX_MODEL_LEN:
        raise ValueError(f"this experiment requires max_model_len={MAX_MODEL_LEN}")
    run_record.setdefault("measurements", [])
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
    initial_user_text, initial_input_ids = build_context_prompt(
        tokenizer, initial_context_target
    )
    if len(initial_input_ids) + max_new_tokens > max_model_len:
        raise ValueError("initial context and generation reserve exceed max_model_len")

    warm_session_id = engine.open_session()
    try:
        warm_results: dict[str, dict[str, Any]] = {}
        try:
            warm_results["session"] = measure_stream_ttft(
                engine,
                initial_input_ids,
                sampling_params,
                session_id=warm_session_id,
            )
            warm_metadata = warm_results["session"]["meta_info"].get("session")
            if not isinstance(warm_metadata, Mapping):
                raise ValueError("warmup Session stream has no Session metadata")
            validate_session_context_metadata(
                warm_metadata,
                policy=FRONTIER_POLICY,
                prior_history_tokens=0,
                suffix_tokens=len(initial_input_ids),
            )
            warm_results["stateless"] = measure_stream_ttft(
                engine,
                initial_input_ids,
                sampling_params,
                session_id=None,
            )
            if (
                warm_results["session"]["output_ids"]
                != warm_results["stateless"]["output_ids"]
            ):
                raise ValueError("warmup output token IDs differ between paths")
            run_record["warmup"] = {
                "measurement_included": False,
                "requests": {
                    path: _request_record(
                        path=path,
                        input_ids=initial_input_ids,
                        result=warm_results[path],
                    )
                    for path in ("session", "stateless")
                },
                "output_ids_match": True,
            }
        except Exception as error:
            _mark_failure(
                run_record,
                error,
                phase="warmup",
                repetition=None,
                turn=None,
                path=None,
                completed_requests={
                    path: _request_record(
                        path=path,
                        input_ids=initial_input_ids,
                        result=result,
                    )
                    for path, result in warm_results.items()
                },
                checkpoint=checkpoint,
            )
            raise
    finally:
        engine.close_session(warm_session_id)

    for repetition_index in range(repetitions):
        repetition = repetition_index + 1
        order = execution_order(repetition_index)
        session_id = engine.open_session()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_text},
        ]
        history_ids = list(initial_input_ids)
        previous_output_ids: list[int] | None = None
        try:
            for turn in range(1, TURNS + 1):
                if turn == 1:
                    full_input_ids = list(initial_input_ids)
                    session_input_ids = list(initial_input_ids)
                    prior_history_tokens = 0
                else:
                    assert previous_output_ids is not None
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content_from_output_ids(
                                    tokenizer, previous_output_ids
                                ),
                            },
                            {"role": "user", "content": NEXT_USER_TEXT},
                        ]
                    )
                    try:
                        session_input_ids, full_input_ids = build_multiturn_inputs(
                            tokenizer, messages, history_ids
                        )
                    except Exception as error:
                        _mark_failure(
                            run_record,
                            error,
                            phase="measurement",
                            repetition=repetition,
                            turn=turn,
                            path=None,
                            completed_requests=None,
                            checkpoint=checkpoint,
                        )
                        raise
                    prior_history_tokens = len(history_ids)
                if len(full_input_ids) + max_new_tokens > max_model_len:
                    error = ValueError(
                        "measured context and generation reserve exceed max_model_len"
                    )
                    _mark_failure(
                        run_record,
                        error,
                        phase="measurement",
                        repetition=repetition,
                        turn=turn,
                        path=None,
                        completed_requests=None,
                        checkpoint=checkpoint,
                    )
                    raise error

                input_by_path = {
                    "session": session_input_ids,
                    "stateless": full_input_ids,
                }
                results: dict[str, dict[str, Any]] = {}
                requests: dict[str, dict[str, Any]] = {}
                for path in order:
                    try:
                        results[path] = measure_stream_ttft(
                            engine,
                            input_by_path[path],
                            sampling_params,
                            session_id=session_id if path == "session" else None,
                        )
                        requests[path] = _request_record(
                            path=path,
                            input_ids=input_by_path[path],
                            result=results[path],
                        )
                        if path == "session":
                            metadata = results[path]["meta_info"].get("session")
                            if not isinstance(metadata, Mapping):
                                raise ValueError(
                                    "Session stream has no Session metadata"
                                )
                            validate_session_context_metadata(
                                metadata,
                                policy=FRONTIER_POLICY,
                                prior_history_tokens=prior_history_tokens,
                                suffix_tokens=len(session_input_ids),
                            )
                    except Exception as error:
                        _mark_failure(
                            run_record,
                            error,
                            phase="measurement",
                            repetition=repetition,
                            turn=turn,
                            path=path,
                            completed_requests=requests,
                            checkpoint=checkpoint,
                        )
                        raise

                output_ids_match = (
                    results["session"]["output_ids"]
                    == results["stateless"]["output_ids"]
                )
                if not output_ids_match:
                    error = ValueError("measured output token IDs differ between paths")
                    requests["session"]["output_ids_match"] = False
                    requests["stateless"]["output_ids_match"] = False
                    _mark_failure(
                        run_record,
                        error,
                        phase="measurement",
                        repetition=repetition,
                        turn=turn,
                        path=None,
                        completed_requests=requests,
                        checkpoint=checkpoint,
                    )
                    raise error

                session_ttft = results["session"]["ttft_ms"]
                stateless_ttft = results["stateless"]["ttft_ms"]
                pair = {
                    "repetition": repetition,
                    "turn": turn,
                    "execution_order": list(order),
                    "full_context_tokens": len(full_input_ids),
                    "session_suffix_tokens": len(session_input_ids),
                    "session": requests["session"],
                    "stateless": requests["stateless"],
                    "output_ids_match": True,
                    "ttft_reduction_pct": 100.0 * (1.0 - session_ttft / stateless_ttft),
                }
                run_record["measurements"].append(pair)
                if checkpoint is not None:
                    checkpoint()
                previous_output_ids = results["session"]["output_ids"]
                history_ids = full_input_ids + previous_output_ids
        finally:
            engine.close_session(session_id)

    run_record["status"] = "completed"


if __name__ == "__main__":
    main()
