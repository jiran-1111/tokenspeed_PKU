#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from benchmarks.course_project.telemetry import capture_npu_smi

CONFIG_PATH = REPO_ROOT / "configs" / "bringup" / "qwen35_4b.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "raw" / "session_kv_perf"
FRONTIER_POLICIES = ("current-frontier",)
SYSTEM_PROMPT = "You are a concise assistant. Keep each reply short."
NEXT_USER_TEXT = "Please continue with a short reply."


def parse_context_lengths(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            "context lengths must be positive comma-separated integers"
        ) from exc
    if (
        not values
        or any(length <= 0 for length in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("context lengths must be positive and unique")
    return values


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def execution_order(repetition_index: int) -> tuple[str, str]:
    if repetition_index < 0:
        raise ValueError("repetition index must be non-negative")
    if repetition_index % 2 == 0:
        return ("session", "stateless")
    return ("stateless", "session")


def expected_session_context_counts(
    *, policy: str, prior_history_tokens: int, suffix_tokens: int
) -> dict[str, int]:
    if policy != "current-frontier":
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


def build_next_turn_inputs(
    tokenizer: Any,
    history_ids: list[int],
    *,
    prior_user_text: str,
    previous_output_ids: list[int],
    next_user_text: str,
) -> tuple[list[int], list[int]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    output_ids_for_content = previous_output_ids
    if (
        previous_output_ids
        and eos_token_id is not None
        and previous_output_ids[-1] == int(eos_token_id)
    ):
        output_ids_for_content = previous_output_ids[:-1]
    assistant_content = "<think>\n" + tokenizer.decode(
        output_ids_for_content, skip_special_tokens=False
    )
    full_input_ids = _token_ids(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prior_user_text},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": next_user_text},
        ],
        add_generation_prompt=True,
    )
    if full_input_ids[: len(history_ids)] != history_ids:
        raise ValueError(
            "tokenizer chat template does not preserve the cached Qwen session history"
        )
    return full_input_ids[len(history_ids) :], full_input_ids


async def _generate(
    engine: Any,
    input_ids: list[int],
    sampling_params: dict[str, int | float],
    *,
    session_id: str | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    result = await asyncio.to_thread(
        engine.generate,
        input_ids=input_ids,
        sampling_params=sampling_params.copy(),
        session_id=session_id,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    output_ids = result.get("output_ids") if isinstance(result, Mapping) else None
    if not isinstance(output_ids, list) or not all(
        isinstance(token_id, int) for token_id in output_ids
    ):
        raise ValueError("Engine result has no raw output_ids list")
    meta_info = result.get("meta_info") or {}
    if not isinstance(meta_info, Mapping):
        raise ValueError("Engine result has invalid meta_info")
    return {
        "output_ids": [int(token_id) for token_id in output_ids],
        "meta_info": dict(meta_info),
        "request_latency_ms": elapsed_ms,
    }


def _request_record(
    *, path: str, input_ids: list[int], result: dict[str, Any]
) -> dict[str, Any]:
    output_ids = result["output_ids"]
    record = {
        "path": path,
        "input_ids": input_ids,
        "input_token_count": len(input_ids),
        "output_ids": output_ids,
        "output_token_count": len(output_ids),
        "request_latency_ms": result["request_latency_ms"],
    }
    if path == "session":
        metadata = result["meta_info"].get("session")
        record["session_metadata"] = (
            dict(metadata) if isinstance(metadata, Mapping) else None
        )
    return record


async def _run_sweep(
    engine: Any,
    tokenizer: Any,
    *,
    context_lengths: list[int],
    repetitions: int,
    frontier_policy: str,
    max_new_tokens: int,
    max_model_len: int,
    run_record: dict[str, Any],
    checkpoint: Callable[[], None] | None = None,
) -> None:
    if frontier_policy not in FRONTIER_POLICIES:
        raise ValueError(f"unsupported frontier policy: {frontier_policy}")
    if not context_lengths or any(length <= 0 for length in context_lengths):
        raise ValueError("context lengths must be positive")
    if repetitions <= 0 or max_new_tokens <= 0:
        raise ValueError("repetitions and max_new_tokens must be positive")
    if any(length + max_new_tokens > max_model_len for length in context_lengths):
        raise ValueError(
            "context length target and generation reserve exceed max_model_len"
        )

    sampling_params: dict[str, int | float] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
    }
    for context_length in context_lengths:
        for repetition_index in range(repetitions):
            session_id = engine.open_session()
            try:
                prior_user_text, initial_input_ids = build_context_prompt(
                    tokenizer, context_length
                )
                if len(initial_input_ids) + max_new_tokens > max_model_len:
                    raise ValueError(
                        "initial context and generation reserve exceed max_model_len"
                    )
                order = execution_order(repetition_index)
                prime_results: dict[str, dict[str, Any]] = {}
                for path in order:
                    prime_results[path] = await _generate(
                        engine,
                        initial_input_ids,
                        sampling_params,
                        session_id=session_id if path == "session" else None,
                    )

                prime_match = (
                    prime_results["session"]["output_ids"]
                    == prime_results["stateless"]["output_ids"]
                )
                prime_record = {
                    "measurement_included": False,
                    "session": _request_record(
                        path="session",
                        input_ids=initial_input_ids,
                        result=prime_results["session"],
                    ),
                    "stateless": _request_record(
                        path="stateless",
                        input_ids=initial_input_ids,
                        result=prime_results["stateless"],
                    ),
                    "output_ids_match": prime_match,
                }
                measurement_record: dict[str, Any] = {
                    "context_length_target": context_length,
                    "initial_context_tokens": len(initial_input_ids),
                    "initial_input_ids": initial_input_ids,
                    "repetition": repetition_index + 1,
                    "execution_order": list(order),
                    "prime": prime_record,
                    "measurement": None,
                }
                run_record["measurements"].append(measurement_record)
                if checkpoint is not None:
                    checkpoint()
                prime_metadata = prime_record["session"]["session_metadata"]
                if not isinstance(prime_metadata, Mapping):
                    raise ValueError(
                        "session generation result has no session metadata"
                    )
                validate_session_context_metadata(
                    prime_metadata,
                    policy=frontier_policy,
                    prior_history_tokens=0,
                    suffix_tokens=len(initial_input_ids),
                )
                if not prime_match:
                    raise ValueError("prime output token IDs differ between paths")

                history_ids = initial_input_ids + prime_results["session"]["output_ids"]
                suffix_ids, full_input_ids = build_next_turn_inputs(
                    tokenizer,
                    history_ids,
                    prior_user_text=prior_user_text,
                    previous_output_ids=prime_results["session"]["output_ids"],
                    next_user_text=NEXT_USER_TEXT,
                )
                if len(full_input_ids) + max_new_tokens > max_model_len:
                    raise ValueError(
                        "measured context and generation reserve exceed max_model_len"
                    )

                measured_results: dict[str, dict[str, Any]] = {}
                measured_inputs = {
                    "session": suffix_ids,
                    "stateless": full_input_ids,
                }
                for path in order:
                    measured_results[path] = await _generate(
                        engine,
                        measured_inputs[path],
                        sampling_params,
                        session_id=session_id if path == "session" else None,
                    )

                measured_session_record = _request_record(
                    path="session",
                    input_ids=suffix_ids,
                    result=measured_results["session"],
                )
                measured_stateless_record = _request_record(
                    path="stateless",
                    input_ids=full_input_ids,
                    result=measured_results["stateless"],
                )
                measurement_match = (
                    measured_results["session"]["output_ids"]
                    == measured_results["stateless"]["output_ids"]
                )
                measurement_record["measurement"] = {
                    "session": measured_session_record,
                    "stateless": measured_stateless_record,
                    "output_ids_match": measurement_match,
                }
                if checkpoint is not None:
                    checkpoint()
                measured_metadata = measured_session_record["session_metadata"]
                if not isinstance(measured_metadata, Mapping):
                    raise ValueError(
                        "session generation result has no session metadata"
                    )
                validate_session_context_metadata(
                    measured_metadata,
                    policy=frontier_policy,
                    prior_history_tokens=len(history_ids),
                    suffix_tokens=len(suffix_ids),
                )
                if not measurement_match:
                    raise ValueError("measured output token IDs differ between paths")
            finally:
                engine.close_session(session_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run serial, paired 4B Session KV and stateless measurements. "
            "The context-length targets refer to the first formatted user input."
        )
    )
    parser.add_argument(
        "--context-lengths",
        type=parse_context_lengths,
        required=True,
        help="comma-separated positive target lengths in formatted input tokens",
    )
    parser.add_argument("--repetitions", type=_positive_int, required=True)
    parser.add_argument("--frontier-policy", choices=FRONTIER_POLICIES, required=True)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _engine_cli_args(serve_command: list[str]) -> list[str]:
    cli_args = serve_command[4:]
    if "--prometheus-port" in cli_args:
        index = cli_args.index("--prometheus-port")
        del cli_args[index : index + 2]
    return cli_args


def _local_model_path(config: dict[str, Any]) -> Path:
    configured_path = config.get("local_model_path")
    if not configured_path:
        raise FileNotFoundError("pinned config has no local_model_path")
    path = Path(configured_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("local model path must be inside the project") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"pinned local model directory is absent: {path}")
    return path


def resolve_output_dir(path: Path) -> Path:
    project_root = REPO_ROOT.resolve()
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"output directory must be inside the project: {resolved}"
        ) from exc
    return resolved


def resolve_output_paths(path: Path) -> tuple[Path, Path]:
    output_dir = resolve_output_dir(path)
    hardware_dir = resolve_output_dir(output_dir / "hardware")
    return output_dir, hardware_dir


def _capture_hardware_snapshot(
    hardware_dir: Path, run_id: str, phase: str
) -> dict[str, bool | str | None]:
    snapshot_path = hardware_dir / f"{run_id}-{phase}.txt"
    available = capture_npu_smi(snapshot_path)
    return {
        "available": available,
        "path": str(snapshot_path) if available else None,
    }


def _write_run_record(output_path: Path, run_record: dict[str, Any]) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main() -> None:
    args = build_parser().parse_args()
    output_dir, hardware_dir = resolve_output_paths(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"qwen35-4b-session-kv-perf-{run_id}.json"
    run_record: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "frontier_policy": args.frontier_policy,
        "context_length_targets": args.context_lengths,
        "repetitions": args.repetitions,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
        },
        "measurement_scope": "serial Engine.generate request latency; not TTFT",
        "generic_prefix_caching": False,
        "kvstore_disabled": True,
        "hardware_before": None,
        "hardware_after": None,
        "measurements": [],
    }
    _write_run_record(output_path, run_record)
    try:
        run_record["hardware_before"] = _capture_hardware_snapshot(
            hardware_dir, run_id, "before"
        )
        _write_run_record(output_path, run_record)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["MODELSCOPE_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        config = load_model_config(CONFIG_PATH)
        local_model_path = _local_model_path(config)
        engine_config = {
            **config,
            "prefix_caching": False,
            "disable_kvstore": True,
        }
        serve_command = build_serve_command(engine_config, host="127.0.0.1", port=8000)
        cli_args = _engine_cli_args(serve_command)
        if Path(cli_args[0]).resolve() != local_model_path:
            raise ValueError("engine launch did not resolve to the pinned local model")
        run_record["model_id"] = config["model_id"]
        run_record["model_revision"] = config["model_revision"]
        run_record["local_model_path"] = str(local_model_path)
        run_record["launch_configuration"] = {
            "model_config": engine_config,
            "engine_cli_args": cli_args,
            "runtime_environment": {
                "ASCEND_RT_VISIBLE_DEVICES": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
            },
            "concurrent_requests": 1,
        }
        _write_run_record(output_path, run_record)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_path),
            revision=config["model_revision"],
            local_files_only=True,
        )
        from tokenspeed.runtime.entrypoints.engine import Engine
        from tokenspeed.runtime.utils.server_args import prepare_server_args

        server_args = prepare_server_args(cli_args)
        with Engine(server_args=server_args) as engine:
            asyncio.run(
                _run_sweep(
                    engine,
                    tokenizer,
                    context_lengths=args.context_lengths,
                    repetitions=args.repetitions,
                    frontier_policy=args.frontier_policy,
                    max_new_tokens=args.max_new_tokens,
                    max_model_len=int(config["max_model_len"]),
                    run_record=run_record,
                    checkpoint=lambda: _write_run_record(output_path, run_record),
                )
            )
        run_record["status"] = "completed"
    except Exception as exc:
        run_record["status"] = "failed"
        run_record["failure_type"] = type(exc).__name__
        raise
    finally:
        run_record["hardware_after"] = _capture_hardware_snapshot(
            hardware_dir, run_id, "after"
        )
        _write_run_record(output_path, run_record)

    print(output_path)


if __name__ == "__main__":
    main()
