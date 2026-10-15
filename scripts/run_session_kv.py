#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from benchmarks.course_project.telemetry import capture_npu_smi

CONFIG_PATH = REPO_ROOT / "configs" / "bringup" / "qwen35_4b.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "raw" / "session_kv"
FRONTIER_POLICIES = ("current-frontier",)
SESSION_TURNS = (
    (
        "What is 17 plus 25? Answer briefly.",
        "Now answer with just the number.",
    ),
    (
        "Name one color commonly associated with the ocean.",
        "Reply with that color in one word.",
    ),
)
SYSTEM_PROMPT = "You are a concise assistant. Keep each reply under 12 words."


def expected_session_context_counts(
    *, policy: str, prior_history_tokens: int, suffix_tokens: int
) -> dict[str, int]:
    if policy == "materialize-final-token":
        raise ValueError("frontier policy materialize-final-token is not implemented")
    if policy not in FRONTIER_POLICIES:
        raise ValueError(f"unknown frontier policy: {policy}")
    if prior_history_tokens < 0 or suffix_tokens < 0:
        raise ValueError("token counts must be non-negative")

    total = prior_history_tokens + suffix_tokens
    if prior_history_tokens == 0:
        reused = 0
    else:
        reused = prior_history_tokens - 1

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
            "session context metadata mismatch: "
            f"policy={policy}, expected={expected}, actual={actual}"
        )


def is_expected_closed_session_rejection(result: Any, session_id: str) -> bool:
    if not isinstance(result, Mapping):
        return False
    meta_info = result.get("meta_info")
    if not isinstance(meta_info, Mapping):
        return False
    finish_reason = meta_info.get("finish_reason")
    if not isinstance(finish_reason, Mapping) or finish_reason.get("type") != "abort":
        return False
    message = finish_reason.get("message")
    expected_message = f"unknown or closed session id {session_id!r}"
    return isinstance(message, str) and expected_message in message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native Session KV turns with stateless greedy controls. "
            "Only current-frontier is implemented; final-token materialization "
            "is not implemented."
        )
    )
    parser.add_argument(
        "--frontier-policy",
        choices=FRONTIER_POLICIES,
        required=True,
        help=(
            "current-frontier leaves the prior final sampled token absent from KV "
            "(the only implemented policy)"
        ),
    )
    parser.add_argument(
        "--overlap-schedule",
        action="store_true",
        help="run with the engine's overlap scheduler enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _engine_cli_args(serve_command: list[str]) -> list[str]:
    cli_args = serve_command[4:]
    if "--prometheus-port" in cli_args:
        prometheus_port_index = cli_args.index("--prometheus-port")
        del cli_args[prometheus_port_index : prometheus_port_index + 2]
    return cli_args


def _build_engine_config_and_cli_args(
    config: dict[str, Any], *, overlap_schedule: bool = False
) -> tuple[dict[str, Any], list[str]]:
    engine_config = {
        **config,
        "prefix_caching": False,
        "disable_kvstore": True,
        "overlap_schedule": overlap_schedule,
    }
    serve_command = build_serve_command(engine_config, host="127.0.0.1", port=8000)
    return engine_config, _engine_cli_args(serve_command)


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


def _initial_suffix(tokenizer: Any, user_text: str) -> list[int]:
    return _token_ids(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        add_generation_prompt=True,
    )


def _next_suffix(
    tokenizer: Any,
    user_text: str,
    *,
    prior_user_text: str,
    previous_output_ids: list[int],
    history_ids: list[int],
) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    output_ids_for_content = previous_output_ids
    if (
        previous_output_ids
        and eos_token_id is not None
        and previous_output_ids[-1] == int(eos_token_id)
    ):
        output_ids_for_content = previous_output_ids[:-1]

    initial_ids = _initial_suffix(tokenizer, prior_user_text)
    without_generation = _token_ids(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prior_user_text},
        ],
        add_generation_prompt=False,
    )
    if initial_ids[: len(without_generation)] != without_generation:
        raise ValueError("generation template does not preserve the initial messages")
    generation_prefix = tokenizer.decode(
        initial_ids[len(without_generation) :], skip_special_tokens=False
    )
    assistant_header = "<|im_start|>assistant\n"
    if not generation_prefix.startswith(assistant_header):
        raise ValueError(
            "generation prefix does not preserve the cached Qwen session history"
        )
    assistant_content = generation_prefix[len(assistant_header) :] + tokenizer.decode(
        output_ids_for_content, skip_special_tokens=False
    )
    full_context_ids = _token_ids(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prior_user_text},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": user_text},
        ],
        add_generation_prompt=True,
    )
    if full_context_ids[: len(history_ids)] != history_ids:
        # Qwen non-thinking templates remove old closed reasoning on re-render.
        # A cached Session cannot delete those tokens: append a new ChatML turn.
        if (
            "</think>" in generation_prefix
            and history_ids == initial_ids + previous_output_ids
        ):
            boundary = (
                "\n"
                if output_ids_for_content != previous_output_ids
                else "<|im_end|>\n"
            )
            return list(
                tokenizer.encode(
                    boundary
                    + "<|im_start|>user\n"
                    + user_text.strip()
                    + "<|im_end|>\n"
                    + generation_prefix,
                    add_special_tokens=False,
                )
            )
        raise ValueError(
            "tokenizer chat template does not preserve the cached Qwen session history"
        )
    return full_context_ids[len(history_ids) :]


async def _generate(
    engine: Any,
    input_ids: list[int],
    sampling_params: dict[str, int | float],
    *,
    request_label: str,
    session_id: str | None,
) -> dict[str, Any]:
    print(f"session-kv request_started label={request_label}", flush=True)
    started_at = time.perf_counter()
    result = await asyncio.to_thread(
        engine.generate,
        input_ids=input_ids,
        sampling_params=sampling_params.copy(),
        session_id=session_id,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    output_ids = result.get("output_ids") if isinstance(result, dict) else None
    if not isinstance(output_ids, list) or not all(
        isinstance(token_id, int) for token_id in output_ids
    ):
        raise ValueError("Engine result has no raw output_ids list")
    meta_info = result.get("meta_info") or {}
    if not isinstance(meta_info, dict):
        raise ValueError("Engine result has invalid meta_info")
    print(
        f"session-kv request_finished label={request_label} "
        f"latency_ms={elapsed_ms:.2f}",
        flush=True,
    )
    return {
        "output_ids": [int(token_id) for token_id in output_ids],
        "meta_info": meta_info,
        "latency_ms": elapsed_ms,
    }


async def _run_sessions(
    engine: Any,
    tokenizer: Any,
    *,
    frontier_policy: str,
    max_new_tokens: int,
    run_record: dict[str, Any],
) -> None:
    sampling_params: dict[str, int | float] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
    }
    session_specs = []
    for index, prompts in enumerate(SESSION_TURNS):
        session_specs.append(
            {
                "label": f"session-{index}",
                "session_id": engine.open_session(),
                "prompts": prompts,
                "history_ids": [],
            }
        )
    session_ids = [spec["session_id"] for spec in session_specs]
    if len(set(session_ids)) != len(session_ids):
        raise RuntimeError("Engine returned duplicate session IDs")
    run_record["session_ids"] = session_ids

    closed_session_ids: set[str] = set()
    try:
        for turn_index in range(len(SESSION_TURNS[0])):
            prepared = []
            for spec in session_specs:
                prompt = spec["prompts"][turn_index]
                if turn_index == 0:
                    suffix_ids = _initial_suffix(tokenizer, prompt)
                else:
                    suffix_ids = _next_suffix(
                        tokenizer,
                        prompt,
                        prior_user_text=spec["prompts"][turn_index - 1],
                        previous_output_ids=spec["previous_output_ids"],
                        history_ids=spec["history_ids"],
                    )
                full_input_ids = spec["history_ids"] + suffix_ids
                prepared.append(
                    {
                        "spec": spec,
                        "suffix_ids": suffix_ids,
                        "full_input_ids": full_input_ids,
                        "prior_history_tokens": len(spec["history_ids"]),
                    }
                )

            requests = []
            for item in prepared:
                requests.extend(
                    (
                        _generate(
                            engine,
                            item["suffix_ids"],
                            sampling_params,
                            request_label=(
                                f"{item['spec']['label']}-turn-{turn_index + 1}-session"
                            ),
                            session_id=item["spec"]["session_id"],
                        ),
                        _generate(
                            engine,
                            item["full_input_ids"],
                            sampling_params,
                            request_label=(
                                f"{item['spec']['label']}-turn-{turn_index + 1}-stateless"
                            ),
                            session_id=None,
                        ),
                    )
                )
            results = await asyncio.gather(*requests)

            for index, item in enumerate(prepared):
                session_result = results[index * 2]
                control_result = results[index * 2 + 1]
                session_metadata = session_result["meta_info"].get("session")
                if not isinstance(session_metadata, Mapping):
                    raise ValueError(
                        "session generation result has no session metadata"
                    )
                output_ids_match = (
                    session_result["output_ids"] == control_result["output_ids"]
                )
                turn_record = {
                    "session_label": item["spec"]["label"],
                    "session_id": item["spec"]["session_id"],
                    "turn_index": turn_index + 1,
                    "input_ids": item["suffix_ids"],
                    "suffix_ids": item["suffix_ids"],
                    "stateless_control_input_ids": item["full_input_ids"],
                    "session_output_ids": session_result["output_ids"],
                    "stateless_control_output_ids": control_result["output_ids"],
                    "session_metadata": dict(session_metadata),
                    "session_latency_ms": session_result["latency_ms"],
                    "stateless_control_latency_ms": control_result["latency_ms"],
                    "output_ids_match": output_ids_match,
                    "context_metadata_validated": False,
                }
                run_record["turns"].append(turn_record)
                validate_session_context_metadata(
                    session_metadata,
                    policy=frontier_policy,
                    prior_history_tokens=item["prior_history_tokens"],
                    suffix_tokens=len(item["suffix_ids"]),
                )
                turn_record["context_metadata_validated"] = True
                if not output_ids_match:
                    raise ValueError(
                        f"session/stateless output token mismatch for "
                        f"{item['spec']['label']} turn {turn_index + 1}"
                    )
                item["spec"]["previous_output_ids"] = session_result["output_ids"]
                item["spec"]["history_ids"] = (
                    item["full_input_ids"] + session_result["output_ids"]
                )

        for session_id in session_ids:
            engine.close_session(session_id)
            closed_session_ids.add(session_id)
        run_record["closed_session_ids"] = sorted(closed_session_ids)

        for session_id in session_ids:
            result = await _generate(
                engine,
                [0],
                sampling_params,
                request_label=f"closed-session-probe-{session_id}",
                session_id=session_id,
            )
            if not is_expected_closed_session_rejection(result, session_id):
                raise RuntimeError(
                    "Engine did not return the expected closed-session abort for "
                    f"{session_id}"
                )
            run_record["closed_session_rejections"].append(
                {
                    "session_id": session_id,
                    "finish_reason": result["meta_info"]["finish_reason"],
                }
            )
        if len(run_record["closed_session_rejections"]) != len(session_ids):
            raise RuntimeError("not all closed session IDs were rejected")
    finally:
        for session_id in session_ids:
            if session_id not in closed_session_ids:
                try:
                    engine.close_session(session_id)
                except Exception:
                    pass


def _capture_hardware_snapshot(
    hardware_dir: Path, run_id: str, phase: str
) -> dict[str, bool | str | None]:
    snapshot_path = hardware_dir / f"{run_id}-{phase}.txt"
    available = capture_npu_smi(snapshot_path)
    return {
        "available": available,
        "path": str(snapshot_path) if available else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    config = load_model_config(CONFIG_PATH)
    local_model_path = _local_model_path(config)
    config, cli_args = _build_engine_config_and_cli_args(
        config, overlap_schedule=args.overlap_schedule
    )
    if Path(cli_args[0]).resolve() != local_model_path:
        raise ValueError("engine launch did not resolve to the pinned local model")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir, hardware_dir = resolve_output_paths(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qwen35-4b-{run_id}.json"
    run_record = {
        "run_id": run_id,
        "status": "running",
        "frontier_policy": args.frontier_policy,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "local_model_path": str(local_model_path),
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
        },
        "launch_configuration": {
            "model_config": config,
            "engine_cli_args": cli_args,
            "runtime_environment": {
                "ASCEND_RT_VISIBLE_DEVICES": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
            },
            "generic_prefix_caching": False,
            "concurrent_sessions": 2,
        },
        "hardware_before": None,
        "hardware_after": None,
        "turns": [],
        "closed_session_rejections": [],
    }
    run_record["hardware_before"] = _capture_hardware_snapshot(
        hardware_dir, run_id, "before"
    )

    try:
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
                _run_sessions(
                    engine,
                    tokenizer,
                    frontier_policy=args.frontier_policy,
                    max_new_tokens=args.max_new_tokens,
                    run_record=run_record,
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
        output_path.write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(output_path)


if __name__ == "__main__":
    main()
