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

import copy
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from urllib.error import HTTPError
from xml.etree import ElementTree

from benchmarks.course_project.core import chat_template_token_ids
from benchmarks.course_project.runner import make_token_sized_text

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_RECIPE_VERSION = "long-prefill-v2"
_CHUNK_SIZES = (4096, 512)
_PROMPT_TARGETS = (256, 2048, 3584)
_MAX_CONTEXT_TOKENS = 4096
_MAX_PROMPT_SEARCH_ATTEMPTS = 32


@dataclass(frozen=True)
class TrialSpec:
    """One planned long-prefill/decode trial."""

    trial_id: str
    condition: str
    prefill_tokens: int
    repetition: int
    chunk_size: int
    execution_order: int


def build_config_variants(base_config: dict) -> dict[int, dict]:
    """Return pinned 4B configs for the two supported prefill chunk sizes."""
    pinned_values = {
        "model_id": "Qwen/Qwen3.5-4B",
        "tp_size": 1,
        "dtype": "bfloat16",
        "max_model_len": _MAX_CONTEXT_TOKENS,
    }
    for field, expected in pinned_values.items():
        if base_config.get(field) != expected:
            raise ValueError(f"long prefill benchmark requires {field}={expected!r}")
    if base_config.get("chunked_prefill_size") not in _CHUNK_SIZES:
        raise ValueError(
            "long prefill benchmark supports chunked_prefill_size 4096 or 512"
        )

    variants = {}
    for chunk_size in _CHUNK_SIZES:
        config = copy.deepcopy(base_config)
        config["chunked_prefill_size"] = chunk_size
        config["prefix_caching"] = False
        config["disable_kvstore"] = True
        variants[chunk_size] = config
    return variants


def local_model_path(config: dict) -> Path:
    """Resolve an existing model directory inside the project tree."""
    configured_path = config.get("local_model_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("long prefill benchmark requires local_model_path")

    path = Path(configured_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved_path = path.resolve()
    project_root = REPO_ROOT.resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("local_model_path must be inside the project tree") from error
    if not resolved_path.is_dir():
        raise ValueError("local_model_path must be an existing directory")
    return resolved_path


def load_local_tokenizer(config: dict):
    """Load the configured tokenizer from the existing project-local checkpoint."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(local_model_path(config)),
        local_files_only=True,
    )


def build_exact_messages(
    tokenizer, target_tokens: int, seed: int | str
) -> list[dict[str, str]]:
    """Build one user message with an exact complete chat-template token count."""
    if target_tokens not in _PROMPT_TARGETS:
        raise ValueError(
            "target_tokens must be 256, 2048, or 3584 and no greater than 4096"
        )

    seed_text = (
        f"Long prefill workload seed {seed}. "
        "Summarize the sequence and preserve every stated detail."
    )

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    # Concatenation may merge tokens across seed boundaries. Supply enough
    # actual encoded tokens before the shared helper truncates candidates.
    for _ in range(target_tokens.bit_length() + 1):
        if len(encode(seed_text)) >= target_tokens:
            break
        seed_text *= 2
    else:
        raise ValueError(f"unable to supply prompt tokens for target {target_tokens}")

    low = 0
    high = target_tokens
    attempts = 0
    tested_candidates: set[int] = set()
    closest_observed: int | None = None
    closest_candidate: int | None = None

    def evaluate(text_tokens: int) -> tuple[list[dict[str, str]], int] | None:
        nonlocal attempts, closest_candidate, closest_observed
        if text_tokens in tested_candidates or attempts >= _MAX_PROMPT_SEARCH_ATTEMPTS:
            return None
        tested_candidates.add(text_tokens)
        attempts += 1
        content = make_token_sized_text(
            target_tokens=text_tokens,
            seed=seed_text,
            encode=encode,
            decode=decode,
        )
        messages = [{"role": "user", "content": content}]
        observed = len(chat_template_token_ids(tokenizer, messages))
        if closest_observed is None or abs(observed - target_tokens) < abs(
            closest_observed - target_tokens
        ):
            closest_observed = observed
            closest_candidate = text_tokens
        return messages, observed

    while low <= high and attempts < _MAX_PROMPT_SEARCH_ATTEMPTS:
        text_tokens = (low + high) // 2
        result = evaluate(text_tokens)
        if result is None:
            break
        messages, observed = result
        if observed == target_tokens and observed <= _MAX_CONTEXT_TOKENS:
            return messages
        if observed > target_tokens:
            high = text_tokens - 1
        else:
            low = text_tokens + 1
        if low > high:
            break

    fallback_center = closest_candidate
    if fallback_center is not None:
        for distance in range(1, _MAX_PROMPT_SEARCH_ATTEMPTS + 1):
            for text_tokens in (
                fallback_center - distance,
                fallback_center + distance,
            ):
                if attempts >= _MAX_PROMPT_SEARCH_ATTEMPTS:
                    break
                if not 0 <= text_tokens <= target_tokens:
                    continue
                result = evaluate(text_tokens)
                if result is None:
                    continue
                messages, observed = result
                if observed == target_tokens and observed <= _MAX_CONTEXT_TOKENS:
                    return messages

    raise ValueError(
        f"unable to build exact prompt for target {target_tokens}; "
        f"closest observed length {closest_observed} (maximum {_MAX_CONTEXT_TOKENS})"
    )


def prompt_sha256(messages: list[dict[str, str]]) -> str:
    """Return a stable hash of prompt messages without exposing their text."""
    serialized = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_result(
    result,
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> tuple[bool, str | None]:
    """Check stream completion and exact server-reported token usage."""
    if not result.stream_completed:
        return False, "stream did not complete"
    if result.input_tokens != expected_input_tokens:
        return (
            False,
            f"input token mismatch: expected {expected_input_tokens}, "
            f"observed {result.input_tokens}",
        )
    if result.output_tokens != expected_output_tokens:
        return (
            False,
            f"output token mismatch: expected {expected_output_tokens}, "
            f"observed {result.output_tokens}",
        )
    return True, None


def _trial_metadata(trial_spec: TrialSpec) -> dict:
    return {
        "schema_version": 1,
        "trial_id": trial_spec.trial_id,
        "condition": trial_spec.condition,
        "prefill_tokens": trial_spec.prefill_tokens,
        "repetition": trial_spec.repetition,
        "chunk_size": trial_spec.chunk_size,
        "execution_order": trial_spec.execution_order,
    }


def _event_records(events) -> list[dict]:
    return [
        {
            "chunk_index": event.chunk_index,
            "field": event.field,
            "epoch_ns": event.epoch_ns,
            "monotonic_ns": event.monotonic_ns,
            "text_length": event.text_length,
        }
        for event in events
    ]


def _request_record(
    *,
    request_role: str,
    result,
    expected_input_tokens: int,
    expected_output_tokens: int,
    max_tokens: int,
    ignore_eos: bool,
    chunk_events: list,
) -> dict:
    valid, invalid_reason = validate_result(
        result,
        expected_input_tokens=expected_input_tokens,
        expected_output_tokens=expected_output_tokens,
    )
    if (
        valid
        and expected_output_tokens > 0
        and not any(event.text_length > 0 for event in chunk_events)
    ):
        valid = False
        invalid_reason = "request_stream_has_no_nonempty_chunk"

    return {
        "request_role": request_role,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "expected_input_tokens": expected_input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "request_send_epoch_ns": result.request_send_epoch_ns,
        "request_send_monotonic_ns": result.request_send_monotonic_ns,
        "failure_observed_epoch_ns": None,
        "failure_observed_monotonic_ns": None,
        "first_chunk_epoch_ns": result.first_chunk_epoch_ns,
        "first_chunk_monotonic_ns": result.first_chunk_monotonic_ns,
        "first_chunk_field": result.first_chunk_field,
        "stream_drain_epoch_ns": result.stream_drain_epoch_ns,
        "stream_drain_monotonic_ns": result.stream_drain_monotonic_ns,
        "ttft_ms": result.ttft_ms,
        "e2e_ms": result.e2e_ms,
        "stream_completed": result.stream_completed,
        "chunk_events": _event_records(chunk_events),
        "http_status": result.http_status,
        "status": "valid" if valid else "invalid",
        "invalid_reason": invalid_reason,
    }


def _failed_request_record(
    *,
    request_role: str,
    expected_input_tokens: int,
    expected_output_tokens: int,
    max_tokens: int,
    ignore_eos: bool,
    error: BaseException,
    chunk_events: list,
    request_send_epoch_ns: int | None,
    request_send_monotonic_ns: int | None,
    failure_observed_epoch_ns: int,
    failure_observed_monotonic_ns: int,
) -> dict:
    event_records = _event_records(chunk_events)
    first_event = event_records[0] if event_records else None
    return {
        "request_role": request_role,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "expected_input_tokens": expected_input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "input_tokens": None,
        "output_tokens": None,
        "request_send_epoch_ns": request_send_epoch_ns,
        "request_send_monotonic_ns": request_send_monotonic_ns,
        "failure_observed_epoch_ns": failure_observed_epoch_ns,
        "failure_observed_monotonic_ns": failure_observed_monotonic_ns,
        "first_chunk_epoch_ns": (
            None if first_event is None else first_event["epoch_ns"]
        ),
        "first_chunk_monotonic_ns": (
            None if first_event is None else first_event["monotonic_ns"]
        ),
        "first_chunk_field": None if first_event is None else first_event["field"],
        "stream_drain_epoch_ns": None,
        "stream_drain_monotonic_ns": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "stream_completed": False,
        "chunk_events": event_records,
        "http_status": (
            error.code
            if isinstance(error, HTTPError) and type(error.code) is int
            else None
        ),
        "status": "invalid",
        "invalid_reason": f"{request_role}_request_failed: {type(error).__name__}",
    }


def _trial_result(
    trial_spec: TrialSpec,
    requests: list[dict],
    *,
    invalid_reasons: list[str],
    chunk_gap_events: list[dict] | None = None,
    requests_overlap: bool | None = None,
) -> dict:
    return {
        **_trial_metadata(trial_spec),
        "status": "invalid" if invalid_reasons else "valid",
        "invalid_reason": "; ".join(invalid_reasons) if invalid_reasons else None,
        "requests_overlap": requests_overlap,
        "requests": requests,
        "chunk_gap_events": chunk_gap_events or [],
    }


def _classify_chunk_gaps(
    chunk_events: list[dict], prefill_send_monotonic_ns: int
) -> list[dict]:
    """Classify client SSE chunk gaps; these are not token ITL measurements."""
    gaps = []
    for previous, current in zip(chunk_events, chunk_events[1:]):
        start_ns = previous["monotonic_ns"]
        end_ns = current["monotonic_ns"]
        if end_ns <= prefill_send_monotonic_ns:
            classification = "before"
        elif start_ns >= prefill_send_monotonic_ns:
            classification = "after"
        else:
            classification = "crosses"
        gaps.append(
            {
                "from_chunk_index": previous["chunk_index"],
                "to_chunk_index": current["chunk_index"],
                "from_monotonic_ns": start_ns,
                "to_monotonic_ns": end_ns,
                "client_sse_chunk_gap_ms": (end_ns - start_ns) / 1_000_000,
                "classification": classification,
            }
        )
    return gaps


def run_single_trial(
    *,
    client,
    trial_spec: TrialSpec,
    request_role: str,
    model: str,
    messages: list[dict[str, str]],
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> dict:
    """Run one HTTP request and return a JSON-safe record without generated text."""
    if request_role not in {"decode", "prefill"}:
        raise ValueError("request_role must be decode or prefill")
    ignore_eos = request_role == "decode" and expected_output_tokens == 128
    chunk_events = []
    request_send_timestamps = None

    def record_request_send(epoch_ns: int, monotonic_ns: int) -> None:
        nonlocal request_send_timestamps
        if request_send_timestamps is None:
            request_send_timestamps = (epoch_ns, monotonic_ns)

    try:
        result = client.chat(
            model=model,
            messages=messages,
            max_tokens=expected_output_tokens,
            on_chunk=chunk_events.append,
            on_request_send=record_request_send,
            ignore_eos=ignore_eos,
        )
        request = _request_record(
            request_role=request_role,
            result=result,
            expected_input_tokens=expected_input_tokens,
            expected_output_tokens=expected_output_tokens,
            max_tokens=expected_output_tokens,
            ignore_eos=ignore_eos,
            chunk_events=chunk_events,
        )
    except Exception as error:
        failure_observed_epoch_ns = time.time_ns()
        failure_observed_monotonic_ns = time.monotonic_ns()
        request = _failed_request_record(
            request_role=request_role,
            expected_input_tokens=expected_input_tokens,
            expected_output_tokens=expected_output_tokens,
            max_tokens=expected_output_tokens,
            ignore_eos=ignore_eos,
            error=error,
            chunk_events=chunk_events,
            request_send_epoch_ns=(
                None if request_send_timestamps is None else request_send_timestamps[0]
            ),
            request_send_monotonic_ns=(
                None if request_send_timestamps is None else request_send_timestamps[1]
            ),
            failure_observed_epoch_ns=failure_observed_epoch_ns,
            failure_observed_monotonic_ns=failure_observed_monotonic_ns,
        )
    invalid_reasons = (
        []
        if request["status"] == "valid"
        else [f"{request_role}: {request['invalid_reason']}"]
    )
    return _trial_result(trial_spec, [request], invalid_reasons=invalid_reasons)


def run_staged_trial(
    *,
    client,
    trial_spec: TrialSpec,
    model: str,
    decode_messages: list[dict[str, str]],
    prefill_messages: list[dict[str, str]],
    expected_decode_input_tokens: int,
    expected_decode_output_tokens: int,
    expected_prefill_input_tokens: int,
    expected_prefill_output_tokens: int,
) -> dict:
    """Start Decode first, then submit Prefill on its first nonempty SSE chunk.

    The callback only submits Prefill to the worker pool, so it never waits on
    Prefill while Decode continues draining its SSE stream. Recorded client SSE
    chunk gaps are arrival intervals, not token ITL measurements.
    """
    decode_events = []
    prefill_events = []
    prefill_future: Future | None = None
    decode_send_timestamps = None
    prefill_send_timestamps = None
    decode_ignore_eos = expected_decode_output_tokens == 128
    requests = []
    invalid_reasons = []
    decode_result = None
    prefill_result = None
    decode_error = None
    prefill_error = None
    decode_failure_observed = None
    prefill_failure_observed = None

    with ThreadPoolExecutor(max_workers=2) as executor:

        def record_decode_send(epoch_ns: int, monotonic_ns: int) -> None:
            nonlocal decode_send_timestamps
            if decode_send_timestamps is None:
                decode_send_timestamps = (epoch_ns, monotonic_ns)

        def record_prefill_send(epoch_ns: int, monotonic_ns: int) -> None:
            nonlocal prefill_send_timestamps
            if prefill_send_timestamps is None:
                prefill_send_timestamps = (epoch_ns, monotonic_ns)

        def submit_prefill(_field: str, _epoch_ns: int, _monotonic_ns: int) -> None:
            nonlocal prefill_future
            if prefill_future is None:
                prefill_future = executor.submit(
                    client.chat,
                    model=model,
                    messages=prefill_messages,
                    max_tokens=expected_prefill_output_tokens,
                    on_chunk=prefill_events.append,
                    on_request_send=record_prefill_send,
                    ignore_eos=False,
                )

        decode_future = executor.submit(
            client.chat,
            model=model,
            messages=decode_messages,
            max_tokens=expected_decode_output_tokens,
            on_first_chunk=submit_prefill,
            on_chunk=decode_events.append,
            on_request_send=record_decode_send,
            ignore_eos=decode_ignore_eos,
        )
        try:
            decode_result = decode_future.result()
        except Exception as error:
            decode_error = error
            decode_failure_observed = (time.time_ns(), time.monotonic_ns())
        if prefill_future is not None:
            try:
                prefill_result = prefill_future.result()
            except Exception as error:
                prefill_error = error
                prefill_failure_observed = (time.time_ns(), time.monotonic_ns())

    if decode_error is not None:
        requests.append(
            _failed_request_record(
                request_role="decode",
                expected_input_tokens=expected_decode_input_tokens,
                expected_output_tokens=expected_decode_output_tokens,
                max_tokens=expected_decode_output_tokens,
                ignore_eos=decode_ignore_eos,
                error=decode_error,
                chunk_events=decode_events,
                request_send_epoch_ns=(
                    None
                    if decode_send_timestamps is None
                    else decode_send_timestamps[0]
                ),
                request_send_monotonic_ns=(
                    None
                    if decode_send_timestamps is None
                    else decode_send_timestamps[1]
                ),
                failure_observed_epoch_ns=decode_failure_observed[0],
                failure_observed_monotonic_ns=decode_failure_observed[1],
            )
        )
        invalid_reasons.append(requests[-1]["invalid_reason"])
    else:
        decode_request = _request_record(
            request_role="decode",
            result=decode_result,
            expected_input_tokens=expected_decode_input_tokens,
            expected_output_tokens=expected_decode_output_tokens,
            max_tokens=expected_decode_output_tokens,
            ignore_eos=decode_ignore_eos,
            chunk_events=decode_events,
        )
        requests.append(decode_request)
        if decode_request["status"] == "invalid":
            invalid_reasons.append(f"decode: {decode_request['invalid_reason']}")

    if prefill_future is None:
        if decode_error is None:
            if not decode_events:
                invalid_reasons.append("decode_stream_has_no_nonempty_chunk")
            else:
                invalid_reasons.append("decode_first_chunk_callback_not_triggered")
    elif prefill_error is not None:
        requests.append(
            _failed_request_record(
                request_role="prefill",
                expected_input_tokens=expected_prefill_input_tokens,
                expected_output_tokens=expected_prefill_output_tokens,
                max_tokens=expected_prefill_output_tokens,
                ignore_eos=False,
                error=prefill_error,
                chunk_events=prefill_events,
                request_send_epoch_ns=(
                    None
                    if prefill_send_timestamps is None
                    else prefill_send_timestamps[0]
                ),
                request_send_monotonic_ns=(
                    None
                    if prefill_send_timestamps is None
                    else prefill_send_timestamps[1]
                ),
                failure_observed_epoch_ns=prefill_failure_observed[0],
                failure_observed_monotonic_ns=prefill_failure_observed[1],
            )
        )
        invalid_reasons.append(requests[-1]["invalid_reason"])
    else:
        prefill_request = _request_record(
            request_role="prefill",
            result=prefill_result,
            expected_input_tokens=expected_prefill_input_tokens,
            expected_output_tokens=expected_prefill_output_tokens,
            max_tokens=expected_prefill_output_tokens,
            ignore_eos=False,
            chunk_events=prefill_events,
        )
        requests.append(prefill_request)
        if prefill_request["status"] == "invalid":
            invalid_reasons.append(f"prefill: {prefill_request['invalid_reason']}")

    requests_overlap = None
    chunk_gap_events = []
    if decode_result is not None and prefill_result is not None:
        requests_overlap = (
            decode_result.request_send_monotonic_ns
            < prefill_result.stream_drain_monotonic_ns
            and prefill_result.request_send_monotonic_ns
            < decode_result.stream_drain_monotonic_ns
        )
        if not requests_overlap:
            invalid_reasons.append("decode_and_prefill_requests_did_not_overlap")
    if prefill_result is not None:
        chunk_gap_events = _classify_chunk_gaps(
            _event_records(decode_events),
            prefill_result.request_send_monotonic_ns,
        )

    return _trial_result(
        trial_spec,
        requests,
        invalid_reasons=invalid_reasons,
        chunk_gap_events=chunk_gap_events,
        requests_overlap=requests_overlap,
    )


def _paired_values(trial: dict) -> dict:
    requests = {request["request_role"]: request for request in trial["requests"]}
    values = {}
    for role in ("decode", "prefill"):
        request = requests.get(role)
        if request is None:
            continue
        values[f"{role}_ttft_ms"] = request["ttft_ms"]
        values[f"{role}_e2e_ms"] = request["e2e_ms"]
    gaps = [
        event["client_sse_chunk_gap_ms"] for event in trial.get("chunk_gap_events", [])
    ]
    values["client_sse_chunk_gap_ms"] = gaps
    values["client_sse_chunk_gap_ms_median"] = median(gaps) if gaps else None
    return values


def compare_paired_trials(trials: list[dict]) -> dict:
    """Pair 4096/512 trials and summarize raw values and 512-minus-4096 deltas.

    Client SSE chunk gaps remain explicitly named as such; they do not measure
    token ITL. Invalid or incomplete pairs are retained but excluded from
    difference summaries.
    """
    grouped = {}
    for trial in trials:
        key = (trial["condition"], trial["prefill_tokens"], trial["repetition"])
        grouped.setdefault(key, {4096: [], 512: []})
        if trial["chunk_size"] in (4096, 512):
            grouped[key][trial["chunk_size"]].append(trial)

    pairs = []
    metric_names = (
        "decode_ttft_ms",
        "decode_e2e_ms",
        "prefill_ttft_ms",
        "prefill_e2e_ms",
        "client_sse_chunk_gap_ms_median",
    )
    for (condition, prefill_tokens, repetition), variants in sorted(grouped.items()):
        reasons = []
        for chunk_size in (4096, 512):
            if not variants[chunk_size]:
                reasons.append(f"missing_chunk_size_{chunk_size}")
            elif len(variants[chunk_size]) > 1:
                reasons.append(f"duplicate_chunk_size_{chunk_size}")
        raw_values = {"chunk_size_4096": None, "chunk_size_512": None}
        for chunk_size in (4096, 512):
            if len(variants[chunk_size]) == 1:
                trial = variants[chunk_size][0]
                raw_values[f"chunk_size_{chunk_size}"] = _paired_values(trial)
                if trial["status"] != "valid":
                    reasons.append("invalid_trial")

        differences = {}
        if not reasons:
            for metric in metric_names:
                value_4096 = raw_values["chunk_size_4096"].get(metric)
                value_512 = raw_values["chunk_size_512"].get(metric)
                if value_4096 is not None and value_512 is not None:
                    differences[metric] = value_512 - value_4096
        pairs.append(
            {
                "pair_id": f"{condition}-p{prefill_tokens}-r{repetition}",
                "condition": condition,
                "prefill_tokens": prefill_tokens,
                "repetition": repetition,
                "status": "invalid" if reasons else "valid",
                "invalid_reason": "; ".join(reasons) if reasons else None,
                "difference_direction": "chunk_size_512_minus_4096",
                "raw_values": raw_values,
                "paired_differences": differences,
            }
        )

    summaries = []
    summary_groups = sorted(
        {(pair["condition"], pair["prefill_tokens"]) for pair in pairs}
    )
    for condition, prefill_tokens in summary_groups:
        scoped_pairs = [
            pair
            for pair in pairs
            if pair["condition"] == condition
            and pair["prefill_tokens"] == prefill_tokens
            and pair["status"] == "valid"
        ]
        for metric in metric_names:
            values = [
                pair["paired_differences"][metric]
                for pair in scoped_pairs
                if metric in pair["paired_differences"]
            ]
            summaries.append(
                {
                    "condition": condition,
                    "prefill_tokens": prefill_tokens,
                    "metric": metric,
                    "difference_direction": "chunk_size_512_minus_4096",
                    "n": len(values),
                    "median": median(values) if values else None,
                    "range": [min(values), max(values)] if values else None,
                }
            )
    return {"pairs": pairs, "summaries": summaries}


_REPORT_CONDITIONS = (
    "decode-only",
    "prefill-only-2048",
    "prefill-only-3584",
    "staged-2048",
    "staged-3584",
)
_REPORT_CONCLUSION_BOUNDARY = (
    "仅描述所测模型、TP1、输入长度、调度设置与 HTTP 服务环境中的客户端 SSE 块间隔；"
    "不推断服务端内部调度或算子、硬件任务因果关系；小样本不支持显著性或普遍性结论。"
)
_REPORT_METRICS = (
    "decode_ttft_ms",
    "decode_e2e_ms",
    "prefill_ttft_ms",
    "prefill_e2e_ms",
    "client_sse_chunk_gap_before_median_ms",
    "client_sse_chunk_gap_after_median_ms",
)
_REPORT_SVG_NS = "http://www.w3.org/2000/svg"
# Adjacent clock calls can include small scheduling skew; 1 ms stays small versus
# the HTTP request durations while absorbing that measurement-boundary difference.
_REPORT_TIMING_TOLERANCE_MS = 1.0


def _report_input_path(path: Path, raw_dir: Path) -> Path:
    project_root = REPO_ROOT.resolve()
    raw_root = raw_dir.resolve(strict=True)
    try:
        raw_root.relative_to(project_root)
    except ValueError:
        raise ValueError("raw run directory must resolve inside the project") from None
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"{path.name} must resolve inside the project") from None
    try:
        resolved.relative_to(raw_root)
    except ValueError:
        raise ValueError(f"{path.name} must resolve inside the raw run") from None
    if not resolved.is_file():
        raise ValueError(f"{path.name} must be a regular file")
    return resolved


def _report_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _report_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL record in {path.name} at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path.name} line {line_number} must be a JSON object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {path.name}") from error
    return records


def _report_positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a positive finite number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def _report_validate_timing_metric(
    value: object, field: str, *, start_ns: int, end_ns: int
) -> None:
    reported_ms = _report_positive_number(value, field)
    timestamp_ms = (end_ns - start_ns) / 1_000_000
    if not math.isclose(
        reported_ms,
        timestamp_ms,
        rel_tol=0.0,
        abs_tol=_REPORT_TIMING_TOLERANCE_MS,
    ):
        raise ValueError(
            f"{field} does not match monotonic timestamps within "
            f"{_REPORT_TIMING_TOLERANCE_MS:g} ms"
        )


def _report_token_count(
    value: object, field: str, *, allow_none: bool = False
) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer token count")
    return value


def _report_monotonic_ns(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _report_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _report_expected_schedule() -> list[dict]:
    chunk_order_by_round = ((4096, 512), (512, 4096), (4096, 512))
    return [
        {"round": repetition, "order": order, "chunk_size": chunk_size}
        for repetition, chunk_order in enumerate(chunk_order_by_round, start=1)
        for order, chunk_size in enumerate(chunk_order, start=1)
    ]


def _report_validate_run(raw_dir: Path) -> tuple[dict, dict[int, str], str]:
    run = _report_json(_report_input_path(raw_dir / "run.json", raw_dir))
    if type(run.get("schema_version")) is not int or run["schema_version"] != 1:
        raise ValueError("run.json schema_version must be 1")
    run_id = run.get("run_id")
    model_id = run.get("model_id")
    model_revision = run.get("model_revision")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.json run_id is missing")
    if model_id != "Qwen/Qwen3.5-4B":
        raise ValueError("run.json model_id must be Qwen/Qwen3.5-4B")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("run.json model_revision is missing")
    _report_sha256(run.get("config_file_sha256"), "run.json config_file_sha256")
    if run.get("conditions") != list(_REPORT_CONDITIONS):
        raise ValueError("run.json conditions do not match the report schema")
    if run.get("schedule") != _report_expected_schedule():
        raise ValueError("run.json schedule does not match the three-round schema")

    config_by_chunk: dict[int, dict] = {}
    hashes_by_chunk: dict[int, str] = {}
    expected_files = set()
    baseline_config = None
    for launch in _report_expected_schedule():
        repetition = launch["round"]
        chunk_size = launch["chunk_size"]
        name = f"config-round-{repetition:02d}-chunk-{chunk_size}.json"
        expected_files.add(name)
        snapshot = _report_json(_report_input_path(raw_dir / name, raw_dir))
        if (
            type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 1
        ):
            raise ValueError(f"{name} schema_version must be 1")
        if snapshot.get("run_id") != run_id:
            raise ValueError(f"{name} run_id does not match run.json")
        if type(snapshot.get("round")) is not int or snapshot["round"] != repetition:
            raise ValueError(f"{name} round does not match its filename")
        expected_launch_order = launch["order"]
        if (
            type(snapshot.get("launch_order")) is not int
            or snapshot["launch_order"] != expected_launch_order
        ):
            raise ValueError(f"{name} launch_order is invalid")
        if snapshot.get("config_file_sha256") != run["config_file_sha256"]:
            raise ValueError(f"{name} config file digest does not match run.json")
        config_sha256 = _report_sha256(
            snapshot.get("config_sha256"), f"{name} config_sha256"
        )
        config = snapshot.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{name} config must be an object")
        served_model_name = config.get("served_model_name")
        if not isinstance(served_model_name, str) or not served_model_name.strip():
            raise ValueError(f"{name} config served_model_name is missing")
        expected_config = {
            "model_id": model_id,
            "model_revision": model_revision,
            "tp_size": 1,
            "dtype": "bfloat16",
            "max_model_len": _MAX_CONTEXT_TOKENS,
            "chunked_prefill_size": chunk_size,
            "prefix_caching": False,
        }
        for field, expected in expected_config.items():
            observed = config.get(field)
            if (
                observed != expected
                or (type(expected) is int and type(observed) is not int)
                or (type(expected) is bool and type(observed) is not bool)
            ):
                raise ValueError(
                    f"{name} config {field} does not match the pinned setting"
                )

        normalized_config = dict(config)
        normalized_config.pop("chunked_prefill_size")
        if baseline_config is None:
            baseline_config = normalized_config
        elif normalized_config != baseline_config:
            raise ValueError("config snapshots differ beyond chunked_prefill_size")
        prior_hash = hashes_by_chunk.setdefault(chunk_size, config_sha256)
        if prior_hash != config_sha256:
            raise ValueError(
                f"config digest changes between rounds for chunk {chunk_size}"
            )
        prior_config = config_by_chunk.setdefault(chunk_size, config)
        if prior_config != config:
            raise ValueError(f"config changes between rounds for chunk {chunk_size}")

    actual_files = {path.name for path in raw_dir.glob("config-round-*-chunk-*.json")}
    if actual_files != expected_files:
        raise ValueError("raw run must contain exactly six round config snapshots")
    return run, hashes_by_chunk, served_model_name


def _report_expected_prefill_tokens(condition: str) -> int:
    if condition == "decode-only":
        return 0
    if condition in _REPORT_CONDITIONS:
        return int(condition.rsplit("-", 1)[1])
    raise ValueError(f"unsupported measurement condition: {condition}")


def _report_request_chunk_events(
    request: dict,
    *,
    role: str,
    expected_output_tokens: int,
    send_ns: int,
    drain_ns: int,
) -> list[dict]:
    events = request.get("chunk_events")
    if not isinstance(events, list):
        raise ValueError(f"valid {role} chunk_events must be a list")
    if expected_output_tokens > 0 and not events:
        raise ValueError(f"valid {role} request has no chunk events")
    if drain_ns <= send_ns:
        raise ValueError(f"valid {role} request lifetime must be positive")

    expected_fields = {
        "chunk_index",
        "field",
        "epoch_ns",
        "monotonic_ns",
        "text_length",
    }
    previous_monotonic_ns = None
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or set(event) != expected_fields:
            raise ValueError(f"{role} chunk event structure is invalid")
        if type(event["chunk_index"]) is not int or event["chunk_index"] != index:
            raise ValueError(f"{role} chunk_index values must increase from one")
        if event["field"] not in ("content", "reasoning_content"):
            raise ValueError(f"{role} chunk event field is invalid")
        if type(event["epoch_ns"]) is not int or event["epoch_ns"] <= 0:
            raise ValueError(f"{role} chunk event epoch_ns must be positive")
        monotonic_ns = _report_monotonic_ns(
            event["monotonic_ns"], f"{role} chunk event monotonic_ns"
        )
        if monotonic_ns <= 0 or (
            previous_monotonic_ns is not None and monotonic_ns <= previous_monotonic_ns
        ):
            raise ValueError(f"{role} chunk event times must strictly increase")
        if not send_ns <= monotonic_ns <= drain_ns:
            raise ValueError(f"{role} chunk event is outside the request interval")
        if type(event["text_length"]) is not int or event["text_length"] <= 0:
            raise ValueError(f"{role} chunk event text_length must be positive")
        previous_monotonic_ns = monotonic_ns
    return events


def _report_request_metrics(
    trial: dict, *, condition: str, prefill_tokens: int
) -> dict[str, float | None]:
    requests = trial.get("requests")
    if not isinstance(requests, list):
        raise ValueError("trial requests must be a list")
    expected_roles = {
        "decode-only": {"decode"},
        "prefill-only-2048": {"prefill"},
        "prefill-only-3584": {"prefill"},
        "staged-2048": {"decode", "prefill"},
        "staged-3584": {"decode", "prefill"},
    }[condition]
    requests_by_role = {}
    request_times = {}
    chunk_events_by_role = {}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("trial request must be an object")
        role = request.get("request_role")
        if (
            not isinstance(role, str)
            or role not in expected_roles
            or role in requests_by_role
        ):
            raise ValueError("trial request roles do not match the condition")
        requests_by_role[role] = request
        expected_input = 256 if role == "decode" else prefill_tokens
        expected_output = 128 if role == "decode" else 1
        observed_expected_input = _report_token_count(
            request.get("expected_input_tokens"), f"{role} expected_input_tokens"
        )
        observed_expected_output = _report_token_count(
            request.get("expected_output_tokens"), f"{role} expected_output_tokens"
        )
        observed_input = _report_token_count(
            request.get("input_tokens"), f"{role} input_tokens", allow_none=True
        )
        observed_output = _report_token_count(
            request.get("output_tokens"), f"{role} output_tokens", allow_none=True
        )
        send_ns = request.get("request_send_monotonic_ns")
        if send_ns is not None:
            _report_monotonic_ns(send_ns, f"{role} request_send_monotonic_ns")
        drain_ns = request.get("stream_drain_monotonic_ns")
        if drain_ns is not None:
            _report_monotonic_ns(drain_ns, f"{role} stream_drain_monotonic_ns")
        first_chunk_ns = request.get("first_chunk_monotonic_ns")
        if first_chunk_ns is not None:
            _report_monotonic_ns(first_chunk_ns, f"{role} first_chunk_monotonic_ns")
        if send_ns is not None and drain_ns is not None and drain_ns <= send_ns:
            raise ValueError(f"{role} request lifetime must be positive")
        request_times[role] = (send_ns, drain_ns)
        if observed_expected_input != expected_input:
            raise ValueError(f"{role} expected_input_tokens do not match the condition")
        if observed_expected_output != expected_output:
            raise ValueError(
                f"{role} expected_output_tokens do not match the condition"
            )
        status = request.get("status")
        if status != "valid" and status != "invalid":
            raise ValueError(f"{role} request status is invalid")
        invalid_reason = request.get("invalid_reason")
        if status == "invalid" and (
            not isinstance(invalid_reason, str) or not invalid_reason.strip()
        ):
            raise ValueError(f"{role} invalid request has no invalid_reason")
        if status == "valid":
            if observed_input != expected_input:
                raise ValueError(f"valid {role} input token count does not match")
            if observed_output != expected_output:
                raise ValueError(f"valid {role} output token count does not match")
            if request.get("stream_completed") is not True:
                raise ValueError(f"valid {role} stream did not complete")
            if send_ns is None:
                raise ValueError(f"valid {role} request has no send timestamp")
            if drain_ns is None:
                raise ValueError(f"valid {role} request has no drain timestamp")
            if first_chunk_ns is None:
                raise ValueError(f"valid {role} request has no first-chunk timestamp")
            if not send_ns <= first_chunk_ns <= drain_ns:
                raise ValueError(
                    f"valid {role} first-chunk timestamp is outside the request interval"
                )
            chunk_events = _report_request_chunk_events(
                request,
                role=role,
                expected_output_tokens=expected_output,
                send_ns=send_ns,
                drain_ns=drain_ns,
            )
            first_chunk_event = chunk_events[0]
            first_chunk_epoch_ns = request.get("first_chunk_epoch_ns")
            if (
                first_chunk_event["monotonic_ns"] != first_chunk_ns
                or type(first_chunk_epoch_ns) is not int
                or first_chunk_epoch_ns != first_chunk_event["epoch_ns"]
                or request.get("first_chunk_field") != first_chunk_event["field"]
            ):
                raise ValueError(
                    f"valid {role} first-chunk metadata does not match its first chunk event"
                )
            chunk_events_by_role[role] = chunk_events
            _report_validate_timing_metric(
                request.get("ttft_ms"),
                f"{role} ttft_ms",
                start_ns=send_ns,
                end_ns=first_chunk_ns,
            )
            _report_validate_timing_metric(
                request.get("e2e_ms"),
                f"{role} e2e_ms",
                start_ns=send_ns,
                end_ns=drain_ns,
            )
        for field in ("ttft_ms", "e2e_ms"):
            value = request.get(field)
            if value is not None:
                _report_positive_number(value, f"{role} {field}")

    trial_status = trial.get("status")
    if trial_status != "valid" and trial_status != "invalid":
        raise ValueError("trial status must be valid or invalid")
    invalid_reason = trial.get("invalid_reason")
    if trial_status == "invalid" and (
        not isinstance(invalid_reason, str) or not invalid_reason.strip()
    ):
        raise ValueError("invalid trial has no invalid_reason")
    if trial_status == "valid":
        if set(requests_by_role) != expected_roles:
            raise ValueError("valid trial is missing a request")
        if any(request["status"] != "valid" for request in requests_by_role.values()):
            raise ValueError("valid trial contains an invalid request")
        for request in requests_by_role.values():
            for field in ("ttft_ms", "e2e_ms"):
                _report_positive_number(request.get(field), f"request {field}")

    staged = condition.startswith("staged-")
    overlap = trial.get("requests_overlap")
    if staged:
        if overlap is not True and overlap is not False and overlap is not None:
            raise ValueError("staged trial requests_overlap must be boolean or null")
        decode_send_ns, decode_drain_ns = request_times.get("decode", (None, None))
        prefill_send_ns, prefill_drain_ns = request_times.get("prefill", (None, None))
        request_times_available = all(
            value is not None
            for value in (
                decode_send_ns,
                decode_drain_ns,
                prefill_send_ns,
                prefill_drain_ns,
            )
        )
        if request_times_available:
            actual_overlap = (
                decode_send_ns < prefill_drain_ns and prefill_send_ns < decode_drain_ns
            )
            if overlap is not actual_overlap:
                raise ValueError(
                    "staged trial requests_overlap does not match request timestamps"
                )
            if trial_status == "valid" and not actual_overlap:
                raise ValueError("valid staged request intervals must overlap")
        elif trial_status == "valid" or overlap is not None:
            raise ValueError(
                "staged request overlap cannot be verified from timestamps"
            )

        decode_request = requests_by_role.get("decode")
        if (
            decode_request is not None
            and decode_request["status"] == "valid"
            and prefill_send_ns is not None
        ):
            decode_events = chunk_events_by_role["decode"]
            if not decode_events:
                raise ValueError("staged Decode request has no first chunk")
            if prefill_send_ns < decode_events[0]["monotonic_ns"]:
                raise ValueError("staged Prefill send precedes the first Decode chunk")
    elif overlap is not None:
        raise ValueError("non-staged trial must not record request overlap")

    metrics: dict[str, float | None] = {}
    for role in ("decode", "prefill"):
        request = requests_by_role.get(role)
        for field, metric_name in (
            ("ttft_ms", f"{role}_ttft_ms"),
            ("e2e_ms", f"{role}_e2e_ms"),
        ):
            metrics[metric_name] = None if request is None else request.get(field)

    events = trial.get("chunk_gap_events", [])
    if not isinstance(events, list):
        raise ValueError("trial chunk_gap_events must be a list")
    if not staged and events:
        raise ValueError("non-staged trial must not contain chunk gap events")
    prefill_request = requests_by_role.get("prefill")
    send_ns = (
        None
        if prefill_request is None
        else prefill_request.get("request_send_monotonic_ns")
    )
    if staged and send_ns is not None:
        send_ns = _report_monotonic_ns(send_ns, "prefill request_send_monotonic_ns")
    if events and send_ns is None:
        raise ValueError("chunk gap events require a Prefill send timestamp")

    decode_request = requests_by_role.get("decode")
    decode_events = [] if decode_request is None else decode_request.get("chunk_events")
    if not isinstance(decode_events, list):
        if events:
            raise ValueError("chunk gap events require Decode chunk events")
        decode_events = []
    adjacent_decode_chunks = {}
    previous_chunk = None
    for index, chunk_event in enumerate(decode_events, start=1):
        if not isinstance(chunk_event, dict):
            raise ValueError("Decode chunk event must be an object")
        chunk_index = chunk_event.get("chunk_index")
        if type(chunk_index) is not int or chunk_index != index:
            raise ValueError("Decode chunk event indices must increase from one")
        chunk_time_ns = _report_monotonic_ns(
            chunk_event.get("monotonic_ns"), "Decode chunk event monotonic_ns"
        )
        if previous_chunk is not None:
            previous_index, previous_time_ns = previous_chunk
            if chunk_time_ns <= previous_time_ns:
                raise ValueError("Decode chunk event times must strictly increase")
            adjacent_decode_chunks[(previous_index, chunk_index)] = (
                previous_time_ns,
                chunk_time_ns,
            )
        previous_chunk = (chunk_index, chunk_time_ns)

    classified: dict[str, list[float]] = {"before": [], "after": []}
    crosses_count = 0
    seen_gap_pairs = set()
    require_gap_coverage = staged and trial_status == "valid"
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("chunk gap event must be an object")
        classification = event.get("classification")
        if classification not in ("before", "after", "crosses"):
            raise ValueError("chunk gap event classification is invalid")
        from_chunk_index = event.get("from_chunk_index")
        to_chunk_index = event.get("to_chunk_index")
        if type(from_chunk_index) is not int or type(to_chunk_index) is not int:
            raise ValueError("chunk gap indices must be integers")
        actual_endpoints = adjacent_decode_chunks.get(
            (from_chunk_index, to_chunk_index)
        )
        if actual_endpoints is None:
            raise ValueError("chunk gap indices are not adjacent decode chunks")
        gap_pair = (from_chunk_index, to_chunk_index)
        if require_gap_coverage and gap_pair in seen_gap_pairs:
            raise ValueError(
                "valid staged gap records must cover every adjacent Decode chunk pair exactly once"
            )
        seen_gap_pairs.add(gap_pair)
        start_ns = _report_monotonic_ns(
            event.get("from_monotonic_ns"), "chunk gap from_monotonic_ns"
        )
        end_ns = _report_monotonic_ns(
            event.get("to_monotonic_ns"), "chunk gap to_monotonic_ns"
        )
        if (start_ns, end_ns) != actual_endpoints:
            raise ValueError(
                "chunk gap endpoints do not match adjacent decode chunk events"
            )
        if end_ns <= start_ns:
            raise ValueError("chunk gap duration must be positive")
        gap = _report_positive_number(
            event.get("client_sse_chunk_gap_ms"),
            "client_sse_chunk_gap_ms",
        )
        duration_ms = (end_ns - start_ns) / 1_000_000
        if not math.isclose(gap, duration_ms, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "client_sse_chunk_gap_ms does not match monotonic duration"
            )
        expected_classification = (
            "before"
            if end_ns <= send_ns
            else "after" if start_ns >= send_ns else "crosses"
        )
        if classification != expected_classification:
            raise ValueError(
                "chunk gap classification does not match monotonic timestamps"
            )
        if trial_status != "valid":
            continue
        if classification == "crosses":
            crosses_count += 1
        else:
            classified[classification].append(gap)
    if require_gap_coverage and seen_gap_pairs != set(adjacent_decode_chunks):
        raise ValueError(
            "valid staged gap records must cover every adjacent Decode chunk pair exactly once"
        )
    metrics["client_sse_chunk_gap_before_median_ms"] = (
        median(classified["before"]) if classified["before"] else None
    )
    metrics["client_sse_chunk_gap_after_median_ms"] = (
        median(classified["after"]) if classified["after"] else None
    )
    metrics["client_sse_chunk_gap_crosses_count"] = crosses_count
    return metrics


def _report_load_trials(
    raw_dir: Path,
    run: dict,
    hashes_by_chunk: dict[int, str],
    served_model_name: str,
) -> list[dict]:
    records = _report_jsonl(_report_input_path(raw_dir / "trials.jsonl", raw_dir))
    if not records:
        raise ValueError("trials.jsonl contains no records")
    run_id = run["run_id"]
    model_id = run["model_id"]
    model_revision = run["model_revision"]
    seen_trial_ids = set()
    measurements = []
    for index, trial in enumerate(records, start=1):
        label = f"trials.jsonl record {index}"
        if type(trial.get("schema_version")) is not int or trial["schema_version"] != 1:
            raise ValueError(f"{label} schema_version must be 1")
        if trial.get("run_id") != run_id:
            raise ValueError(f"{label} run_id does not match run.json")
        if (
            trial.get("model_id") != model_id
            or trial.get("model_revision") != model_revision
        ):
            raise ValueError(f"{label} model or revision does not match run.json")
        trial_id = trial.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen_trial_ids:
            raise ValueError(f"{label} trial_id is missing or duplicated")
        seen_trial_ids.add(trial_id)
        condition = trial.get("condition")
        if condition not in _REPORT_CONDITIONS:
            if isinstance(condition, str) and condition.startswith("warmup-"):
                continue
            raise ValueError(f"{label} has an unsupported condition")
        chunk_size = trial.get("chunk_size")
        if type(chunk_size) is not int or chunk_size not in _CHUNK_SIZES:
            raise ValueError(f"{label} chunk_size is invalid")
        repetition = trial.get("repetition")
        if type(repetition) is not int or not 1 <= repetition <= 3:
            raise ValueError(f"{label} repetition must be an integer between 1 and 3")
        round_number = trial.get("round")
        if type(round_number) is not int or round_number != repetition:
            raise ValueError(f"{label} round must be an integer matching repetition")
        prefill_tokens = _report_expected_prefill_tokens(condition)
        if (
            _report_token_count(trial.get("prefill_tokens"), f"{label} prefill_tokens")
            != prefill_tokens
        ):
            raise ValueError(f"{label} prefill_tokens do not match the condition")
        expected_id = f"{condition}-r{repetition}-c{chunk_size}"
        if trial_id != expected_id:
            raise ValueError(f"{label} trial_id does not match its condition and round")
        if trial.get("pair_id") != f"{condition}-r{repetition}":
            raise ValueError(f"{label} pair_id does not match its condition and round")
        if trial.get("config_sha256") != hashes_by_chunk[chunk_size]:
            raise ValueError(f"{label} config digest does not match its chunk snapshot")
        if (
            not isinstance(trial.get("served_model_name"), str)
            or not trial["served_model_name"]
        ):
            raise ValueError(f"{label} served_model_name is missing")
        if trial["served_model_name"] != served_model_name:
            raise ValueError(
                f"{label} served_model_name does not match the config snapshot"
            )
        _report_request_metrics(
            trial,
            condition=condition,
            prefill_tokens=prefill_tokens,
        )
        measurements.append(trial)
    return measurements


def _report_trial_values(
    trial: dict | None, condition: str
) -> dict[str, float | int | None]:
    if trial is None or trial["status"] != "valid":
        return {metric: None for metric in _REPORT_METRICS} | {
            "client_sse_chunk_gap_crosses_count": None
        }
    return _report_request_metrics(
        trial,
        condition=condition,
        prefill_tokens=trial["prefill_tokens"],
    )


def _report_pairs(trials: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict[int, dict]] = {}
    for trial in trials:
        key = (trial["condition"], trial["repetition"])
        variants = grouped.setdefault(key, {})
        chunk_size = trial["chunk_size"]
        if chunk_size in variants:
            raise ValueError(
                f"duplicate {trial['condition']} repetition {trial['repetition']} "
                f"for chunk size {chunk_size}"
            )
        variants[chunk_size] = trial

    pairs = []
    for (condition, repetition), variants in sorted(
        grouped.items(),
        key=lambda item: (_REPORT_CONDITIONS.index(item[0][0]), item[0][1]),
    ):
        reasons = []
        for chunk_size in _CHUNK_SIZES:
            if chunk_size not in variants:
                reasons.append(f"chunk_size_{chunk_size}: missing sample")
            elif variants[chunk_size]["status"] != "valid":
                reason = variants[chunk_size].get("invalid_reason") or "invalid trial"
                reasons.append(f"chunk_size_{chunk_size}: {reason}")
        valid = not reasons
        values = {
            chunk_size: _report_trial_values(
                variants.get(chunk_size),
                condition,
            )
            for chunk_size in _CHUNK_SIZES
        }
        differences = {}
        if valid:
            for metric in _REPORT_METRICS:
                value_4096 = values[4096][metric]
                value_512 = values[512][metric]
                if value_4096 is not None and value_512 is not None:
                    differences[metric] = value_512 - value_4096
        pairs.append(
            {
                "condition": condition,
                "prefill_tokens": _report_expected_prefill_tokens(condition),
                "repetition": repetition,
                "status": "valid" if valid else "invalid",
                "invalid_reason": "; ".join(reasons) if reasons else "",
                "valid": valid,
                "values": values,
                "paired_differences": differences,
            }
        )

    for condition in _REPORT_CONDITIONS:
        condition_pairs = [
            pair for pair in pairs if pair["condition"] == condition and pair["valid"]
        ]
        for metric in _report_metric_names(condition):
            metric_n = sum(
                pair["values"][4096][metric] is not None
                and pair["values"][512][metric] is not None
                for pair in condition_pairs
            )
            if metric_n < 3:
                for pair in condition_pairs:
                    pair["paired_differences"].pop(metric, None)
    return pairs


def _report_csv_text(rows: list[dict], fieldnames: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    repr(value)
                    if isinstance(value, float)
                    else (
                        "True"
                        if value is True
                        else (
                            "False"
                            if value is False
                            else "" if value is None else value
                        )
                    )
                )
                for field, value in row.items()
            }
        )
    return stream.getvalue()


def _report_paired_rows(pairs: list[dict]) -> tuple[list[dict], tuple[str, ...]]:
    fields = [
        "condition",
        "prefill_tokens",
        "repetition",
        "status",
        "invalid_reason",
    ]
    for chunk_size in _CHUNK_SIZES:
        fields.extend(f"chunk_size_{chunk_size}_{metric}" for metric in _REPORT_METRICS)
        fields.append(f"chunk_size_{chunk_size}_client_sse_chunk_gap_crosses_count")
    fields.extend(
        f"paired_difference_{metric}_512_minus_4096" for metric in _REPORT_METRICS
    )

    rows = []
    for pair in pairs:
        row = {field: pair[field] for field in fields[:5]}
        for chunk_size in _CHUNK_SIZES:
            for metric, value in pair["values"][chunk_size].items():
                row[f"chunk_size_{chunk_size}_{metric}"] = value
        for metric in _REPORT_METRICS:
            row[f"paired_difference_{metric}_512_minus_4096"] = pair[
                "paired_differences"
            ].get(metric)
        rows.append(row)
    return rows, tuple(fields)


def _report_metric_names(condition: str) -> tuple[str, ...]:
    if condition == "decode-only":
        return ("decode_ttft_ms", "decode_e2e_ms")
    if condition.startswith("prefill-only-"):
        return ("prefill_ttft_ms", "prefill_e2e_ms")
    return (
        "decode_ttft_ms",
        "decode_e2e_ms",
        "prefill_ttft_ms",
        "prefill_e2e_ms",
        "client_sse_chunk_gap_before_median_ms",
        "client_sse_chunk_gap_after_median_ms",
    )


def _report_summary_rows(pairs: list[dict]) -> tuple[list[dict], tuple[str, ...]]:
    fields = (
        "condition",
        "prefill_tokens",
        "metric",
        "unit",
        "n_chunk_size_4096",
        "median_chunk_size_4096",
        "min_chunk_size_4096",
        "max_chunk_size_4096",
        "n_chunk_size_512",
        "median_chunk_size_512",
        "min_chunk_size_512",
        "max_chunk_size_512",
        "n_pairs",
        "paired_difference_direction",
        "paired_difference_median",
        "paired_difference_min",
        "paired_difference_max",
        "conclusion_boundary",
    )
    rows = []
    for condition in _REPORT_CONDITIONS:
        condition_pairs = [
            pair for pair in pairs if pair["condition"] == condition and pair["valid"]
        ]
        for metric in _report_metric_names(condition):
            values_by_chunk = {
                chunk_size: [
                    pair["values"][chunk_size][metric]
                    for pair in condition_pairs
                    if pair["values"][chunk_size][metric] is not None
                ]
                for chunk_size in _CHUNK_SIZES
            }
            differences = [
                pair["values"][512][metric] - pair["values"][4096][metric]
                for pair in condition_pairs
                if pair["values"][512][metric] is not None
                and pair["values"][4096][metric] is not None
            ]
            sufficient = len(differences) >= 3
            row = {
                "condition": condition,
                "prefill_tokens": _report_expected_prefill_tokens(condition),
                "metric": metric,
                "unit": "ms",
                "n_pairs": len(differences),
                "paired_difference_direction": "chunk_size_512_minus_4096",
                "paired_difference_median": median(differences) if sufficient else None,
                "paired_difference_min": min(differences) if sufficient else None,
                "paired_difference_max": max(differences) if sufficient else None,
                "conclusion_boundary": (
                    _REPORT_CONCLUSION_BOUNDARY
                    if sufficient
                    else f"数据不足（该指标有效配对 n={len(differences)}，至少需要3）；"
                    "不形成中位数、范围或配对差值比较。"
                    f"{_REPORT_CONCLUSION_BOUNDARY}"
                ),
            }
            for chunk_size in _CHUNK_SIZES:
                values = values_by_chunk[chunk_size]
                row[f"n_chunk_size_{chunk_size}"] = len(values)
                row[f"median_chunk_size_{chunk_size}"] = (
                    median(values) if sufficient else None
                )
                row[f"min_chunk_size_{chunk_size}"] = (
                    min(values) if sufficient else None
                )
                row[f"max_chunk_size_{chunk_size}"] = (
                    max(values) if sufficient else None
                )
            rows.append(row)
    return rows, fields


def _report_gaps_svg(pairs: list[dict]) -> str:
    ElementTree.register_namespace("", _REPORT_SVG_NS)
    width, height = 1000, 600
    left, right, top, bottom = 90, 40, 70, 120
    plot_bottom = height - bottom
    plot_width = width - left - right
    plot_height = plot_bottom - top
    categories = [
        (condition, phase)
        for condition in ("staged-2048", "staged-3584")
        for phase in ("before", "after")
    ]
    metric_for_phase = {
        "before": "client_sse_chunk_gap_before_median_ms",
        "after": "client_sse_chunk_gap_after_median_ms",
    }
    values = [
        pair["values"][chunk_size][metric_for_phase[phase]]
        for pair in pairs
        if pair["valid"] and pair["condition"].startswith("staged-")
        for chunk_size in _CHUNK_SIZES
        for phase in ("before", "after")
        if pair["values"][chunk_size][metric_for_phase[phase]] is not None
    ]
    max_value = max(values, default=1.0)
    y_max = max_value * 1.1

    def y_position(value: float) -> float:
        return plot_bottom - value / y_max * plot_height

    def x_position(category_index: int, chunk_size: int) -> float:
        center = left + plot_width * (category_index + 0.5) / len(categories)
        return center + (-12 if chunk_size == 4096 else 12)

    root = ElementTree.Element(
        f"{{{_REPORT_SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    ElementTree.SubElement(root, f"{{{_REPORT_SVG_NS}}}title").text = (
        "Prefill 前后客户端 SSE 块间隔"
    )
    ElementTree.SubElement(root, f"{{{_REPORT_SVG_NS}}}desc").text = (
        "展示每轮有效配对的客户端 SSE 块间隔原始点和中位数；跨越 Prefill 发送时刻的间隔不计入前后段。"
    )
    ElementTree.SubElement(
        root,
        f"{{{_REPORT_SVG_NS}}}text",
        {"x": str(left), "y": "36", "class": "chart-title"},
    ).text = "客户端 SSE 块间隔（毫秒）"
    for tick in range(5):
        value = y_max * tick / 4
        y = y_position(value)
        ElementTree.SubElement(
            root,
            f"{{{_REPORT_SVG_NS}}}line",
            {
                "x1": str(left),
                "x2": str(width - right),
                "y1": repr(y),
                "y2": repr(y),
                "class": "grid-line",
            },
        )
        ElementTree.SubElement(
            root,
            f"{{{_REPORT_SVG_NS}}}text",
            {
                "x": str(left - 12),
                "y": repr(y + 4),
                "text-anchor": "end",
                "class": "y-label",
            },
        ).text = f"{value:.1f}"

    colors = {4096: "#2563eb", 512: "#dc2626"}
    for category_index, (condition, phase) in enumerate(categories):
        center = left + plot_width * (category_index + 0.5) / len(categories)
        ElementTree.SubElement(
            root,
            f"{{{_REPORT_SVG_NS}}}text",
            {
                "x": repr(center),
                "y": str(plot_bottom + 26),
                "text-anchor": "middle",
                "class": "category-label",
            },
        ).text = f"{condition} / Prefill {phase}"
        metric = metric_for_phase[phase]
        paired_repetitions = {
            pair["repetition"]
            for pair in pairs
            if pair["valid"]
            and pair["condition"] == condition
            and pair["values"][4096][metric] is not None
            and pair["values"][512][metric] is not None
        }
        sufficient = len(paired_repetitions) >= 3
        for chunk_size in _CHUNK_SIZES:
            observations = [
                (pair["repetition"], pair["values"][chunk_size][metric])
                for pair in pairs
                if pair["valid"]
                and pair["condition"] == condition
                and pair["values"][chunk_size][metric] is not None
            ]
            x = x_position(category_index, chunk_size)
            for repetition, value in observations:
                ElementTree.SubElement(
                    root,
                    f"{{{_REPORT_SVG_NS}}}circle",
                    {
                        "cx": repr(x),
                        "cy": repr(y_position(value)),
                        "r": "5",
                        "fill": colors[chunk_size],
                        "class": "observation",
                        "data-condition": condition,
                        "data-phase": phase,
                        "data-chunk-size": str(chunk_size),
                        "data-repetition": str(repetition),
                        "data-value-ms": repr(value),
                    },
                )
            median_observations = [
                value
                for repetition, value in observations
                if repetition in paired_repetitions
            ]
            if sufficient:
                center_value = median(median_observations)
                y = y_position(center_value)
                ElementTree.SubElement(
                    root,
                    f"{{{_REPORT_SVG_NS}}}polygon",
                    {
                        "points": (f"{x},{y - 7} {x + 7},{y} {x},{y + 7} {x - 7},{y}"),
                        "fill": colors[chunk_size],
                        "class": "median",
                        "data-condition": condition,
                        "data-phase": phase,
                        "data-chunk-size": str(chunk_size),
                        "data-value-ms": repr(center_value),
                    },
                )
        if not sufficient:
            ElementTree.SubElement(
                root,
                f"{{{_REPORT_SVG_NS}}}text",
                {
                    "x": repr(center),
                    "y": str(plot_bottom + 55),
                    "text-anchor": "middle",
                    "class": "data-insufficient",
                    "data-condition": condition,
                    "data-phase": phase,
                    "data-n": str(len(paired_repetitions)),
                },
            ).text = f"数据不足（配对 n={len(paired_repetitions)}，不比较）"

    for index, chunk_size in enumerate(_CHUNK_SIZES):
        x = left + 15 + index * 180
        ElementTree.SubElement(
            root,
            f"{{{_REPORT_SVG_NS}}}circle",
            {
                "cx": str(x),
                "cy": str(height - 24),
                "r": "5",
                "fill": colors[chunk_size],
            },
        )
        ElementTree.SubElement(
            root,
            f"{{{_REPORT_SVG_NS}}}text",
            {"x": str(x + 10), "y": str(height - 20), "class": "legend-label"},
        ).text = f"chunked_prefill_size={chunk_size}"
    stream = io.StringIO()
    ElementTree.ElementTree(root).write(stream, encoding="unicode")
    return stream.getvalue()


def analyze_run(raw_dir: Path, output_dir: Path) -> list[Path]:
    """Analyze one project-local raw run and create paired CSV and SVG reports."""
    project_root = REPO_ROOT.resolve()

    def resolve_project_path(path: Path, label: str, *, strict: bool = False) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        resolved = candidate.resolve(strict=strict)
        try:
            resolved.relative_to(project_root)
        except ValueError:
            raise ValueError(f"{label} must stay inside the project") from None
        return resolved

    raw_path = resolve_project_path(raw_dir, "raw run directory", strict=True)
    if not raw_path.is_dir():
        raise ValueError("raw run path must be a directory")
    output_path = resolve_project_path(output_dir, "output directory")
    try:
        output_path.relative_to(raw_path)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the raw run directory")
    if output_path.exists():
        raise FileExistsError("output directory already exists")

    run, hashes_by_chunk, served_model_name = _report_validate_run(raw_path)
    trials = _report_load_trials(raw_path, run, hashes_by_chunk, served_model_name)
    pairs = _report_pairs(trials)
    paired_rows, paired_fields = _report_paired_rows(pairs)
    summary_rows, summary_fields = _report_summary_rows(pairs)
    artifacts = {
        "paired_trials.csv": _report_csv_text(paired_rows, paired_fields),
        "summary.csv": _report_csv_text(summary_rows, summary_fields),
        "decode_chunk_gaps.svg": _report_gaps_svg(pairs),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=".long-prefill-decode-", dir=output_path.parent)
    )
    try:
        for name, content in artifacts.items():
            path = staging_path / name
            with path.open("x", encoding="utf-8", newline="") as output:
                output.write(content)
        if output_path.exists():
            raise FileExistsError("output directory already exists")
        staging_path.rename(output_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise
    return [output_path / name for name in artifacts]
