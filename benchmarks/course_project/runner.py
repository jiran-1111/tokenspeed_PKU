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

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import fmean

from benchmarks.course_project.core import OpenAIChatClient

MOCK_TOOL_RESULT = (
    "Mock tool result: Search returned two relevant documents. Continue with the task."
)


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    messages: list[dict[str, str]]
    workload: str
    session_id: str | None = None
    turn_index: int | None = None
    tool_wait_ms: int = 0
    mock_tool_result_injected: bool = False


def build_agent_turn_request(
    *,
    request_id: str,
    session_id: str,
    turn_index: int,
    history: list[dict[str, str]],
    user_message: str,
    tool_wait_policy: str,
    tool_wait_ms: int,
) -> BenchmarkRequest:
    if tool_wait_policy not in {"all", "one-session"}:
        raise ValueError(f"Unsupported tool wait policy: {tool_wait_policy}")

    inject_mock_result = (
        turn_index > 1 and tool_wait_policy == "one-session" and session_id == "session-0"
    )
    waits = turn_index > 1 and (
        tool_wait_policy == "all" or session_id == "session-0"
    )
    content = user_message
    if inject_mock_result:
        content = f"{content}\n\n{MOCK_TOOL_RESULT}"
    return BenchmarkRequest(
        request_id=request_id,
        messages=[*history, {"role": "user", "content": content}],
        workload="agent",
        session_id=session_id,
        turn_index=turn_index,
        tool_wait_ms=tool_wait_ms if waits else 0,
        mock_tool_result_injected=inject_mock_result,
    )


def make_token_sized_text(
    *,
    target_tokens: int,
    seed: str,
    encode: Callable[[str], list],
    decode: Callable[[list], str],
) -> str:
    seed_tokens = encode(seed)
    if not seed_tokens:
        raise ValueError("seed must encode to at least one token")
    repeats = (target_tokens + len(seed_tokens) - 1) // len(seed_tokens)
    token_ids = encode(seed * repeats)[:target_tokens]
    return decode(token_ids)


def build_agent_session_contexts(
    *,
    concurrency: int,
    total_tokens: int,
    shared_tokens: int,
    repetition: int,
    encode: Callable[[str], list],
    decode: Callable[[list], str],
) -> dict[str, str]:
    shared_text = make_token_sized_text(
        target_tokens=shared_tokens,
        seed=f"R{repetition:06d} shared system policy and tool schema. ",
        encode=encode,
        decode=decode,
    )
    unique_tokens = total_tokens - shared_tokens
    return {
        f"session-{session_index}": shared_text
        + make_token_sized_text(
            target_tokens=unique_tokens,
            seed=f"S{session_index:06d}R{repetition:06d} private context. ",
            encode=encode,
            decode=decode,
        )
        for session_index in range(concurrency)
    }


def run_concurrent_requests(
    *,
    client: OpenAIChatClient,
    requests: list[BenchmarkRequest],
    concurrency: int,
    model: str,
    max_tokens: int,
) -> list[dict]:
    def execute(request: BenchmarkRequest) -> dict:
        if request.tool_wait_ms:
            time.sleep(request.tool_wait_ms / 1000)
        result = client.chat(
            model=model,
            messages=request.messages,
            max_tokens=max_tokens,
        )
        chunk_values = list(result.inter_chunk_latencies_ms)
        return {
            "request_id": request.request_id,
            "workload": request.workload,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "tool_wait_ms": request.tool_wait_ms,
            "mock_tool_result_injected": request.mock_tool_result_injected,
            "concurrency": concurrency,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "itl_p95_ms": None,
            "chunk_latency_p95_ms": _percentile(chunk_values, 0.95) if chunk_values else None,
            "inter_chunk_latencies_ms": chunk_values,
            "latency_observation": "nonempty_text_chunk_arrival",
            "e2e_ms": result.e2e_ms,
            "output_text": result.text,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(execute, requests))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_records(records: list[dict], *, wall_time_seconds: float) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty benchmark run")
    total_tokens = sum(
        record["input_tokens"] + record["output_tokens"] for record in records
    )
    chunk_values = [
        value
        for record in records
        for value in record.get("inter_chunk_latencies_ms", [])
    ]
    return {
        "request_count": len(records),
        "ttft_mean_ms": fmean(record["ttft_ms"] for record in records),
        "ttft_p50_ms": _percentile([record["ttft_ms"] for record in records], 0.50),
        "ttft_p95_ms": _percentile([record["ttft_ms"] for record in records], 0.95),
        "tpot_mean_ms": fmean(record["tpot_ms"] for record in records),
        "tpot_p50_ms": _percentile([record["tpot_ms"] for record in records], 0.50),
        "tpot_p95_ms": _percentile([record["tpot_ms"] for record in records], 0.95),
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "chunk_latency_p50_ms": _percentile(chunk_values, 0.50) if chunk_values else None,
        "chunk_latency_p95_ms": _percentile(chunk_values, 0.95) if chunk_values else None,
        "latency_observation": "nonempty_text_chunk_arrival",
        "e2e_mean_ms": fmean(record["e2e_ms"] for record in records),
        "e2e_p50_ms": _percentile([record["e2e_ms"] for record in records], 0.50),
        "e2e_p95_ms": _percentile([record["e2e_ms"] for record in records], 0.95),
        "request_throughput": len(records) / wall_time_seconds,
        "token_throughput": total_tokens / wall_time_seconds,
        "input_token_throughput": sum(record["input_tokens"] for record in records)
        / wall_time_seconds,
        "output_token_throughput": sum(record["output_tokens"] for record in records)
        / wall_time_seconds,
        "wall_time_seconds": wall_time_seconds,
    }
