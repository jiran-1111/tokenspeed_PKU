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

import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_REQUIRED_MODEL_FIELDS = (
    "model_id",
    "served_model_name",
    "device",
    "tp_size",
    "max_model_len",
    "dtype",
    "max_num_seqs",
    "max_total_tokens",
    "chunked_prefill_size",
    "block_size",
    "memory_utilization",
    "graph_capture_sizes",
    "prefix_caching",
    "enable_metrics",
    "prometheus_port",
    "tokenspeed_metrics_port",
    "mixed_batch",
    "overlap_schedule",
    "enforce_eager",
    "model_revision",
    "mm_attention_backend",
)


@dataclass(frozen=True)
class AgentRequest:
    session_id: str
    turn_index: int
    messages: list[dict[str, str]]
    tool_wait_ms: int


@dataclass(frozen=True)
class StreamChunkEvent:
    chunk_index: int
    field: str
    epoch_ns: int
    monotonic_ns: int
    text_length: int


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    inter_chunk_latencies_ms: tuple[float, ...] = ()
    request_send_epoch_ns: int = 0
    request_send_monotonic_ns: int = 0
    first_chunk_epoch_ns: int | None = None
    first_chunk_monotonic_ns: int | None = None
    stream_drain_epoch_ns: int = 0
    stream_drain_monotonic_ns: int = 0
    first_chunk_field: str | None = None
    stream_completed: bool = False
    chunk_events: tuple[StreamChunkEvent, ...] = ()
    http_status: int | None = None


@dataclass(frozen=True)
class TokenSpeedGenerateResult:
    text: str
    output_ids: tuple[int, ...]
    input_tokens: int
    output_tokens: int
    e2e_ms: float


def load_model_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_model_config(config)
    return config


def model_load_path(config: dict) -> str:
    local_model_path = config.get("local_model_path")
    if local_model_path is None:
        return config["model_id"]

    path = Path(local_model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return str(path) if path.is_dir() else config["model_id"]


def chat_template_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Return token IDs for one chat template across tokenizer return formats."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def metrics_server_url(config: dict) -> str:
    return f"http://127.0.0.1:{config['tokenspeed_metrics_port']}"


def _validate_model_config(config: dict) -> None:
    for field in _REQUIRED_MODEL_FIELDS:
        if field not in config:
            raise ValueError(f"Missing required model config field: {field}")
    if "hf_overrides" in config and not isinstance(config["hf_overrides"], dict):
        raise ValueError("hf_overrides must be a dictionary")


def build_serve_command(config: dict, *, host: str, port: int) -> list[str]:
    _validate_model_config(config)
    graph_capture_sizes = [str(size) for size in config["graph_capture_sizes"]]
    command = [
        "python",
        "-m",
        "tokenspeed.cli",
        "serve",
        model_load_path(config),
        "--revision",
        config["model_revision"],
        "--served-model-name",
        config["served_model_name"],
        "--device",
        config["device"],
        "--dtype",
        config["dtype"],
        "--kv-cache-dtype",
        "auto",
        "--attention-backend",
        "mha",
        "--mm-attention-backend",
        config["mm_attention_backend"],
        "--sampling-backend",
        "greedy",
        "--disable-prefill-graph",
        "--disable-pdl",
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-total-tokens",
        str(config["max_total_tokens"]),
        "--chunked-prefill-size",
        str(config["chunked_prefill_size"]),
        "--prefix-granularity",
        str(config["block_size"]),
        "--max-cudagraph-capture-size",
        str(max(config["graph_capture_sizes"])),
        "--cudagraph-capture-sizes",
        *graph_capture_sizes,
        "--world-size",
        str(config["tp_size"]),
        "--gpu-memory-utilization",
        str(config["memory_utilization"]),
        "--disable-autotune",
    ]
    command.append(
        "--enable-prefix-caching"
        if config["prefix_caching"]
        else "--disable-prefix-caching"
    )
    if config["enable_metrics"]:
        command.append("--enable-metrics")
    if config.get("disable_kvstore", False):
        command.append("--disable-kvstore")
    if config["mixed_batch"]:
        command.append("--enable-mixed-batch")
    if not config["overlap_schedule"]:
        command.append("--disable-overlap-schedule")
    if config["enforce_eager"]:
        command.append("--enforce-eager")
    command.extend(["--host", host, "--port", str(port)])
    command.extend(["--prometheus-port", str(config["prometheus_port"])])
    command.extend(["--metrics-port", str(config["tokenspeed_metrics_port"])])
    if "hf_overrides" in config:
        command.extend(["--hf-overrides", json.dumps(config["hf_overrides"])])
    return command


def calculate_request_metrics(
    *,
    started_at: float,
    first_token_at: float,
    ended_at: float,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, float | int]:
    e2e_seconds = ended_at - started_at
    decode_intervals = max(output_tokens - 1, 1)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "e2e_ms": e2e_seconds * 1000,
        "tpot_ms": (ended_at - first_token_at) * 1000 / decode_intervals,
        "output_tokens_per_second": output_tokens / e2e_seconds,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        timeout_seconds: float = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        on_first_chunk: Callable[[str, int, int], None] | None = None,
        on_chunk: Callable[[StreamChunkEvent], None] | None = None,
        on_request_send: Callable[[int, int], None] | None = None,
        ignore_eos: bool = False,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "do_sample": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started_at = time.perf_counter()
        first_token_at = None
        last_token_at = None
        inter_chunk_latencies_ms = []
        chunk_events = []
        text_parts = []
        input_tokens = 0
        output_tokens = 0
        first_chunk_epoch_ns = None
        first_chunk_monotonic_ns = None
        first_chunk_field = None
        stream_completed = False
        request_send_epoch_ns = time.time_ns()
        request_send_monotonic_ns = time.monotonic_ns()
        if on_request_send is not None:
            on_request_send(request_send_epoch_ns, request_send_monotonic_ns)
        http_status = None
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            response_status = response.status
            if type(response_status) is int:
                http_status = response_status
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    stream_completed = True
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    for field in ("content", "reasoning_content"):
                        piece = delta.get(field)
                        if not piece:
                            continue
                        token_at = time.perf_counter()
                        chunk_event = StreamChunkEvent(
                            chunk_index=len(chunk_events) + 1,
                            field=field,
                            epoch_ns=time.time_ns(),
                            monotonic_ns=time.monotonic_ns(),
                            text_length=len(piece),
                        )
                        chunk_events.append(chunk_event)
                        if first_token_at is None:
                            first_token_at = token_at
                            first_chunk_epoch_ns = chunk_event.epoch_ns
                            first_chunk_monotonic_ns = chunk_event.monotonic_ns
                            first_chunk_field = field
                            if on_first_chunk is not None:
                                on_first_chunk(
                                    first_chunk_field,
                                    first_chunk_epoch_ns,
                                    first_chunk_monotonic_ns,
                                )
                        if on_chunk is not None:
                            on_chunk(chunk_event)
                        if last_token_at is not None:
                            inter_chunk_latencies_ms.append(
                                (token_at - last_token_at) * 1000
                            )
                        last_token_at = token_at
                        text_parts.append(piece)
        ended_at = time.perf_counter()
        stream_drain_epoch_ns = time.time_ns()
        stream_drain_monotonic_ns = time.monotonic_ns()
        if first_token_at is None:
            first_token_at = ended_at
        metrics = calculate_request_metrics(
            started_at=started_at,
            first_token_at=first_token_at,
            ended_at=ended_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return ChatResult(
            text="".join(text_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=float(metrics["ttft_ms"]),
            e2e_ms=float(metrics["e2e_ms"]),
            tpot_ms=float(metrics["tpot_ms"]),
            inter_chunk_latencies_ms=tuple(inter_chunk_latencies_ms),
            request_send_epoch_ns=request_send_epoch_ns,
            request_send_monotonic_ns=request_send_monotonic_ns,
            first_chunk_epoch_ns=first_chunk_epoch_ns,
            first_chunk_monotonic_ns=first_chunk_monotonic_ns,
            stream_drain_epoch_ns=stream_drain_epoch_ns,
            stream_drain_monotonic_ns=stream_drain_monotonic_ns,
            first_chunk_field=first_chunk_field,
            stream_completed=stream_completed,
            chunk_events=tuple(chunk_events),
            http_status=http_status,
        )


class TokenSpeedGenerateClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 600) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.hostname is None:
            raise ValueError("base_url must include a host")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port >= 65535:
            raise ValueError("base_url port leaves no room for the control server port")
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        self.generate_url = urlunsplit(
            parsed._replace(
                netloc=f"{host}:{port + 1}",
                path="/generate",
                query="",
                fragment="",
            )
        )
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        input_token_ids: list[int],
        max_tokens: int,
    ) -> TokenSpeedGenerateResult:
        payload = {
            "input_ids": input_token_ids,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
            "stream": False,
        }
        request = urllib.request.Request(
            self.generate_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started_at = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            output = json.load(response)
        ended_at = time.perf_counter()
        output_ids = output.get("output_ids")
        if not isinstance(output_ids, list) or not all(
            isinstance(token_id, int) for token_id in output_ids
        ):
            raise ValueError("TokenSpeed /generate response has no output_ids list")
        meta_info = output.get("meta_info") or {}
        return TokenSpeedGenerateResult(
            text=str(output.get("text", "")),
            output_ids=tuple(output_ids),
            input_tokens=int(meta_info.get("prompt_tokens", len(input_token_ids))),
            output_tokens=len(output_ids),
            e2e_ms=(ended_at - started_at) * 1000,
        )


def build_agent_conversation(
    *,
    session_id: str,
    system_prompt: str,
    tool_schema: str,
    user_turns: list[str],
    assistant_turns: list[str],
    tool_wait_ms: list[int],
) -> list[AgentRequest]:
    if len(tool_wait_ms) != len(user_turns):
        raise ValueError("tool_wait_ms must contain one value per user turn")
    if len(assistant_turns) > max(len(user_turns) - 1, 0):
        raise ValueError("assistant_turns may only provide history between user turns")

    system_content = f"{system_prompt}\n\nAvailable tools:\n{tool_schema}"
    history = [{"role": "system", "content": system_content}]
    requests = []
    for index, user_content in enumerate(user_turns):
        history.append({"role": "user", "content": user_content})
        requests.append(
            AgentRequest(
                session_id=session_id,
                turn_index=index + 1,
                messages=[message.copy() for message in history],
                tool_wait_ms=tool_wait_ms[index],
            )
        )
        if index < len(assistant_turns):
            history.append({"role": "assistant", "content": assistant_turns[index]})
    return requests
