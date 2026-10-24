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

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.course_project.native_performance import parse_target_hbm


def validate_protocol(protocol: dict, *, max_model_len: int) -> None:
    """Reject incomplete or invalid matrix conditions before Engine startup."""
    for name in ("input_lengths", "concurrency"):
        values = protocol.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain distinct positive integers")
    for name in (
        "max_new_tokens",
        "warmup_requests",
        "requests_per_cell",
        "repetitions",
        "engine_port",
    ):
        value = protocol.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if protocol["engine_port"] > 65534:
        raise ValueError("engine_port must leave room for the adjacent port")
    if max(protocol["input_lengths"]) + protocol["max_new_tokens"] > max_model_len:
        raise ValueError("matrix exceeds model context capacity")
    if min(protocol["warmup_requests"], protocol["requests_per_cell"]) < max(
        protocol["concurrency"]
    ):
        raise ValueError("request count must cover the requested concurrency")
    interval = protocol.get("hbm_interval_seconds")
    if (
        type(interval) not in (int, float)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("HBM interval must be positive and finite")
    if not isinstance(protocol.get("seed_text"), str) or not protocol["seed_text"]:
        raise ValueError("seed_text must be nonempty")


def build_inputs(seed_ids: list[int], *, lengths: list[int]) -> dict[str, list[int]]:
    """Repeat locally encoded seed IDs and slice to exact input lengths."""
    if not seed_ids or any(type(token) is not int or token < 0 for token in seed_ids):
        raise ValueError("seed must contain nonnegative integer token IDs")
    if any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("input lengths must be positive integers")
    return {
        str(length): (seed_ids * math.ceil(length / len(seed_ids)))[:length]
        for length in lengths
    }


@contextmanager
def capture_profile(engine: Any, *, output_dir: Path, tp_size: int, record: dict):
    """Capture all batches separately from normal performance and always stop.

    A successful capture must export exactly one trace for each TP rank.
    """
    profile_dir = output_dir / "profile"
    capture = {"clock_alignment": "unverified", "controls": [], "traces": []}
    record["profile"] = capture

    def control(action: str) -> None:
        event = {
            "action": action,
            "before_perf_counter_ns": time.perf_counter_ns(),
            "before_epoch_ns": time.time_ns(),
            "acknowledged": False,
        }
        capture["controls"].append(event)
        try:
            operation = (
                engine.tokenizer_manager.start_profile(
                    output_dir=str(profile_dir),
                    activities=["CPU", "NPU", "MEM"],
                    with_stack=False,
                    record_shapes=True,
                    profile_id="hw2",
                )
                if action == "start"
                else engine.tokenizer_manager.stop_profile()
            )
            result = engine.llm.run(operation)
            if result is None or result.success is not True:
                raise RuntimeError(f"profile {action} was not acknowledged")
            event.update({"acknowledged": True, "message": result.message})
        except BaseException as error:
            event["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            event["after_perf_counter_ns"] = time.perf_counter_ns()
            event["after_epoch_ns"] = time.time_ns()

    try:
        control("start")
        yield
    finally:
        try:
            control("stop")
        finally:
            for path in sorted(profile_dir.rglob("*.trace.json")):
                match = re.fullmatch(r"hw2-TP(\d+)\.trace\.json", path.name)
                capture["traces"].append(
                    {
                        "path": path.relative_to(output_dir).as_posix(),
                        "rank": int(match.group(1)) if match else None,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    if {trace["rank"] for trace in capture["traces"]} != set(range(tp_size)) or len(
        capture["traces"]
    ) != tp_size:
        raise ValueError("profile traces must contain exactly one file per TP rank")


def measure_request(engine: Any, *, input_ids: list[int], max_new_tokens: int) -> dict:
    """Collect real stream chunks, retaining partial output and stream errors."""
    started = time.perf_counter_ns()
    output_ids, chunks, errors = [], [], []
    try:
        stream = engine.generate(
            input_ids=list(input_ids),
            sampling_params={
                "temperature": 0,
                "ignore_eos": True,
                "max_new_tokens": max_new_tokens,
                "stream_interval": 1,
            },
            session_id=None,
            stream=True,
        )
        for chunk in stream:
            observed = time.perf_counter_ns()
            ids = chunk.get("output_ids") if isinstance(chunk, Mapping) else None
            if not isinstance(ids, list) or any(
                type(token) is not int or token < 0 for token in ids
            ):
                # Continue draining: the synchronous Engine bridge owns a live request.
                errors.append("invalid stream output_ids")
                continue
            output_ids.extend(ids)
            if ids:
                chunks.append({"observed_ns": observed, "output_tokens": len(ids)})
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    ended = time.perf_counter_ns()
    if len(output_ids) != max_new_tokens:
        errors.append(f"expected {max_new_tokens} output tokens, got {len(output_ids)}")
    first = chunks[0]["observed_ns"] if chunks else None
    return {
        "input_ids": list(input_ids),
        "output_ids": output_ids,
        "started_ns": started,
        "finished_ns": ended,
        "ttft_ms": (first - started) / 1e6 if first is not None else None,
        "e2e_ms": (ended - started) / 1e6,
        "chunks": chunks,
        "inter_chunk_ms": [
            (right["observed_ns"] - left["observed_ns"]) / 1e6
            for left, right in zip(chunks, chunks[1:])
        ],
        "status": "failed" if errors else "passed",
        "error": "; ".join(errors) if errors else None,
    }


def run_matrix(
    engine: Any,
    *,
    protocol: dict,
    inputs: dict[str, list[int]],
    output_dir: Path,
    npu_id: int,
    chip_logic_ids: dict[int, int],
    query_hbm: Callable[[], str],
    record: dict,
) -> None:
    """Measure bounded concurrent batches with separate warmup and HBM sampling."""
    validate_protocol(
        protocol,
        max_model_len=max(protocol["input_lengths"]) + protocol["max_new_tokens"],
    )
    for length in protocol["input_lengths"]:
        if len(inputs[str(length)]) != length:
            raise ValueError("actual input length differs from the protocol")
    record.update({"batches": [], "hbm_samples": [], "sampler_errors": []})
    stopped, ready = threading.Event(), threading.Event()

    def sample_hbm() -> None:
        record["sampler_started_ns"] = time.perf_counter_ns()
        try:
            with (output_dir / "hbm_samples.jsonl").open("x", encoding="utf-8") as file:
                while not stopped.is_set():
                    started = time.perf_counter_ns()
                    raw = query_hbm()
                    raw_name = f'npu_smi_{len(record["hbm_samples"]):05d}.txt'
                    with (output_dir / raw_name).open("x", encoding="utf-8") as table:
                        table.write(raw)
                    chips = parse_target_hbm(
                        raw, npu_id=npu_id, chip_logic_ids=chip_logic_ids
                    )
                    sample = {
                        "started_ns": started,
                        "finished_ns": time.perf_counter_ns(),
                        "raw_table": raw_name,
                        "chips": chips,
                    }
                    record["hbm_samples"].append(sample)
                    file.write(json.dumps(sample) + "\n")
                    file.flush()
                    ready.set()
                    stopped.wait(protocol["hbm_interval_seconds"])
        except Exception as error:
            record["sampler_errors"].append(
                {
                    "observed_ns": time.perf_counter_ns(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            record["sampler_finished_ns"] = time.perf_counter_ns()
            ready.set()

    sampler = threading.Thread(target=sample_hbm, name="tp-matrix-hbm")
    sampler.start()
    try:
        if not ready.wait(15) or record["sampler_errors"]:
            raise ValueError("HBM sampler failed before measurement")
        with (output_dir / "requests.jsonl").open("x", encoding="utf-8") as requests:
            for length in protocol["input_lengths"]:
                for concurrency in protocol["concurrency"]:
                    conditions = [("warmup", 0, protocol["warmup_requests"])] + [
                        ("formal", index + 1, protocol["requests_per_cell"])
                        for index in range(protocol["repetitions"])
                    ]
                    for phase, repetition, count in conditions:
                        batch = {
                            "input_length": length,
                            "concurrency": concurrency,
                            "phase": phase,
                            "repetition": repetition,
                            "requests": 0,
                            "output_tokens": 0,
                            "status": "started",
                            "started_ns": time.perf_counter_ns(),
                        }
                        record["batches"].append(batch)
                        failed = False
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            futures = {
                                executor.submit(
                                    measure_request,
                                    engine,
                                    input_ids=inputs[str(length)],
                                    max_new_tokens=protocol["max_new_tokens"],
                                ): index
                                for index in range(count)
                            }
                            for future in as_completed(futures):
                                result = {
                                    **future.result(),
                                    "input_length": length,
                                    "concurrency": concurrency,
                                    "phase": phase,
                                    "repetition": repetition,
                                    "request_index": futures[future],
                                }
                                requests.write(json.dumps(result) + "\n")
                                requests.flush()
                                batch["requests"] += 1
                                batch["output_tokens"] += len(result["output_ids"])
                                failed |= result["status"] != "passed"
                        batch["finished_ns"] = time.perf_counter_ns()
                        batch["wall_ns"] = batch["finished_ns"] - batch["started_ns"]
                        batch["output_tokens_per_second"] = (
                            batch["output_tokens"] * 1e9 / batch["wall_ns"]
                        )
                        batch["status"] = "failed" if failed else "passed"
                        if failed or record["sampler_errors"]:
                            raise ValueError("matrix request or HBM sampling failed")
    finally:
        stopped.set()
        sampler.join(timeout=15)
        if sampler.is_alive():
            record["sampler_errors"].append({"error": "HBM sampler did not stop"})
        for batch in record["batches"]:
            contained = [
                sample
                for sample in record["hbm_samples"]
                if sample["started_ns"] >= batch["started_ns"]
                and sample["finished_ns"] <= batch.get("finished_ns", 0)
            ]
            batch["hbm_contained_samples"] = len(contained)
            batch["hbm_observed_peak_mb_per_chip"] = (
                {
                    chip: max(sample["chips"][chip]["used_mb"] for sample in contained)
                    for chip in chip_logic_ids
                }
                if contained
                else None
            )
    if record["sampler_errors"]:
        raise ValueError("HBM sampling failed")
