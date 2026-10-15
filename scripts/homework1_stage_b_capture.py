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

"""Isolated Stage B capture; never modify logits to force continuation tokens."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any


def validate_frontier(
    case: dict, *, step: int, input_ids: list[int], positions: list[int]
) -> None:
    """Require the original prefill or the exact next fixed token and position."""
    if not 0 <= step <= 16:
        raise ValueError("Unexpected continuation step")
    expected_ids = (
        case["input_ids"] if step == 0 else [case["continuation_ids"][step - 1]]
    )
    expected_positions = (
        list(range(len(expected_ids)))
        if step == 0
        else [len(case["input_ids"]) + step - 1]
    )
    if input_ids != expected_ids or positions != expected_positions:
        raise ValueError(f"Frozen frontier mismatch at step {step}")


def install(
    runner_class: type, executor_class: type, *, torch: Any, arm_path: Path
) -> None:
    """Install process-local, opt-in capture and post-sampling token replacement."""
    original_forward = runner_class.forward
    original_sampling = executor_class._run_sampling
    case_id = None
    step = 0
    pending = None

    @wraps(original_forward)
    def forward(self, ctx, input_ids, positions, *args, **kwargs):
        nonlocal case_id, step, pending
        if not arm_path.exists():
            return original_forward(self, ctx, input_ids, positions, *args, **kwargs)
        case = json.loads(arm_path.read_text(encoding="utf-8"))
        if case["case_id"] != case_id:
            if pending is not None:
                raise ValueError("Previous forward was not sampled")
            case_id, step = case["case_id"], 0
        ids = input_ids.detach().cpu().reshape(-1).tolist()
        position_values = positions.detach().cpu()
        position_axes = position_values.tolist()
        if position_values.ndim == 2:
            if position_values.shape[0] != 3 or any(
                not torch.equal(position_values[0], position_values[axis])
                for axis in (1, 2)
            ):
                raise ValueError("Text MRoPE requires three identical position axes")
            pos = position_axes[0]
        elif position_values.ndim == 1:
            pos = position_axes
        else:
            raise ValueError("Unexpected position tensor rank")
        validate_frontier(case, step=step, input_ids=ids, positions=pos)
        result = original_forward(self, ctx, input_ids, positions, *args, **kwargs)
        logits = result.next_token_logits
        if tuple(logits.shape) != (1, 248320):
            raise ValueError("Expected one full vocabulary vector")
        vector = logits[0].detach().cpu().float().clone()
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("Nonfinite raw logits before NaN guard")
        greedy = int(vector.argmax().item())
        record = {
            "case_id": case_id,
            "step": step,
            "input_ids": ids,
            "positions": pos,
            "position_axes": position_axes,
            "prefix_length": len(case["input_ids"]) + step,
            "logits": vector,
            "raw_greedy_id": greedy,
        }
        path = Path(case["output"]) / f"step_{step:02d}.pt"
        with path.open("xb") as stream:
            torch.save(record, stream)
        pending = (case["continuation_ids"][step] if step < 16 else None, greedy, path)
        step += 1
        return result

    @wraps(original_sampling)
    def sampling(self, logits_output, sampling_info, ctx, candidates=None):
        nonlocal pending
        tokens, accepts = original_sampling(
            self, logits_output, sampling_info, ctx, candidates
        )
        if not arm_path.exists():
            return tokens, accepts
        if pending is None or tokens.numel() != 1 or accepts.numel() != 1:
            raise ValueError("Sampling did not match one captured vector")
        forced, greedy, path = pending
        if int(tokens.item()) != greedy or int(accepts.item()) != 1:
            raise ValueError("Original greedy sampler disagrees with raw logits")
        emitted = tokens.clone()
        if forced is not None:
            emitted.fill_(forced)
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "raw_greedy_id": greedy,
                    "emitted_id": int(emitted.item()),
                    "forced": forced is not None,
                },
                stream,
            )
        pending = None
        return emitted, accepts

    runner_class.forward = forward
    executor_class._run_sampling = sampling
