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

from collections.abc import Iterable
from dataclasses import dataclass

from benchmarks.course_project.core import ChatResult, TokenSpeedGenerateResult


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    category: str
    turns: list[str]


def build_correctness_cases() -> list[CorrectnessCase]:
    long_context = "".join(
        f"第{index}段：北京拥有悠久历史、现代交通网络和丰富的文化遗产。"
        for index in range(1, 61)
    )
    return [
        CorrectnessCase(
            name="chinese_beijing",
            category="chinese",
            turns=["你好，请用一句话介绍北京。"],
        ),
        CorrectnessCase(
            name="english_capital",
            category="english",
            turns=["What is the capital of France? Answer in one short sentence."],
        ),
        CorrectnessCase(
            name="math_arithmetic",
            category="math",
            turns=["计算 37 × 19，并只给出结果。"],
        ),
        CorrectnessCase(
            name="code_python",
            category="code",
            turns=[
                "Write a Python function that returns the sum of a list of integers."
            ],
        ),
        CorrectnessCase(
            name="long_context_recall",
            category="long_context",
            turns=[
                f"阅读以下材料：\n{long_context}\n请概括材料反复描述的城市及三个特点。"
            ],
        ),
        CorrectnessCase(
            name="multi_turn_memory",
            category="multi_turn",
            turns=[
                "请记住项目代号是天枢，不要解释。",
                "这个项目使用 Ascend 910C。请同时记住硬件型号。",
                "请回答项目代号和硬件型号。",
            ],
        ),
    ]


def make_correctness_record(
    *,
    case: CorrectnessCase,
    turn_index: int,
    input_token_ids: list[int],
    retokenized_output_ids: list[int],
    max_new_tokens: int,
    result: ChatResult | TokenSpeedGenerateResult,
    engine_output_token_ids: list[int] | None = None,
) -> dict:
    return {
        "case": case.name,
        "category": case.category,
        "turn_index": turn_index,
        "client_input_token_ids": input_token_ids,
        "input_token_ids_source": "client_chat_template",
        "output_token_ids": engine_output_token_ids,
        "retokenized_output_ids": retokenized_output_ids,
        "output_token_ids_source": (
            "tokenspeed_generate_output_ids"
            if engine_output_token_ids is not None
            else "unavailable_from_chat_response"
        ),
        "token_correctness_verified": False,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "output_text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "ttft_ms": getattr(result, "ttft_ms", None),
        "tpot_ms": getattr(result, "tpot_ms", None),
        "inter_chunk_latencies_ms": list(
            getattr(result, "inter_chunk_latencies_ms", ())
        ),
        "e2e_ms": result.e2e_ms,
    }


def compare_token_ids(engine_ids: list[int], reference_ids: list[int]) -> dict:
    mismatch_index = next(
        (
            index
            for index, (engine_id, reference_id) in enumerate(
                zip(engine_ids, reference_ids)
            )
            if engine_id != reference_id
        ),
        None,
    )
    if mismatch_index is None and len(engine_ids) != len(reference_ids):
        mismatch_index = min(len(engine_ids), len(reference_ids))
    return {
        "exact_match": mismatch_index is None,
        "engine_token_count": len(engine_ids),
        "reference_token_count": len(reference_ids),
        "first_mismatch_index": mismatch_index,
    }


def collect_stop_token_ids(*values: int | Iterable[int] | None) -> list[int]:
    stop_ids: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            stop_ids.add(value)
        else:
            stop_ids.update(int(token_id) for token_id in value)
    return sorted(stop_ids)


def truncate_at_stop_token(
    token_ids: list[int], stop_token_ids: Iterable[int]
) -> list[int]:
    stop_ids = set(stop_token_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[:index]
    return token_ids[:]


def compare_token_ids_with_stop_alignment(
    engine_ids: list[int], reference_ids: list[int], stop_token_ids: Iterable[int]
) -> dict:
    raw = compare_token_ids(engine_ids, reference_ids)
    aligned = compare_token_ids(
        truncate_at_stop_token(engine_ids, stop_token_ids),
        truncate_at_stop_token(reference_ids, stop_token_ids),
    )
    return {
        **raw,
        "stop_aligned_exact_match": aligned["exact_match"],
        "stop_aligned_engine_token_count": aligned["engine_token_count"],
        "stop_aligned_reference_token_count": aligned["reference_token_count"],
        "stop_aligned_first_mismatch_index": aligned["first_mismatch_index"],
    }


def build_reference_attention_mask(input_ids: list[int]) -> list[int]:
    return [1] * len(input_ids)


def run_decode_diagnostic(
    *,
    client,
    input_token_ids: list[int],
    max_new_tokens: int,
    mode: str,
    stop_token_ids: Iterable[int] = (),
) -> tuple[list[int], int]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if mode == "continuous":
        result = client.generate(
            input_token_ids=input_token_ids,
            max_tokens=max_new_tokens,
        )
        return list(result.output_ids), 1
    if mode != "full_recompute":
        raise ValueError(f"unsupported decode diagnostic mode: {mode}")

    generated_ids: list[int] = []
    stop_ids = set(stop_token_ids)
    request_count = 0
    for _ in range(max_new_tokens):
        result = client.generate(
            input_token_ids=input_token_ids + generated_ids,
            max_tokens=1,
        )
        request_count += 1
        step_ids = list(result.output_ids)
        if len(step_ids) > 1:
            raise ValueError("one-token diagnostic request returned multiple tokens")
        if not step_ids:
            break
        generated_ids.append(step_ids[0])
        if step_ids[0] in stop_ids:
            break
    return generated_ids, request_count
