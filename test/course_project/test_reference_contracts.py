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

import pytest

from benchmarks.course_project.reference_contracts import (
    forward_windows,
    reference_weight_dtypes,
)


@pytest.mark.parametrize(
    "name", ["A_log", "dt_bias", "model.layers.0.linear_attn.A_log", "x.dt_bias"]
)
@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_decay_parameters_stay_float32(name: str, compute_dtype: str) -> None:
    assert reference_weight_dtypes(name=name, compute_dtype=compute_dtype) == (
        "float32",
        "float32",
    )


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "lm_head.weight",
        "x.A_log.weight",
        "x.dt_bias.weight",
    ],
)
@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_raw_norm_and_ordinary_weights_use_bfloat16_projection(
    name: str, compute_dtype: str
) -> None:
    assert reference_weight_dtypes(name=name, compute_dtype=compute_dtype) == (
        "bfloat16",
        compute_dtype,
    )


@pytest.mark.parametrize("name", ["", "   ", None])
def test_invalid_weight_name(name: str | None) -> None:
    with pytest.raises(ValueError):
        reference_weight_dtypes(name=name, compute_dtype="float32")


@pytest.mark.parametrize("compute_dtype", ["float16", None])
def test_invalid_compute_dtype(compute_dtype: str | None) -> None:
    with pytest.raises(ValueError):
        reference_weight_dtypes(name="lm_head.weight", compute_dtype=compute_dtype)


@pytest.mark.parametrize("length", [1, 8, 127, 128, 129, 512, 1024, 2048])
def test_windows_preserve_common_prefix_and_absolute_positions(length: int) -> None:
    prompt = list(range(length))
    suffix = list(range(3000, 3016))
    rows = forward_windows(input_ids=prompt, continuation_ids=suffix)
    assert len(rows) == 17
    for step, row in enumerate(rows):
        expected_prefix = [*prompt, *suffix[:step]]
        assert row == {
            "step": step,
            "prefix_length": length + step,
            "cached_input_ids": prompt if step == 0 else [suffix[step - 1]],
            "cached_position_ids": (
                list(range(length)) if step == 0 else [length + step - 1]
            ),
            "recomputed_input_ids": expected_prefix,
            "recomputed_position_ids": list(range(len(expected_prefix))),
        }
    assert rows[1]["cached_input_ids"] == [3000]
    assert rows[1]["cached_position_ids"] == [length]
    assert prompt == list(range(length))
    assert suffix == list(range(3000, 3016))


@pytest.mark.parametrize("input_ids", [[], [-1], [True], [1.0]])
def test_invalid_prompt_ids(input_ids: list[int]) -> None:
    with pytest.raises(ValueError):
        forward_windows(input_ids=input_ids, continuation_ids=list(range(16)))


@pytest.mark.parametrize(
    "continuation_ids",
    [
        [],
        list(range(15)),
        list(range(17)),
        [-1] + list(range(15)),
        [True] + list(range(15)),
        [1.0] + list(range(15)),
    ],
)
def test_invalid_continuation_ids(continuation_ids: list[int]) -> None:
    with pytest.raises(ValueError):
        forward_windows(input_ids=[0], continuation_ids=continuation_ids)


def test_original_lists_and_all_returned_lists_are_independent() -> None:
    prompt = [7, 8]
    suffix = list(range(16))
    rows = forward_windows(input_ids=prompt, continuation_ids=suffix)
    list_fields = [
        "cached_input_ids",
        "cached_position_ids",
        "recomputed_input_ids",
        "recomputed_position_ids",
    ]
    returned_lists = [row[field] for row in rows for field in list_fields]
    assert len({id(value) for value in returned_lists}) == len(returned_lists)
    assert all(value is not prompt and value is not suffix for value in returned_lists)
    prompt[0] = 999
    suffix[0] = 999
    assert rows[0]["cached_input_ids"] == [7, 8]
    assert rows[1]["cached_input_ids"] == [0]
    assert rows[1]["recomputed_input_ids"] == [7, 8, 0]
    for index, value in enumerate(returned_lists):
        snapshots = [other.copy() for other in returned_lists]
        value.append(123456)
        for other_index, other in enumerate(returned_lists):
            if other_index != index:
                assert other == snapshots[other_index]
    assert prompt == [999, 8]
    assert suffix == [999, *range(1, 16)]
