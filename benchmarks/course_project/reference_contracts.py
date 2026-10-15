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


def reference_weight_dtypes(*, name: str, compute_dtype: str) -> tuple[str, str]:
    """Return (checkpoint projection, reference compute) dtypes for Qwen3.5-0.8B.

    These describe raw HF learned weights projected to engine-equivalent values,
    not every engine tensor's actual storage. Gemma's derived FP32 gamma is
    1 + raw checkpoint.float(); the HF raw weight must not have 1 pre-added.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    if compute_dtype not in ("float32", "bfloat16"):
        raise ValueError("compute_dtype must be float32 or bfloat16")
    projection_dtype = (
        "float32" if name.rsplit(".", 1)[-1] in ("A_log", "dt_bias") else "bfloat16"
    )
    return projection_dtype, (
        "float32" if compute_dtype == "float32" else projection_dtype
    )


def forward_windows(*, input_ids: list[int], continuation_ids: list[int]) -> list[dict]:
    """Return Prefill and 16 teacher-forced cached/recomputed forward windows.

    IDs must be nonnegative integers; the frozen manifest/collector checks the
    tokenizer's vocabulary upper bound. Each returned list is independent.
    """
    if not input_ids:
        raise ValueError("input_ids must contain a nonempty prompt")
    if len(continuation_ids) != 16:
        raise ValueError("continuation_ids must contain exactly 16 IDs")
    if any(
        type(token_id) is not int or token_id < 0
        for token_id in [*input_ids, *continuation_ids]
    ):
        raise ValueError("IDs must be nonnegative integers excluding bool")
    length = len(input_ids)
    return [
        {
            "step": step,
            "prefix_length": length + step,
            "cached_input_ids": (
                input_ids.copy() if step == 0 else [continuation_ids[step - 1]]
            ),
            "cached_position_ids": (
                list(range(length)) if step == 0 else [length + step - 1]
            ),
            "recomputed_input_ids": input_ids + continuation_ids[:step],
            "recomputed_position_ids": list(range(length + step)),
        }
        for step in range(17)
    ]
