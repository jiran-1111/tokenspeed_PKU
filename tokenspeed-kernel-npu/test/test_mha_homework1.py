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

"""Actual-model paged MHA boundaries with an independent CPU FP32 oracle."""

import math
from collections.abc import Callable
from typing import Literal

import pytest

torch = pytest.importorskip("torch")

# Frozen before the first run: the existing MHA contract, not the GDN contract.
ATOL = 0.02
RTOL = 0.02


def _paged_rows(
    cache: torch.Tensor, table: torch.Tensor, batch: int, length: int
) -> torch.Tensor:
    assert cache.device.type == table.device.type == "cpu"
    return cache[table[batch].long()].flatten(0, 1)[:length].float()


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    assert q.device.type == k.device.type == v.device.type == "cpu"
    q, k, v = q.float(), k.float(), v.float()
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        query_positions = torch.arange(q.shape[0]) + k.shape[0] - q.shape[0]
        future = torch.arange(k.shape[0])[None, :] > query_positions[:, None]
        scores.masked_fill_(future[None, :, :], -torch.inf)
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


def test_cpu_reference_page_order_and_truncation() -> None:
    cache = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2, 1, 1)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    torch.testing.assert_close(
        _paged_rows(cache, table, 0, 3).flatten(),
        torch.tensor([4.0, 5.0, 0.0]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _paged_rows(cache, table, 1, 3).flatten(),
        torch.tensor([6.0, 7.0, 2.0]),
        atol=0,
        rtol=0,
    )


def test_cpu_reference_bottom_right_causal_and_gqa() -> None:
    q = torch.zeros(2, 4, 1, dtype=torch.bfloat16)
    k = torch.zeros(3, 2, 1, dtype=torch.bfloat16)
    v = torch.tensor([[2, 20], [4, 40], [12, 120]], dtype=torch.bfloat16)[..., None]
    expected = torch.tensor([[3, 3, 30, 30], [6, 6, 60, 60]])[..., None].float()
    torch.testing.assert_close(_attention(q, k, v, causal=True), expected)
    torch.testing.assert_close(_attention(q[:1], k, v, causal=False), expected[1:2])


def test_cpu_reference_scaled_softmax() -> None:
    q = torch.tensor([[[2, 0, 0, 0]]], dtype=torch.bfloat16)
    k = torch.tensor([[[0, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=torch.bfloat16)
    v = torch.tensor([[[0, 0, 0, 0]], [[1, 1, 1, 1]]], dtype=torch.bfloat16)
    expected = torch.full((1, 1, 4), math.e / (1 + math.e))
    torch.testing.assert_close(_attention(q, k, v, causal=False), expected)


@pytest.mark.parametrize("length", [127, 128, 129])
@pytest.mark.parametrize("mode", ["extend", "decode"])
def test_mha_homework1_paged_boundary(
    length: int,
    mode: Literal["extend", "decode"],
    record_property: Callable[[str, object], None],
) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend MHA tests require an NPU")
    from tokenspeed_kernel_npu.ops.mha import (
        mha_decode_with_kvcache,
        mha_extend_with_kvcache,
    )

    generator = torch.Generator(device="cpu").manual_seed(311 + length)
    q_lengths = [2, 3] if mode == "extend" else [1, 1]
    q = torch.randn(sum(q_lengths), 8, 256, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 2, 256, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(4, 128, 2, 256, generator=generator, dtype=torch.bfloat16)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([length, length], dtype=torch.int32)
    cumulative_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], dtype=torch.int32)
    cumulative_kv = torch.tensor([0, length, 2 * length], dtype=torch.int32)
    expected_parts = []
    start = 0
    for batch, q_length in enumerate(q_lengths):
        expected_parts.append(
            _attention(
                q[start : start + q_length],
                _paged_rows(k, table, batch, length),
                _paged_rows(v, table, batch, length),
                causal=mode == "extend",
            )
        )
        start += q_length
    expected = torch.cat(expected_parts)

    inputs = [q, k, v, table, lengths, cumulative_q, cumulative_kv]
    device_inputs = [tensor.to("npu") for tensor in inputs]
    snapshots = [tensor.clone() for tensor in device_inputs]
    q_npu, k_npu, v_npu, table_npu, lengths_npu, cq_npu, ckv_npu = device_inputs
    options = dict(
        window_left=-1,
        logit_cap=0.0,
        sinks=None,
        return_lse=False,
        softmax_scale=1 / math.sqrt(256),
        q_scale=None,
        k_scale=None,
        v_scale=None,
        enable_pdl=False,
    )
    # The cache already includes the current tokens; these APIs only read it.
    if mode == "extend":
        output = mha_extend_with_kvcache(
            q_npu,
            cq_npu,
            ckv_npu,
            k_npu,
            v_npu,
            table_npu,
            lengths_npu,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=length,
            is_causal=True,
            **options,
        )
    else:
        output = mha_decode_with_kvcache(
            q_npu,
            k_npu,
            v_npu,
            table_npu,
            lengths_npu,
            max_seqlen_k=length,
            max_seqlen_q=1,
            **options,
        )
    actual = output.cpu().float()
    record_property("shape", str(tuple(actual.shape)))
    assert actual.shape == q.shape
    finite = bool(torch.isfinite(actual).all())
    difference = actual - expected
    record_property("max_abs", float(difference.abs().max()))
    record_property(
        "relative_l2", float(difference.norm() / expected.norm().clamp_min(1e-12))
    )
    record_property("finite", finite)
    assert finite
    for tensor, snapshot in zip(device_inputs, snapshots):
        assert torch.equal(tensor, snapshot), "MHA modified an input tensor"
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
