"""0.8B geometry, chunk boundaries, and interleaved recurrent state checks."""

import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")

OUTPUT_ATOL, OUTPUT_RTOL = 2e-3, 2e-2
STATE_ATOL, STATE_RTOL = 3e-4, 1e-2


def _step(q, k, v, state, decay, beta):
    q, k, v = q.float(), k.float(), v.float()
    q_norm, k_norm = q.square().sum(-1, keepdim=True), k.square().sum(-1, keepdim=True)
    q_norm, k_norm = q_norm + 1e-6, k_norm + 1e-6
    q, k = q * q_norm.rsqrt(), k * k_norm.rsqrt()
    decayed = state * decay.exp()[:, None, None]
    delta = beta[:, None] * (v - torch.einsum("hk,hkv->hv", k, decayed))
    updated = decayed + k[:, :, None] * delta[:, None, :]
    return torch.einsum("hk,hkv->hv", q, updated) * 128**-0.5, updated


def _check(actual, expected, *, state, label, record_property):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all()
    difference = actual - expected
    relative_l2 = float(difference.norm() / expected.norm().clamp_min(1e-12))
    record_property(
        label,
        json.dumps(
            {"max_abs": float(difference.abs().max()), "relative_l2": relative_l2}
        ),
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=STATE_ATOL if state else OUTPUT_ATOL,
        rtol=STATE_RTOL if state else OUTPUT_RTOL,
    )


@pytest.mark.parametrize("ratio", [0, 0.5, 1, 2])
def test_prefill_reference_norm_epsilon(ratio: float) -> None:
    amplitude = (ratio * 1e-6 / 128) ** 0.5
    q = torch.full((16, 128), amplitude, dtype=torch.bfloat16)
    k, v = q.clone(), torch.ones_like(q)
    state = torch.zeros(16, 128, 128, dtype=torch.float32)
    decay = torch.zeros(16, dtype=torch.float32)
    beta = torch.full((16,), 0.5, dtype=torch.float32)
    output, state = _step(q, k, v, state, decay, beta)
    q64, k64 = q.double(), k.double()
    q64 = q64 / (q64.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k64 = k64 / (k64.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    expected_state = k64[:, :, None] * (0.5 * v.double())[:, None, :]
    expected_output = (q64[:, :, None] * expected_state).sum(1) * 128**-0.5
    torch.testing.assert_close(output.double(), expected_output, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(state.double(), expected_state, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("ratio", [0, 0.5, 1, 2])
def test_real_heads_prefill_norm_epsilon(ratio: float, record_property) -> None:
    amplitude = (ratio * 1e-6 / 128) ** 0.5
    q = torch.full((1, 1, 16, 128), amplitude, dtype=torch.bfloat16)
    k, v = q.clone(), torch.ones_like(q)
    g = torch.zeros(1, 1, 16, dtype=torch.float32)
    beta = torch.full((1, 1, 16), 0.5, dtype=torch.float32)
    initial = torch.zeros(1, 16, 128, 128, dtype=torch.float32)
    output, state = _step(
        q[0, 0], k[0, 0], v[0, 0], initial[0].transpose(-1, -2), g[0, 0], beta[0, 0]
    )
    actual = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=128**-0.5,
        initial_state=initial.npu(),
        cu_seqlens=torch.tensor([0, 1], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    _check(
        actual.out,
        output.reshape(1, 1, 16, 128).to(torch.bfloat16),
        state=False,
        label="epsilon_output",
        record_property=record_property,
    )
    _check(
        actual.final_state,
        state.transpose(-1, -2).unsqueeze(0),
        state=True,
        label="epsilon_state",
        record_property=record_property,
    )


@pytest.mark.parametrize("length", [1, 8, 127, 128, 129])
def test_real_heads_prefill_boundaries(length, record_property):
    generator = torch.Generator().manual_seed(113 + length)
    q, k, v = [
        torch.randn(1, length, 16, 128, generator=generator).to(torch.bfloat16)
        for _ in range(3)
    ]
    g = F.logsigmoid(torch.randn(1, length, 16, generator=generator))
    beta = torch.randn(1, length, 16, generator=generator).sigmoid()
    initial = torch.randn(1, 16, 128, 128, generator=generator) * 0.01
    state = initial[0].transpose(-1, -2).clone()
    outputs = []
    for index in range(length):
        output, state = _step(
            q[0, index], k[0, index], v[0, index], state, g[0, index], beta[0, index]
        )
        outputs.append(output)
    actual = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=128**-0.5,
        initial_state=initial.npu(),
        cu_seqlens=torch.tensor([0, length], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    _check(
        actual.out,
        torch.stack(outputs).unsqueeze(0).to(torch.bfloat16),
        state=False,
        label="output_error",
        record_property=record_property,
    )
    _check(
        actual.final_state,
        state.transpose(-1, -2).unsqueeze(0),
        state=True,
        label="state_error",
        record_property=record_property,
    )


def test_multistep_decode_interleaved_slots(record_property):
    generator = torch.Generator().manual_seed(241)
    original = torch.randn(5, 16, 128, 128, generator=generator) * 0.01
    pool = original.npu()
    references = {slot: original[slot].transpose(-1, -2).clone() for slot in (1, 3)}
    A_log = torch.randn(16, generator=generator)
    dt_bias = torch.randn(16, generator=generator)
    for step in range(8):
        for slot in (1, 3):
            q, k, v = [
                torch.randn(1, 1, 16, 128, generator=generator).to(torch.bfloat16)
                for _ in range(3)
            ]
            a, b = [torch.randn(1, 1, 16, generator=generator) for _ in range(2)]
            decay = -A_log.exp() * F.softplus(a.reshape(16) + dt_bias)
            output, references[slot] = _step(
                q[0, 0],
                k[0, 0],
                v[0, 0],
                references[slot],
                decay,
                b.reshape(16).sigmoid(),
            )
            before = pool.cpu().clone()
            actual = gdn_decode_step(
                q.npu(),
                k.npu(),
                v.npu(),
                A_log=A_log.npu(),
                a=a.npu(),
                dt_bias=dt_bias.npu(),
                b=b.npu(),
                initial_state=pool,
                initial_state_indices=torch.tensor(
                    [slot], device="npu", dtype=torch.int32
                ),
                scale=128**-0.5,
                solution="triton",
            )
            after = pool.cpu()
            other = [index for index in range(5) if index != slot]
            torch.testing.assert_close(after[other], before[other], atol=0, rtol=0)
            _check(
                actual.reshape(16, 128),
                output.to(torch.bfloat16),
                state=False,
                label=f"output_{step}_{slot}",
                record_property=record_property,
            )
            _check(
                after[slot],
                references[slot].transpose(-1, -2),
                state=True,
                label=f"state_{step}_{slot}",
                record_property=record_property,
            )


def test_decode_separate_read_write_slots(record_property):
    generator = torch.Generator().manual_seed(347)
    original = torch.randn(5, 16, 128, 128, generator=generator) * 0.01
    pool = original.npu()
    q, k, v = [
        torch.randn(1, 1, 16, 128, generator=generator).to(torch.bfloat16)
        for _ in range(3)
    ]
    A_log, dt_bias = [torch.randn(16, generator=generator) for _ in range(2)]
    a, b = [torch.randn(1, 1, 16, generator=generator) for _ in range(2)]
    decay = -A_log.exp() * F.softplus(a.reshape(16) + dt_bias)
    expected, state = _step(
        q[0, 0],
        k[0, 0],
        v[0, 0],
        original[1].transpose(-1, -2),
        decay,
        b.reshape(16).sigmoid(),
    )
    actual = gdn_decode_step(
        q.npu(),
        k.npu(),
        v.npu(),
        A_log=A_log.npu(),
        a=a.npu(),
        dt_bias=dt_bias.npu(),
        b=b.npu(),
        initial_state=pool,
        initial_state_indices=torch.tensor([1], device="npu", dtype=torch.int32),
        output_state_indices=torch.tensor([3], device="npu", dtype=torch.int32),
        scale=128**-0.5,
        solution="triton",
    )
    after = pool.cpu()
    torch.testing.assert_close(
        after[[0, 1, 2, 4]], original[[0, 1, 2, 4]], atol=0, rtol=0
    )
    _check(
        actual.reshape(16, 128),
        expected.to(torch.bfloat16),
        state=False,
        label="dual_index_output",
        record_property=record_property,
    )
    _check(
        after[3],
        state.transpose(-1, -2),
        state=True,
        label="dual_index_state",
        record_property=record_property,
    )
