from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float()
    expected_f = expected.float()
    return float((actual_f - expected_f).norm() / expected_f.norm().clamp_min(1e-30))


def _require_npu() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    return torch.device("npu:0")


@pytest.mark.parametrize("seq_len", [19, 32])
def test_npu_chunk_prefill_tracks_decode_for_full_and_partial_chunks(seq_len: int):
    device = _require_npu()
    torch.manual_seed(20261010 + seq_len)
    q_heads = value_heads = 2
    head_dim = 128
    q = torch.randn(1, seq_len, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = (
        torch.randn(
            1, seq_len, value_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.5
    )
    a = torch.randn(1, seq_len, value_heads, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(value_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn_like(A_log) * 0.1
    initial_state = (
        torch.randn(
            1, value_heads, head_dim, head_dim, device=device, dtype=torch.float32
        )
        * 0.02
    )

    x = a.float() + dt_bias
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = (-torch.exp(A_log) * softplus).float()
    beta = torch.sigmoid(b.float())
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    prefill = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=head_dim**-0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )

    state_pool = torch.zeros(
        2, value_heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    state_pool[0].copy_(initial_state[0])
    read_index = torch.tensor([0], device=device, dtype=torch.int32)
    write_index = torch.tensor([1], device=device, dtype=torch.int32)
    decode_outputs = []
    for token in range(seq_len):
        decode_outputs.append(
            gdn_decode_step(
                q[:, token : token + 1],
                k[:, token : token + 1],
                v[:, token : token + 1],
                A_log=A_log,
                a=a[:, token : token + 1],
                dt_bias=dt_bias,
                b=b[:, token : token + 1],
                initial_state=state_pool,
                initial_state_indices=read_index,
                output_state_indices=write_index,
                scale=head_dim**-0.5,
                use_qk_l2norm=True,
                solution="triton",
            )
        )
        state_pool[0].copy_(state_pool[1])
    decode_output = torch.cat(decode_outputs, dim=1)

    output_error = _relative_l2(prefill.out, decode_output)
    state_error = _relative_l2(prefill.final_state, state_pool[:1])
    assert (
        output_error < 5e-4 and state_error < 2e-4
    ), f"output/state relative-L2 errors: {output_error:.6g}/{state_error:.6g}"
