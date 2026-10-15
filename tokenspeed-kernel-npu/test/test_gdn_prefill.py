import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def test_gdn_chunk_prefill_triton_matches_reference():
    torch.manual_seed(17)
    dtype = torch.bfloat16
    seq_len = 8
    q = torch.randn(1, seq_len, 1, 128, device="npu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q) * 0.5
    g = F.logsigmoid(torch.randn(1, seq_len, 1, device="npu"))
    beta = torch.rand(1, seq_len, 1, device="npu", dtype=dtype).sigmoid()
    initial_state = torch.randn(1, 1, 128, 128, device="npu", dtype=dtype) * 0.01
    cu_seqlens = torch.tensor([0, seq_len], device="npu", dtype=torch.int32)

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=128**-0.5,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        solution="triton",
    )

    q_ref, k_ref, v_ref = q.float().cpu(), k.float().cpu(), v.float().cpu()
    g_ref, beta_ref = g.cpu(), beta.float().cpu()
    state = initial_state[0, 0].float().cpu().transpose(-2, -1).clone()
    expected = torch.empty_like(v_ref)
    for token in range(seq_len):
        q_t = q_ref[0, token, 0]
        k_t = k_ref[0, token, 0]
        q_t = q_t * torch.rsqrt(q_t.square().sum() + 1e-6)
        k_t = k_t * torch.rsqrt(k_t.square().sum() + 1e-6)
        state = torch.exp(g_ref[0, token, 0]) * state
        delta = beta_ref[0, token, 0] * (v_ref[0, token, 0] - k_t @ state)
        state = state + k_t[:, None] * delta[None, :]
        expected[0, token, 0] = 128**-0.5 * (q_t @ state)

    torch.testing.assert_close(
        result.out.cpu().float(), expected.to(dtype).float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        result.final_state.cpu().float(),
        state.transpose(-2, -1).unsqueeze(0).unsqueeze(0).to(dtype).float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_gdn_chunk_prefill_supports_qwen35_head_count():
    torch.manual_seed(19)
    dtype = torch.bfloat16
    seq_len = 128
    num_heads = 16
    q = torch.randn(1, seq_len, num_heads, 128, device="npu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn(1, seq_len, num_heads, device="npu"))
    beta = torch.rand(1, seq_len, num_heads, device="npu", dtype=dtype).sigmoid()
    initial_state = torch.zeros(1, num_heads, 128, 128, device="npu", dtype=dtype)
    cu_seqlens = torch.tensor([0, seq_len], device="npu", dtype=torch.int32)

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=128**-0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        solution="triton",
    )

    assert result.out.shape == v.shape
    assert result.final_state.shape == initial_state.shape
    nonfinite = torch.nonzero(~torch.isfinite(result.out), as_tuple=False)
    assert nonfinite.numel() == 0, nonfinite[:5].cpu().tolist()
    assert torch.isfinite(result.final_state).all()
