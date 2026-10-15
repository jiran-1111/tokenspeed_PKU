"""FP32 effective gamma must not promote BF16 Q/K cache operands."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("rows", [1, 129])
def test_qk_fp32_gamma_keeps_bf16_and_matches_canonical_reference(rows) -> None:
    generator = torch.Generator().manual_seed(811)
    q = torch.randn(rows, 8 * 256, generator=generator).to(torch.bfloat16)
    k = torch.randn(rows, 2 * 256, generator=generator).to(torch.bfloat16)
    gamma = 1.0 + torch.full((256,), 0.003, dtype=torch.bfloat16).float()
    actual_q, actual_k = qk_rmsnorm(q.npu(), k.npu(), gamma.npu(), gamma.npu(), 1e-6)
    for actual, source, heads in ((actual_q, q, 8), (actual_k, k, 2)):
        assert actual.dtype == source.dtype
        values = source.reshape(rows, heads, 256).float()
        expected = (
            (
                values
                * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                * gamma
            )
            .to(source.dtype)
            .reshape(source.shape)
        )
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.02)


def test_qk_packed_noncontiguous_matches_single_rounding() -> None:
    # Repeated values keep the CPU/NPU reduction unambiguous. Exact BF16
    # output detects early gamma or normalization-scale rounding.
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    packed_cpu = values.repeat(3, (8 + 2) * 256 // 4)
    q, k = packed_cpu.npu().split([8 * 256, 2 * 256], dim=-1)
    assert not q.is_contiguous() and not k.is_contiguous()
    gamma = 1.0 + torch.full((256,), 0.003, dtype=torch.bfloat16).float()
    actual_q, actual_k = qk_rmsnorm(q, k, gamma.npu(), gamma.npu(), 1e-6)
    expected_values = (
        values.float() * torch.rsqrt(values.float().square().mean() + 1e-6) * gamma[0]
    ).to(torch.bfloat16)
    for actual, heads in ((actual_q, 8), (actual_k, 2)):
        expected = expected_values.repeat(3, heads * 256 // 4)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
