from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an available Ascend NPU"
)


@pytest.mark.parametrize("with_residual", [False, True])
def test_gemma_rmsnorm_matches_reference_on_npu(with_residual: bool) -> None:
    eps = 1e-6
    hidden_size = 16
    norm = GemmaRMSNorm(hidden_size, eps=eps).to("npu")
    weight = torch.linspace(-0.5, 0.5, hidden_size, dtype=torch.float32, device="npu")
    norm._weight_loader(norm.weight, weight)

    x = torch.randn((4, hidden_size), dtype=torch.bfloat16, device="npu")
    residual = torch.randn_like(x) if with_residual else None
    combined = x if residual is None else x + residual
    combined_fp32 = combined.float()
    expected = (
        combined_fp32
        * torch.rsqrt(combined_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * (weight + torch.ones_like(weight))
    ).to(x.dtype)

    actual = norm(x, residual)

    if residual is None:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.01, atol=0.01)
    else:
        actual_output, updated_residual = actual
        torch.testing.assert_close(
            actual_output.cpu(), expected.cpu(), rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(updated_residual.cpu(), combined.cpu())
