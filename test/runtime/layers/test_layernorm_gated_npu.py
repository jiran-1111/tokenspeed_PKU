import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Gated RMSNorm tests require an NPU"
)


def test_gated_rmsnorm_runs_on_npu():
    torch.manual_seed(13)
    x = torch.randn(2, 128, device="npu", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device="npu", dtype=torch.float32)

    output = rmsnorm_fn(x, weight, z=z, eps=1e-6, weights_independent=True)

    x_ref, z_ref, weight_ref = x.float().cpu(), z.float().cpu(), weight.cpu()
    expected = (
        x_ref
        * torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight_ref
        * F.silu(z_ref)
    )
    torch.testing.assert_close(
        output.cpu().float(), expected.to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2
    )


def test_gated_rmsnorm_handles_rows_beyond_core_dim_limit_on_npu():
    torch.manual_seed(17)
    x = torch.randn(65536, 8, device="npu", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(8, device="npu", dtype=torch.float32)

    output = rmsnorm_fn(x, weight, z=z, eps=1e-6, weights_independent=True)

    x_ref, z_ref, weight_ref = x.float().cpu(), z.float().cpu(), weight.cpu()
    expected = (
        x_ref
        * torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight_ref
        * F.silu(z_ref)
    )
    torch.testing.assert_close(
        output.cpu().float(), expected.to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2
    )
