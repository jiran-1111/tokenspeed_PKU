"""Canonical Qwen3.5 gamma must add one in FP32 even for BF16 weights."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
from tokenspeed.runtime.layers.layernorm import (
    GemmaRMSNorm,
    _torch_allreduce_residual_rmsnorm,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("with_residual", [False, True])
def test_bf16_loaded_gamma_preserves_canonical_fp32_addition(with_residual) -> None:
    width = 256
    norm = GemmaRMSNorm(width).to(device="npu", dtype=torch.bfloat16)
    weight = torch.full((width,), 0.003, dtype=torch.bfloat16)
    norm._weight_loader(norm.weight, weight.npu())
    # The pinned Transformers Qwen3_5RMSNorm uses 1 + weight.float().
    gamma = 1.0 + weight.float()
    assert norm.gemma_weight.dtype == torch.float32
    torch.testing.assert_close(norm.gemma_weight.cpu(), gamma, atol=0, rtol=0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16).repeat(1, width // 4)
    residual = x * 0.5 if with_residual else None
    combined = x if residual is None else x + residual
    expected = (
        combined.float()
        * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
        * gamma
    ).to(torch.bfloat16)
    residual_npu = residual.npu() if residual is not None else None
    actual = norm(x.npu(), residual_npu)
    if with_residual:
        actual, updated = actual
        assert updated is residual_npu
        torch.testing.assert_close(updated.cpu(), combined, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-3)


def test_allreduce_helper_keeps_fp32_gamma_and_residual_identity(monkeypatch) -> None:
    # Replace communication only: this checks the actual NPU norm helper,
    # not multi-rank HCCL transport or an end-to-end TP model.
    group = object()

    def double_input(tensor, *, group):
        assert group is expected_group
        tensor.mul_(2)

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", double_input)
    x_cpu = (
        torch.arange(256, dtype=torch.float32).reshape(1, 256).to(torch.bfloat16) / 256
    )
    residual_cpu = x_cpu * 0.5
    gamma_cpu = 1.0 + torch.full((256,), 0.003, dtype=torch.bfloat16).float()
    combined = x_cpu * 2 + residual_cpu
    expected = (
        combined.float()
        * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
        * gamma_cpu
    ).to(torch.bfloat16)
    residual = residual_cpu.npu()
    output, updated, handle = _torch_allreduce_residual_rmsnorm(
        input_tensor=x_cpu.npu(),
        residual=residual,
        weight=gamma_cpu.npu(),
        group=group,
        eps=1e-6,
    )
    assert updated is residual
    assert handle is None
    torch.testing.assert_close(updated.cpu(), combined, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu(), expected, atol=1e-4, rtol=1e-3)
