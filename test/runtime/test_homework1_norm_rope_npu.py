"""Real 0.8B Norm widths and MRoPE path, with independent CPU references."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.rotary_embedding import MRotaryEmbedding

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def _compare(actual, expected, label, record_property, *, rope=False) -> None:
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all()
    diff = actual - expected
    record_property(
        label,
        json.dumps(
            {
                "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
                "relative_l2": float(diff.norm() / expected.norm().clamp_min(1e-12)),
            }
        ),
    )
    if expected.numel() and expected.dtype != actual.dtype:
        raise AssertionError("comparison must use FP32 metrics")
    atol, rtol = (0.02, 0.02) if rope else (0.002, 0.02)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("width", [256, 1024])
@pytest.mark.parametrize("rows", [0, 1, 129, 1024])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gemma", [False, True])
def test_actual_norm_shapes(
    width, rows, with_residual, dtype, gemma, record_property
) -> None:
    generator = torch.Generator().manual_seed(671 + width + rows)
    x = torch.randn(rows, width, generator=generator).to(dtype)
    residual = (
        torch.randn(rows, width, generator=generator).to(dtype)
        if with_residual
        else None
    )
    weight = (torch.randn(width, generator=generator) * 0.1).to(dtype)
    combined = x if residual is None else x + residual
    gamma = weight.float() + 1 if gemma else weight
    expected = (
        combined.float()
        * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
        * gamma.float()
    ).to(dtype)
    if gemma:
        norm = GemmaRMSNorm(width).to(device="npu", dtype=dtype)
        norm._weight_loader(norm.weight, weight.npu())
        actual = norm(x.npu(), residual.npu() if residual is not None else None)
    else:
        actual = rmsnorm(
            x.npu(),
            gamma.npu(),
            1e-6,
            residual=residual.npu() if residual is not None else None,
        )
    if with_residual:
        actual, actual_residual = actual
        torch.testing.assert_close(actual_residual.cpu(), combined, atol=0, rtol=0)
    if dtype == torch.float32:
        # The legacy 0.02 envelope was too loose to verify a FP32 path.
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
    _compare(actual, expected, "norm_error", record_property)


@pytest.mark.parametrize("mode", ["text_1d", "text_3d", "different_axes"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_actual_mrope_partial_dim(mode, dtype, record_property) -> None:
    positions = torch.tensor([0, 1, 127, 128, 4095])
    if mode != "text_1d":
        positions = positions.repeat(3, 1)
        if mode == "different_axes":
            positions[1] = (positions[1] + 7) % 4096
            positions[2] = (positions[2] + 13) % 4096
    rope = MRotaryEmbedding(
        256,
        64,
        4096,
        10000000,
        True,
        dtype,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    ).to("npu")
    generator = torch.Generator().manual_seed(719)
    packed = torch.randn(5, (8 + 2) * 256, generator=generator).to(dtype)
    q, k = packed.npu().split([8 * 256, 2 * 256], dim=-1)
    actual_q, actual_k = rope(positions.npu(), q, k)
    inverse = 1.0 / (10000000 ** (torch.arange(0, 64, 2).float() / 64))
    if positions.ndim == 1:
        angles = positions.float()[:, None] * inverse
    else:
        selected = torch.empty(5, 32)
        for index in range(32):
            axis = (
                1
                if index % 3 == 1 and index < 33
                else (2 if index % 3 == 2 and index < 30 else 0)
            )
            selected[:, index] = positions[axis].float()
        angles = selected * inverse
    # Match the module's coefficient quantization; perform the reference arithmetic in FP32.
    cos, sin = (
        angles.cos().to(dtype).float()[:, None],
        angles.sin().to(dtype).float()[:, None],
    )
    for label, source, actual, heads in (
        ("q", packed[:, : 8 * 256], actual_q, 8),
        ("k", packed[:, 8 * 256 :], actual_k, 2),
    ):
        source = source.reshape(5, heads, 256)
        first, second = source[..., :32].float(), source[..., 32:64].float()
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), -1
        ).to(dtype)
        expected = torch.cat((rotated, source[..., 64:]), -1).reshape(5, -1)
        if dtype == torch.float32:
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
        _compare(actual, expected, label + "_rope_error", record_property, rope=True)
        torch.testing.assert_close(
            actual.cpu().reshape(5, heads, 256)[..., 64:],
            source[..., 64:],
            atol=0,
            rtol=0,
        )
