"""Test the actual runtime convolution, not CUDA-only conv dispatch."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("channels", [16, 6143, 6144])
def test_causal_conv_multistep_slot_isolation(channels, record_property):
    generator = torch.Generator().manual_seed(421 + channels)
    # Runtime contract: channels contiguous, history as the other stride.
    original = (
        torch.randn(5, 3, channels, generator=generator)
        .to(torch.bfloat16)
        .transpose(1, 2)
    )
    pool = original.npu()
    weight = torch.randn(channels, 4, generator=generator).to(torch.bfloat16)
    bias = torch.randn(channels, generator=generator).to(torch.bfloat16)
    references = {slot: original[slot].clone() for slot in (1, 3)}
    for step in range(8):
        for slot in (1, 3):
            x = torch.randn(1, channels, generator=generator).to(torch.bfloat16)
            window = torch.cat((references[slot], x[0, :, None]), dim=1)
            expected = torch.nn.functional.silu(
                (window.float() * weight.float()).sum(-1) + bias.float()
            ).to(torch.bfloat16)
            references[slot] = window[:, 1:].clone()
            before = pool.cpu().clone()
            actual = (
                causal_conv1d_update(
                    x.npu(),
                    pool,
                    weight.npu(),
                    bias=bias.npu(),
                    activation="silu",
                    conv_state_indices=torch.tensor(
                        [slot], device="npu", dtype=torch.int32
                    ),
                    validate_data=True,
                )
                .float()
                .cpu()
                .reshape(channels)
            )
            assert torch.isfinite(actual).all()
            diff = actual - expected.float()
            record_property(
                f"error_{step}_{slot}",
                json.dumps(
                    {
                        "max_abs": float(diff.abs().max()),
                        "relative_l2": float(
                            diff.norm() / expected.float().norm().clamp_min(1e-12)
                        ),
                    }
                ),
            )
            torch.testing.assert_close(actual, expected.float(), atol=2e-3, rtol=2e-2)
            after = pool.cpu()
            torch.testing.assert_close(after[slot], references[slot], atol=0, rtol=0)
            others = [index for index in range(5) if index != slot]
            torch.testing.assert_close(after[others], before[others], atol=0, rtol=0)
