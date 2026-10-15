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

"""Compare Ascend kernels against independent CPU/Transformers references."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.experimental_qwen35 import (
    attention,
    causal_conv,
    gated_delta,
    rms_norm,
    rotary,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

DEVICE = os.environ["QWEN35_TEST_DEVICE"]
torch.set_num_threads(4)


@pytest.mark.parametrize("length", [1, 3, 4, 17, 64, 65, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_outputs_and_multistep_state(length, dtype):
    torch.manual_seed(20261015 + length)
    b, h, kd, vd = 1, 16, 128, 128
    q, k, v = [torch.randn(b, length, h, d, dtype=dtype) for d in (kd, kd, vd)]
    g = -F.softplus(torch.randn(b, length, h))
    beta = torch.rand(b, length, h).to(dtype)
    initial = torch.randn(b, h, kd, vd) * 0.02
    expected, expected_state = torch_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    args = [x.to(DEVICE) for x in (q, k, v, g, beta, initial)]
    before = args[-1].clone()
    out, state = gated_delta(*args, algorithm="chunk", chunk_size=64)
    atol = 3e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=3e-3)
    torch.testing.assert_close(state.cpu(), expected_state, atol=3e-5, rtol=3e-3)
    torch.testing.assert_close(args[-1], before, atol=0, rtol=0)
    step_state = args[-1]
    pieces = []
    for t in range(length):
        part, step_state = gated_delta(
            *(x[:, t : t + 1] for x in args[:5]),
            step_state,
            algorithm="recurrent",
            chunk_size=64
        )
        pieces.append(part)
    torch.testing.assert_close(torch.cat(pieces, dim=1), out, atol=atol, rtol=3e-3)
    torch.testing.assert_close(step_state, state, atol=3e-5, rtol=3e-3)


@pytest.mark.parametrize("length", [1, 3, 4, 17, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conv_window_and_continuation(length, dtype):
    torch.manual_seed(length)
    x = torch.randn(1, length, 6144, dtype=dtype)
    weight = torch.randn(6144, 1, 4, dtype=dtype) * 0.1
    initial = torch.randn(1, 6144, 3, dtype=dtype)
    joined = torch.cat((initial, x.transpose(1, 2)), dim=-1)
    ref = F.silu(
        F.conv1d(joined.float(), weight.float(), groups=6144).to(dtype)
    ).transpose(1, 2)
    xd, wd, sd = [a.to(DEVICE) for a in (x, weight, initial)]
    actual, final = causal_conv(xd, wd, sd)
    atol = 1e-6 if dtype == torch.float32 else 4e-3
    torch.testing.assert_close(actual.cpu(), ref, atol=atol, rtol=1e-3)
    torch.testing.assert_close(final.cpu(), joined[:, :, -3:], atol=0, rtol=0)
    pieces = []
    for t in range(length):
        y, sd = causal_conv(xd[:, t : t + 1], wd, sd)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces, dim=1), actual, atol=0, rtol=0)
    torch.testing.assert_close(sd, final, atol=0, rtol=0)


@pytest.mark.parametrize(
    "offset,length", [(0, 1), (0, 17), (7, 1), (7, 9), (0, 65), (0, 128)]
)
def test_attention_and_partial_rope(offset, length):
    torch.manual_seed(length)
    q = torch.randn(1, 8, length, 256)
    k, v = [torch.randn(1, 2, offset + length, 256) for _ in range(2)]
    rows = torch.arange(length) + offset
    mask = torch.arange(offset + length)[None, :] <= rows[:, None]
    ref = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1), attn_mask=mask
    )
    actual = attention(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), offset=offset)
    torch.testing.assert_close(actual.cpu(), ref, atol=2e-5, rtol=2e-4)
    angle = torch.randn(1, 1, length, 32)
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1)
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1)
    expected = q.clone()
    expected[..., :32] = q[..., :32] * angle.cos() - q[..., 32:64] * angle.sin()
    expected[..., 32:64] = q[..., 32:64] * angle.cos() + q[..., :32] * angle.sin()
    torch.testing.assert_close(
        rotary(q.to(DEVICE), cos.to(DEVICE), sin.to(DEVICE)).cpu(),
        expected,
        atol=2e-6,
        rtol=1e-5,
    )


def test_zero_centered_norm():
    torch.manual_seed(4)
    x = torch.randn(7, 1024, dtype=torch.bfloat16)
    w = torch.randn(1024, dtype=torch.bfloat16)
    expected = (
        x.double()
        * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
        * (1 + w.double())
    ).to(x.dtype)
    actual = rms_norm(x.to(DEVICE), w.to(DEVICE), eps=1e-6, zero_centered=True)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.032, rtol=0.008)


def test_kernel_import_does_not_load_optional_gpu_compilers():
    env = {
        **os.environ,
        "TOKENSPEED_KERNEL_PROFILE": "0",
        "TOKENSPEED_KERNEL_CAPTURE_SHAPES": "0",
    }
    code = "import tokenspeed_kernel,sys; assert 'tokenspeed_kernel._triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.parametrize("length", [64, 65, 129])
def test_delta_near_parallel_keys_slow_decay(length):
    torch.manual_seed(91)
    key = torch.randn(1, 1, 16, 128).expand(1, length, 16, 128).clone()
    key = key + torch.randn_like(key) * 1e-3
    q = key.clone()
    v = torch.randn_like(key)
    g = torch.full((1, length, 16), -1e-8)
    beta = torch.full_like(g, 0.99)
    initial = torch.randn(1, 16, 128, 128) * 0.01
    expected, expected_state = torch_recurrent_gated_delta_rule(
        q,
        key,
        v,
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = gated_delta(
        *(x.to(DEVICE) for x in (q, key, v, g, beta, initial)),
        algorithm="chunk",
        chunk_size=64
    )
    torch.testing.assert_close(out.cpu(), expected, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(state.cpu(), expected_state, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize(
    "variable", ["TOKENSPEED_KERNEL_PROFILE", "TOKENSPEED_KERNEL_CAPTURE_SHAPES"]
)
def test_optional_profile_still_bootstraps_when_requested(variable):
    env = {
        **os.environ,
        "TOKENSPEED_KERNEL_PROFILE": "0",
        "TOKENSPEED_KERNEL_CAPTURE_SHAPES": "0",
        variable: "true",
    }
    code = "import sys,types; m=types.ModuleType('tokenspeed_kernel.profiling'); calls=[]; m.bootstrap_profiling_from_env=lambda:calls.append(1); sys.modules[m.__name__]=m; import tokenspeed_kernel; assert calls==[1]"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
