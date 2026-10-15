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

"""Portable ATen kernels for the single-request Qwen3.5 correctness harness.

Every operation stays on the input device. Ascend uses torch_npu's ATen
dispatch; no CUDA/Triton imports or host copies are needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x, weight, *, eps: float, zero_centered: bool):
    scale = weight.float() + 1 if zero_centered else weight.float()
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * scale).to(
        x.dtype
    )


def gated_norm(x, gate, weight, *, eps: float):
    xf = x.float()
    normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    )
    return (normalized * weight * F.silu(gate.float())).to(x.dtype)


def causal_conv(x, weight, state):
    """[B,T,C], [C,1,W], [B,C,W-1] -> activated output and new window.

    Explicit short-filter accumulation avoids unsupported depthwise Conv1D
    paths. State is functional: callers can safely retain an earlier snapshot.
    """
    width = weight.shape[-1]
    history = torch.cat((state, x.transpose(1, 2)), dim=-1)
    out = torch.zeros_like(x, dtype=torch.float32)
    for j in range(width):
        out = (
            out
            + history[:, :, j : j + x.shape[1]].transpose(1, 2).float()
            * weight[:, 0, j].float()
        )
    return F.silu(out.to(x.dtype)), history[:, :, -(width - 1) :].contiguous().clone()


def _unit_lower_inverse(negative_lower):
    """Invert a unit triangular matrix by stable block substitution.

    Forming high powers of the full lower triangle causes catastrophic
    cancellation for near-parallel keys and beta near one. Block substitution
    multiplies already solved diagonal blocks, avoiding those large powers.
    """
    n = negative_lower.shape[-1]
    padded = 1 << (n - 1).bit_length()
    matrix = F.pad(negative_lower, (0, padded - n, 0, padded - n))
    batch, heads = matrix.shape[:2]
    inverse = torch.ones(
        batch, heads, padded, 1, 1, device=matrix.device, dtype=matrix.dtype
    )
    width = 1
    while width < padded:
        block = width * 2
        groups = padded // block
        diagonal = (
            matrix.reshape(batch, heads, groups, block, groups, block)
            .diagonal(dim1=2, dim2=4)
            .movedim(-1, 2)
        )
        left, right = inverse[:, :, 0::2], inverse[:, :, 1::2]
        cross = (right @ diagonal[..., width:, :width]) @ left
        upper = torch.cat((left, torch.zeros_like(left)), dim=-1)
        lower = torch.cat((cross, right), dim=-1)
        inverse = torch.cat((upper, lower), dim=-2)
        width = block
    return inverse[:, :, 0, :n, :n]


def gated_delta(q, k, v, g, beta, state, *, algorithm: str, chunk_size: int):
    """Gated delta rule with FP32 [B,H,K,V] state and normalized Q/K.

    Chunk mode solves the strictly triangular delta system by block
    substitution. This avoids linalg.solve_triangular CPU fallback on
    Ascend. Both paths implement all updates, including a partial last chunk.
    """
    dtype = q.dtype
    q, k, v = [x.transpose(1, 2).float() for x in (q, k, v)]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    g, beta = g.transpose(1, 2).float(), beta.transpose(1, 2).float()
    state = state.float()
    outputs = []
    if algorithm == "recurrent":
        for t in range(q.shape[2]):
            kt, qt = k[:, :, t], q[:, :, t]
            state = state * g[:, :, t].exp()[..., None, None]
            delta = (v[:, :, t] - (state * kt[..., None]).sum(-2)) * beta[:, :, t, None]
            state = state + kt[..., None] * delta[..., None, :]
            outputs.append((state * qt[..., None]).sum(-2).unsqueeze(2))
    elif algorithm == "chunk":
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for start in range(0, q.shape[2], chunk_size):
            qc, kc, vc = [x[:, :, start : start + chunk_size] for x in (q, k, v)]
            bc = beta[:, :, start : start + chunk_size]
            gc = g[:, :, start : start + chunk_size].cumsum(-1)
            n = qc.shape[2]
            upper = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
            decay = (
                (gc[..., :, None] - gc[..., None, :])
                .masked_fill(upper, float("-inf"))
                .exp()
            )
            kb = kc * bc[..., None]
            lower = -((kb @ kc.transpose(-1, -2)) * decay).tril(-1)
            inverse = _unit_lower_inverse(lower)
            u = inverse @ (vc * bc[..., None])
            w = inverse @ (kb * gc.exp()[..., None])
            delta = u - w @ state
            attn = (qc @ kc.transpose(-1, -2)) * decay
            outputs.append((qc * gc.exp()[..., None]) @ state + attn @ delta)
            tail_k = kc * (gc[..., -1:] - gc).exp()[..., None]
            state = (
                state * gc[..., -1, None, None].exp() + tail_k.transpose(-1, -2) @ delta
            )
    else:
        raise ValueError(f"Unknown GDN algorithm: {algorithm}")
    return torch.cat(outputs, dim=2).transpose(1, 2).to(dtype), state


def rotary(x, cos, sin):
    dim = cos.shape[-1]
    left, right = x[..., : dim // 2], x[..., dim // 2 : dim]
    rotated = torch.cat((-right, left), dim=-1)
    return torch.cat((x[..., :dim] * cos + rotated * sin, x[..., dim:]), dim=-1)


def attention(q, k, v, *, offset: int):
    """Causal GQA, inputs [B,H,T,D], including offset chunk continuation."""
    repeats = q.shape[1] // k.shape[1]
    k, v = [x.repeat_interleave(repeats, dim=1) for x in (k, v)]
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    rows = torch.arange(q.shape[2], device=q.device) + offset
    cols = torch.arange(k.shape[2], device=q.device)
    scores = scores.masked_fill(cols[None, :] > rows[:, None], torch.finfo(q.dtype).min)
    probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
    return probs @ v
