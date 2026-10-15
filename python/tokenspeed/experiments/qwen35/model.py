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

"""Complete Qwen3.5 text forward for the Ascend course baseline.

This harness owns per-request state, never scheduler slots. It is not a second
serving backend and does not claim LCM, prefix caching or retraction support.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenspeed_kernel.ops.experimental_qwen35 import (
    attention,
    causal_conv,
    gated_delta,
    gated_norm,
    rms_norm,
    rotary,
)


@dataclass
class RequestState:
    position: int = 0
    kv: dict = field(default_factory=dict)
    conv: dict = field(default_factory=dict)
    recurrent: dict = field(default_factory=dict)


class TextModel:
    def __init__(
        self,
        model_path,
        *,
        device: str,
        dtype: torch.dtype,
        gdn_algorithm: str,
        chunk_size: int,
        profile: bool,
    ):
        self.path = Path(model_path)
        self.config = json.loads((self.path / "config.json").read_text())["text_config"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.algorithm = gdn_algorithm
        self.chunk_size = chunk_size
        self.profile = profile
        self.profile_records = []
        self.profile_phase = "initialization"
        self.weights = {}
        self.loaded_keys = []
        self.ignored_keys = []
        if gdn_algorithm not in {"recurrent", "chunk"}:
            raise ValueError(f"Unsupported GDN algorithm: {gdn_algorithm}")
        c = self.config
        if (
            c["attention_bias"]
            or c["hidden_act"] != "silu"
            or not c["tie_word_embeddings"]
        ):
            raise ValueError(
                "This harness requires bias-free SiLU with tied embeddings"
            )
        if c["rope_parameters"]["rope_type"] != "default":
            raise ValueError("Only default text RoPE is supported")
        for shard in sorted(self.path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name.startswith("model.language_model."):
                        key = name.removeprefix("model.language_model.")
                        self.weights[key] = f.get_tensor(name).to(
                            device=self.device, dtype=dtype
                        )
                        self.loaded_keys.append(name)
                    else:
                        self.ignored_keys.append(name)
        if not self.weights:
            raise ValueError("No Qwen3.5 language-model weights found")
        self._validate_weights()
        rd = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv_freq = 1.0 / (
            c["rope_parameters"]["rope_theta"]
            ** (torch.arange(0, rd, 2, device=self.device).float() / rd)
        )

    def _validate_weights(self):
        c = self.config
        h, d, kv = c["hidden_size"], c["head_dim"], c["num_key_value_heads"]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        expected = {"embed_tokens.weight": (c["vocab_size"], h), "norm.weight": (h,)}
        for i, kind in enumerate(c["layer_types"]):
            p = f"layers.{i}."
            expected.update(
                {
                    p + "input_layernorm.weight": (h,),
                    p + "post_attention_layernorm.weight": (h,),
                }
            )
            for proj, shape in {
                "gate": (c["intermediate_size"], h),
                "up": (c["intermediate_size"], h),
                "down": (h, c["intermediate_size"]),
            }.items():
                expected[p + f"mlp.{proj}_proj.weight"] = shape
            if kind == "linear_attention":
                shapes = {
                    "in_proj_qkv.weight": (2 * kh * kd + vh * vd, h),
                    "in_proj_z.weight": (vh * vd, h),
                    "in_proj_a.weight": (vh, h),
                    "in_proj_b.weight": (vh, h),
                    "A_log": (vh,),
                    "dt_bias": (vh,),
                    "norm.weight": (vd,),
                    "out_proj.weight": (h, vh * vd),
                    "conv1d.weight": (
                        2 * kh * kd + vh * vd,
                        1,
                        c["linear_conv_kernel_dim"],
                    ),
                }
                expected.update({p + "linear_attn." + k: v for k, v in shapes.items()})
            elif kind == "full_attention":
                shapes = {
                    "q_proj.weight": (2 * c["num_attention_heads"] * d, h),
                    "k_proj.weight": (kv * d, h),
                    "v_proj.weight": (kv * d, h),
                    "o_proj.weight": (h, c["num_attention_heads"] * d),
                    "q_norm.weight": (d,),
                    "k_norm.weight": (d,),
                }
                expected.update({p + "self_attn." + k: v for k, v in shapes.items()})
            else:
                raise ValueError(f"Unsupported layer: {kind}")
        if set(expected) != set(self.weights):
            raise ValueError(f"Weight key mismatch: {set(expected)^set(self.weights)}")
        for name, shape in expected.items():
            if tuple(self.weights[name].shape) != shape:
                raise ValueError(f"Weight shape mismatch: {name}")

    def scope(self, name):
        return self._timed_scope(name) if self.profile else nullcontext()

    @contextmanager
    def _timed_scope(self, name):
        device_api = torch.get_device_module(self.device)
        device_api.synchronize()
        start = time.perf_counter()
        yield
        device_api.synchronize()
        self.profile_records.append(
            {
                "phase": self.profile_phase,
                "stage": name,
                "elapsed_ms": 1000 * (time.perf_counter() - start),
            }
        )

    def linear(self, x, name):
        with self.scope("qwen35/projection"):
            return F.linear(x, self.weights[name + ".weight"])

    def norm(self, x, name):
        with self.scope("qwen35/norm"):
            return rms_norm(
                x,
                self.weights[name + ".weight"],
                eps=self.config["rms_norm_eps"],
                zero_centered=True,
            )

    def gdn(self, x, i, state):
        c, w = self.config, self.weights
        p = f"layers.{i}.linear_attn."
        bsz, length, _ = x.shape
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        qkv = self.linear(x, p + "in_proj_qkv")
        z = self.linear(x, p + "in_proj_z").view(bsz, length, vh, vd)
        a, b = self.linear(x, p + "in_proj_a"), self.linear(x, p + "in_proj_b")
        if i not in state.conv:
            state.conv[i] = x.new_zeros(
                bsz, qkv.shape[-1], c["linear_conv_kernel_dim"] - 1
            )
            state.recurrent[i] = torch.zeros(
                bsz, vh, kd, vd, device=x.device, dtype=torch.float32
            )
        with self.scope("qwen35/conv"):
            qkv, state.conv[i] = causal_conv(qkv, w[p + "conv1d.weight"], state.conv[i])
        q, k, v = qkv.split([kh * kd, kh * kd, vh * vd], dim=-1)
        q, k = [
            t.reshape(bsz, length, kh, kd).repeat_interleave(vh // kh, dim=2)
            for t in (q, k)
        ]
        v = v.reshape(bsz, length, vh, vd)
        g = -w[p + "A_log"].float().exp() * F.softplus(a.float() + w[p + "dt_bias"])
        with self.scope("qwen35/gdn"):
            y, state.recurrent[i] = gated_delta(
                q,
                k,
                v,
                g,
                b.sigmoid(),
                state.recurrent[i],
                algorithm=self.algorithm,
                chunk_size=self.chunk_size,
            )
        with self.scope("qwen35/gated_norm"):
            y = gated_norm(y, z, w[p + "norm.weight"], eps=c["rms_norm_eps"])
        return self.linear(y.reshape(bsz, length, -1), p + "out_proj")

    def mha(self, x, i, state, cos, sin):
        c = self.config
        p = f"layers.{i}.self_attn."
        b, t, _ = x.shape
        d = c["head_dim"]
        q, gate = self.linear(x, p + "q_proj").view(b, t, -1, 2 * d).chunk(2, dim=-1)
        q = self.norm(q, p + "q_norm").transpose(1, 2)
        k = self.norm(
            self.linear(x, p + "k_proj").view(b, t, -1, d), p + "k_norm"
        ).transpose(1, 2)
        v = self.linear(x, p + "v_proj").view(b, t, -1, d).transpose(1, 2)
        with self.scope("qwen35/rope"):
            q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if i in state.kv:
            oldk, oldv = state.kv[i]
            k, v = torch.cat((oldk, k), dim=2), torch.cat((oldv, v), dim=2)
        state.kv[i] = k, v
        with self.scope("qwen35/attention"):
            y = (
                attention(q, k, v, offset=state.position)
                .transpose(1, 2)
                .reshape(b, t, -1)
            )
        y = y * gate.reshape(b, t, -1).sigmoid()
        return self.linear(y, p + "o_proj")

    @torch.inference_mode()
    def forward(self, ids, state, *, all_logits: bool):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Only one non-empty, unpadded text request is supported")
        if ids.device != self.device:
            raise ValueError(f"Input must reside on {self.device}")
        if state.position + ids.shape[1] > self.config["max_position_embeddings"]:
            raise ValueError("Context length exceeds model configuration")
        with self.scope("qwen35/embedding"):
            x = F.embedding(ids, self.weights["embed_tokens.weight"])
        positions = torch.arange(
            state.position, state.position + ids.shape[1], device=x.device
        ).float()
        freq = positions[:, None] * self.inv_freq[None, :]
        freq = torch.cat((freq, freq), dim=-1)[None, None]
        cos, sin = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
        for i, kind in enumerate(self.config["layer_types"]):
            p = f"layers.{i}."
            nx = self.norm(x, p + "input_layernorm")
            y = (
                self.gdn(nx, i, state)
                if kind == "linear_attention"
                else self.mha(nx, i, state, cos, sin)
            )
            x = x + y
            nx = self.norm(x, p + "post_attention_layernorm")
            gate, up = self.linear(nx, p + "mlp.gate_proj"), self.linear(
                nx, p + "mlp.up_proj"
            )
            x = x + self.linear(F.silu(gate) * up, p + "mlp.down_proj")
        x = self.norm(x, "norm")
        state.position += ids.shape[1]
        with self.scope("qwen35/lm_head"):
            logits = F.linear(
                x if all_logits else x[:, -1:], self.weights["embed_tokens.weight"]
            )
        return logits.float()
