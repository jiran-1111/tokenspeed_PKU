"""Opt-in TP1 Prefill layer boundaries; call from the guarded logits hook.

Only a diagnostic: D2H synchronization changes timing. Never benchmark with it.
"""

import hashlib
import json
import os
from functools import wraps
from pathlib import Path


def install(target_mode):
    directory = os.environ.get("TOKENSPEED_DIAG_LAYERWISE_DIR")
    if not directory:
        return
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
    from tokenspeed.runtime.models.qwen3_5 import (
        Qwen3_5AttentionDecoderLayer,
        Qwen3_5LinearDecoderLayer,
    )

    root = Path(directory)
    seen = set()

    def save_layerwise(name, tensor):
        if name in seen:
            raise RuntimeError(f"duplicate layerwise boundary: {name}")
        cpu = tensor.detach().float().cpu().contiguous()
        raw = cpu.numpy().tobytes()
        path = root / f"{name}.f32"
        with path.open("xb") as stream:
            stream.write(raw)
        with (root / "manifest.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "name": name,
                        "shape": list(cpu.shape),
                        "source_dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                + "\n"
            )
        seen.add(name)

    def attach_layer0_hooks(layer):
        handles = []
        for name, module in (
            ("input-norm", layer.input_layernorm),
            ("gdn-output", layer.linear_attn),
            ("post-attn-norm", layer.post_attention_layernorm),
            ("mlp-output", layer.mlp),
        ):

            def capture(module, args, result, name=name):
                if target_mode() is not None:
                    value = result[0] if isinstance(result, tuple) else result
                    save_layerwise("layer-0-" + name, value)

            handles.append(module.register_forward_hook(capture))
        return handles

    def wrap_layerwise_capture(cls):
        original = cls.forward

        @wraps(original)
        def forward(self, *args, **kwargs):
            capture = target_mode() is not None
            if capture:
                hidden, residual = kwargs["hidden_states"], kwargs["residual"]
                combined = hidden + residual if residual is not None else hidden
                save_layerwise(f"layer-{self.layer_id}-input", combined)
            handles = (
                attach_layer0_hooks(self) if capture and self.layer_id == 0 else []
            )
            try:
                result = original(self, *args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            if capture:
                hidden, residual = result
                combined = hidden + residual if residual is not None else hidden
                save_layerwise(f"layer-{self.layer_id}-output", combined)
            return result

        cls.forward = forward

    for cls in (Qwen3_5LinearDecoderLayer, Qwen3_5AttentionDecoderLayer):
        wrap_layerwise_capture(cls)
    original_logits = LogitsProcessor._get_logits

    @wraps(original_logits)
    def logits(self, hidden_states, *args, **kwargs):
        if target_mode() is not None:
            save_layerwise("final-norm", hidden_states)
        return original_logits(self, hidden_states, *args, **kwargs)

    LogitsProcessor._get_logits = logits
