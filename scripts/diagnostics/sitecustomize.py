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

"""Opt-in decode capture; add this directory to PYTHONPATH to install.

Requires TOKENSPEED_DIAG_ARM_FILE and at least one of
TOKENSPEED_DIAG_LOGITS_PATH / TOKENSPEED_DIAG_STATES_PATH. The arm file selects
continuous or full_recompute. Target position/input and layer-0 raw guards remain
in effect. Existing conv_state/ssm_state fields and arrays are post-state;
Layer-0 *_before fields and arrays snapshot the input page before the backend runs.
Invalid layer-0 input pages (including the no-initial-state sentinel) produce a
capture_error without interrupting the backend or indexing a negative page.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _clone_state_page(pages: torch.Tensor, page: int) -> torch.Tensor:
    return pages[page].detach().clone()


def _install_decode_capture() -> None:
    output_path = os.environ.get("TOKENSPEED_DIAG_LOGITS_PATH")
    state_output_path = os.environ.get("TOKENSPEED_DIAG_STATES_PATH")
    arm_path = os.environ.get("TOKENSPEED_DIAG_ARM_FILE")
    raw_output_dir = (
        Path(state_output_path).with_name(
            Path(state_output_path).stem.removesuffix("-states") + "-layer0-arrays"
        )
        if state_output_path
        else None
    )
    if not arm_path or not (output_path or state_output_path):
        return

    import torch

    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

    target_position = int(os.environ.get("TOKENSPEED_DIAG_TARGET_POSITION", "1010"))
    target_input_id = int(os.environ.get("TOKENSPEED_DIAG_TARGET_INPUT_ID", "271"))

    local = threading.local()
    write_lock = threading.Lock()
    call_counts: dict[tuple[int, str], int] = {}

    def active_mode() -> str | None:
        try:
            value = Path(arm_path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return value or None

    def append_record(record: dict[str, object]) -> None:
        if not output_path:
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with write_lock, Path(output_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()

    def append_state_record(record: dict[str, object]) -> None:
        if not state_output_path:
            return
        Path(state_output_path).parent.mkdir(parents=True, exist_ok=True)
        with write_lock, Path(state_output_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()

    def target_mode() -> str | None:
        mode = active_mode()
        forward = getattr(local, "forward_meta", None)
        if mode is None or forward is None:
            return None
        if (
            forward.get("position_last") != target_position
            or forward.get("input_id_last") != target_input_id
        ):
            return None
        return mode

    def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        cpu_tensor = tensor.detach().contiguous().to(device="cpu")
        raw = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        values = cpu_tensor.float().reshape(-1)
        finite = torch.isfinite(values)
        finite_values = values[finite]
        return {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "numel": int(cpu_tensor.numel()),
            "finite_numel": int(finite.sum().item()),
            "min": float(finite_values.min().item()) if finite_values.numel() else None,
            "max": float(finite_values.max().item()) if finite_values.numel() else None,
            "l2_norm": (
                float(torch.linalg.vector_norm(finite_values).item())
                if finite_values.numel()
                else None
            ),
        }

    def write_raw_tensor(
        tensor: torch.Tensor, mode: str, name: str
    ) -> dict[str, object]:
        if raw_output_dir is None:
            raise RuntimeError("raw tensor output directory is unavailable")
        if mode not in {"continuous", "full_recompute"}:
            raise RuntimeError("raw tensor capture has an unexpected mode")
        cpu_tensor = tensor.detach().contiguous().to(device="cpu")
        raw = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        path = raw_output_dir / f"{mode}-layer0-{name}.bin"
        with path.open("xb") as stream:
            stream.write(raw)
        return {
            "file": path.name,
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "numel": int(cpu_tensor.numel()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def capture_mha(self, layer, token_to_kv_pool, mode: str, op: str, bs: int) -> None:
        metadata = (
            self.forward_decode_metadata
            if op == "decode"
            else self.forward_extend_metadata
        )
        if bs < 1:
            raise RuntimeError("MHA capture saw an empty batch")
        seq_len = int(metadata.seq_lens[0].item())
        page_count = (seq_len + self.kernel_page_size - 1) // self.kernel_page_size
        page_ids = metadata.page_table[0, :page_count]
        if page_ids.numel() != page_count or (page_ids < 0).any().item():
            raise RuntimeError("MHA capture found an invalid logical page table")
        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        page_ids = page_ids.to(device=k_cache.device, dtype=torch.long)
        key = k_cache.index_select(0, page_ids).reshape(
            -1, k_cache.shape[-2], k_cache.shape[-1]
        )[:seq_len]
        value = v_cache.index_select(0, page_ids.to(v_cache.device)).reshape(
            -1, v_cache.shape[-2], v_cache.shape[-1]
        )[:seq_len]
        append_state_record(
            {
                "capture_mode": mode,
                "backend": "mha",
                "operation": op,
                "layer_id": int(layer.layer_id),
                "row": 0,
                "seq_len": seq_len,
                "page_ids": [int(page_id) for page_id in page_ids.cpu().tolist()],
                "forward": getattr(local, "forward_meta", None),
                "key": tensor_summary(key),
                "value": tensor_summary(value),
                "mxfp8": bool(self.is_mxfp8),
            }
        )

    def capture_mamba(
        self,
        layer_id: int,
        mode: str,
        op: str,
        backend_inputs: dict[str, object],
        backend_output: torch.Tensor,
        state_before: dict[str, torch.Tensor],
    ) -> None:
        state_in, state_out, conv_states, ssm_states = self._layer_state(layer_id)
        state_page = int(state_out[0].item())
        input_page = int(state_in[0].item())
        if state_page < 0:
            raise RuntimeError("GDN capture found an invalid output state page")
        record = {
            "capture_mode": mode,
            "backend": "gdn_state",
            "operation": op,
            "layer_id": int(layer_id),
            "row": 0,
            "input_page": input_page,
            "output_page": state_page,
            "forward": getattr(local, "forward_meta", None),
            "conv_state": tensor_summary(conv_states[state_page]),
            "ssm_state": tensor_summary(ssm_states[state_page]),
            **{name: tensor_summary(value) for name, value in state_before.items()},
        }
        if int(layer_id) == 0:
            last_row_inputs = {
                name: value[-1:]
                for name, value in backend_inputs.items()
                if name in {"mixed_qkv", "z"}
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
            }
            record["backend_input_summaries"] = {
                name: tensor_summary(value) for name, value in last_row_inputs.items()
            }
            backend_last_output = (
                backend_output[:, -1:]
                if backend_output.ndim >= 2 and backend_output.shape[0] == 1
                else backend_output[-1:]
            )
            if backend_output.ndim > 0:
                record["backend_output_summary"] = tensor_summary(backend_last_output)
            if raw_output_dir is not None:
                try:
                    raw_tensors = {
                        "conv_state": conv_states[state_page],
                        "ssm_state": ssm_states[state_page],
                        **state_before,
                        **last_row_inputs,
                    }
                    if backend_output.ndim > 0:
                        raw_tensors["backend_output"] = backend_last_output
                    record["raw_arrays"] = {
                        name: write_raw_tensor(tensor, mode, name)
                        for name, tensor in raw_tensors.items()
                    }
                except Exception as error:
                    record["raw_capture_error"] = f"{type(error).__name__}: {error}"
        append_state_record(record)

    def wrap_state_capture(cls, method_name: str, backend: str) -> None:
        original = getattr(cls, method_name)

        @wraps(original)
        def wrapped(self, *args, **kwargs):
            layer = kwargs.get("layer")
            if layer is None and len(args) > 3:
                layer = args[3]
            layer_id = kwargs.get("layer_id", getattr(layer, "layer_id", None))
            mode = target_mode()
            backend_inputs: dict[str, object] = {}
            state_before: dict[str, torch.Tensor] = {}
            before_error: Exception | None = None
            if (
                backend == "gdn"
                and mode is not None
                and layer_id is not None
                and int(layer_id) == 0
                and (backend, int(layer_id))
                not in getattr(local, "state_capture_seen", set())
            ):
                try:
                    state_in, _, conv_states, ssm_states = self._layer_state(
                        int(layer_id)
                    )
                    input_page = int(state_in[0].item())
                    if input_page < 0:
                        raise RuntimeError(
                            "GDN capture found an invalid input state page"
                        )
                    state_before = {
                        "conv_state_before": _clone_state_page(conv_states, input_page),
                        "ssm_state_before": _clone_state_page(ssm_states, input_page),
                    }
                except Exception as error:
                    before_error = error
            if backend == "gdn" and mode is not None and layer_id == 0:
                for name in ("mixed_qkv", "z"):
                    value = kwargs.get(name)
                    if isinstance(value, torch.Tensor) and value.ndim > 0:
                        backend_inputs[name] = value[-1:].detach().clone()

            result = original(self, *args, **kwargs)
            mode = target_mode()
            if mode is None:
                return result
            if layer_id is None:
                return result
            seen = getattr(local, "state_capture_seen", set())
            key = (backend, int(layer_id))
            if key in seen:
                return result
            seen.add(key)
            local.state_capture_seen = seen
            try:
                if backend == "mha":
                    token_to_kv_pool = kwargs.get("token_to_kv_pool")
                    if token_to_kv_pool is None and len(args) > 5:
                        token_to_kv_pool = args[5]
                    bs = kwargs.get("bs")
                    if bs is None and len(args) > 6:
                        bs = args[6]
                    capture_mha(
                        self,
                        layer,
                        token_to_kv_pool,
                        mode,
                        method_name.removeprefix("forward_"),
                        int(bs),
                    )
                else:
                    if before_error is not None:
                        raise before_error
                    capture_mamba(
                        self,
                        int(layer_id),
                        mode,
                        method_name.removeprefix("forward_"),
                        backend_inputs,
                        result,
                        state_before,
                    )
            except Exception as error:
                append_state_record(
                    {
                        "capture_mode": mode,
                        "backend": backend,
                        "operation": method_name,
                        "layer_id": int(layer_id),
                        "capture_error": f"{type(error).__name__}: {error}",
                    }
                )
            return result

        setattr(cls, method_name, wrapped)

    if state_output_path:
        for method_name in ("forward_decode", "forward_extend"):
            wrap_state_capture(MHAAttnBackend, method_name, "mha")
            wrap_state_capture(MambaAttnBackend, method_name, "gdn")

    original_model_forward = ModelRunner.forward

    @wraps(original_model_forward)
    def model_forward(self, ctx, input_ids, positions, *args, **kwargs):
        mode = active_mode()
        local.forward_meta = None
        if mode is not None:
            position_values = (
                positions.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
            )
            input_values = (
                input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
            )
            forward_mode = getattr(ctx, "forward_mode", None)
            local.forward_meta = {
                "forward_mode": getattr(forward_mode, "name", str(forward_mode)),
                "input_num_tokens": int(
                    getattr(ctx, "input_num_tokens", input_values.numel())
                ),
                "position_count": int(position_values.numel()),
                "position_min": (
                    int(position_values.min().item())
                    if position_values.numel()
                    else None
                ),
                "position_max": (
                    int(position_values.max().item())
                    if position_values.numel()
                    else None
                ),
                "position_last": (
                    int(position_values[-1].item()) if position_values.numel() else None
                ),
                "input_id_last": (
                    int(input_values[-1].item()) if input_values.numel() else None
                ),
            }
            local.state_capture_seen = set()
        return original_model_forward(self, ctx, input_ids, positions, *args, **kwargs)

    ModelRunner.forward = model_forward

    original_get_logits = LogitsProcessor._get_logits

    @wraps(original_get_logits)
    def get_logits(
        self, hidden_states, lm_head, logits_metadata, embedding_bias=None, plan=None
    ):
        logits = original_get_logits(
            self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=embedding_bias,
            plan=plan,
        )
        mode = active_mode()
        forward = getattr(local, "forward_meta", None)
        if mode is None or logits.numel() == 0 or target_mode() is None:
            return logits

        rows = logits.reshape(-1, logits.shape[-1])
        values, token_ids = torch.topk(rows[-1].float(), k=min(2, rows.shape[-1]))
        pid = os.getpid()
        key = (pid, mode)
        with write_lock:
            call_index = call_counts.get(key, 0)
            call_counts[key] = call_index + 1

        metadata_mode = getattr(logits_metadata, "forward_mode", None)
        append_record(
            {
                "capture_mode": mode,
                "pid": pid,
                "call_index": call_index,
                "forward_mode": getattr(metadata_mode, "name", str(metadata_mode)),
                "forward": forward,
                "logits_rows": int(rows.shape[0]),
                "vocab_size": int(rows.shape[-1]),
                "top_token_ids": [int(value) for value in token_ids.tolist()],
                "top_logits": [float(value) for value in values.tolist()],
                "top1_top2_margin": (
                    float((values[0] - values[1]).item())
                    if values.numel() > 1
                    else None
                ),
            }
        )
        return logits

    LogitsProcessor._get_logits = get_logits


_install_decode_capture()
