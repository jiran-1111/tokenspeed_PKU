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

"""Read-only capture of the real BF16 input and weight of the final projection."""

from __future__ import annotations

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any


def install_head_capture(
    processor_class: type,
    *,
    torch: Any,
    arm_path: Path,
    vocab_size: int,
    hidden_size: int,
) -> None:
    """Capture only explicitly armed, dense BF16 TP1 projections, without replacing outputs."""
    original = processor_class._get_logits
    case_id: str | None = None
    step = 0
    weight_identity: tuple | None = None

    @wraps(original)
    def get_logits(
        self, hidden_states, lm_head, logits_metadata, embedding_bias=None, plan=None
    ):
        nonlocal case_id, step, weight_identity
        if not arm_path.exists():
            return original(
                self, hidden_states, lm_head, logits_metadata, embedding_bias, plan
            )
        if (
            self.tp_size != 1
            or self.logit_scale is not None
            or self.final_logit_softcapping
            or self._use_fused_lm_head
            or embedding_bias is not None
            or plan is not None
            or tuple(hidden_states.shape) != (1, hidden_size)
            or tuple(lm_head.weight.shape) != (vocab_size, hidden_size)
            or hidden_states.dtype != torch.bfloat16
            or lm_head.weight.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Unsupported projection semantics for the frozen diagnostic"
            )
        case = json.loads(arm_path.read_text(encoding="utf-8"))
        if case["case_id"] != case_id:
            case_id, step = case["case_id"], 0
        if not 0 <= step <= 16:
            raise ValueError("Unexpected projection count")
        identity = (
            lm_head.weight.data_ptr(),
            tuple(lm_head.weight.shape),
            lm_head.weight.dtype,
        )
        if weight_identity is None:
            weight = lm_head.weight.detach().cpu().contiguous().clone()
            sha = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
            root = Path(case["output"]).parent
            with (root / "head_weight.pt").open("xb") as stream:
                torch.save(weight, stream)
            with (root / "head_identity.json").open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "shape": list(weight.shape),
                        "dtype": str(weight.dtype),
                        "tensor_sha256": sha,
                    },
                    stream,
                )
            weight_identity = identity
        elif identity != weight_identity:
            raise ValueError("Projection weight storage changed during capture")
        hidden = hidden_states.detach().cpu().clone()
        result = original(
            self, hidden_states, lm_head, logits_metadata, embedding_bias, plan
        )
        path = Path(case["output"]) / f"hidden_{step:02d}.pt"
        with path.open("xb") as stream:
            torch.save({"case_id": case_id, "step": step, "hidden": hidden}, stream)
        step += 1
        return result

    processor_class._get_logits = get_logits
