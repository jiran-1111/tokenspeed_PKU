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

"""CPU contracts for read-only, opt-in projection capture."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch


def setup_capture(tmp_path):
    assert (
        importlib.util.find_spec("scripts.homework1_head_capture") is not None
    ), "Projection capture helper is missing"
    from scripts.homework1_head_capture import install_head_capture

    calls = []

    class Processor:
        def __init__(self):
            self.tp_size = 1
            self.logit_scale = None
            self.final_logit_softcapping = None
            self._use_fused_lm_head = False

        def _get_logits(
            self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=None,
            plan=None,
        ):
            calls.append(
                (hidden_states, lm_head, logits_metadata, embedding_bias, plan)
            )
            return hidden_states @ lm_head.weight.T

    arm = tmp_path / "arm.json"
    install_head_capture(
        Processor, torch=torch, arm_path=arm, vocab_size=7, hidden_size=4
    )
    return Processor(), calls, arm


def test_unarmed_preserves_every_argument_and_result(tmp_path):
    processor, calls, arm = setup_capture(tmp_path)
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    head = SimpleNamespace(weight=torch.ones(7, 4, dtype=torch.bfloat16))
    marker, bias, plan = object(), object(), object()
    result = processor._get_logits(hidden, head, marker, bias, plan)
    assert calls == [(hidden, head, marker, bias, plan)]
    assert result.tolist() == [[4] * 7]
    assert list(tmp_path.iterdir()) == []


def arm_case(arm, case_id):
    output = arm.parent / case_id
    output.mkdir(exist_ok=True)
    arm.write_text(json.dumps({"case_id": case_id, "output": str(output)}))
    return output


def test_armed_copies_hidden_and_weight_without_changing_logits(tmp_path):
    processor, calls, arm = setup_capture(tmp_path)
    case = arm_case(arm, "case_00")
    hidden = torch.arange(4).reshape(1, 4).bfloat16()
    weight = torch.arange(28).reshape(7, 4).bfloat16()
    head = SimpleNamespace(weight=weight)
    expected = hidden @ weight.T
    result = processor._get_logits(hidden, head, None)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert calls[0][0] is hidden and calls[0][1] is head
    saved = torch.load(case / "hidden_00.pt", weights_only=True)
    assert saved["case_id"] == "case_00" and saved["step"] == 0
    hidden.zero_()
    torch.testing.assert_close(
        saved["hidden"], torch.arange(4).reshape(1, 4).bfloat16(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "head_weight.pt", weights_only=True),
        weight,
        atol=0,
        rtol=0,
    )
    processor._get_logits(hidden, head, None)
    assert (case / "hidden_01.pt").is_file()
    arm_case(arm, "case_01")
    processor._get_logits(hidden, head, None)
    assert (tmp_path / "case_01/hidden_00.pt").is_file()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("tp_size", 2),
        ("logit_scale", 2),
        ("final_logit_softcapping", 1),
        ("_use_fused_lm_head", True),
    ],
)
def test_rejects_projection_semantics_outside_protocol(tmp_path, setting, value):
    processor, calls, arm = setup_capture(tmp_path)
    arm_case(arm, "case_00")
    processor.__dict__[setting] = value
    with pytest.raises(ValueError, match="projection"):
        processor._get_logits(
            torch.ones(1, 4, dtype=torch.bfloat16),
            SimpleNamespace(weight=torch.ones(7, 4, dtype=torch.bfloat16)),
            None,
        )
    assert calls == []
