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

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_session_kv import _load_method

from benchmarks.course_project.core import build_serve_command, load_model_config

ROOT = Path(__file__).resolve().parents[2]
LOADER = "python/tokenspeed/runtime/execution/weight_loader.py"


def test_explicit_hf_overrides_are_forwarded_without_dropping_other_fields():
    config = load_model_config(ROOT / "configs/baseline/qwen35_9b.yaml")
    assert "--hf-overrides" not in build_serve_command(
        config, host="127.0.0.1", port=8000
    )
    overrides = {"npu_lm_head_dtype": "float32", "rope_theta": 10000000}
    command = build_serve_command(
        {**config, "hf_overrides": overrides}, host="127.0.0.1", port=8000
    )
    assert json.loads(command[command.index("--hf-overrides") + 1]) == overrides


def test_hf_overrides_rejects_non_dictionary():
    config = load_model_config(ROOT / "configs/baseline/qwen35_9b.yaml")
    with pytest.raises(ValueError, match="hf_overrides"):
        build_serve_command(
            {**config, "hf_overrides": "float32"}, host="127.0.0.1", port=8000
        )


def test_candidate_changes_only_explicit_head_precision():
    baseline = load_model_config(ROOT / "configs/baseline/qwen35_9b.yaml")
    candidate = load_model_config(ROOT / "configs/candidates/qwen35_9b_fp32_head.yaml")
    assert candidate.pop("hf_overrides") == {"npu_lm_head_dtype": "float32"}
    assert candidate == baseline


def test_head_conversion_is_inside_weight_lifetime_before_engram_initialization():
    tree = ast.parse((ROOT / LOADER).read_text(encoding="utf-8"))
    loader = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightLoader"
    )
    method = next(
        n
        for n in loader.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    region = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call)
            and isinstance(i.context_expr.func, ast.Attribute)
            and i.context_expr.func.attr == "region"
            for i in n.items
        )
    )
    calls = [
        n
        for n in ast.walk(region)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_configure_npu_lm_head"
    ]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} == {"device", "requested_dtype"}


@pytest.fixture
def policy():
    torch = pytest.importorskip("torch")
    return torch, _load_method(
        LOADER,
        None,
        "_configure_npu_lm_head",
        torch=torch,
        should_apply_lm_head_quant_method=lambda head, method: method is not None,
    )


def make_model(torch):
    class Qwen3_5ForConditionalGeneration:
        def __init__(self):
            self.config = SimpleNamespace(tie_word_embeddings=False)
            self.encoder_only = False
            self.lm_head = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.tensor([[1, 0], [1, 1]], dtype=torch.bfloat16)
                ),
                quant_method=None,
            )

    return Qwen3_5ForConditionalGeneration()


def test_fp32_expansion_preserves_parameter_values_and_recovers_near_tie(policy):
    torch, configure = policy
    model = make_model(torch)
    parameter = model.lm_head.weight
    before = parameter.detach().clone()
    hidden = torch.tensor([[30.25, 0.035]], dtype=torch.bfloat16)
    assert (hidden @ before.T).argmax().item() == 0
    configure(model, requested_dtype="float32", device="npu")
    assert model.lm_head.weight is parameter
    assert parameter.dtype == torch.float32
    assert torch.equal(parameter.to(torch.bfloat16), before)
    projected = hidden.float() @ parameter.T
    assert projected.dtype == torch.float32
    assert projected.argmax().item() == 1


def test_omitted_override_does_not_inspect_or_modify_other_models(policy):
    _, configure = policy
    configure(object(), requested_dtype=None, device="cpu")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("device", "NPU"),
        ("dtype", "float32"),
        ("model", "Qwen3.5"),
        ("tied", "untied"),
        ("encoder", "LM head"),
        ("quantized", "unquantized"),
        ("weight_dtype", "BF16"),
    ],
)
def test_unsupported_explicit_requests_fail_instead_of_being_ignored(
    policy, change, message
):
    torch, configure = policy
    model = make_model(torch)
    device, dtype = "npu", "float32"
    if change == "device":
        device = "cpu"
    if change == "dtype":
        dtype = "float16"
    if change == "model":
        model = object()
    if change == "tied":
        model.config.tie_word_embeddings = True
    if change == "encoder":
        model.lm_head = None
    if change == "quantized":
        model.lm_head.quant_method = object()
    if change == "weight_dtype":
        model.lm_head.weight.data = model.lm_head.weight.data.float()
    with pytest.raises(ValueError, match=message):
        configure(model, requested_dtype=dtype, device=device)
