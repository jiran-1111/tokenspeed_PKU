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

import pytest

from scripts import homework1_stage_b_capture as capture


def test_frozen_input_frontiers():
    case = {"input_ids": [3, 4], "continuation_ids": list(range(16))}
    for step in range(17):
        ids = case["input_ids"] if step == 0 else [step - 1]
        positions = [0, 1] if step == 0 else [step + 1]
        capture.validate_frontier(case, step=step, input_ids=ids, positions=positions)
    with pytest.raises(ValueError):
        capture.validate_frontier(case, step=1, input_ids=[99], positions=[2])
    with pytest.raises(ValueError):
        capture.validate_frontier(case, step=1, input_ids=[0], positions=[3])
    with pytest.raises(ValueError):
        capture.validate_frontier(case, step=17, input_ids=[0], positions=[18])


def test_capture_and_force_preserve_logits(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Runner:
        def forward(self, ctx, input_ids, positions):
            return SimpleNamespace(
                next_token_logits=torch.arange(248320).reshape(1, -1).float()
            )

    class Executor:
        def _run_sampling(self, logits_output, sampling_info, ctx, candidates):
            return torch.tensor([248319], dtype=torch.int32), torch.tensor([1])

    arm = tmp_path / "arm.json"
    import json

    arm.write_text(
        json.dumps(
            {
                "case_id": "fixture",
                "input_ids": [9],
                "continuation_ids": list(range(16)),
                "output": str(tmp_path),
            }
        )
    )
    capture.install(Runner, Executor, torch=torch, arm_path=arm)
    for step in range(17):
        ids = [9] if step == 0 else [step - 1]
        out = Runner().forward(None, torch.tensor(ids), torch.tensor([step]))
        before = out.next_token_logits.clone()
        tokens, accepts = Executor()._run_sampling(out, None, None, None)
        assert tokens.item() == (step if step < 16 else 248319)
        assert accepts.item() == 1 and torch.equal(out.next_token_logits, before)
    files = list(tmp_path.glob("step_*.pt"))
    assert len(files) == 17
    row = torch.load(tmp_path / "step_00.pt", weights_only=True)
    assert row["raw_greedy_id"] == 248319 and row["logits"].shape == (248320,)


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "wrong_sampler"])
def test_invalid_vectors_or_sampler_are_rejected(tmp_path, failure):
    torch = pytest.importorskip("torch")
    import json
    from types import SimpleNamespace

    class Runner:
        def forward(self, ctx, input_ids, positions):
            values = torch.zeros(1, 248319 if failure == "shape" else 248320)
            if failure == "nonfinite":
                values[0, 0] = float("nan")
            return SimpleNamespace(next_token_logits=values)

    class Executor:
        def _run_sampling(self, logits_output, sampling_info, ctx, candidates):
            return torch.tensor([1]), torch.tensor([1])

    arm = tmp_path / "arm.json"
    arm.write_text(
        json.dumps(
            {
                "case_id": "bad",
                "input_ids": [9],
                "continuation_ids": list(range(16)),
                "output": str(tmp_path),
            }
        )
    )
    capture.install(Runner, Executor, torch=torch, arm_path=arm)
    with pytest.raises(ValueError):
        out = Runner().forward(None, torch.tensor([9]), torch.tensor([0]))
        Executor()._run_sampling(out, None, None, None)


def test_unarmed_hook_does_not_capture(tmp_path):
    class Runner:
        def forward(self, *args):
            return "original"

    class Executor:
        def _run_sampling(self, *args):
            return "tokens", "accepts"

    capture.install(Runner, Executor, torch=None, arm_path=tmp_path / "absent")
    assert Runner().forward(None, None, None) == "original"
    assert Executor()._run_sampling(None, None, None) == ("tokens", "accepts")


@pytest.mark.parametrize("bad_axis", [False, True])
def test_text_mrope_preserves_all_three_position_axes(tmp_path, bad_axis):
    torch = pytest.importorskip("torch")
    import json
    from types import SimpleNamespace

    class Runner:
        def forward(self, ctx, input_ids, positions):
            return SimpleNamespace(next_token_logits=torch.zeros(1, 248320))

    class Executor:
        def _run_sampling(self, logits_output, sampling_info, ctx, candidates):
            return torch.tensor([0]), torch.tensor([1])

    arm = tmp_path / "arm.json"
    arm.write_text(
        json.dumps(
            {
                "case_id": "mrope",
                "input_ids": [9, 8],
                "continuation_ids": list(range(16)),
                "output": str(tmp_path),
            }
        )
    )
    capture.install(Runner, Executor, torch=torch, arm_path=arm)
    positions = torch.tensor([[0, 1], [0, 1], [0, 1]])
    if bad_axis:
        positions[1, 1] = 2
        with pytest.raises(ValueError):
            Runner().forward(None, torch.tensor([9, 8]), positions)
    else:
        out = Runner().forward(None, torch.tensor([9, 8]), positions)
        Executor()._run_sampling(out, None, None, None)
        saved = torch.load(tmp_path / "step_00.pt", weights_only=True)
        assert saved["positions"] == [0, 1]
        assert saved["position_axes"] == [[0, 1], [0, 1], [0, 1]]
