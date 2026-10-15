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

import copy

import pytest

from benchmarks.course_project.calibration_inputs import build_manifest
from scripts import collect_homework1_validation_reference as validation


def test_original_validation_rows_in_original_order():
    manifest = build_manifest(
        encode=lambda text: [ord(c) for c in text],
        vocab_size=100000,
        tokenizer_revision="fixed-rev",
    )
    before = copy.deepcopy(manifest)
    rows = validation.validation_cases(
        manifest,
        revision="fixed-rev",
        expected_sha=manifest["manifest_sha256"],
        vocab_size=100000,
    )
    assert rows == [c for c in manifest["cases"] if c["group"] == "validation"]
    assert len(rows) == 16 and manifest == before


def test_validation_rejects_tampered_manifest():
    manifest = build_manifest(
        encode=lambda text: [ord(c) for c in text],
        vocab_size=100000,
        tokenizer_revision="fixed-rev",
    )
    manifest["cases"][-1]["continuation_ids"][0] += 1
    with pytest.raises(ValueError, match="SHA256"):
        validation.validation_cases(
            manifest,
            revision="fixed-rev",
            expected_sha=manifest["manifest_sha256"],
            vocab_size=100000,
        )


def test_cached_reference_consumes_exact_frontiers(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    seen = []

    class Model:
        def __call__(self, **kwargs):
            ids = kwargs["input_ids"].tolist()[0]
            positions = kwargs["position_ids"].tolist()[0]
            seen.append((ids, positions, kwargs["past_key_values"]))
            length = positions[-1] + 1
            cache = SimpleNamespace(get_seq_length=lambda: length)
            return SimpleNamespace(
                logits=torch.zeros(1, 1, 248320), past_key_values=cache
            )

    case = {
        "case_id": "fixture",
        "input_ids": [9, 8],
        "continuation_ids": list(range(16)),
    }
    validation.collect_cached_case(
        Model(),
        case,
        output=tmp_path,
        index=0,
        device="cpu",
        torch=torch,
        linear_layer_type=object,
    )
    assert len(seen) == 17 and seen[0] == ([9, 8], [0, 1], None)
    assert all(
        ids == [step - 1] and positions == [step + 1] and past is not None
        for step, (ids, positions, past) in enumerate(seen[1:], 1)
    )
    saved = torch.load(tmp_path / "case_00.pt", weights_only=True)
    assert saved["cached_logits"].shape == (17, 248320)
    assert (
        saved["input_ids"] == case["input_ids"]
        and saved["suffix_ids"] == case["continuation_ids"]
    )
