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

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.course_project.calibration_inputs import build_manifest
from scripts import collect_homework1_references as collector


def rehash(manifest: dict) -> str:
    content = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    sha = hashlib.sha256(
        json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest["manifest_sha256"] = sha
    return sha


@pytest.fixture
def manifest() -> dict:
    return build_manifest(
        encode=lambda text: [ord(char) for char in text],
        vocab_size=100000,
        tokenizer_revision="fixed-rev",
    )


def select(manifest: dict) -> list[dict]:
    return collector.calibration_cases(
        manifest,
        revision="fixed-rev",
        expected_sha=manifest["manifest_sha256"],
        vocab_size=100000,
    )


def test_selection(manifest: dict) -> None:
    cases = select(manifest)
    assert len(cases) == 16
    assert all(case["group"] == "calibration" for case in cases)
    assert sorted(case["input_length"] for case in cases) == sorted(
        [1, 8, 127, 128, 129, 512, 1024, 2048] * 2
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "model",
        "count",
        "group",
        "duplicate",
        "vocab",
        "suffix",
        "length",
        "steps",
    ],
)
def test_rehashed_invalid_inputs(manifest: dict, mutation: str) -> None:
    if mutation == "revision":
        manifest["tokenizer_revision"] = "other"
    elif mutation == "model":
        manifest["model_id"] = "other"
    elif mutation == "count":
        manifest["cases"].pop()
    elif mutation == "group":
        manifest["cases"][0]["group"] = "validation"
    elif mutation == "duplicate":
        manifest["cases"][1]["input_ids"] = manifest["cases"][0]["input_ids"].copy()
    elif mutation == "vocab":
        manifest["cases"][0]["input_ids"][0] = 100000
    elif mutation == "suffix":
        manifest["cases"][0]["continuation_ids"].pop()
    elif mutation == "length":
        manifest["cases"][0]["input_length"] = 8
    else:
        manifest["continuation_steps"] = 15
    rehash(manifest)
    with pytest.raises(ValueError):
        select(manifest)


def test_tamper_and_external_sha(manifest: dict) -> None:
    with pytest.raises(ValueError):
        collector.calibration_cases(
            manifest, revision="fixed-rev", expected_sha="0" * 64, vocab_size=100000
        )
    manifest["cases"][0]["input_ids"][0] += 1
    with pytest.raises(ValueError):
        select(manifest)


def test_output_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        collector.new_output_path(collector.REPO_ROOT.parent / "outside-reference")
    with pytest.raises(ValueError):
        collector.new_output_path(collector.REPO_ROOT)
    with pytest.raises(FileExistsError):
        collector.new_output_path(Path(__file__))
    new = collector.REPO_ROOT / ".tmp" / "never-created-collection"
    assert collector.new_output_path(new) == new.resolve()
    assert not new.exists()


def test_symlink_escape_and_dangling(tmp_path: Path) -> None:
    link = collector.REPO_ROOT / ".tmp" / f"link-{tmp_path.name}"
    link.parent.mkdir(exist_ok=True)
    try:
        link.symlink_to(collector.REPO_ROOT.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    try:
        with pytest.raises(ValueError):
            collector.new_output_path(link / "new")
        with pytest.raises((ValueError, FileExistsError)):
            collector.new_output_path(link)
        link.unlink()
        link.symlink_to(
            collector.REPO_ROOT / ".tmp" / f"missing-{tmp_path.name}",
            target_is_directory=True,
        )
        with pytest.raises(FileExistsError):
            collector.new_output_path(link)
    finally:
        link.unlink()


def test_exclusive_json_and_file_hash(tmp_path: Path) -> None:
    output = tmp_path / "one.json"
    collector.write_json(output, {"汉": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"汉": 1}
    assert collector.sha_file(output) == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        collector.write_json(output, {})


def test_hf_loading_sets_preserve_exact_values_as_json(tmp_path: Path) -> None:
    raw = {
        "missing_keys": {"z", "a"},
        "unexpected_keys": {"model.visual.weight"},
        "mismatched_keys": {("weight", (2, 3), (3, 2))},
        "error_msgs": ["example"],
    }
    normalized = collector.normalize_loading_info(raw)
    collector.write_json(tmp_path / "loading.json", normalized)
    saved = json.loads((tmp_path / "loading.json").read_text())
    assert saved == {
        "missing_keys": ["a", "z"],
        "unexpected_keys": ["model.visual.weight"],
        "mismatched_keys": [["weight", [2, 3], [3, 2]]],
        "error_msgs": ["example"],
    }
    assert raw["missing_keys"] == {"z", "a"}
    empty = {
        "missing_keys": set(),
        "unexpected_keys": set(),
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    assert collector.normalize_loading_info(empty) == {key: [] for key in empty}
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        with pytest.raises(ValueError):
            collector.normalize_loading_info({**empty, field: []})
    with pytest.raises(ValueError):
        collector.normalize_loading_info({**empty, "error_msgs": set()})


@pytest.mark.parametrize(
    "reference,npu,threads",
    [
        ("cpu-fp32-eager", "0", "1"),
        ("npu-bf16-sdpa", None, "1"),
        ("npu-bf16-sdpa", "-1", "1"),
        ("cpu-fp32-sdpa", None, "0"),
    ],
)
def test_cli_invalid_device_and_threads(
    tmp_path: Path, reference: str, npu: str | None, threads: str
) -> None:
    argv = [
        "--reference",
        reference,
        "--model",
        str(tmp_path),
        "--revision",
        "r",
        "--manifest",
        str(tmp_path / "missing"),
        "--manifest-sha",
        "0" * 64,
        "--output",
        str(tmp_path / "output"),
        "--threads",
        threads,
    ]
    if npu is not None:
        argv += ["--npu-device", npu]
    with pytest.raises(SystemExit):
        collector.parse_args(argv)
    assert not (tmp_path / "output").exists()


def test_all_cli_choices_are_required() -> None:
    with pytest.raises(SystemExit):
        collector.parse_args([])


@pytest.mark.parametrize(
    "omitted",
    [
        "--reference",
        "--model",
        "--revision",
        "--manifest",
        "--manifest-sha",
        "--output",
        "--threads",
    ],
)
def test_individually_required_cli_arguments(omitted: str) -> None:
    argv = [
        "--reference",
        "cpu-fp32-eager",
        "--model",
        "model",
        "--revision",
        "r",
        "--manifest",
        "manifest",
        "--manifest-sha",
        "sha",
        "--output",
        "output",
        "--threads",
        "1",
    ]
    index = argv.index(omitted)
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        collector.parse_args(argv)


@pytest.mark.parametrize("reference", collector.REFERENCES)
def test_explicit_valid_cli_references(reference: str) -> None:
    argv = [
        "--reference",
        reference,
        "--model",
        "model",
        "--revision",
        "r",
        "--manifest",
        "manifest",
        "--manifest-sha",
        "sha",
        "--output",
        "output",
        "--threads",
        "2",
    ]
    if reference.startswith("npu"):
        argv += ["--npu-device", "0"]
    args = collector.parse_args(argv)
    assert args.reference == reference and args.threads == 2
    assert args.npu_device == (0 if reference.startswith("npu") else None)


def test_guard_before_torch_and_output(tmp_path: Path, manifest: dict) -> None:
    model = tmp_path / "model"
    model.mkdir()
    collector.write_json(model / "config.json", {"text_config": {"vocab_size": 100000}})
    frozen = tmp_path / "manifest.json"
    collector.write_json(frozen, manifest)
    output = collector.REPO_ROOT / ".tmp" / f"guard-output-{tmp_path.name}"
    program = "import sys; from scripts.collect_homework1_references import main;\ntry: main()\nexcept ValueError: pass\nelse: raise AssertionError('guard did not fail')\nassert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--reference",
            "cpu-fp32-eager",
            "--model",
            str(model),
            "--revision",
            "fixed-rev",
            "--manifest",
            str(frozen),
            "--manifest-sha",
            "0" * 64,
            "--output",
            str(output),
            "--threads",
            "1",
        ],
        cwd=collector.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_tensor_comparison() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([0.0, 0.0])
    exact = collector.compare_tensor(reference, reference, torch=torch)
    assert exact["max_abs"] == exact["relative_l2"] == 0
    different = collector.compare_tensor(
        torch.tensor([3.0, 4.0]), reference, torch=torch
    )
    assert different["max_abs"] == 4
    assert different["relative_l2"] == pytest.approx(5e12)
    for invalid in (
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([float("inf"), 0.0]),
        torch.tensor([0.0]),
    ):
        with pytest.raises(ValueError):
            collector.compare_tensor(invalid, reference, torch=torch)
        with pytest.raises(ValueError):
            collector.compare_tensor(reference, invalid, torch=torch)


def test_projection_identity() -> None:
    torch = pytest.importorskip("torch")
    identities = []
    for dtype in ("float32", "bfloat16"):
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.data = torch.tensor([[1.003, 2.006], [3.009, 4.012]])
        identities.append(
            collector.project_weights(model, compute_dtype=dtype, torch=torch)
        )
    assert (
        identities[0]["global_weight_sha256"] == identities[1]["global_weight_sha256"]
    )
    for dtype in ("float32", "bfloat16"):
        model = torch.nn.Module()
        model.register_parameter("A_log", torch.nn.Parameter(torch.tensor([1.003])))
        model.register_parameter("dt_bias", torch.nn.Parameter(torch.tensor([2.006])))
        identity = collector.project_weights(model, compute_dtype=dtype, torch=torch)
        assert all(
            row["projection_dtype"] == "float32"
            and row["actual_dtype"] == "torch.float32"
            for row in identity["parameters"]
        )


def test_case_collection_windows_and_raw_states(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Layer:
        def __init__(self, value: int) -> None:
            self.conv_states = torch.tensor([value], dtype=torch.float32)
            self.recurrent_states = torch.tensor([[value]], dtype=torch.float32)

    class Cache:
        def __init__(self, length: int) -> None:
            self.length = length
            self.layers = [Layer(length) for _ in range(18)]

        def get_seq_length(self) -> int:
            return self.length

    calls = []

    class TextConfig:
        vocab_size = 7

        def get_text_config(self) -> "TextConfig":
            return self

    class Model:
        config = TextConfig()

        def __call__(self, **kwargs: object) -> object:
            length = kwargs["attention_mask"].shape[1]
            calls.append(kwargs)
            assert kwargs["use_cache"] is True and kwargs["logits_to_keep"] == 1
            assert kwargs["attention_mask"].tolist() == [[1] * length]
            assert kwargs["position_ids"][0, -1].item() == length - 1
            return SimpleNamespace(
                logits=torch.arange(7.0).reshape(1, 1, 7), past_key_values=Cache(length)
            )

    case = {"case_id": "tiny", "input_ids": [1, 2], "continuation_ids": [3] * 16}
    collector.collect_case(
        Model(),
        case,
        output=tmp_path,
        index=0,
        device="cpu",
        torch=torch,
        linear_layer_type=Layer,
    )
    assert len(calls) == 34
    for step in range(17):
        cached, recomputed = calls[step * 2 : step * 2 + 2]
        assert cached["input_ids"].shape[1] == (2 if step == 0 else 1)
        assert recomputed["input_ids"].shape[1] == 2 + step
        assert cached["input_ids"].tolist() == ([[1, 2]] if step == 0 else [[3]])
        assert cached["position_ids"].tolist() == (
            [[0, 1]] if step == 0 else [[step + 1]]
        )
        assert recomputed["input_ids"].tolist() == [[1, 2] + [3] * step]
        assert recomputed["position_ids"].tolist() == [list(range(2 + step))]
        if step == 0:
            assert cached["past_key_values"] is None
        else:
            assert cached["past_key_values"].get_seq_length() == step + 1
        assert recomputed["past_key_values"] is None
    saved = torch.load(tmp_path / "case_00.pt", weights_only=True)
    assert saved["cached_logits"].shape == (17, 7)
    assert len(saved["final_gdn_states"]) == 36
    assert all(
        set(pair) == {"cached", "recomputed"}
        for pair in saved["final_gdn_states"].values()
    )
    assert all(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and (tensor == 18).all()
        for pair in saved["final_gdn_states"].values()
        for tensor in pair.values()
    )
    observations = json.loads((tmp_path / "case_00.json").read_text())
    assert len(observations["steps"]) == 17
    assert all(len(step["states"]) == 36 for step in observations["steps"])
    assert [(row["step"], row["prefix_length"]) for row in observations["steps"]] == [
        (step, 2 + step) for step in range(17)
    ]
    with pytest.raises(FileExistsError):
        collector.collect_case(
            Model(),
            case,
            output=tmp_path,
            index=0,
            device="cpu",
            torch=torch,
            linear_layer_type=Layer,
        )


def test_failure_is_preserved(
    tmp_path: Path, manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    collector.write_json(model / "config.json", {"text_config": {"vocab_size": 100000}})
    frozen = tmp_path / "inputs.json"
    collector.write_json(frozen, manifest)
    output = tmp_path / "failure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--reference",
            "cpu-fp32-eager",
            "--model",
            str(model),
            "--revision",
            "fixed-rev",
            "--manifest",
            str(frozen),
            "--manifest-sha",
            manifest["manifest_sha256"],
            "--output",
            str(output),
            "--threads",
            "1",
        ],
    )

    def fail(args: object, cases: list[dict], *, output: Path, collect_case_fn) -> None:
        assert collect_case_fn is collector.collect_case
        collector.write_json(output / "completed.json", {"saved": True})
        raise ValueError("deliberate collection failure")

    monkeypatch.setattr(collector, "collect", fail)
    with pytest.raises(ValueError, match="deliberate"):
        collector.main()
    failure = json.loads((output / "failed.json").read_text())
    assert failure == {
        "exception_type": "ValueError",
        "message": "deliberate collection failure",
        "completed_files": 2,
    }
    assert (output / "completed.json").exists()
    previous = (output / "failed.json").read_bytes()
    with pytest.raises(FileExistsError):
        collector.main()
    assert (output / "failed.json").read_bytes() == previous
