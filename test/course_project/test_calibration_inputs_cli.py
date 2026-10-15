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

import builtins
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_homework1_calibration_inputs.py"


def run_cli(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    assert SCRIPT.is_file(), "The calibration inputs CLI has not been implemented"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.fixture
def block_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"transformers", "torch", "torch_npu"}:
            attempted.append(name)
            raise AssertionError(f"Unexpected dependency import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return attempted


@pytest.fixture
def model(tmp_path: Path) -> Path:
    assert tmp_path.resolve().is_relative_to(REPO_ROOT)
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"text_config": {"vocab_size": 100000}}), encoding="utf-8"
    )
    return directory


class CharacterTokenizer:
    """Offline adapter for the external tokenizer API; build/write remain real."""

    def __init__(self, *, length: int, encoding: str) -> None:
        self.length = length
        self.encoding = encoding
        self.special_token_arguments: list[bool] = []

    def __len__(self) -> int:
        return self.length

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.special_token_arguments.append(add_special_tokens)
        if self.encoding == "short":
            return [1]
        if self.encoding == "illegal":
            return [100000] * len(text)
        return [ord(character) for character in text]


def install_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tokenizer: CharacterTokenizer
) -> list[tuple[Path, str, bool]]:
    calls: list[tuple[Path, str, bool]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(
            model: Path, *, revision: str, local_files_only: bool
        ) -> CharacterTokenizer:
            calls.append((model, revision, local_files_only))
            return tokenizer

    module = ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def arguments(model: Path, output: Path) -> list[str]:
    return ["--model", str(model), "--revision", "fixed-rev", "--output", str(output)]


def test_help_does_not_import_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    block_dependencies: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, ["--help"])
    assert error.value.code == 0
    assert "no model inference" in capsys.readouterr().out
    assert block_dependencies == []


@pytest.mark.parametrize("missing", ["--model", "--revision", "--output"])
def test_behavior_arguments_are_required(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    block_dependencies: list[str],
) -> None:
    supplied = arguments(tmp_path / "absent-model", tmp_path / "inputs.json")
    index = supplied.index(missing)
    del supplied[index : index + 2]
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, supplied)
    assert error.value.code == 2
    assert block_dependencies == []


def test_existing_output_is_preserved_before_config_or_dependency_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block_dependencies: list[str]
) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"existing content")
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, arguments(tmp_path / "absent-model", output))
    assert output.read_bytes() == b"existing content"
    assert block_dependencies == []


def test_outside_output_is_rejected_before_config_or_dependency_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block_dependencies: list[str]
) -> None:
    output = REPO_ROOT.parent / f"{tmp_path.name}-outside" / "inputs.json"
    assert not output.parent.exists()
    with pytest.raises(ValueError, match="repository"):
        run_cli(monkeypatch, arguments(tmp_path / "absent-model", output))
    assert not output.parent.exists()
    assert block_dependencies == []


def test_symlink_escape_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block_dependencies: list[str]
) -> None:
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(REPO_ROOT.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlink creation is unavailable")
    output = link / f"{tmp_path.name}-outside" / "inputs.json"
    assert not output.parent.exists()
    with pytest.raises(ValueError, match="repository"):
        run_cli(monkeypatch, arguments(tmp_path / "absent-model", output))
    assert not output.parent.exists()
    assert block_dependencies == []


@pytest.mark.parametrize("tokenizer_length", [1, 200000])
def test_offline_success_uses_config_vocabulary_and_stable_real_manifest(
    tokenizer_length: int,
    monkeypatch: pytest.MonkeyPatch,
    model: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    tokenizer = CharacterTokenizer(length=tokenizer_length, encoding="characters")
    calls = install_tokenizer(monkeypatch, tokenizer)
    output = tmp_path / "new-directory" / "inputs.json"
    run_cli(monkeypatch, arguments(model, output))
    original = output.read_bytes()
    manifest = json.loads(original)
    assert calls == [(model, "fixed-rev", True)]
    assert tokenizer.special_token_arguments == [False] * 33
    assert manifest["tokenizer_revision"] == "fixed-rev"
    assert len(manifest["cases"]) == 32
    for group in ("calibration", "validation"):
        cases = [case for case in manifest["cases"] if case["group"] == group]
        assert sorted(case["input_length"] for case in cases) == sorted(
            [1, 8, 127, 128, 129, 512, 1024, 2048] * 2
        )
    for case in manifest["cases"]:
        assert (
            case["input_ids"]
            == [ord(char) for char in case["source_text"]][: case["input_length"]]
        )
        assert (
            case["continuation_ids"]
            == [ord(char) for char in case["continuation_text"]][:16]
        )
    content = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    canonical = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    expected_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert manifest["manifest_sha256"] == expected_digest
    assert capsys.readouterr().out.strip() == f"32 cases; SHA256 {expected_digest}"
    second = output.parent / "second.json"
    run_cli(monkeypatch, arguments(model, second))
    assert second.read_bytes() == original
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, arguments(model, output))
    assert output.read_bytes() == original
    assert len(calls) == 2


@pytest.mark.parametrize("encoding", ["illegal", "short"])
def test_invalid_encoding_leaves_no_output_or_parent(
    encoding: str, monkeypatch: pytest.MonkeyPatch, model: Path, tmp_path: Path
) -> None:
    install_tokenizer(monkeypatch, CharacterTokenizer(length=200000, encoding=encoding))
    output = tmp_path / "not-created" / "inputs.json"
    with pytest.raises(ValueError, match="vocabulary|too short"):
        run_cli(monkeypatch, arguments(model, output))
    assert not output.parent.exists()
