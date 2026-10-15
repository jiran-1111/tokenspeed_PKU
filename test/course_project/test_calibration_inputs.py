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
from pathlib import Path

import pytest

from benchmarks.course_project.calibration_inputs import build_manifest, write_manifest


def encode(text: str) -> list[int]:
    return [ord(char) for char in text]


def digest(value: object) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@pytest.fixture
def manifest() -> dict:
    return build_manifest(
        encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev"
    )


def test_manifest_matrix_and_group_disjointness(manifest: dict) -> None:
    assert manifest["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert manifest["tokenizer_revision"] == "fixed-rev"
    assert manifest["continuation_steps"] == 16
    cases = manifest["cases"]
    assert len(cases) == 32
    for group in ("calibration", "validation"):
        rows = [row for row in cases if row["group"] == group]
        assert sorted(row["input_length"] for row in rows) == sorted(
            [1, 8, 127, 128, 129, 512, 1024, 2048] * 2
        )
        assert all(len(row["input_ids"]) == row["input_length"] for row in rows)
        assert all(len(row["continuation_ids"]) == 16 for row in rows)
    assert len({row["case_id"] for row in cases}) == 32
    assert len({tuple(row["input_ids"]) for row in cases}) == 32
    assert {row["source_family"] for row in cases if row["group"] == "calibration"} == {
        "cal_weather_math",
        "cal_python_bits",
    }
    assert {row["source_family"] for row in cases if row["group"] == "validation"} == {
        "val_history_counts",
        "val_sql_botany",
    }


def test_sources_and_continuation_are_actual_fixed_text(manifest: dict) -> None:
    families = {
        "cal_weather_math": (
            "甲校准气象数学记录",
            lambda i: f"站点{i}: 温度={(i * 7) % 31}，降水={(i * 11) % 97}；检查整数和{i + i * i}。",
        ),
        "cal_python_bits": (
            "Blue calibration Python bit laboratory",
            lambda i: f"record_{i}: value={i * 13}; mask={i % 17}; result=value ^ mask; assert result >= 0",
        ),
        "val_history_counts": (
            "乙验证历史清点档案",
            lambda i: f"档案{i}: 年份={1500 + i % 500}，卷数={i % 23 + 1}，合计={(i % 23 + 1) * 3}；核对分类。",
        ),
        "val_sql_botany": (
            "Green validation SQL botanical ledger",
            lambda i: f"plant_{i}: species=fern_{i % 19}; leaves={i % 29}; SELECT SUM(leaves) FROM garden_{i % 7};",
        ),
    }
    continuation = (
        "\nContinuation: verify the ledger, then state one arithmetic check and a short explanation. "
        "后续检查记录并给出一项计算。"
    )
    for row in manifest["cases"]:
        header, record = families[row["source_family"]]
        expected = "\n".join(
            [header, f"input_length={row['input_length']}"]
            + [record(i) for i in range(1, 1025)]
        )
        assert row["source_text"] == expected
        assert row["input_ids"] == encode(expected)[: row["input_length"]]
        assert row["continuation_text"] == continuation
        assert row["continuation_ids"] == encode(continuation)[:16]


def test_content_hashes_are_reproducible(manifest: dict) -> None:
    assert manifest == build_manifest(
        encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev"
    )
    content = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    assert manifest["manifest_sha256"] == digest(content)
    for row in manifest["cases"]:
        assert (
            row["source_text_sha256"]
            == hashlib.sha256(row["source_text"].encode("utf-8")).hexdigest()
        )
        assert row["input_ids_sha256"] == digest(row["input_ids"])
        assert row["continuation_ids_sha256"] == digest(row["continuation_ids"])


def test_write_refuses_existing_output(manifest: dict, tmp_path: Path) -> None:
    output = tmp_path / "inputs.json"
    write_manifest(manifest, output)
    original = output.read_bytes()
    assert json.loads(original.decode("utf-8")) == manifest
    with pytest.raises(FileExistsError):
        write_manifest(manifest, output)
    assert output.read_bytes() == original
    second = tmp_path / "second.json"
    write_manifest(manifest, second)
    assert second.read_bytes() == original


def test_write_refuses_tampered_manifest(manifest: dict, tmp_path: Path) -> None:
    manifest["cases"][0]["input_ids"][0] += 1
    output = tmp_path / "inputs.json"
    with pytest.raises(ValueError):
        write_manifest(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "ids", [[], [-1] * 3000, [True] * 3000, [100000] * 3000, [1.0] * 3000]
)
def test_invalid_or_short_tokenization_rejected(ids: list) -> None:
    with pytest.raises(ValueError):
        build_manifest(
            encode=lambda text: ids, vocab_size=100000, tokenizer_revision="fixed-rev"
        )


@pytest.mark.parametrize("vocab_size", [0, -1, True, 1.0, "100000"])
def test_invalid_vocab_size_rejected(vocab_size: object) -> None:
    with pytest.raises(ValueError):
        build_manifest(
            encode=encode, vocab_size=vocab_size, tokenizer_revision="fixed-rev"
        )


@pytest.mark.parametrize("revision", ["", "   ", None, 1])
def test_invalid_revision_rejected(revision: object) -> None:
    with pytest.raises(ValueError):
        build_manifest(encode=encode, vocab_size=100000, tokenizer_revision=revision)


def test_duplicate_native_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest(
            encode=lambda text: [1] * 3000,
            vocab_size=100000,
            tokenizer_revision="fixed-rev",
        )


@pytest.mark.parametrize("short_source", [True, False])
def test_source_and_continuation_short_encoding_rejected(short_source: bool) -> None:
    def short_encode(text: str) -> list[int]:
        if text.startswith("\nContinuation:") == short_source:
            return encode(text)
        return encode(text)[:8]

    with pytest.raises(ValueError, match="short"):
        build_manifest(
            encode=short_encode, vocab_size=100000, tokenizer_revision="fixed-rev"
        )


def test_invalid_id_after_prefix_is_rejected() -> None:
    def invalid_tail(text: str) -> list[int]:
        return encode(text) + [True]

    with pytest.raises(ValueError):
        build_manifest(
            encode=invalid_tail, vocab_size=100000, tokenizer_revision="fixed-rev"
        )
