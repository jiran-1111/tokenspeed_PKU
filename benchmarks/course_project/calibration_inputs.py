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

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

_INPUT_LENGTHS = (1, 8, 127, 128, 129, 512, 1024, 2048)
_CONTINUATION_STEPS = 16
_CONTINUATION_TEXT = (
    "\nContinuation: verify the ledger, then state one arithmetic check and a short explanation. "
    "后续检查记录并给出一项计算。"
)
_FAMILIES: dict[str, tuple[tuple[str, str, Callable[[int], str]], ...]] = {
    "calibration": (
        (
            "cal_weather_math",
            "甲校准气象数学记录",
            lambda i: f"站点{i}: 温度={(i * 7) % 31}，降水={(i * 11) % 97}；检查整数和{i + i * i}。",
        ),
        (
            "cal_python_bits",
            "Blue calibration Python bit laboratory",
            lambda i: f"record_{i}: value={i * 13}; mask={i % 17}; result=value ^ mask; assert result >= 0",
        ),
    ),
    "validation": (
        (
            "val_history_counts",
            "乙验证历史清点档案",
            lambda i: f"档案{i}: 年份={1500 + i % 500}，卷数={i % 23 + 1}，合计={(i % 23 + 1) * 3}；核对分类。",
        ),
        (
            "val_sql_botany",
            "Green validation SQL botanical ledger",
            lambda i: f"plant_{i}: species=fern_{i % 19}; leaves={i % 29}; SELECT SUM(leaves) FROM garden_{i % 7};",
        ),
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encode_prefix(
    *, encode: Callable[[str], list[int]], text: str, length: int, vocab_size: int
) -> list[int]:
    ids = encode(text)
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or not 0 <= token_id < vocab_size
        for token_id in ids
    ):
        raise ValueError("Encoded IDs must be non-bool integers within the vocabulary")
    if len(ids) < length:
        raise ValueError(
            f"Encoded text is too short: need {length} IDs, got {len(ids)}"
        )
    return ids[:length]


def build_manifest(
    *, encode: Callable[[str], list[int]], vocab_size: int, tokenizer_revision: str
) -> dict:
    """Build the fixed native input manifest without loading a tokenizer or model.

    Args:
        encode: Encode full source text into token IDs without special tokens.
        vocab_size: Positive model vocabulary bound for every encoded ID.
        tokenizer_revision: Nonempty fixed tokenizer identity recorded verbatim.

    Returns:
        JSON-serializable input cases, continuation IDs, and canonical SHA256 hashes.

    Raises:
        ValueError: Invalid identity, short/invalid tokenization, or duplicate inputs.
    """
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise ValueError("vocab_size must be a positive non-bool integer")
    if not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
        raise ValueError("tokenizer_revision must be a nonempty string")

    continuation_ids = _encode_prefix(
        encode=encode,
        text=_CONTINUATION_TEXT,
        length=_CONTINUATION_STEPS,
        vocab_size=vocab_size,
    )
    continuation_hash = _text_sha256(_canonical_json(continuation_ids))
    cases: list[dict] = []
    seen_inputs: set[tuple[int, ...]] = set()
    for group, families in _FAMILIES.items():
        for length in _INPUT_LENGTHS:
            for variant, (family, header, record) in enumerate(families):
                source_text = "\n".join(
                    [header, f"input_length={length}"]
                    + [record(i) for i in range(1, 1025)]
                )
                input_ids = _encode_prefix(
                    encode=encode,
                    text=source_text,
                    length=length,
                    vocab_size=vocab_size,
                )
                input_tuple = tuple(input_ids)
                if input_tuple in seen_inputs:
                    raise ValueError(
                        f"duplicate native input IDs for {group}/{length}/{variant}"
                    )
                seen_inputs.add(input_tuple)
                cases.append(
                    {
                        "case_id": f"{group}/{length}/{variant}",
                        "group": group,
                        "input_length": length,
                        "source_family": family,
                        "source_text": source_text,
                        "source_text_sha256": _text_sha256(source_text),
                        "input_ids": input_ids,
                        "input_ids_sha256": _text_sha256(_canonical_json(input_ids)),
                        "continuation_text": _CONTINUATION_TEXT,
                        "continuation_ids": continuation_ids.copy(),
                        "continuation_ids_sha256": continuation_hash,
                    }
                )
    manifest = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "tokenizer_revision": tokenizer_revision,
        "continuation_steps": _CONTINUATION_STEPS,
        "cases": cases,
    }
    manifest["manifest_sha256"] = _text_sha256(_canonical_json(manifest))
    return manifest


def write_manifest(manifest: dict, output: Path) -> None:
    """Validate manifest SHA256 and create output exclusively as stable UTF-8 JSON.

    Args:
        manifest: Previously built manifest with its original manifest_sha256.
        output: New file path in an existing parent directory.

    Raises:
        ValueError: The manifest content no longer matches its stored SHA256.
        FileExistsError: The output already exists; its content is preserved.
    """
    content = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _text_sha256(_canonical_json(content)):
        raise ValueError("Manifest SHA256 does not match its content")
    serialized = _canonical_json(manifest)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.write("\n")
