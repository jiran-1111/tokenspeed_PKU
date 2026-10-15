#!/usr/bin/env python3
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

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import load_model_config, model_load_path
from benchmarks.course_project.correctness import (
    build_reference_attention_mask,
    collect_stop_token_ids,
    compare_token_ids_with_stop_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TokenSpeed output IDs with a Transformers greedy reference."
    )
    parser.add_argument("config", type=Path, help="JSON-compatible YAML model config")
    parser.add_argument("input", type=Path, help="TokenSpeed correctness JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override each record's max_new_tokens; required for older records without it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("correctness input contains no records")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    for row in rows:
        if not isinstance(row.get("output_token_ids"), list):
            raise ValueError("correctness input is missing engine output_token_ids")

    import torch
    import torch_npu  # noqa: F401 - registers the NPU device with PyTorch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available for the reference comparison")
    torch.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config),
        revision=config["model_revision"],
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path(config),
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("npu")
    model.eval()

    comparisons = []
    for row in rows:
        params = row.get("sampling_params") or {}
        max_new_tokens = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else params.get("max_new_tokens")
        )
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be supplied by the record or --max-new-tokens"
            )
        input_ids = [int(token_id) for token_id in row["client_input_token_ids"]]
        stop_token_ids = collect_stop_token_ids(
            model.config.eos_token_id,
            model.generation_config.eos_token_id,
            tokenizer.eos_token_id,
            getattr(tokenizer, "additional_stop_token_ids", None),
            params.get("stop_token_ids"),
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="npu")
        attention_mask = torch.tensor(
            [build_reference_attention_mask(input_ids)],
            dtype=torch.long,
            device="npu",
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None and stop_token_ids:
            pad_token_id = stop_token_ids[0]

        generation_kwargs = {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "pad_token_id": pad_token_id,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
        }
        if stop_token_ids:
            generation_kwargs["eos_token_id"] = stop_token_ids
        with torch.inference_mode():
            sequence = model.generate(**generation_kwargs)[0]
        reference_ids = [
            int(token_id) for token_id in sequence[len(input_ids) :].tolist()
        ]
        engine_ids = [int(token_id) for token_id in row["output_token_ids"]]
        comparison = compare_token_ids_with_stop_alignment(
            engine_ids, reference_ids, stop_token_ids
        )
        comparisons.append(
            {
                "case": row["case"],
                "turn_index": row["turn_index"],
                "engine_output_token_ids": engine_ids,
                "reference_output_token_ids": reference_ids,
                "stop_token_ids": stop_token_ids,
                **comparison,
            }
        )

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_reference_compare.json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "reference": "Transformers AutoModelForCausalLM.generate",
        "device": "npu:0",
        "sampling": {"do_sample": False, "num_beams": 1},
        "record_count": len(comparisons),
        "exact_match_count": sum(item["exact_match"] for item in comparisons),
        "stop_aligned_exact_match_count": sum(
            item["stop_aligned_exact_match"] for item in comparisons
        ),
        "comparison_policy": "exact_match compares raw IDs; stop_aligned_exact_match truncates both sequences at the first configured EOS/stop token",
        "comparisons": comparisons,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
