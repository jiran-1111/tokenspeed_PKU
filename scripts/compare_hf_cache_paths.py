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
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import load_model_config, model_load_path
from benchmarks.course_project.correctness import collect_stop_token_ids


def build_exact_prefix(
    input_token_ids: list[int],
    engine_output_ids: list[int],
    reference_output_ids: list[int],
    *,
    expected_target_index: int | None = None,
) -> tuple[list[int], int]:
    """Return prompt + shared output prefix and its first divergent output index."""
    mismatch_index = next(
        (
            index
            for index, (engine_id, reference_id) in enumerate(
                zip(engine_output_ids, reference_output_ids)
            )
            if engine_id != reference_id
        ),
        None,
    )
    if mismatch_index is None:
        if len(engine_output_ids) != len(reference_output_ids):
            mismatch_index = min(len(engine_output_ids), len(reference_output_ids))
        else:
            raise ValueError("outputs have no reference mismatch to diagnose")
    if mismatch_index >= min(len(engine_output_ids), len(reference_output_ids)):
        raise ValueError("first mismatch has no target token in both outputs")
    if expected_target_index is not None and expected_target_index != mismatch_index:
        raise ValueError(
            "reported target index does not equal the first mismatch "
            f"({expected_target_index} != {mismatch_index})"
        )
    if engine_output_ids[:mismatch_index] != reference_output_ids[:mismatch_index]:
        raise ValueError("engine and reference output prefixes do not match")
    return input_token_ids + engine_output_ids[:mismatch_index], mismatch_index


def build_prefix_allowed_tokens_fn(
    prompt_length: int,
    forced_output_ids: list[int],
    *,
    target_index: int,
    vocab_size: int,
):
    """Force prior output tokens, then leave the target-step vocabulary open."""
    if prompt_length < 1:
        raise ValueError("prompt_length must be at least 1")
    if target_index < 0 or len(forced_output_ids) != target_index:
        raise ValueError("forced_output_ids must contain exactly the prior tokens")
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")

    def allow_recorded_token(batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        input_length = (
            int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
        )
        output_index = input_length - prompt_length
        if output_index < 0 or output_index > target_index:
            raise ValueError("generation requested a token outside the forced prefix")
        if output_index == target_index:
            return list(range(vocab_size))
        return [int(forced_output_ids[output_index])]

    return allow_recorded_token


def validate_generated_target_prefix(
    generated_token_ids: list[int],
    forced_prefix_ids: list[int],
    *,
    target_index: int,
) -> int:
    if target_index < 0 or len(forced_prefix_ids) != target_index:
        raise ValueError("forced prefix length must equal the target output index")
    if len(generated_token_ids) != target_index + 1:
        raise ValueError("generation did not stop at exactly the target step")
    if generated_token_ids[:target_index] != forced_prefix_ids:
        raise ValueError("generated tokens do not reproduce the forced prefix")
    return int(generated_token_ids[target_index])


def normalize_top_k(token_ids: list[int], logits: list[float], *, k: int = 2) -> dict:
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(token_ids) != len(logits) or len(token_ids) < k:
        raise ValueError("top-k token IDs and logits must contain at least k pairs")
    return {
        "top_token_ids": [int(token_id) for token_id in token_ids[:k]],
        "top_logits": [float(value) for value in logits[:k]],
    }


def get_transformers_version(transformers_module: Any) -> str:
    return str(getattr(transformers_module, "__version__", "unknown"))


def select_tokenspeed_topk(
    rows: list[dict], *, input_token_id: int, position: int
) -> dict[str, dict]:
    aligned: dict[str, dict] = {}
    for mode in ("continuous", "full_recompute"):
        matches = [
            row
            for row in rows
            if row.get("capture_mode") == mode
            and (row.get("forward") or {}).get("input_id_last") == input_token_id
            and (row.get("forward") or {}).get("position_last") == position
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one TokenSpeed {mode} row for "
                f"input_token_id={input_token_id}, position={position}; "
                f"found {len(matches)}"
            )
        row = matches[0]
        top_k = normalize_top_k(
            row.get("top_token_ids") or [], row.get("top_logits") or []
        )
        aligned[mode] = top_k
    return aligned


def write_json_no_overwrite(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"input contains no records: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"input must contain JSON objects: {path}")
    return rows


def _select_case(
    input_rows: list[dict], comparison_rows: list[dict], case: str, turn: int
) -> tuple[dict, dict]:
    source_matches = [
        row
        for row in input_rows
        if row.get("case") == case and row.get("turn_index") == turn
    ]
    compare_matches = [
        row
        for row in comparison_rows
        if row.get("case") == case and row.get("turn_index") == turn
    ]
    if len(source_matches) != 1 or len(compare_matches) != 1:
        raise ValueError(
            f"expected exactly one source and comparison row for "
            f"case={case!r}, turn={turn}; found "
            f"{len(source_matches)} and {len(compare_matches)}"
        )
    return source_matches[0], compare_matches[0]


def _validate_case(source: dict, comparison: dict) -> tuple[list[int], int, int, int]:
    input_token_ids = source.get("client_input_token_ids")
    engine_ids = source.get("output_token_ids")
    reference_ids = comparison.get("reference_output_token_ids")
    compare_engine_ids = comparison.get("engine_output_token_ids")
    if not all(
        isinstance(ids, list) and all(isinstance(token_id, int) for token_id in ids)
        for ids in (input_token_ids, engine_ids, reference_ids, compare_engine_ids)
    ):
        raise ValueError("source and comparison rows must contain token ID arrays")
    if engine_ids != compare_engine_ids:
        raise ValueError(
            "reference comparison does not match the selected engine output"
        )
    prefix, target_index = build_exact_prefix(
        input_token_ids,
        engine_ids,
        reference_ids,
        expected_target_index=comparison.get("first_mismatch_index"),
    )
    if len(prefix) == 0:
        raise ValueError("exact prefix is empty")
    return prefix, target_index, engine_ids[target_index], reference_ids[target_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Transformers cached and full-recompute logits at the first "
            "TokenSpeed/reference divergence."
        )
    )
    parser.add_argument("config", type=Path, help="JSON-compatible YAML model config")
    parser.add_argument("input", type=Path, help="Native-ID correctness JSONL")
    parser.add_argument(
        "comparison", type=Path, help="Transformers reference comparison JSON"
    )
    parser.add_argument(
        "tokenspeed_logits", type=Path, help="TokenSpeed top-k logits JSONL"
    )
    parser.add_argument("--case", default="long_context_recall")
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw" / "correctness" / "decode_diagnostic",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _capture_target_top_k(
    *,
    model: Any,
    torch: Any,
    input_token_ids: list[int],
    forced_output_ids: list[int],
    target_index: int,
    use_cache: bool,
    pad_token_id: int,
) -> dict:
    input_tensor = torch.tensor([input_token_ids], dtype=torch.long, device="npu:0")
    attention_mask = torch.ones_like(input_tensor)
    generation_kwargs = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "pad_token_id": pad_token_id,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": target_index + 1,
        "prefix_allowed_tokens_fn": build_prefix_allowed_tokens_fn(
            len(input_token_ids),
            forced_output_ids[:target_index],
            target_index=target_index,
            vocab_size=int(model.config.vocab_size),
        ),
        "use_cache": use_cache,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    with torch.inference_mode():
        generation = model.generate(**generation_kwargs)
    generated_ids = [
        int(token_id)
        for token_id in generation.sequences[0][len(input_token_ids) :].tolist()
    ]
    target_generated_id = validate_generated_target_prefix(
        generated_ids,
        forced_output_ids[:target_index],
        target_index=target_index,
    )
    step_logits = getattr(generation, "logits", None)
    if step_logits is None or len(step_logits) != target_index + 1:
        actual_count = 0 if step_logits is None else len(step_logits)
        raise RuntimeError(
            "Transformers did not return logits through the target step "
            f"(expected {target_index + 1}, got {actual_count})"
        )
    target_logits = step_logits[target_index][0]
    count = min(2, int(target_logits.shape[-1]))
    values, token_ids = torch.topk(target_logits, k=count)
    return {
        "generated_target_token_id": target_generated_id,
        **normalize_top_k(token_ids.tolist(), values.tolist(), k=count),
    }


def main() -> None:
    args = parse_args()
    if args.turn < 1:
        raise ValueError("--turn must be at least 1")
    config = load_model_config(args.config)
    input_rows = _read_jsonl(args.input)
    tokenspeed_rows = _read_jsonl(args.tokenspeed_logits)
    comparison_data = json.loads(args.comparison.read_text(encoding="utf-8"))
    if comparison_data.get("model_id") != config["model_id"]:
        raise ValueError("reference comparison model_id does not match config")
    if comparison_data.get("model_revision") != config["model_revision"]:
        raise ValueError("reference comparison revision does not match config")
    source, comparison = _select_case(
        input_rows, comparison_data.get("comparisons", []), args.case, args.turn
    )
    prefix, target_index, engine_target_id, reference_target_id = _validate_case(
        source, comparison
    )
    input_token_id = prefix[-1]
    target_position = len(prefix) - 1
    tokenspeed_top_k = select_tokenspeed_topk(
        tokenspeed_rows, input_token_id=input_token_id, position=target_position
    )
    if args.output is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = args.output_dir / (
            f"{config['served_model_name']}-{args.case}-turn{args.turn}-"
            f"hf-cache-paths-{run_id}.json"
        )
    else:
        output_path = args.output
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    import torch
    import torch_npu  # noqa: F401 - registers the NPU device with PyTorch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available for the reference comparison")
    torch.npu.set_device(0)
    model_path = model_load_path(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=config["model_revision"],
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("npu:0")
    model.eval()

    stop_token_ids = collect_stop_token_ids(
        model.config.eos_token_id,
        model.generation_config.eos_token_id,
        tokenizer.eos_token_id,
        getattr(tokenizer, "additional_stop_token_ids", None),
    )
    generated_prefix = source["output_token_ids"][:target_index]
    stop_token_set = set(stop_token_ids)
    if any(token_id in stop_token_set for token_id in generated_prefix):
        raise ValueError("target prefix contains a stop token before the target step")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None and stop_token_ids:
        pad_token_id = stop_token_ids[0]
    if pad_token_id is None:
        raise ValueError("no pad token ID is available for HF generation")

    hf_cached = _capture_target_top_k(
        model=model,
        torch=torch,
        input_token_ids=source["client_input_token_ids"],
        forced_output_ids=source["output_token_ids"],
        target_index=target_index,
        use_cache=True,
        pad_token_id=int(pad_token_id),
    )
    hf_full_recompute = _capture_target_top_k(
        model=model,
        torch=torch,
        input_token_ids=source["client_input_token_ids"],
        forced_output_ids=source["output_token_ids"],
        target_index=target_index,
        use_cache=False,
        pad_token_id=int(pad_token_id),
    )
    prefix_digest = hashlib.sha256(
        json.dumps(prefix, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "reference": "Transformers AutoModelForCausalLM.generate",
        "transformers_version": get_transformers_version(transformers),
        "device": "npu:0",
        "case": args.case,
        "turn_index": args.turn,
        "target_output_index": target_index,
        "first_divergence_verified": True,
        "engine_target_token_id": engine_target_id,
        "reference_target_token_id": reference_target_id,
        "sampling": {"do_sample": False, "num_beams": 1},
        "hf_logits": "generate.output_logits before logits processors",
        "exact_prefix": {
            "token_count": len(prefix),
            "sha256": prefix_digest,
            "last_input_token_id": input_token_id,
            "target_position": target_position,
        },
        "top_k": 2,
        "hf": {
            "use_cache_true": {"use_cache": True, **hf_cached},
            "use_cache_false": {"use_cache": False, **hf_full_recompute},
        },
        "tokenspeed": tokenspeed_top_k,
        "source_capture": str(args.input),
        "reference_comparison": str(args.comparison),
        "tokenspeed_logits_capture": str(args.tokenspeed_logits),
    }
    write_json_no_overwrite(output_path, result)
    print(output_path)


if __name__ == "__main__":
    main()
