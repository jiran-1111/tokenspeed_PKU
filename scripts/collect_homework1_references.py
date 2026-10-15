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

"""Collect offline calibration references; numerical acceptance remains pending."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.reference_contracts import (
    forward_windows,
    reference_weight_dtypes,
)

REFERENCES = ("cpu-fp32-eager", "cpu-fp32-sdpa", "cpu-bf16-sdpa", "npu-bf16-sdpa")


def calibration_cases(
    manifest: dict, *, revision: str, expected_sha: str, vocab_size: int
) -> list[dict]:
    """Validate frozen identity and the entire input matrix; return calibration rows."""
    content = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if digest != expected_sha or digest != manifest["manifest_sha256"]:
        raise ValueError("Manifest SHA256 mismatch")
    if (
        manifest["model_id"] != "Qwen/Qwen3.5-0.8B"
        or manifest["tokenizer_revision"] != revision
    ):
        raise ValueError("Model or tokenizer revision mismatch")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    cases = manifest["cases"]
    if (
        len(cases) != 32
        or type(manifest["continuation_steps"]) is not int
        or manifest["continuation_steps"] != 16
    ):
        raise ValueError("Expected 32 cases and 16 continuation steps")
    for case in cases:
        forward_windows(
            input_ids=case["input_ids"], continuation_ids=case["continuation_ids"]
        )
        if type(case["input_length"]) is not int or case["input_length"] != len(
            case["input_ids"]
        ):
            raise ValueError("Input length mismatch")
        if any(
            token >= vocab_size
            for token in case["input_ids"] + case["continuation_ids"]
        ):
            raise ValueError("Token ID exceeds vocabulary")
    if len({tuple(case["input_ids"]) for case in cases}) != 32:
        raise ValueError("Duplicate native input IDs")
    if len({case["case_id"] for case in cases}) != 32:
        raise ValueError("Duplicate case IDs")
    expected = Counter([1, 8, 127, 128, 129, 512, 1024, 2048] * 2)
    for group in ("calibration", "validation"):
        if (
            Counter(case["input_length"] for case in cases if case["group"] == group)
            != expected
        ):
            raise ValueError(f"Invalid {group} case matrix")
    return [case for case in cases if case["group"] == "calibration"]


def new_output_path(output: Path) -> Path:
    """Resolve a new repository-contained directory without creating anything."""
    resolved = output.resolve()
    if resolved == REPO_ROOT or not resolved.is_relative_to(REPO_ROOT):
        raise ValueError("Output must be inside repository and not its root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    return resolved


def sha_file(path: Path) -> str:
    """Hash one file using bounded 1 MiB reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Create a UTF-8 JSON file exclusively; preserve all existing files."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def normalize_loading_info(loading: dict) -> dict:
    """Preserve HF 5.12 loading values as JSON-compatible sorted lists."""
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if type(loading[field]) is not set:
            raise ValueError(f"Expected HF 5.12 loading set: {field}")
    if type(loading["error_msgs"]) is not list:
        raise ValueError("Expected HF 5.12 loading error_msgs list")
    return {
        **loading,
        "missing_keys": sorted(loading["missing_keys"]),
        "unexpected_keys": sorted(loading["unexpected_keys"]),
        "mismatched_keys": sorted(loading["mismatched_keys"]),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse explicit reference protocol and validate CPU/NPU argument ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", choices=REFERENCES, required=True)
    for name in ("model", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("revision", "manifest-sha"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--npu-device", type=int)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.reference.startswith("npu"):
        if args.npu_device is None or args.npu_device < 0:
            parser.error("NPU reference requires a nonnegative --npu-device")
    elif args.npu_device is not None:
        parser.error("CPU reference forbids --npu-device")
    return args


def compare_tensor(actual: Any, reference: Any, *, torch: Any) -> dict:
    """Measure finite equal-shape tensors in CPU float64 without acceptance thresholds."""
    if actual.shape != reference.shape:
        raise ValueError("Tensor shape mismatch")
    left, right = actual.detach().cpu().double(), reference.detach().cpu().double()
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("Nonfinite tensor")
    delta = left - right
    norm = torch.linalg.vector_norm(right).item()
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "max_abs": delta.abs().max().item(),
        "relative_l2": torch.linalg.vector_norm(delta).item() / max(norm, 1e-12),
        "reference_norm": norm,
        "finite": True,
    }


def weight_identity(model: Any, *, compute_dtype: str, torch: Any) -> dict:
    """Hash canonical FP32 CPU raw parameter values one parameter at a time."""
    rows, global_digest = [], hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        projection, compute = reference_weight_dtypes(
            name=name, compute_dtype=compute_dtype
        )
        if (
            parameter.dtype
            != {"float32": torch.float32, "bfloat16": torch.bfloat16}[compute]
        ):
            raise ValueError(f"Unexpected actual parameter dtype: {name}")
        canonical = parameter.detach().cpu().float().contiguous()
        if not bool(torch.isfinite(canonical).all()):
            raise ValueError(f"Nonfinite parameter: {name}")
        sha = hashlib.sha256(canonical.numpy().tobytes()).hexdigest()
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "projection_dtype": projection,
            "actual_dtype": str(parameter.dtype),
            "sha256": sha,
        }
        global_digest.update(
            json.dumps(
                {"name": name, "shape": row["shape"], "sha256": sha},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        rows.append(row)
    return {"parameters": rows, "global_weight_sha256": global_digest.hexdigest()}


def project_weights(model: Any, *, compute_dtype: str, torch: Any) -> dict:
    """Project checkpoint parameters per contract, preserving raw Gemma weights."""
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    for name, parameter in sorted(model.named_parameters()):
        projection, compute = reference_weight_dtypes(
            name=name, compute_dtype=compute_dtype
        )
        projected = parameter.detach().to(dtypes[projection])
        parameter.data = projected.to(dtypes[compute])
        if not torch.equal(parameter.detach().to(dtypes[projection]), projected):
            raise ValueError(f"Projection roundtrip failed: {name}")
    return weight_identity(model, compute_dtype=compute_dtype, torch=torch)


def collect_case(
    model: Any,
    case: dict,
    *,
    output: Path,
    index: int,
    device: str,
    torch: Any,
    linear_layer_type: type,
) -> None:
    """Collect 17 paired forwards and final raw GDN states for one isolated case."""
    tensor_path, json_path = (
        output / f"case_{index:02d}.pt",
        output / f"case_{index:02d}.json",
    )
    if (
        tensor_path.exists()
        or tensor_path.is_symlink()
        or json_path.exists()
        or json_path.is_symlink()
    ):
        raise FileExistsError(f"Case output already exists: {index}")
    cached, cached_logits, recomputed_logits, observations, final_states = (
        None,
        [],
        [],
        [],
        {},
    )
    with torch.inference_mode():
        for window in forward_windows(
            input_ids=case["input_ids"], continuation_ids=case["continuation_ids"]
        ):
            values = []
            for path, past in (("cached", cached), ("recomputed", None)):
                result = model(
                    input_ids=torch.tensor(
                        [window[f"{path}_input_ids"]], device=device
                    ),
                    position_ids=torch.tensor(
                        [window[f"{path}_position_ids"]], device=device
                    ),
                    attention_mask=torch.ones(
                        (1, window["prefix_length"]), dtype=torch.long, device=device
                    ),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
                logits = result.logits[0, -1].detach().float().cpu().clone()
                if logits.shape != (model.config.get_text_config().vocab_size,):
                    raise ValueError("Logit vector does not match model vocabulary")
                if result.past_key_values.get_seq_length() != window["prefix_length"]:
                    raise ValueError("Cache prefix length mismatch")
                values.append((logits, result.past_key_values))
                del result
            left, cached = values[0]
            right, recomputed = values[1]
            observation = {
                "step": window["step"],
                "prefix_length": window["prefix_length"],
                "logits": compare_tensor(left, right, torch=torch),
                "states": [],
            }
            for path, vector in (("cached", left), ("recomputed", right)):
                top = torch.topk(vector, 5)
                observation[f"{path}_top5"] = {
                    "ids": top.indices.tolist(),
                    "values": top.values.tolist(),
                    "margin": (top.values[0] - top.values[1]).item(),
                }
            linear_count = 0
            for layer_index, (cached_layer, recomputed_layer) in enumerate(
                zip(cached.layers, recomputed.layers, strict=True)
            ):
                if type(cached_layer) is not type(recomputed_layer):
                    raise ValueError("Cache layer types differ")
                if isinstance(cached_layer, linear_layer_type):
                    linear_count += 1
                    for field, actual, reference in (
                        (
                            "conv_states",
                            cached_layer.conv_states,
                            recomputed_layer.conv_states,
                        ),
                        (
                            "recurrent_states",
                            cached_layer.recurrent_states,
                            recomputed_layer.recurrent_states,
                        ),
                    ):
                        if (
                            actual is None
                            or reference is None
                            or actual.dtype != reference.dtype
                            or type(actual) is not type(reference)
                        ):
                            raise ValueError("Missing or incompatible GDN state")
                        comparison = compare_tensor(actual, reference, torch=torch)
                        comparison.update({"layer": layer_index, "field": field})
                        observation["states"].append(comparison)
                        if window["step"] == 16:
                            final_states[f"{layer_index}.{field}"] = {
                                "cached": actual.detach().cpu().clone(),
                                "recomputed": reference.detach().cpu().clone(),
                            }
            if linear_count != 18:
                raise ValueError("Expected 18 GDN cache layers")
            cached_logits.append(left)
            recomputed_logits.append(right)
            observations.append(observation)
            del values, recomputed, cached_layer, recomputed_layer, actual, reference
    payload = {
        "case_id": case["case_id"],
        "input_ids": case["input_ids"],
        "suffix_ids": case["continuation_ids"],
        "cached_logits": torch.stack(cached_logits),
        "recomputed_logits": torch.stack(recomputed_logits),
        "final_gdn_states": final_states,
    }
    with tensor_path.open("xb") as handle:
        torch.save(payload, handle)
    write_json(json_path, {"case_id": case["case_id"], "steps": observations})
    print(f"Collected {index + 1}/16: {case['case_id']}", flush=True)


def collect(
    args: argparse.Namespace,
    cases: list[dict],
    *,
    output: Path,
    collect_case_fn: Callable[..., None],
) -> None:
    """Load local HF fallback implementation, freeze identity, and collect calibration."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, cache_utils
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    torch.set_num_threads(args.threads)
    hf.causal_conv1d_fn = None
    hf.causal_conv1d_update = None
    hf.chunk_gated_delta_rule = None
    hf.fused_recurrent_gated_delta_rule = None
    hf.FusedRMSNormGated = None
    hf.is_fast_path_available = False
    device = "cpu"
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    if args.reference.startswith("npu"):
        import torch_npu

        torch.npu.set_device(args.npu_device)
        device = f"npu:{args.npu_device}"
        versions["torch_npu"] = torch_npu.__version__
    attention = "eager" if args.reference.endswith("eager") else "sdpa"
    compute_dtype = "bfloat16" if "bf16" in args.reference else "float32"
    index_path = args.model / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    sources = [
        args.model / "config.json",
        index_path,
        args.manifest,
        Path(hf.__file__),
        Path(cache_utils.__file__),
        Path(__file__),
        REPO_ROOT / "benchmarks/course_project/reference_contracts.py",
    ]
    for shard in sorted(set(weight_map.values())):
        shard_path = (args.model / shard).resolve()
        if not shard_path.is_relative_to(args.model.resolve()):
            raise ValueError("Weight shard escapes model directory")
        sources.append(shard_path)
    hashes = {str(path.resolve()): sha_file(path) for path in sources}
    model, raw_loading = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=attention,
        output_loading_info=True,
    )
    loading = normalize_loading_info(raw_loading)
    write_json(output / "loading.json", loading)
    if any(loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete or mismatched model loading; see loading.json")
    if any(
        not key.startswith(("model.visual.", "visual."))
        for key in loading["unexpected_keys"]
    ):
        raise ValueError("Unexpected non-vision checkpoint keys; see loading.json")
    identity = project_weights(model, compute_dtype=compute_dtype, torch=torch)
    model.to(device).eval()
    if weight_identity(model, compute_dtype=compute_dtype, torch=torch) != identity:
        raise ValueError("Moving model changed weight identity")
    write_json(
        output / "identity.json",
        {
            "weights": identity,
            "source_sha256": hashes,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": model.config.to_dict(),
            "attention": model.config.get_text_config()._attn_implementation,
            "threads": torch.get_num_threads(),
            "versions": versions,
            "device": device,
            "loading": loading,
            "linear_attention_implementation": "HF Torch fallback",
        },
    )
    for index, case in enumerate(cases):
        collect_case_fn(
            model,
            case,
            output=output,
            index=index,
            device=device,
            torch=torch,
            linear_layer_type=LinearAttentionCacheLayerMixin,
        )
    if {str(path.resolve()): sha_file(path) for path in sources} != hashes:
        raise ValueError("Source identity changed during collection")
    write_json(
        output / "result.json",
        {
            "status": "collected",
            "case_count": 16,
            "vectors_per_path": 272,
            "reference_numerical_trust": "pending assessment",
            "engine_validation": "not executed",
        },
    )


def main() -> None:
    """Run standard-library guards before imports or creating the output directory."""
    args = parse_args(None)
    output = new_output_path(args.output)
    with (args.model / "config.json").open(encoding="utf-8") as handle:
        vocab_size = json.load(handle)["text_config"]["vocab_size"]
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases = calibration_cases(
        manifest,
        revision=args.revision,
        expected_sha=args.manifest_sha,
        vocab_size=vocab_size,
    )
    manifest_file_sha = sha_file(args.manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "manifest_file_sha256": manifest_file_sha,
            "status": "collecting",
        },
    )
    try:
        collect(args, cases, output=output, collect_case_fn=collect_case)
    except Exception as error:
        write_json(
            output / "failed.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "completed_files": sum(path.is_file() for path in output.iterdir()),
            },
        )
        raise


if __name__ == "__main__":
    main()
