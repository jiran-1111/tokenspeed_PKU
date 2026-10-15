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

"""Explicit single-chip offline inference, benchmark and correctness commands."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tokenspeed.experiments.qwen35.model import RequestState, TextModel


def sync():
    torch.npu.synchronize()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def metrics(actual, reference):
    a, b = actual.cpu().double().flatten(), reference.cpu().double().flatten()
    return {
        "max_abs": float((a - b).abs().max()),
        "rmse": float((a - b).square().mean().sqrt()),
        "relative_rmse": float(
            (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-9)
        ),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "top1_agreement": float(
            (actual.cpu().argmax(-1) == reference.cpu().argmax(-1)).float().mean()
        ),
    }


def synthetic_ids(tokenizer, length):
    base = tokenizer.encode(
        "Artificial intelligence systems combine computation, memory, and communication. ",
        add_special_tokens=False,
    )
    return torch.tensor((base * ((length + len(base) - 1) // len(base)))[:length]).view(
        1, -1
    )


@torch.inference_mode()
def generate(model, ids, count, *, stop_eos: bool):
    state = RequestState()
    token_ids = []
    durations = []
    model.profile_phase = "prefill"
    sync()
    start = time.perf_counter()
    logits = model.forward(ids, state, all_logits=False)
    token = logits[:, -1].argmax(-1, keepdim=True)
    sync()
    prefill = time.perf_counter() - start
    token_ids.append(int(token.item()))
    for _ in range(count - 1):
        if stop_eos and token_ids[-1] == model.config["eos_token_id"]:
            break
        sync()
        start = time.perf_counter()
        model.profile_phase = "decode"
        logits = model.forward(token, state, all_logits=False)
        token = logits[:, -1].argmax(-1, keepdim=True)
        sync()
        durations.append(time.perf_counter() - start)
        token_ids.append(int(token.item()))
    return {
        "input_tokens": ids.numel(),
        "generated_tokens": len(token_ids),
        "token_ids": token_ids,
        "ttft_ms": 1000 * prefill,
        "decode_step_ms": [1000 * t for t in durations],
        "decode_mean_ms": 1000 * statistics.mean(durations) if durations else None,
        "peak_allocated_mib": torch.npu.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.npu.max_memory_reserved() / 2**20,
    }


@torch.inference_mode()
def validate(model, tokenizer, output):
    # Official Transformers is an independent implementation, running on CPU
    # in FP32 with the same checkpoint values (no NPU implementation patching).
    config = Qwen3_5TextConfig(**model.config)
    config._attn_implementation = "eager"
    ref = Qwen3_5ForCausalLM(config).eval().float()
    weights = {f"model.{k}": v.cpu().float() for k, v in model.weights.items()}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    ref.load_state_dict(weights, strict=True)
    del weights
    records = []
    limits = (
        {"max_abs": 0.5, "relative_rmse": 0.02, "cosine": 0.999}
        if model.dtype == torch.bfloat16
        else {"max_abs": 0.004, "relative_rmse": 0.0003, "cosine": 0.99999}
    )
    for length in (1, 17, 65, 128, 512):
        ids = synthetic_ids(tokenizer, length)
        state = RequestState()
        got = model.forward(ids.to(model.device), state, all_logits=False)
        expected = ref(ids, use_cache=True, logits_to_keep=1)
        for step in range(5):
            m = metrics(got, expected.logits)
            record = {"input_tokens": length, "decode_step": step, **m}
            record["pass"] = (
                m["max_abs"] <= limits["max_abs"]
                and m["relative_rmse"] <= limits["relative_rmse"]
                and m["cosine"] >= limits["cosine"]
            )
            records.append(record)
            print(record, flush=True)
            if step != 4:
                token = expected.logits[:, -1].argmax(-1, keepdim=True)
                got = model.forward(token.to(model.device), state, all_logits=False)
                expected = ref(
                    token,
                    past_key_values=expected.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
    # A -> B -> A uses independently owned request state. Interleaved calls
    # exercise isolation, rather than merely rerunning a fresh process.
    a = synthetic_ids(tokenizer, 17).to(model.device)
    b = synthetic_ids(tokenizer, 9).flip(-1).to(model.device)
    first = RequestState()
    fresh = RequestState()
    original = model.forward(a, first, all_logits=False)
    model.forward(b, RequestState(), all_logits=False)
    repeat = model.forward(a, fresh, all_logits=False)
    torch.testing.assert_close(original, repeat, atol=0, rtol=0)
    next_id = original[:, -1].argmax(-1, keepdim=True)
    one = model.forward(next_id, first, all_logits=False)
    two = model.forward(next_id, fresh, all_logits=False)
    torch.testing.assert_close(one, two, atol=0, rtol=0)
    # Full prefix vs arbitrary chunk continuation (including all state types).
    whole_state = RequestState()
    split_state = RequestState()
    whole = model.forward(a, whole_state, all_logits=False)
    model.forward(a[:, :3], split_state, all_logits=False)
    split = model.forward(a[:, 3:], split_state, all_logits=False)
    chunk = metrics(split, whole)
    result = {
        "reference": "Transformers 5.17.0 Qwen3_5ForCausalLM CPU FP32, eager attention",
        "dtype": str(model.dtype),
        "limits": limits,
        "records": records,
        "request_isolation": "bitwise pass",
        "chunk_continuation": chunk,
        "loaded_weight_tensors": len(model.loaded_keys),
        "ignored_non_text_tensors": len(model.ignored_keys),
    }
    result["pass"] = (
        all(r["pass"] for r in records)
        and chunk["relative_rmse"] <= limits["relative_rmse"]
        and chunk["cosine"] >= limits["cosine"]
    )
    save(output, result)
    if not result["pass"]:
        raise AssertionError(f"Model validation failed; see {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=["generate", "validate", "benchmark", "profile"],
    )
    p.add_argument("--dtype", required=True, choices=["bfloat16", "float32"])
    p.add_argument("--gdn-algorithm", required=True, choices=["chunk", "recurrent"])
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="请用一句话介绍北京。")
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args()
    if len(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    ) != 1 or not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Explicitly expose exactly one free physical chip")
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise ValueError("Exactly one visible chip is required")
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = TextModel(
        args.model,
        device="npu:0",
        dtype=getattr(torch, args.dtype),
        gdn_algorithm=args.gdn_algorithm,
        chunk_size=args.chunk_size,
        profile=args.mode == "profile",
    )
    print(
        f"Loaded {len(model.weights)} text tensors on {model.device}; physical chip {os.environ['ASCEND_RT_VISIBLE_DEVICES']}",
        flush=True,
    )
    if args.mode == "validate":
        validate(model, tokenizer, args.output)
    elif args.mode == "generate":
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        result = generate(model, ids.to(model.device), args.new_tokens, stop_eos=True)
        result["prompt"] = args.prompt
        result["text"] = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
        save(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.mode == "benchmark":
        results = []
        for length in (32, 512):
            ids = synthetic_ids(tokenizer, length).to(model.device)
            generate(model, ids, 4, stop_eos=False)
            for count in (8, 32):
                for rep in range(args.repetitions):
                    gc.collect()
                    torch.npu.reset_peak_memory_stats()
                    result = generate(model, ids, count, stop_eos=False)
                    result["repeat"] = rep
                    results.append(result)
                    print(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in {"token_ids", "decode_step_ms"}
                        },
                        flush=True,
                    )
                    save(
                        args.output,
                        {
                            "dtype": args.dtype,
                            "algorithm": args.gdn_algorithm,
                            "chunk_size": args.chunk_size,
                            "physical_chip": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                            "results": results,
                        },
                    )
    else:
        ids = synthetic_ids(tokenizer, 512).to(model.device)
        generate(model, ids, 3, stop_eos=False)
        model.profile_records.clear()
        run = generate(model, ids, 3, stop_eos=False)
        totals = {}
        for row in model.profile_records:
            key = f"{row['phase']}/{row['stage']}"
            item = totals.setdefault(key, {"calls": 0, "elapsed_ms": 0.0})
            item["calls"] += 1
            item["elapsed_ms"] += row["elapsed_ms"]
        result = {
            "physical_chip": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            "method": "Intrusive per-stage host wall time, synchronized before/after each scope. Includes dispatch, compute and synchronization; not raw device kernel time.",
            "run": run,
            "totals": totals,
            "records": model.profile_records,
        }
        save(args.output, result)
        print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
