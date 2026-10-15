"""Compare HF cached and recomputed logits at one recorded divergence."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("capture", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", default="code_python")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.capture.read_text().splitlines()]
    row = next(r for r in rows if r["case"] == args.case)
    compare = json.loads(args.comparison.read_text())
    item = next(r for r in compare["comparisons"] if r["case"] == args.case)
    index = item["first_mismatch_index"]
    assert (
        item["engine_output_token_ids"][:index]
        == item["reference_output_token_ids"][:index]
    )
    prompt = row["client_input_token_ids"]
    prefix = item["reference_output_token_ids"][:index]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .to("npu:0")
        .eval()
    )
    observations = {}
    with torch.inference_mode():
        inputs = torch.tensor([prompt], device="npu:0")
        result = model(input_ids=inputs, use_cache=True)
        cache = result.past_key_values
        if prefix:
            del result
        for token in prefix:
            result = model(
                input_ids=torch.tensor([[token]], device="npu:0"),
                past_key_values=cache,
                use_cache=True,
            )
            cache = result.past_key_values
        values, ids = result.logits[0, -1].float().topk(5)
        observations["cached"] = {"ids": ids.tolist(), "logits": values.tolist()}
        del result, cache
        result = model(
            input_ids=torch.tensor([prompt + prefix], device="npu:0"), use_cache=False
        )
        values, ids = result.logits[0, -1].float().topk(5)
        observations["recomputed"] = {"ids": ids.tolist(), "logits": values.tolist()}
    payload = {
        "case": args.case,
        "index": index,
        "prefix_tokens": len(prompt) + index,
        "observations": observations,
        "boundary": "HF BF16 diagnostic only; no tolerance waiver",
    }
    with args.output.open("x") as output:
        json.dump(payload, output, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
