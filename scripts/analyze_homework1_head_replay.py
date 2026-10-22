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

"""CPU FP32 final-projection replay at all frozen HW1 input frontiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch


def errors(actual, expected):
    actual, expected = actual.double(), expected.double()
    difference = actual - expected
    return dict(
        relative_l2=float(difference.norm() / expected.norm()),
        max_abs=float(difference.abs().max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "old-engine", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vocab-block", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.threads != 1 or args.vocab_block != 4096:
        raise ValueError(
            "Requires the predeclared single-thread, 4096-row CPU protocol"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    weight = torch.load(
        args.capture / "head_weight.pt", map_location="cpu", weights_only=True
    )
    if tuple(weight.shape) != (248320, 1024) or weight.dtype != torch.bfloat16:
        raise ValueError("Unexpected captured head")
    bf16_hash, fp32_hash = hashlib.sha256(), hashlib.sha256()
    for offset in range(0, len(weight), args.vocab_block):
        block = weight[offset : offset + args.vocab_block].contiguous()
        bf16_hash.update(block.view(torch.uint8).numpy().tobytes())
        fp32_hash.update(block.float().numpy().tobytes())
    identity = json.loads((args.reference / "identity.json").read_text())
    parameter = next(
        p
        for p in identity["weights"]["parameters"]
        if p["name"] == "model.embed_tokens.weight"
    )
    head_identity = json.loads((args.capture / "head_identity.json").read_text())
    if (
        bf16_hash.hexdigest() != head_identity["tensor_sha256"]
        or fp32_hash.hexdigest() != parameter["sha256"]
    ):
        raise ValueError(
            "Actual head weight differs from the same projected reference embedding"
        )
    hidden, records, references = [], [], []
    for case in range(16):
        ref = torch.load(
            args.reference / f"case_{case:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if tuple(ref["cached_logits"].shape) != (17, 248320):
            raise ValueError("Incomplete full reference")
        for step in range(17):
            folder = args.capture / f"case_{case:02d}"
            original = torch.load(
                folder / f"step_{step:02d}.pt", map_location="cpu", weights_only=True
            )
            old = torch.load(
                args.old_engine / f"case_{case:02d}/step_{step:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            middle = torch.load(
                folder / f"hidden_{step:02d}.pt", map_location="cpu", weights_only=True
            )
            if (
                middle["case_id"] != original["case_id"]
                or middle["step"] != step
                or original["input_ids"] != old["input_ids"]
                or original["positions"] != old["positions"]
            ):
                raise ValueError("Hidden/logits/frontier identity mismatch")
            hidden.append(middle["hidden"].reshape(1024))
            actual = original["logits"]
            reference = ref["cached_logits"][step]
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise ValueError("Nonfinite logits")
            references.append(reference)
            records.append(
                dict(
                    case=case,
                    case_id=original["case_id"],
                    step=step,
                    original_argmax=int(old["logits"].argmax()),
                    captured_argmax=int(actual.argmax()),
                    reference_argmax=int(reference.argmax()),
                    capture_bitwise_equal=torch.equal(actual, old["logits"]),
                    capture_vs_old_max_abs=float((actual - old["logits"]).abs().max()),
                    captured_relative_l2=errors(actual, reference)["relative_l2"],
                    captured_max_abs=errors(actual, reference)["max_abs"],
                )
            )
    hidden = torch.stack(hidden).float().contiguous()
    if not torch.isfinite(hidden).all():
        raise ValueError("Nonfinite hidden")
    logits = torch.empty(272, 248320, dtype=torch.float32)
    for offset in range(0, len(weight), args.vocab_block):
        logits[:, offset : offset + args.vocab_block] = (
            hidden @ weight[offset : offset + args.vocab_block].float().T
        )
    torch.save(
        dict(hidden_fp32=hidden, replay_logits=logits), args.output / "replay_logits.pt"
    )
    for row, actual, reference, middle in zip(records, logits, references, hidden):
        if not torch.isfinite(actual).all():
            raise ValueError("Nonfinite replay")
        error = errors(actual, reference)
        row.update(
            replay_argmax=int(actual.argmax()),
            replay_relative_l2=error["relative_l2"],
            replay_max_abs=error["max_abs"],
        )
        l2, absolute = (
            (0.04160680933973707, 0.4934459924697876)
            if row["step"] == 0
            else (0.027781371644764415, 0.4458632469177246)
        )
        row["captured_gate_pass"] = (
            row["captured_relative_l2"] <= l2 and row["captured_max_abs"] <= absolute
        )
        row["replay_gate_pass"] = (
            error["relative_l2"] <= l2 and error["max_abs"] <= absolute
        )
        candidates = sorted(
            {row["reference_argmax"], row["captured_argmax"], row["replay_argmax"]}
        )
        scores64 = weight[candidates].double() @ middle.double()
        scores32 = actual[candidates].double()
        row["candidate_fp32_vs_fp64_max_abs"] = float((scores64 - scores32).abs().max())
        row["candidate_fp64_argmax"] = candidates[int(scores64.argmax())]
        row["reference_top2_gap"] = float(reference.topk(2).values.diff().abs()[0])
    with (args.output / "vectors.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    old_bad = [x for x in records if x["original_argmax"] != x["reference_argmax"]]
    new_bad = [x for x in records if x["replay_argmax"] != x["reference_argmax"]]
    summary = dict(
        vector_count=len(records),
        old_argmax_mismatches=len(old_bad),
        captured_argmax_mismatches=sum(
            x["captured_argmax"] != x["reference_argmax"] for x in records
        ),
        replay_argmax_mismatches=len(new_bad),
        old_mismatches_resolved=sum(
            x["replay_argmax"] == x["reference_argmax"] for x in old_bad
        ),
        newly_introduced_mismatches=sum(
            x["original_argmax"] == x["reference_argmax"] for x in new_bad
        ),
        capture_bitwise_equal_vectors=sum(x["capture_bitwise_equal"] for x in records),
        capture_vs_old_max_abs=max(x["capture_vs_old_max_abs"] for x in records),
        candidate_fp32_vs_fp64_max_abs=max(
            x["candidate_fp32_vs_fp64_max_abs"] for x in records
        ),
        candidate_fp32_fp64_argmax_disagreements=sum(
            x["replay_argmax"] != x["candidate_fp64_argmax"] for x in records
        ),
        captured_gate_failures=sum(not x["captured_gate_pass"] for x in records),
        replay_gate_failures=sum(not x["replay_gate_pass"] for x in records),
        captured_old_one_percent_failures=sum(
            x["captured_relative_l2"] > 0.01 for x in records
        ),
        replay_one_percent_failures=sum(
            x["replay_relative_l2"] > 0.01 for x in records
        ),
        by_phase={
            phase: {
                name: max(
                    x[name] for x in records if (x["step"] == 0) == (phase == "prefill")
                )
                for name in [
                    "captured_relative_l2",
                    "captured_max_abs",
                    "replay_relative_l2",
                    "replay_max_abs",
                ]
            }
            for phase in ["prefill", "decode"]
        },
        actual_head_bf16_sha256=bf16_hash.hexdigest(),
        projected_head_fp32_sha256=fp32_hash.hexdigest(),
        elapsed_seconds=time.perf_counter() - start,
        threads=args.threads,
        vocab_block=args.vocab_block,
        scope="Only final dense projection recomputed on CPU FP32 from actual service BF16 hidden values; candidate-only FP64 check; not full-model FP32 or free greedy",
    )
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
