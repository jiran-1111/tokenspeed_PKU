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

"""Audit every frozen Stage B vector; keep frozen and historical gates separate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts import collect_homework1_references as collector
from scripts.collect_homework1_validation_reference import validation_cases
from scripts.homework1_stage_b_capture import validate_frontier

GATES = {
    "prefill": (0.04160680933973707, 0.4934459924697876),
    "decode": (0.027781371644764415, 0.4458632469177246),
}


def gate_result(metrics: dict, *, step: int) -> dict:
    """Apply the frozen rectangular gate and the separate original 1% gate."""
    if not 0 <= step <= 16:
        raise ValueError("Step outside frozen protocol")
    import math

    l2, absolute = metrics["relative_l2"], metrics["max_abs"]
    if not metrics["finite"] or not all(
        math.isfinite(v) and v >= 0 for v in (l2, absolute)
    ):
        raise ValueError("Invalid numerical metrics")
    limit_l2, limit_absolute = GATES["prefill" if step == 0 else "decode"]
    return {
        "frozen_pass": l2 <= limit_l2 and absolute <= limit_absolute,
        "original_1pct_pass": l2 <= 0.01 and absolute <= 0.5,
        "relative_l2_limit": limit_l2,
        "max_abs_limit": limit_absolute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "engine", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = collector.new_output_path(args.output)
    if (
        collector.sha_file(args.manifest)
        != "23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb"
    ):
        raise ValueError("Frozen manifest file mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = validation_cases(
        manifest,
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        expected_sha="f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936",
        vocab_size=248320,
    )
    identity = json.loads((args.reference / "identity.json").read_text())
    if (
        identity["weights"]["global_weight_sha256"]
        != "c2084362ae79d1d0f3d61a9781598296fae0944efe4e6fd04888a48692c57884"
    ):
        raise ValueError("Projected reference weights mismatch")
    if identity["attention"] != "eager" or identity["device"] != "cpu":
        raise ValueError("Primary reference implementation mismatch")
    for root in (args.reference, args.engine):
        result = json.loads((root / "result.json").read_text())
        if result["case_count"] != 16 or result["status"] != "collected":
            raise ValueError("Incomplete collection")
    import torch

    rows, hashes = [], {}
    for index, case in enumerate(cases):
        reference_path = args.reference / f"case_{index:02d}.pt"
        ref = torch.load(reference_path, map_location="cpu", weights_only=True)
        hashes[str(reference_path)] = collector.sha_file(reference_path)
        if (
            ref["case_id"] != case["case_id"]
            or ref["input_ids"] != case["input_ids"]
            or ref["suffix_ids"] != case["continuation_ids"]
        ):
            raise ValueError("Reference native IDs mismatch")
        if tuple(ref["cached_logits"].shape) != (17, 248320):
            raise ValueError("Missing full reference vectors")
        for step in range(17):
            path = args.engine / f"case_{index:02d}" / f"step_{step:02d}.pt"
            raw = torch.load(path, map_location="cpu", weights_only=True)
            hashes[str(path)] = collector.sha_file(path)
            if (
                raw["case_id"] != case["case_id"]
                or raw["step"] != step
                or raw["prefix_length"] != len(case["input_ids"]) + step
            ):
                raise ValueError("Engine identity or prefix mismatch")
            validate_frontier(
                case, step=step, input_ids=raw["input_ids"], positions=raw["positions"]
            )
            if raw["position_axes"] != [raw["positions"]] * 3:
                raise ValueError("Engine text MRoPE axes mismatch")
            if tuple(raw["logits"].shape) != (248320,):
                raise ValueError("Missing full engine vector")
            metrics = collector.compare_tensor(
                raw["logits"], ref["cached_logits"][step], torch=torch
            )
            sample = json.loads(path.with_suffix(".json").read_text())
            expected = (
                case["continuation_ids"][step] if step < 16 else raw["raw_greedy_id"]
            )
            if (
                sample["raw_greedy_id"] != int(raw["logits"].argmax())
                or sample["emitted_id"] != expected
                or sample["forced"] != (step < 16)
            ):
                raise ValueError("Sampler provenance mismatch")
            rows.append(
                {
                    "case_id": case["case_id"],
                    "step": step,
                    "stage": "prefill" if step == 0 else "decode",
                    "relative_l2": metrics["relative_l2"],
                    "max_abs": metrics["max_abs"],
                    "engine_argmax": raw["raw_greedy_id"],
                    "reference_argmax": int(ref["cached_logits"][step].argmax()),
                    **gate_result(metrics, step=step),
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "vectors.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "case_count": 16,
        "vector_count": len(rows),
        "gates": GATES,
        "all_frozen_pass": all(row["frozen_pass"] for row in rows),
        "frozen_failures": [row for row in rows if not row["frozen_pass"]],
        "original_1pct_failures": sum(not row["original_1pct_pass"] for row in rows),
        "argmax_mismatches_at_forced_frontiers": sum(
            row["engine_argmax"] != row["reference_argmax"] for row in rows
        ),
        "scope": "16 frozen isolated trajectories; no general error bound; historical 1% failures remain unchanged",
    }
    collector.write_json(output / "summary.json", summary)
    collector.write_json(
        output / "provenance.json",
        {
            "raw_sha256": hashes,
            "analyzer_sha256": collector.sha_file(Path(__file__)),
            "frozen_record_sha256": collector.sha_file(
                REPO_ROOT / "docs/project/homework1_frozen_numerical_gate_20261015.md"
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
