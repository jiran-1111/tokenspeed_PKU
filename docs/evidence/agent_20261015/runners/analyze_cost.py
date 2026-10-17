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

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("raw")
p.add_argument("output")
p.add_argument("--retry-root", type=Path)
args = p.parse_args()
root = Path(args.raw)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
assert json.loads((root / "result.json").read_text())["arms"] == 6
conditions = [(128, 32), (128, 128), (1024, 32), (1024, 128)]
arms = []
summary = {}
hashes = {}
all_rows = {}
configs = {}
excluded = []
for repetition in range(3):
    for mode in ("baseline", "candidate"):
        selected_root = root
        if args.retry_root is not None and (repetition, mode) == (2, "candidate"):
            failed = json.loads((root / "r2_candidate/result.json").read_text())
            assert failed["sampler_errors"] == ["AssertionError: -9"]
            assert (
                json.loads((args.retry_root / "result.json").read_text())["status"]
                == "collected"
            )
            excluded.append(
                {
                    "arm": "r2_candidate",
                    "reason": "sampler interrupted during Engine teardown (-9); raw retained; one protocol-failure retry, no metric-based selection",
                    "excluded_formal_requests": 16,
                }
            )
            selected_root = args.retry_root
        arm = selected_root / f"r{repetition}_{mode}"
        result = json.loads((arm / "result.json").read_text())
        assert result["status"] == "collected" and result["source_unchanged"]
        assert (
            result["source_manifest_sha256"]
            == "ef7b3e028233517746147f36da92efea679a29ba50471df4574e3cac9c103a27"
        )
        assert result["sampler_errors"] == []
        configs[repetition, mode] = result["config"]
        assert re.search(
            r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)",
            (arm / "post_exit_processes.txt").read_text(),
        )
        assert (
            json.loads((selected_root / f"r{repetition}_{mode}_exit.json").read_text())[
                "exit_code"
            ]
            == 0
        )
        rows = [
            json.loads(line)
            for line in (arm / "requests.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 16 and Counter(
            (r["input_tokens"], r["output_tokens"]) for r in rows
        ) == Counter({c: 4 for c in conditions})
        all_rows[repetition, mode] = {
            (r["input_tokens"], r["output_tokens"], r["request_index"]): r for r in rows
        }
        for row in rows:
            n = row["output_tokens"]
            assert (
                len(row["output_ids"]) == n
                and row["stream_chunk_sizes"] == [1] * n
                and len(row["inter_token_ms"]) == n - 1
            )
            assert all(
                math.isfinite(row[k]) and row[k] > 0
                for k in ("ttft_ms", "tpot_ms", "e2e_ms")
            )
            assert math.isclose(
                statistics.mean(row["inter_token_ms"]), row["tpot_ms"], abs_tol=1e-9
            )
            assert row["e2e_ms"] >= row["ttft_ms"] + row["tpot_ms"] * (n - 1) - 1e-6
            assert (
                row["last_meta"]["prompt_tokens"] == row["input_tokens"]
                and row["last_meta"]["completion_tokens"] == n
            )
        samples = [
            json.loads(line)
            for line in (arm / "hbm_samples.jsonl").read_text().splitlines()
        ]
        assert len(samples) == result["hbm_sample_count"] and len(samples) > 5
        assert all(
            s["returncode"] == 0 and isinstance(s["hbm_mb"], int) and s["hbm_mb"] > 0
            for s in samples
        )
        formal = [s for s in samples if s["phase"].startswith("formal_")]
        assert formal
        assert {s["phase"] for s in formal} == {
            f"formal_{i}_{o}" for i, o in conditions
        }
        medians = {
            f"{i}/{o}": {
                metric: statistics.median(
                    r[metric]
                    for r in rows
                    if (r["input_tokens"], r["output_tokens"]) == (i, o)
                )
                for metric in ("ttft_ms", "tpot_ms", "e2e_ms")
            }
            for i, o in conditions
        }
        arms.append(
            {
                "mode": mode,
                "repetition": repetition,
                "formal_requests": 16,
                "condition_p50_ms": medians,
                "hbm_samples": len(samples),
                "first_hbm_mb": samples[0]["hbm_mb"],
                "observed_peak_hbm_mb": max(s["hbm_mb"] for s in samples),
                "formal_observed_peak_hbm_mb": max(s["hbm_mb"] for s in formal),
                "max_query_seconds": max(s["query_seconds"] for s in samples),
                "max_sample_start_gap_seconds": max(
                    (b["monotonic_ns"] - a["monotonic_ns"]) / 1e9
                    for a, b in zip(samples, samples[1:])
                ),
            }
        )
for i, o in conditions:
    condition = f"{i}/{o}"
    summary[condition] = {}
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        base = [
            a["condition_p50_ms"][condition][metric]
            for a in arms
            if a["mode"] == "baseline"
        ]
        candidate = [
            a["condition_p50_ms"][condition][metric]
            for a in arms
            if a["mode"] == "candidate"
        ]
        summary[condition][metric] = {
            "baseline_mean_of_3_run_p50": statistics.mean(base),
            "candidate_mean_of_3_run_p50": statistics.mean(candidate),
            "baseline_run_p50": base,
            "candidate_run_p50": candidate,
            "candidate_over_baseline_percent": 100
            * (statistics.mean(candidate) / statistics.mean(base) - 1),
            "paired_run_change_percent": [
                100 * (b / a - 1) for a, b in zip(base, candidate)
            ],
        }
for path in sorted(root.rglob("*")):
    if path.is_file():
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
same_output_pairs = 0
if args.retry_root is not None:
    for path in sorted(args.retry_root.rglob("*")):
        if path.is_file():
            hashes[f"retry/{path.relative_to(args.retry_root).as_posix()}"] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
for repetition in range(3):
    candidate = dict(configs[repetition, "candidate"])
    assert candidate.pop("hf_overrides") == {"npu_lm_head_dtype": "float32"}
    assert candidate == configs[repetition, "baseline"]
    for key, row in all_rows[repetition, "baseline"].items():
        same_output_pairs += (
            row["output_ids"] == all_rows[repetition, "candidate"][key]["output_ids"]
        )
result = {
    "source_manifest_sha256": "ef7b3e028233517746147f36da92efea679a29ba50471df4574e3cac9c103a27",
    "arms": arms,
    "formal_requests": 96,
    "excluded_attempts": excluded,
    "same_output_id_pairs": same_output_pairs,
    "paired_requests": 48,
    "summary": summary,
    "logical_head_extra_bytes": 2034237440,
    "limitations": [
        "3 independent runs per precision; descriptive statistics only",
        "Synthetic repeated token ID1000 measures cost, not semantic quality",
        "Periodic target-chip HBM is observed occupancy, not instantaneous peak or KV-only memory",
        "Other chips may run unrelated tasks; no isolation claim for shared host/card resources",
        "If excluded_attempts is nonempty, one candidate arm retried after teardown sampling failure; pairing elapsed time differs",
    ],
}
(out / "cost_summary.json").write_text(json.dumps(result, indent=2) + "\n")
(out / "raw_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(
    json.dumps(
        {
            "validated_arms": len(arms),
            "formal_requests": 96,
            "hbm_samples": sum(a["hbm_samples"] for a in arms),
            "summary": summary,
        },
        indent=2,
    )
)
