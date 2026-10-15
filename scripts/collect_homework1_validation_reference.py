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

"""Collect the frozen Stage B validation reference without changing Stage A."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.reference_contracts import forward_windows
from scripts import collect_homework1_references as collector


def collect_cached_case(
    model,
    case: dict,
    *,
    output: Path,
    index: int,
    device: str,
    torch,
    linear_layer_type: type,
) -> None:
    """Collect the primary cached reference; no recomputation or state audit."""
    cached, vectors = None, []
    with torch.inference_mode():
        for window in forward_windows(
            input_ids=case["input_ids"], continuation_ids=case["continuation_ids"]
        ):
            result = model(
                input_ids=torch.tensor([window["cached_input_ids"]], device=device),
                position_ids=torch.tensor(
                    [window["cached_position_ids"]], device=device
                ),
                attention_mask=torch.ones(
                    (1, window["prefix_length"]), dtype=torch.long, device=device
                ),
                past_key_values=cached,
                use_cache=True,
                logits_to_keep=1,
            )
            vector = result.logits[0, -1].detach().cpu().float().clone()
            if vector.shape != (248320,) or not bool(torch.isfinite(vector).all()):
                raise ValueError("Invalid full reference vector")
            cached = result.past_key_values
            if cached.get_seq_length() != window["prefix_length"]:
                raise ValueError("Reference cache frontier mismatch")
            vectors.append(vector)
    with (output / f"case_{index:02d}.pt").open("xb") as stream:
        torch.save(
            {
                "case_id": case["case_id"],
                "input_ids": case["input_ids"],
                "suffix_ids": case["continuation_ids"],
                "cached_logits": torch.stack(vectors),
            },
            stream,
        )
    print(f"Collected {index + 1}/16: {case['case_id']}", flush=True)


def validation_cases(
    manifest: dict, *, revision: str, expected_sha: str, vocab_size: int
) -> list[dict]:
    """Validate the complete frozen matrix and return original validation rows."""
    collector.calibration_cases(
        manifest, revision=revision, expected_sha=expected_sha, vocab_size=vocab_size
    )
    return [case for case in manifest["cases"] if case["group"] == "validation"]


def main() -> None:
    args = collector.parse_args(None)
    if args.reference != "cpu-fp32-eager":
        raise ValueError("Stage B primary reference requires cpu-fp32-eager")
    output = collector.new_output_path(args.output)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    cases = validation_cases(
        manifest,
        revision=args.revision,
        expected_sha=args.manifest_sha,
        vocab_size=config["text_config"]["vocab_size"],
    )
    file_sha = collector.sha_file(args.manifest)
    if file_sha != "23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb":
        raise ValueError("Frozen manifest file SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    collector.write_json(
        output / "protocol.json",
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "manifest_file_sha256": file_sha,
            "group": "validation",
            "entrypoint_sha256": collector.sha_file(Path(__file__)),
            "status": "collecting",
        },
    )
    try:
        collector.collect(
            args, cases, output=output, collect_case_fn=collect_cached_case
        )
    except Exception as error:
        collector.write_json(
            output / "failed.json",
            {"exception_type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
