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
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import (
    TokenSpeedGenerateClient,
    append_jsonl,
    load_model_config,
    model_load_path,
)
from benchmarks.course_project.correctness import (
    collect_stop_token_ids,
    run_decode_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare continuous decode with full-prompt recomputation."
    )
    parser.add_argument("config", type=Path, help="JSON-compatible YAML model config")
    parser.add_argument("input", type=Path, help="Native-ID correctness JSONL capture")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--case", default="long_context_recall")
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("both", "continuous", "full_recompute"),
        default="both",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw" / "correctness" / "decode_diagnostic",
    )
    return parser.parse_args()


def select_decode_diagnostic_modes(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return "continuous", "full_recompute"
    if mode in {"continuous", "full_recompute"}:
        return (mode,)
    raise ValueError(f"unsupported decode diagnostic mode: {mode}")


def main() -> None:
    args = parse_args()
    if args.turn < 1:
        raise ValueError("--turn must be at least 1")
    config = load_model_config(args.config)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if row.get("case") == args.case and row.get("turn_index") == args.turn
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row for case={args.case!r}, turn={args.turn}; "
            f"found {len(matches)}"
        )
    source = matches[0]
    input_token_ids = source.get("client_input_token_ids")
    if not isinstance(input_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in input_token_ids
    ):
        raise ValueError("selected record has no client_input_token_ids")
    max_new_tokens = args.max_new_tokens or (source.get("sampling_params") or {}).get(
        "max_new_tokens"
    )
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be supplied by the record or CLI")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config),
        revision=config["model_revision"],
        local_files_only=True,
    )
    stop_token_ids = collect_stop_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "additional_stop_token_ids", None),
    )
    client = TokenSpeedGenerateClient(args.base_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = select_decode_diagnostic_modes(args.mode)
    output_paths = [
        args.output_dir
        / f"{config['served_model_name']}-{args.case}-turn{args.turn}-{mode}-{run_id}.jsonl"
        for mode in modes
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite an existing diagnostic output")

    for mode, output_path in zip(modes, output_paths):
        output_ids, request_count = run_decode_diagnostic(
            client=client,
            input_token_ids=input_token_ids,
            max_new_tokens=max_new_tokens,
            mode=mode,
            stop_token_ids=stop_token_ids,
        )
        append_jsonl(
            output_path,
            {
                "case": source["case"],
                "category": source.get("category"),
                "turn_index": source["turn_index"],
                "client_input_token_ids": input_token_ids,
                "input_token_ids_source": "copied_from_native_id_capture",
                "output_token_ids": output_ids,
                "output_token_ids_source": "tokenspeed_generate_output_ids",
                "token_correctness_verified": False,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "stop_token_ids": stop_token_ids,
                },
                "decode_mode": mode,
                "request_count": request_count,
                "source_capture": str(args.input),
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "served_model_name": config["served_model_name"],
            },
        )
        print(output_path)


if __name__ == "__main__":
    main()
