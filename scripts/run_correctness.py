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
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import (
    TokenSpeedGenerateClient,
    append_jsonl,
    chat_template_token_ids,
    load_model_config,
    model_load_path,
)
from benchmarks.course_project.correctness import (
    build_correctness_cases,
    make_correctness_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Qwen3.5 correctness cases."
    )
    parser.add_argument("config", type=Path, help="JSON-compatible YAML model config")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        REPO_ROOT
        / "results"
        / "raw"
        / "correctness"
        / f"{config['served_model_name']}-{run_id}.jsonl"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config), revision=config["model_revision"]
    )
    client = TokenSpeedGenerateClient(args.base_url)
    for case in build_correctness_cases():
        messages = []
        for turn_index, prompt in enumerate(case.turns, start=1):
            messages.append({"role": "user", "content": prompt})
            input_token_ids = chat_template_token_ids(tokenizer, messages)
            result = client.generate(
                input_token_ids=input_token_ids,
                max_tokens=args.max_tokens,
            )
            retokenized_output_ids = tokenizer.encode(
                result.text, add_special_tokens=False
            )
            record = make_correctness_record(
                case=case,
                turn_index=turn_index,
                input_token_ids=input_token_ids,
                retokenized_output_ids=list(retokenized_output_ids),
                max_new_tokens=args.max_tokens,
                result=result,
                engine_output_token_ids=list(result.output_ids),
            )
            record.update(
                {
                    "run_id": run_id,
                    "model_id": config["model_id"],
                    "model_revision": config["model_revision"],
                    "served_model_name": config["served_model_name"],
                }
            )
            append_jsonl(output_path, record)
            messages.append({"role": "assistant", "content": result.text})

    print(output_path)


if __name__ == "__main__":
    main()
