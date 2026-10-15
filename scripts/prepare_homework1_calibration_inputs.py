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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.calibration_inputs import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze homework1 calibration and validation native IDs; no model inference"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.output.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Output must be inside the repository")

    with (args.model / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    vocab_size = config["text_config"]["vocab_size"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    manifest = build_manifest(
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        vocab_size=vocab_size,
        tokenizer_revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, args.output)
    print(f"{len(manifest['cases'])} cases; SHA256 {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
