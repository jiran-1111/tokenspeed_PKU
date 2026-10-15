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
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a course-project Qwen3.5 configuration."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--npu-profiler-level",
        choices=("Level0", "Level1"),
        default="Level0",
        help="Torch-NPU profiler level for the launched service",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    command = build_serve_command(config, host=args.host, port=args.port)
    if args.dry_run:
        print(shlex.join(command))
        return
    child_env = os.environ.copy()
    child_env["TOKENSPEED_NPU_PROFILER_LEVEL"] = args.npu_profiler_level
    subprocess.run(command, check=True, env=child_env)


if __name__ == "__main__":
    main()
