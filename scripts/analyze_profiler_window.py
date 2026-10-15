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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.profiler_analysis import (
    analyze_single_request_ttft_window,
    write_single_request_ttft_window_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one request's TTFT window against Torch/CANN profiler timelines. "
            "All paths must stay inside the TokenSpeed project; existing analysis "
            "output is never overwritten."
        )
    )
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def project_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        raise ValueError(f"{label} must stay inside the TokenSpeed project") from None
    return resolved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        raw_run_dir = project_path(args.raw_run_dir, label="raw run directory")
        profile_dir = project_path(args.profile_dir, label="profile directory")
        analysis_dir = project_path(args.analysis_dir, label="analysis directory")
        if not raw_run_dir.is_dir():
            raise FileNotFoundError(f"Raw run directory does not exist: {raw_run_dir}")
        if not profile_dir.is_dir():
            raise FileNotFoundError(f"Profile directory does not exist: {profile_dir}")
        if analysis_dir.exists():
            raise FileExistsError(f"Analysis output already exists: {analysis_dir}")
        report = analyze_single_request_ttft_window(raw_run_dir, profile_dir)
        output_path = write_single_request_ttft_window_csv(report, analysis_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(output_path)


if __name__ == "__main__":
    main()
