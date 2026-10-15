#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.course_project.profiler_analysis import analyze_root


def _project_path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "path must stay inside the TokenSpeed project"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize existing Ascend Level1 profiler captures without modifying them."
    )
    parser.add_argument(
        "--input-root",
        type=_project_path,
        default=PROJECT_ROOT / "results/profiles/qwen35-0.8b/agent_level1_cann",
    )
    parser.add_argument(
        "--output",
        type=_project_path,
        default=PROJECT_ROOT / "results/analysis/agent_level1_cann_summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the output JSON when it already exists; input captures remain read-only",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")

    summary = analyze_root(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with args.output.open(mode, encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {summary['capture_count']} capture summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
