#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.long_prefill_decode import analyze_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one project-local long-prefill/decode raw run. "
            "The output directory must not already exist."
        )
    )
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _project_path(path: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise ValueError(f"{label} must stay inside the TokenSpeed project") from None
    return resolved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        raw_dir = _project_path(args.raw_run_dir, "raw run directory")
        output_dir = _project_path(args.output_dir, "output directory")
        if not raw_dir.is_dir():
            raise FileNotFoundError("Raw run directory does not exist")
        if output_dir.exists():
            raise FileExistsError("Output directory already exists")
        output_paths = analyze_run(raw_dir, output_dir)
    except FileExistsError:
        print("error: output directory already exists", file=sys.stderr)
        raise SystemExit(2) from None
    except FileNotFoundError:
        print(
            "error: raw run directory or required input file does not exist",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except OSError:
        print(
            "error: unable to access report inputs or write output files",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None

    for path in output_paths:
        print(path.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
