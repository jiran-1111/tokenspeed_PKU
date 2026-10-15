#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/use_project_env.sh"
python "${repo_root}/scripts/run_profiler.py" "$@"
