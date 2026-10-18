#!/usr/bin/env bash
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

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Activate this environment with: source scripts/use_project_env.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if ! command -v conda >/dev/null 2>&1 && [[ -x /opt/miniconda3/bin/conda ]]; then
    export PATH="/opt/miniconda3/bin:${PATH}"
fi
if ! CONDA_BASE="$(conda info --base)"; then
    echo "Conda is required. Install Conda, then run scripts/setup_env.sh." >&2
    return 1
fi
if [[ ! -d "${PROJECT_ROOT}/.conda-env/conda-meta" ]]; then
    echo "Project Conda environment is missing. Run scripts/setup_env.sh first." >&2
    return 1
fi

CANN_ENV="${CANN_ENV:-${TOKENSPEED_CANN_ROOT:-${CONDA_BASE}/Ascend/cann-8.5.0}/set_env.sh}"
if [[ ! -r "${CANN_ENV}" || "$(basename -- "${CANN_ENV}")" != set_env.sh ]]; then
    echo "CANN_ENV must name a readable CANN set_env.sh: ${CANN_ENV}" >&2
    return 1
fi
RESOLVED_CANN_ROOT="$(cd -- "$(dirname -- "${CANN_ENV}")" && pwd -P)"
if [[ -n "${TOKENSPEED_CANN_ROOT:-}" ]]; then
    if [[ ! -d "${TOKENSPEED_CANN_ROOT}" ]] ||
        [[ "$(cd -- "${TOKENSPEED_CANN_ROOT}" && pwd -P)" != "${RESOLVED_CANN_ROOT}" ]]; then
        echo "CANN_ENV and TOKENSPEED_CANN_ROOT refer to different CANN installations." >&2
        return 1
    fi
fi
export TOKENSPEED_CANN_ROOT="${RESOLVED_CANN_ROOT}"
export CANN_ENV="${TOKENSPEED_CANN_ROOT}/set_env.sh"

# Preserve the caller's nounset setting while loading the vendor script.
if [[ $- == *u* ]]; then
    set +u
    # shellcheck disable=SC1090
    if ! source "${CANN_ENV}"; then
        set -u
        return 1
    fi
    set -u
else
    # shellcheck disable=SC1090
    source "${CANN_ENV}" || return 1
fi

export CONDA_ENVS_PATH="${PROJECT_ROOT}/.conda-envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/.conda-pkgs"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/.cache/torch-extensions"
export TRITON_HOME="${PROJECT_ROOT}/.cache/triton"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export ASCEND_WORK_PATH="${PROJECT_ROOT}/.tmp/ascend_work"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/.cache/numba"
export TMPDIR="${PROJECT_ROOT}/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/tokenspeed-kernel/python:${PROJECT_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"

mkdir -p \
    "${PROJECT_ROOT}/.conda-envs" \
    "${PROJECT_ROOT}/.conda-pkgs" \
    "${PROJECT_ROOT}/.cache" \
    "${PROJECT_ROOT}/.tmp" \
    "${PROJECT_ROOT}/models" \
    "${PROJECT_ROOT}/data" \
    "${PROJECT_ROOT}/results/raw" \
    "${PROJECT_ROOT}/results/plots" || return 1

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh" || return 1
conda activate "${PROJECT_ROOT}/.conda-env" || return 1
if [[ "${CONDA_PREFIX:-}" != "${PROJECT_ROOT}/.conda-env" ]]; then
    echo "Conda did not activate the repository environment." >&2
    return 1
fi
