#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if ! command -v conda >/dev/null 2>&1 && [[ -x /opt/miniconda3/bin/conda ]]; then
    export PATH="/opt/miniconda3/bin:${PATH}"
fi
CONDA_BASE="$(conda info --base)"
CANN_ENV="${CANN_ENV:-${CONDA_BASE}/Ascend/cann-8.5.0/set_env.sh}"
if [[ -f "${CANN_ENV}" ]]; then
    # CANN's environment script expects unset variables to expand as empty strings.
    if [[ $- == *u* ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${CANN_ENV}"
        set -u
    else
        # shellcheck disable=SC1090
        source "${CANN_ENV}"
    fi
fi

export CONDA_ENVS_PATH="${PROJECT_ROOT}/.conda-envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/.conda-pkgs,${CONDA_BASE}/pkgs"
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
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/tokenspeed-kernel/python:${PROJECT_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"

mkdir -p \
    "${PROJECT_ROOT}/.conda-envs" \
    "${PROJECT_ROOT}/.conda-pkgs" \
    "${PROJECT_ROOT}/.cache" \
    "${PROJECT_ROOT}/.tmp" \
    "${PROJECT_ROOT}/models" \
    "${PROJECT_ROOT}/data" \
    "${PROJECT_ROOT}/results/raw" \
    "${PROJECT_ROOT}/results/plots"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${PROJECT_ROOT}/.conda-env"
