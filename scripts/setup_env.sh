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
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if ! command -v conda >/dev/null 2>&1 && [[ -x /opt/miniconda3/bin/conda ]]; then
    export PATH="/opt/miniconda3/bin:${PATH}"
fi
CONDA_BASE="$(conda info --base)"

# Validate both entry points before creating an environment or installing packages.
CANN_ENV="${CANN_ENV:-${TOKENSPEED_CANN_ROOT:-${CONDA_BASE}/Ascend/cann-8.5.0}/set_env.sh}"
if [[ ! -r "${CANN_ENV}" || "$(basename -- "${CANN_ENV}")" != set_env.sh ]]; then
    echo "CANN_ENV must name a readable CANN set_env.sh: ${CANN_ENV}" >&2
    exit 1
fi
RESOLVED_CANN_ROOT="$(cd -- "$(dirname -- "${CANN_ENV}")" && pwd -P)"
if [[ -n "${TOKENSPEED_CANN_ROOT:-}" ]]; then
    if [[ ! -d "${TOKENSPEED_CANN_ROOT}" ]] ||
        [[ "$(cd -- "${TOKENSPEED_CANN_ROOT}" && pwd -P)" != "${RESOLVED_CANN_ROOT}" ]]; then
        echo "CANN_ENV and TOKENSPEED_CANN_ROOT refer to different CANN installations." >&2
        exit 1
    fi
fi
export TOKENSPEED_CANN_ROOT="${RESOLVED_CANN_ROOT}"
export CANN_ENV="${TOKENSPEED_CANN_ROOT}/set_env.sh"

# CANN's environment script expects unset variables to expand as empty strings.
set +u
# shellcheck disable=SC1090
source "${CANN_ENV}"
set -u

export CONDA_ENVS_PATH="${PROJECT_ROOT}/.conda-envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/.conda-pkgs"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export TMPDIR="${PROJECT_ROOT}/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

mkdir -p \
    "${CONDA_ENVS_PATH}" \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "${TMPDIR}"

ENV_PREFIX="${PROJECT_ROOT}/.conda-env"
if [[ ! -d "${ENV_PREFIX}/conda-meta" ]]; then
    conda create --prefix "${ENV_PREFIX}" --yes python=3.11 pip
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/use_project_env.sh"

ASCEND_PYPI="${TOKENSPEED_ASCEND_PYPI:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
python -m pip install \
    "torch==2.9.0" \
    "torch-npu==2.9.0.post2" \
    --extra-index-url "${ASCEND_PYPI}"

bash "${PROJECT_ROOT}/test/ci_system/install_triton_ascend.sh"
python -m pip install --no-deps -e "${PROJECT_ROOT}/tokenspeed-kernel/python"
python -m pip install -e "${PROJECT_ROOT}/tokenspeed-scheduler"
python -m pip install -e "${PROJECT_ROOT}/python"

echo "Environment ready: ${ENV_PREFIX}"
echo "Activate with: source scripts/use_project_env.sh"
