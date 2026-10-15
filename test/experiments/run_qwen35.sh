#!/usr/bin/env bash
# Explicit environment and chip selection; no installs and no shared mutations.
set -eo pipefail
if [[ $# -lt 2 ]]; then
    echo 'Usage: run_qwen35.sh PHYSICAL_CHIP {test|generate|validate|benchmark|profile|gdn-benchmark} [options...]' >&2
    exit 2
fi
chip=$1
mode=$2
shift 2
if [[ ! "$chip" =~ ^[0-7]$ ]]; then echo 'Expected a physical chip ID from 0 to 7' >&2; exit 2; fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate aics
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/u2601213425/hw1/.venv-qwen35/bin/activate
export ASCEND_RT_VISIBLE_DEVICES="$chip"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo/python:$repo/tokenspeed-kernel/python:$repo/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export QWEN35_TEST_DEVICE=npu:0
echo "Physical chip: $chip; logical device: npu:0"
occupancy=$(npu-smi info -t proc-mem -i "$((chip/2))" -c "$((chip%2))")
echo "$occupancy"
if ! grep -q 'No process in device' <<< "$occupancy"; then
    echo "Chip $chip is occupied; select a free chip before running." >&2
    exit 1
fi
# Advisory per-user lock prevents overlapping runs from this harness.
exec 9>"/home/u2601213425/hw1/.qwen35-chip-${chip}.lock"
flock -n 9 || { echo 'Another harness run owns this chip'; exit 1; }
if [[ "$mode" == test ]]; then
    exec python -m pytest -q test/experiments/test_qwen35_aten.py "$@"
elif [[ "$mode" == gdn-benchmark ]]; then
    exec python -u test/experiments/bench_qwen35_gdn.py "$@"
else
    exec python -u -m tokenspeed.experiments.qwen35.run \
        --model /home/u2601213425/hw1/models/Qwen3.5-0.8B --mode "$mode" "$@"
fi
