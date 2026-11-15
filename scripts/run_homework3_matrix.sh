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
: "${ASCEND_RT_VISIBLE_DEVICES:?Select two chips explicitly}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
output=${1:?Specify a new matrix output directory}
mkdir "$output"
npu-smi info -m >"$output/mapping.txt"
npu-smi info >"$output/before.txt"
for model in 0.8b 4b; do
    for pattern in high low; do
        for variant in off on-small on-large; do
            mode=on
            ratio=2
            case "$variant" in
                off) mode=off ;;
                on-small) ratio=0.3 ;;
            esac
            name="$model-tp2-$pattern-$variant"
            set +e
            timeout --kill-after=15s 360s bash scripts/run_homework3.sh \
                --config "configs/homework3/qwen35_${model}_tp2.yaml" \
                --offloading "$mode" --host-ratio "$ratio" --pattern "$pattern" \
                --cycles 4 --output-dir "$output/$name" >"$output/$name.log" 2>&1
            rc=$?
            set -e
            printf '%s\t%s\n' "$name" "$rc" >>"$output/status.tsv"
            if [[ "$rc" != 0 ]]; then
                exit "$rc"
            fi
        done
    done
done
npu-smi info >"$output/after.txt"
