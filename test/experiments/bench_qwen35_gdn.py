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

"""Compare complete recurrent and chunk GDN on one already selected chip."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from tokenspeed_kernel.ops.experimental_qwen35 import gated_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    records = []
    for length in (1, 32, 512):
        torch.manual_seed(length)
        q, k, v = [
            torch.randn(1, length, 16, 128, dtype=torch.bfloat16).to("npu:0")
            for _ in range(3)
        ]
        g = torch.full((1, length, 16), -0.01, device="npu:0")
        beta = torch.full((1, length, 16), 0.9, device="npu:0", dtype=torch.bfloat16)
        initial = torch.zeros(1, 16, 128, 128, device="npu:0")
        inputs = (q, k, v, g, beta, initial)
        expected = gated_delta(*inputs, algorithm="recurrent", chunk_size=64)
        for algorithm in ("recurrent", "chunk"):
            actual = gated_delta(*inputs, algorithm=algorithm, chunk_size=64)
            torch.testing.assert_close(actual[0], expected[0], atol=2e-3, rtol=3e-3)
            torch.testing.assert_close(actual[1], expected[1], atol=3e-5, rtol=3e-3)
            times = []
            for _ in range(args.repetitions):
                torch.npu.synchronize()
                start = time.perf_counter()
                gated_delta(*inputs, algorithm=algorithm, chunk_size=64)
                torch.npu.synchronize()
                times.append(1000 * (time.perf_counter() - start))
            row = {
                "tokens": length,
                "algorithm": algorithm,
                "median_ms": statistics.median(times),
                "runs_ms": times,
            }
            records.append(row)
            print(row, flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "physical_chip": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                "shape": "[1,T,16,128]",
                "dtype": "BF16 with FP32 state",
                "chunk_size": 64,
                "results": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
