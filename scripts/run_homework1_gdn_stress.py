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

"""Fixed correlated-key GDN regression with full output/state evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step


def reference(q, k, v, g, beta, initial):
    """CPU FP32 token recurrence; state is K by V, unlike the NPU pool."""
    state = initial[0].transpose(-1, -2).clone()
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[0, t].float(), k[0, t].float(), v[0, t].float()
        qt = qt * (qt.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        kt = kt * (kt.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        state = state * g[0, t].exp()[:, None, None]
        error = (vt - torch.bmm(kt[:, None, :], state).squeeze(1)) * beta[0, t, :, None]
        state = state + kt[:, :, None] * error[:, None, :]
        outputs.append(torch.bmm(qt[:, None, :], state).squeeze(1) * 128**-0.5)
    return torch.stack(outputs).unsqueeze(0), state.transpose(-1, -2).unsqueeze(0)


def metrics(actual, expected, *, state):
    actual, expected = actual.cpu().float(), expected.cpu().float()
    atol, rtol = (3e-4, 1e-2) if state else (2e-3, 2e-2)
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    failures = int((difference > atol + rtol * expected.abs()).sum())
    return dict(
        finite=finite,
        passed=finite and failures == 0,
        failed_elements=failures,
        max_abs=float(difference.max()),
        relative_l2=float(difference.norm() / expected.norm().clamp_min(1e-12)),
        atol=atol,
        rtol=rtol,
    )


def prefill(q, k, v, g, beta, initial):
    result = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=128**-0.5,
        initial_state=initial.npu(),
        cu_seqlens=torch.tensor([0, q.shape[1]], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    return result.out.cpu(), result.final_state.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(20261022)
    key = torch.randn(1, 1, 16, 128, generator=generator)
    q = (key + 0.05 * torch.randn(1, 129, 16, 128, generator=generator)).bfloat16()
    k = (key + 0.01 * torch.randn(1, 129, 16, 128, generator=generator)).bfloat16()
    v = torch.randn(1, 129, 16, 128, generator=generator).bfloat16()
    initial = torch.randn(1, 16, 128, 128, generator=generator) * 0.01
    A_log = torch.full((16,), math.log(1e-8 / math.log(2)))
    dt_bias = torch.zeros(16)
    a = torch.zeros(1, 129, 16)
    b = torch.full((1, 129, 16), math.log(99))
    g = -A_log.exp()[None, None, :] * F.softplus(a + dt_bias)
    beta = b.sigmoid()
    protocol = dict(
        seed=20261022,
        lengths=[64, 65, 127, 128, 129],
        shape=[1, 129, 16, 128],
        q_noise=0.05,
        k_noise=0.01,
        initial_state_std=0.01,
        g=float(g[0, 0, 0]),
        beta=float(beta[0, 0, 0]),
        output_atol=0.002,
        output_rtol=0.02,
        state_atol=0.0003,
        state_rtol=0.01,
        split="32 for length64; otherwise64",
        decode_read=1,
        decode_write=3,
        reference_state_orientation="K,V",
        npu_state_orientation="V,K",
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.save(
        dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial=initial,
            A_log=A_log,
            dt_bias=dt_bias,
            a=a,
            b=b,
        ),
        args.output / "inputs.pt",
    )
    results = []
    for length in protocol["lengths"]:
        inputs = [x[:, :length] for x in (q, k, v, g, beta)]
        expected_out, expected_state = reference(*inputs, initial)
        full_out, full_state = prefill(*inputs, initial)
        split = 32 if length == 64 else 64
        first_out, first_state = prefill(*(x[:, :split] for x in inputs), initial)
        second_out, second_state = prefill(*(x[:, split:] for x in inputs), first_state)
        segment_out = torch.cat([first_out, second_out], dim=1)
        original_pool = torch.randn(5, 16, 128, 128, generator=generator) * 0.01
        original_pool[1] = initial[0]
        pool = original_pool.npu()
        decoded = []
        untouched = True
        for t in range(length):
            read = 1 if t == 0 else 3
            output = gdn_decode_step(
                q[:, t : t + 1].npu(),
                k[:, t : t + 1].npu(),
                v[:, t : t + 1].npu(),
                A_log=A_log.npu(),
                a=a[:, t : t + 1].npu(),
                dt_bias=dt_bias.npu(),
                b=b[:, t : t + 1].npu(),
                initial_state=pool,
                initial_state_indices=torch.tensor(
                    [read], device="npu", dtype=torch.int32
                ),
                output_state_indices=torch.tensor([3], device="npu", dtype=torch.int32),
                scale=128**-0.5,
                solution="triton",
            )
            decoded.append(output.cpu().reshape(1, 1, 16, 128))
            untouched = untouched and torch.equal(
                pool.cpu()[[0, 1, 2, 4]], original_pool[[0, 1, 2, 4]]
            )
        decode_out = torch.cat(decoded, dim=1)
        decode_state = pool[3:4].cpu()
        hybrid_pool_cpu = original_pool.clone()
        hybrid_pool_cpu[3] = first_state[0]
        hybrid_pool = hybrid_pool_cpu.npu()
        hybrid_outputs = [first_out]
        for t in range(split, length):
            output = gdn_decode_step(
                q[:, t : t + 1].npu(),
                k[:, t : t + 1].npu(),
                v[:, t : t + 1].npu(),
                A_log=A_log.npu(),
                a=a[:, t : t + 1].npu(),
                dt_bias=dt_bias.npu(),
                b=b[:, t : t + 1].npu(),
                initial_state=hybrid_pool,
                initial_state_indices=torch.tensor(
                    [3], device="npu", dtype=torch.int32
                ),
                output_state_indices=torch.tensor([3], device="npu", dtype=torch.int32),
                scale=128**-0.5,
                solution="triton",
            )
            hybrid_outputs.append(output.cpu().reshape(1, 1, 16, 128))
            untouched = untouched and torch.equal(
                hybrid_pool.cpu()[[0, 1, 2, 4]], hybrid_pool_cpu[[0, 1, 2, 4]]
            )
        hybrid_out = torch.cat(hybrid_outputs, dim=1)
        hybrid_state = hybrid_pool[3:4].cpu()
        checks = {
            "prefill_output": metrics(full_out, expected_out.bfloat16(), state=False),
            "prefill_state": metrics(full_state, expected_state, state=True),
            "segmented_output": metrics(
                segment_out, expected_out.bfloat16(), state=False
            ),
            "segmented_state": metrics(second_state, expected_state, state=True),
            "segmented_vs_full_output": metrics(segment_out, full_out, state=False),
            "segmented_vs_full_state": metrics(second_state, full_state, state=True),
            "decode_output": metrics(decode_out, expected_out.bfloat16(), state=False),
            "decode_state": metrics(decode_state, expected_state, state=True),
            "decode_vs_prefill_output": metrics(decode_out, full_out, state=False),
            "decode_vs_prefill_state": metrics(decode_state, full_state, state=True),
            "prefill_then_decode_output": metrics(
                hybrid_out, expected_out.bfloat16(), state=False
            ),
            "prefill_then_decode_state": metrics(
                hybrid_state, expected_state, state=True
            ),
            "prefill_then_decode_vs_full_output": metrics(
                hybrid_out, full_out, state=False
            ),
            "prefill_then_decode_vs_full_state": metrics(
                hybrid_state, full_state, state=True
            ),
        }
        result = dict(
            length=length,
            untouched_slots_exact=bool(untouched),
            checks=checks,
            passed=bool(untouched) and all(x["passed"] for x in checks.values()),
        )
        torch.save(
            dict(
                reference_output=expected_out,
                reference_state=expected_state,
                full_output=full_out,
                full_state=full_state,
                segmented_output=segment_out,
                segmented_state=second_state,
                decode_output=decode_out,
                decode_state=decode_state,
                hybrid_output=hybrid_out,
                hybrid_state=hybrid_state,
            ),
            args.output / f"length_{length}.pt",
        )
        (args.output / f"length_{length}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        results.append(result)
        print(
            json.dumps(
                dict(
                    length=length,
                    passed=result["passed"],
                    failed_checks=[
                        n for n, value in checks.items() if not value["passed"]
                    ],
                )
            ),
            flush=True,
        )
    summary = dict(
        passed=all(x["passed"] for x in results),
        lengths=results,
        elapsed_seconds=time.perf_counter() - start,
        inputs_sha256=hashlib.sha256(
            (args.output / "inputs.pt").read_bytes()
        ).hexdigest(),
    )
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
