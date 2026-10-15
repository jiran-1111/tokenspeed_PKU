"""Bounded two-rank HCCL diagnostic; no model or downloads."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=60))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for count in (1, 4096):
                tensor = torch.full(
                    (count,), rank + 1, dtype=dtype, device=f"npu:{rank}"
                )
                dist.all_reduce(tensor)
                torch.npu.synchronize()
                assert torch.all(tensor == 3).item()
                print(
                    f"rank={rank} dtype={dtype} count={count} all_reduce=PASS",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
