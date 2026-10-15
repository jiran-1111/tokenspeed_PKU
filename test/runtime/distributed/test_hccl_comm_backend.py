"""Focused HCCL communication backend tests."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _all_gather_worker(rank: int, port: int) -> None:
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        group = (0, 1)
        pg_manager.register_process_group("hccl", group, dist.group.WORLD)
        inp = torch.tensor([rank, rank + 10], dtype=torch.int32, device=f"npu:{rank}")
        output = torch.empty(4, dtype=torch.int32, device=inp.device)

        HcclBackend().all_gather_single(output, inp, group)

        torch.testing.assert_close(
            output.cpu(), torch.tensor([0, 10, 1, 11], dtype=torch.int32)
        )

        inp = torch.tensor(
            [rank * 10 + value for value in range(4)],
            dtype=torch.int32,
            device=f"npu:{rank}",
        )
        output = HcclBackend().reduce_scatter(inp, group)
        expected = torch.tensor([10, 12] if rank == 0 else [14, 16], dtype=torch.int32)
        torch.testing.assert_close(output.cpu(), expected)

        inp = torch.tensor(
            [[(rank + 1) * 10], [(rank + 2) * 10], [(rank + 3) * 10]],
            dtype=torch.int32,
            device=f"npu:{rank}",
        )
        output = HcclBackend().token_reduce_scatter(inp, group, [1, 2])
        expected = torch.tensor([30] if rank == 0 else [50, 70], dtype=torch.int32)
        torch.testing.assert_close(output.cpu().flatten(), expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_RUN_HCCL_TESTS") != "1",
    reason="set TOKENSPEED_RUN_HCCL_TESTS=1 to run the two-NPU HCCL test",
)
def test_all_gather_single_uses_a_supported_hccl_collective() -> None:
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < 2:
        pytest.skip("requires two visible NPUs")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(_all_gather_worker, args=(port,), nprocs=2, join=True)
