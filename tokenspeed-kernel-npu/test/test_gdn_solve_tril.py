import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn._triton.solve_tril import solve_tril

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def test_solve_tril_32x32_matches_reference():
    torch.manual_seed(23)
    block_size = 32
    matrix = (
        torch.randn(block_size, block_size, device="npu", dtype=torch.float32) * 0.05
    )
    matrix = torch.tril(matrix, diagonal=-1).unsqueeze(0).unsqueeze(2)
    cu_seqlens = torch.tensor([0, block_size], device="npu", dtype=torch.int32)

    result = solve_tril(matrix, cu_seqlens=cu_seqlens, output_dtype=torch.bfloat16)

    matrix_ref = matrix[0, :, 0, :].float().cpu()
    identity = torch.eye(block_size)
    expected = torch.linalg.solve_triangular(
        identity + matrix_ref, identity, upper=False
    )
    torch.testing.assert_close(
        result[0, :, 0, :].float().cpu(), expected, atol=1e-2, rtol=1e-2
    )
