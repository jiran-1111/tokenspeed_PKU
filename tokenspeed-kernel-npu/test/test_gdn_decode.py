import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def test_gdn_decode_step_triton_matches_reference():
    torch.manual_seed(11)
    dtype = torch.bfloat16
    q = torch.randn(1, 1, 1, 128, device="npu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 1, 128, device="npu", dtype=dtype)
    a = torch.randn(1, 1, 1, device="npu")
    b = torch.randn_like(a)
    A_log = torch.randn(1, device="npu")
    dt_bias = torch.randn_like(A_log)
    pool = torch.randn(2, 1, 128, 128, device="npu", dtype=dtype) * 0.02
    initial_state = pool[0, 0].float().cpu().transpose(-2, -1)

    output = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=torch.tensor([0], device="npu", dtype=torch.int32),
        scale=128**-0.5,
        solution="triton",
    )

    q_ref, k_ref, v_ref = (tensor.float().cpu().reshape(128) for tensor in (q, k, v))
    q_ref = q_ref / torch.sqrt(q_ref.square().sum() + 1e-6)
    k_ref = k_ref / torch.sqrt(k_ref.square().sum() + 1e-6)
    decay = -torch.exp(A_log.cpu()) * F.softplus(a.cpu() + dt_bias.cpu())
    beta = torch.sigmoid(b.cpu()).reshape(())
    decayed_state = initial_state * torch.exp(decay.reshape(()))
    delta = beta * (v_ref - k_ref @ decayed_state)
    expected_state = decayed_state + k_ref[:, None] * delta[None, :]
    expected_output = (q_ref[:, None] * expected_state).sum(dim=0) * 128**-0.5

    torch.testing.assert_close(
        output.cpu().float().reshape(128), expected_output, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        pool[0, 0].cpu().float(),
        expected_state.transpose(-2, -1).to(dtype).float(),
        atol=3e-2,
        rtol=3e-2,
    )
