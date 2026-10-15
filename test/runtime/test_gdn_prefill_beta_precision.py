from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_mamba_npu_prefill_computes_beta_in_fp32_before_sigmoid(monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    seq_len, heads, head_dim = 3, 2, 16
    query = torch.randn(
        1, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    recurrent_state = torch.zeros(
        1, heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    a = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.zeros(heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    boundaries = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    captured: dict[str, torch.Tensor] = {}

    def capture_beta(q, k, v, g, beta, **kwargs):
        captured["beta"] = beta
        return SimpleNamespace(
            out=torch.zeros_like(v), final_state=kwargs["initial_state"]
        )

    monkeypatch.setattr(mamba, "gdn_chunk_prefill", capture_beta)
    backend = object.__new__(mamba.MambaAttnBackend)
    backend._prefill_scan(
        query,
        key,
        value,
        recurrent_state,
        boundaries,
        A_log=A_log,
        dt_bias=dt_bias,
        a=a,
        b=b,
        g_raw=None,
        f_a_out=None,
        f_b_weight=None,
        beta_raw=None,
        seq_len=seq_len,
        num_real_tokens=seq_len,
        lower_bound=None,
        inputs_packed=False,
    )

    expected = torch.sigmoid(b.float()).unsqueeze(0)
    assert captured["beta"].dtype == torch.float32
    torch.testing.assert_close(captured["beta"], expected, rtol=0, atol=0)
