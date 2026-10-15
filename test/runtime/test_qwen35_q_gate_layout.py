import torch

import tokenspeed.runtime.models.qwen3_5 as qwen35


def test_qwen35_q_gate_is_split_per_head():
    q_gate = torch.arange(16).reshape(1, 16)
    split = getattr(qwen35, "_split_q_gate_by_head", None)

    assert callable(split), "Qwen3.5 Q/gate split helper is missing"

    q, gate = split(q_gate, num_heads=2, head_dim=4)

    torch.testing.assert_close(q, torch.tensor([[0, 1, 2, 3, 8, 9, 10, 11]]))
    torch.testing.assert_close(gate, torch.tensor([[4, 5, 6, 7, 12, 13, 14, 15]]))
