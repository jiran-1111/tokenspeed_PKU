import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_npu_adapter_imports_tokenspeed_triton_profiler():
    adapter = ROOT / "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/_triton.py"
    tree = ast.parse(adapter.read_text(), filename=str(adapter))

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tokenspeed_triton"
        and any(
            alias.name == "profiler" and alias.asname == "proton"
            for alias in node.names
        )
        for node in ast.walk(tree)
    ), "NPU adapter must import proton from tokenspeed_triton.profiler"
