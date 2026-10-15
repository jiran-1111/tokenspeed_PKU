import ast
from pathlib import Path


def test_runtime_triton_uses_kernel_adapter():
    source = (
        Path(__file__).parents[3]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "utils"
        / "triton.py"
    ).read_text()
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tokenspeed_kernel._triton"
        and {alias.name for alias in node.names} >= {"tl", "triton"}
        for node in ast.walk(tree)
    )
