"""Exercise the cached-prefix probe's zero-token control-flow boundary."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("prefix", [[], [12, 13]])
def test_cached_probe_keeps_prefill_result_when_prefix_is_empty(prefix):
    path = (
        Path(__file__).resolve().parents[2] / "scripts/probe_reference_prefix_logits.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    block = next(n for n in main.body if isinstance(n, ast.With))
    end = next(
        i
        for i, n in enumerate(block.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and isinstance(n.targets[0].value, ast.Name)
        and n.targets[0].value.id == "observations"
    )
    calls = []

    class Logits:
        def __getitem__(self, index):
            return self

        def float(self):
            return self

        def topk(self, count):
            return (
                SimpleNamespace(tolist=lambda: [30.0, 29.0]),
                SimpleNamespace(tolist=lambda: [7, 8]),
            )

    def model(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(past_key_values=object(), logits=Logits())

    namespace = {
        "torch": SimpleNamespace(tensor=lambda x, **kwargs: x),
        "prompt": [1, 2],
        "prefix": prefix,
        "model": model,
        "observations": {},
    }
    exec(
        compile(
            ast.Module(body=block.body[: end + 1], type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    assert len(calls) == len(prefix) + 1
    assert namespace["observations"]["cached"] == {
        "ids": [7, 8],
        "logits": [30.0, 29.0],
    }
