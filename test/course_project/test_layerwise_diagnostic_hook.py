"""The temporary layer hook must keep TP1 residual semantics and guards."""

import ast
from pathlib import Path


def test_layerwise_wrapper_recombines_residual_only_when_armed():
    path = Path(__file__).resolve().parents[2] / "scripts/homework1_layerwise_hook.py"
    tree = ast.parse(path.read_text())
    install = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install"
    )
    definitions = [
        n
        for n in install.body
        if isinstance(n, ast.FunctionDef) and n.name == "wrap_layerwise_capture"
    ]
    assert len(definitions) == 1, "missing guarded layerwise capture wrapper"
    mode = [None]
    observations = []

    class Values:
        def __init__(self, values):
            self.values = values

        def __add__(self, other):
            return Values([a + b for a, b in zip(self.values, other.values)])

    class Layer:
        layer_id = 3

        def forward(self, *, hidden_states, residual, **kwargs):
            return hidden_states, residual

    namespace = {
        "target_mode": lambda: mode[0],
        "wraps": lambda f: lambda g: g,
        "save_layerwise": lambda name, tensor: observations.append(
            (name, tensor.values)
        ),
        "attach_layer0_hooks": lambda layer: [],
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["wrap_layerwise_capture"](Layer)
    layer = Layer()
    hidden, residual = Values([1, 2]), Values([3, 4])
    result = layer.forward(hidden_states=hidden, residual=residual)
    assert result == (hidden, residual) and observations == []
    mode[0] = "full_recompute"
    result = layer.forward(hidden_states=hidden, residual=residual)
    assert result == (hidden, residual)
    assert observations == [("layer-3-input", [4, 6]), ("layer-3-output", [4, 6])]
    observations.clear()
    layer.forward(hidden_states=hidden, residual=None)
    assert observations == [("layer-3-input", [1, 2]), ("layer-3-output", [1, 2])]


def test_layer0_component_hooks_are_guarded_and_capture_tuple_output():
    path = Path(__file__).resolve().parents[2] / "scripts/homework1_layerwise_hook.py"
    install = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "install"
    )
    definitions = [
        n
        for n in install.body
        if isinstance(n, ast.FunctionDef) and n.name == "attach_layer0_hooks"
    ]
    assert len(definitions) == 1, "missing layer0 component boundary hooks"
    observations, hooks = [], []
    armed = [False]

    class Module:
        def register_forward_hook(self, hook):
            hooks.append(hook)
            return hook

    class Layer:
        input_layernorm = Module()
        linear_attn = Module()
        post_attention_layernorm = Module()
        mlp = Module()

    namespace = {
        "target_mode": lambda: "full_recompute" if armed[0] else None,
        "save_layerwise": lambda name, tensor: observations.append((name, tensor)),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    assert len(namespace["attach_layer0_hooks"](Layer())) == 4
    value = object()
    hooks[0](None, None, (value, object()))
    assert observations == []
    armed[0] = True
    for hook in hooks:
        hook(None, None, (value, object()))
    assert observations == [
        ("layer-0-" + name, value)
        for name in ["input-norm", "gdn-output", "post-attn-norm", "mlp-output"]
    ]
