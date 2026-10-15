import ast
from pathlib import Path
from types import SimpleNamespace


def test_request_handler_allows_last_context_token():
    root = Path(__file__).resolve().parents[2]
    folder = root / "python/tokenspeed/runtime/engine"
    tree = ast.parse((folder / "event_loop.py").read_text(encoding="utf-8"))
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "RequestHandler"
    )
    limit_expr = next(k.value for k in call.keywords if k.arg == "max_req_len")
    limit = eval(
        compile(ast.Expression(limit_expr), "limit", "eval"),
        {"self": SimpleNamespace(max_model_len=4096)},
    )
    tree = ast.parse((folder / "request_handler.py").read_text(encoding="utf-8"))
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "max_new_tokens"
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "min"
    )
    for input_len in (1, 128, 4094, 4095):
        state = SimpleNamespace(
            prompt_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(max_new_tokens=4096 - input_len),
        )
        cap = eval(
            compile(ast.Expression(assignment.value), "cap", "eval"),
            {"self": SimpleNamespace(max_req_len=limit), "req_state": state},
        )
        assert cap == 4096 - input_len
