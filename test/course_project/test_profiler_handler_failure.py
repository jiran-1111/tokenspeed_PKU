"""Focused profiler-handler behavior without importing runtime dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_start_profile(create_profiler: Mock):
    source_path = (
        REPO_ROOT
        / "python"
        / "tokenspeed"
        / "runtime"
        / "engine"
        / "request_handler.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    request_handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RequestHandler"
    )
    method = next(
        node
        for node in request_handler.body
        if isinstance(node, ast.FunctionDef) and node.name == "start_profile"
    )
    method.decorator_list = []
    method.returns = None
    for argument in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
        argument.annotation = None

    namespace = {
        "ProfileReqOutput": lambda *, success, message: SimpleNamespace(
            success=success, message=message
        ),
        "resolve_profiler_backend": lambda **_kwargs: SimpleNamespace(is_npu=True),
        "current_platform": lambda: SimpleNamespace(is_npu=True),
        "torch": object(),
        "create_profiler": create_profiler,
    }
    isolated = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(isolated), str(source_path), "exec"),
        namespace,
    )
    return namespace["start_profile"]


class TestProfilerHandlerFailure(TestCase):
    def test_unsupported_profiler_configuration_returns_failure_without_activation(
        self,
    ):
        create_profiler = Mock(
            side_effect=RuntimeError("Level1 profiler configuration unsupported")
        )
        start_profile = _load_start_profile(create_profiler)
        handler = SimpleNamespace(
            profiler_activities=["CPU", "NPU"],
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=True,
            torch_profiler=None,
            torch_profiler_is_npu=False,
            profile_in_progress=False,
        )

        result = start_profile(handler)

        self.assertFalse(result.success)
        self.assertIn("Level1 profiler configuration unsupported", result.message)
        self.assertFalse(handler.profile_in_progress)
        self.assertIsNone(handler.torch_profiler)
        create_profiler.assert_called_once()
