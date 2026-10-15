# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for rank-local startup trace markers."""

import ast
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tokenspeed.runtime.execution.startup_trace import (
    trace_startup_operation,
)


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _trace_stages(node: ast.FunctionDef) -> set[str]:
    stages = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if (
            not isinstance(function, ast.Name)
            or function.id != "trace_startup_operation"
        ):
            continue
        for keyword in child.keywords:
            if keyword.arg == "stage" and isinstance(keyword.value, ast.Constant):
                stages.add(keyword.value.value)
    return stages


def test_operation_logs_boundaries_and_returns_result():
    logger = Mock()
    operation = Mock(return_value="ready")

    result = trace_startup_operation(
        logger,
        rank=1,
        stage="attention_kv_construction",
        operation=operation,
        detail="initial",
    )

    assert result == "ready"
    operation.assert_called_once_with()
    assert [call.args[0] for call in logger.info.call_args_list] == [
        "startup_trace rank=1 stage=attention_kv_construction "
        "event=start detail=initial",
        "startup_trace rank=1 stage=attention_kv_construction "
        "event=complete detail=initial",
    ]


def test_operation_error_propagates_without_complete_marker():
    logger = Mock()
    operation = Mock(side_effect=RuntimeError("diagnostic failure"))

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        trace_startup_operation(
            logger,
            rank=0,
            stage="graph_capture",
            operation=operation,
            detail="batch_size=1",
        )

    operation.assert_called_once_with()
    assert [call.args[0] for call in logger.info.call_args_list] == [
        "startup_trace rank=0 stage=graph_capture " "event=start detail=batch_size=1",
    ]


def test_device_builder_traces_startup_boundaries():
    node = _function_node(
        ROOT / "python/tokenspeed/runtime/execution/device.py",
        "build_device_side",
    )

    assert {
        "model_runner_init",
        "multimodal_runtime_warmup",
        "communication_runtime_prepare",
        "attention_kv_construction",
        "model_executor_create",
        "kernel_autotune",
        "graph_capture",
    } <= _trace_stages(node)


def test_decode_graph_syncs_and_barriers_are_rank_traced():
    node = _function_node(
        ROOT / "python/tokenspeed/runtime/execution/forward_step.py",
        "_capture_one",
    )
    direct_barrier_calls = []
    direct_sync_calls = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Attribute) and child.func.attr == "barrier":
            direct_barrier_calls.append(child)
        if isinstance(child.func, ast.Attribute) and child.func.attr == "synchronize":
            direct_sync_calls.append(child)

    assert not direct_barrier_calls
    assert not direct_sync_calls
    assert {
        "graph_capture_device_sync",
        "graph_capture_distributed_barrier",
    } <= _trace_stages(node)
