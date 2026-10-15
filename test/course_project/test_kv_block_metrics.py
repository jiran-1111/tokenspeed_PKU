import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.course_project import telemetry


@pytest.fixture
def collector(monkeypatch):
    env = SimpleNamespace(
        envs=SimpleNamespace(
            TOKENSPEED_TEST_REQUEST_TIME_STATS=SimpleNamespace(get=lambda: False)
        )
    )
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils.env", env)
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/metrics/collector.py"
    )
    spec = importlib.util.spec_from_file_location("kv_metrics_test_collector", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("active,cached,total", [(3, 2, 10), (0, 4, 10), (0, 0, 0)])
def test_scheduler_snapshot_exports_lcm_block_partition(
    collector, active, cached, total
):
    registry = pytest.importorskip("prometheus_client").CollectorRegistry()
    metrics = collector.EngineMetrics({"tp_rank": "0"}, enabled=True, registry=registry)
    metrics.record_scheduler_iteration(
        running=1,
        waiting=2,
        num_active_pages=active,
        num_cached_pages=cached,
        num_total_pages=total,
        num_iteration_tokens=1,
    )
    expected = {
        "total": total,
        "active": active,
        "cached": cached,
        "empty": total - active - cached,
        "allocated": active + cached,
    }
    for name, value in expected.items():
        assert (
            registry.get_sample_value(
                "tokenspeed:kv_lcm_blocks_" + name, {"tp_rank": "0"}
            )
            == value
        )


def test_benchmark_records_block_counts_and_load_peaks():
    payload = "\n".join(
        f'tokenspeed:kv_lcm_blocks_{name}{{tp_rank="0"}} {value}'
        for name, value in {
            "total": 10,
            "active": 3,
            "cached": 2,
            "empty": 5,
            "allocated": 5,
        }.items()
    )
    snapshot = telemetry.parse_prometheus_metrics(payload)
    assert snapshot["kv_lcm_blocks_allocated"] == 5
    sampler = telemetry.ServerMetricsSampler("http://unused")
    sampler._samples = [snapshot, {**snapshot, "kv_lcm_blocks_active": 4}]
    result = telemetry.add_metric_delta({}, snapshot, snapshot, sampler._peaks())
    assert result["kv_lcm_blocks_total"] == 10
    assert result["kv_lcm_blocks_cached"] == 2
    assert result["kv_lcm_blocks_active_peak"] == 4


def test_old_metrics_endpoint_does_not_fabricate_block_counts():
    result = telemetry.add_metric_delta({}, {}, {"kv_cache_usage_ratio": 0.5})
    assert not any(key.startswith("kv_lcm_blocks_") for key in result)


@pytest.mark.parametrize("active,empty", [(3, 5), (0, 6), (0, 10)])
def test_event_loop_snapshot_reaches_metrics_without_losing_cached_blocks(
    collector, active, empty
):
    registry = pytest.importorskip("prometheus_client").CollectorRegistry()
    metrics = collector.EngineMetrics({"tp_rank": "0"}, enabled=True, registry=registry)
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/engine/event_loop.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    event_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    methods = [
        node
        for node in event_loop.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_get_scheduler_stats", "_record_scheduler_iteration_metrics"}
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    loop = SimpleNamespace(
        scheduler=SimpleNamespace(
            empty_lcm_blocks=lambda: empty,
            active_lcm_blocks=lambda: active,
            waiting_size=lambda: 2,
        ),
        _scheduler_cache_geometry=SimpleNamespace(num_usable_pages=10),
        _num_running=lambda: 1,
        metrics=metrics,
    )
    stats = namespace["_get_scheduler_stats"](loop)
    namespace["_record_scheduler_iteration_metrics"](loop, stats, 1)
    for state, expected in {
        "active": active,
        "empty": empty,
        "cached": 10 - empty - active,
        "allocated": 10 - empty,
    }.items():
        assert (
            registry.get_sample_value(
                "tokenspeed:kv_lcm_blocks_" + state, {"tp_rank": "0"}
            )
            == expected
        )
