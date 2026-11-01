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

import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_native_device_guard_is_available():
    assert (
        importlib.util.find_spec("benchmarks.course_project.native_devices") is not None
    )


def table(*, pids=(), used=1000):
    rows = "\n".join(f"| 5 1 | {pid} | python | 100 |" for pid in pids)
    return (
        f"| 5 Ascend910 | OK | irrelevant |\n"
        f"| 1 11 | 0000:03:00.0 | 0 / 0 {used} / 65536 |\n"
        "+---+\n| NPU Chip | Process id | Process name | Process memory(MB) |\n"
        "+---+\n" + (rows or "| No running processes found |") + "\n+---+\n"
    )


@pytest.fixture
def guard_env(monkeypatch, tmp_path):
    import benchmarks.course_project.native_devices as devices

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "python").mkdir()
    source = repo / "python/main.py"
    source.write_text("pass\n")
    manifest = repo / "sources.json"
    manifest.write_text(
        json.dumps({"python/main.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    out = repo / "results"
    out.mkdir()
    state = {
        "table": table(),
        "identities": {100: (7, "1000"), 200: (8, "2000"), 300: (7, "3000")},
        "parents": {300: 100},
        "signals": [],
        "threads": [],
        "locks": [],
    }

    class Thread:
        def __init__(self, *, target, **kwargs):
            self.target = target
            state["threads"].append(self)

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    def query(args):
        if args == ["-m"]:
            return "5 0 10 10 Ascend910\n5 1 11 11 Ascend910\n"
        return state["table"]

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(
            LOCK_EX=1, LOCK_NB=2, flock=lambda *args: state["locks"].append(args)
        ),
    )
    monkeypatch.setenv("TOKENSPEED_HW2_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "11")
    monkeypatch.setattr(devices, "_query", query)
    monkeypatch.setattr(devices, "_identity", lambda pid: state["identities"][pid])
    monkeypatch.setattr(devices, "_parent", lambda pid: state["parents"].get(pid, 1))
    monkeypatch.setattr(devices.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        devices.os, "kill", lambda pid, sig: state["signals"].append((pid, sig))
    )
    monkeypatch.setattr(devices.threading, "Thread", Thread)
    args = dict(
        npu_id=5,
        chip_logic_ids={1: 11},
        tp_size=1,
        resource_mode="shared",
        min_free_mb=8192,
        output_dir=out,
        source_manifest=manifest,
        receipt={},
    )
    return devices, repo, args, state


def test_explicit_device_policy_and_nondefault_tp1_chip(guard_env):
    devices, repo, args, state = guard_env
    assert devices.parse_chip_logic_ids("1:11") == {1: 11}
    with devices.native_hardware_lease(repo, **args):
        pass
    assert args["receipt"]["engine_children_released"]
    assert args["receipt"]["sources_unchanged"]
    assert state["locks"]
    assert (repo.parent / ".tmp/hw2-npu.lock").exists()
    assert args["receipt"]["after_target_idle"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"tp_size": 2}, "count"),
        ({"chip_logic_ids": {1: 12}}, "mapping"),
        ({"min_free_mb": None}, "8192"),
        ({"min_free_mb": 8191}, "8192"),
        ({"resource_mode": "auto"}, "resource mode"),
    ],
)
def test_invalid_device_and_shared_policies_fail_closed(guard_env, change, match):
    devices, repo, args, state = guard_env
    args.update(change)
    with pytest.raises(ValueError, match=match):
        with devices.native_hardware_lease(repo, **args):
            pytest.fail("invalid target entered engine body")


@pytest.mark.parametrize("fault", ["unknown", "reuse", "memory", "truncated"])
def test_monitor_preserves_failure_table_and_signals_only_collector(guard_env, fault):
    devices, repo, args, state = guard_env
    state["table"] = table(pids=(200,))
    with pytest.raises(ValueError):
        with devices.native_hardware_lease(repo, **args):
            if fault == "unknown":
                state["table"] = table(pids=(200, 999))
                state["identities"][999] = (8, "9990")
            elif fault == "reuse":
                state["identities"][200] = (8, "2001")
            elif fault == "memory":
                state["table"] = table(pids=(200,), used=60000)
            else:
                state["table"] = "query truncated"
            state["threads"][0].target()
    receipt = args["receipt"]
    assert state["signals"] == [(100, signal.SIGINT)]
    assert receipt["violations"] and receipt["sources_unchanged"]
    assert any("error" in item for item in receipt["samples"])
    assert any(
        path.read_text() == state["table"]
        for path in args["output_dir"].glob("guard_*.txt")
    )


def test_background_is_preserved_and_owned_children_must_release(
    guard_env, monkeypatch
):
    devices, repo, args, state = guard_env
    state["table"] = table(pids=(200,))
    with devices.native_hardware_lease(repo, **args):
        state["table"] = table(pids=(100, 200, 300))
        # One observation verifies the worker ancestry and immutable identity.
        # Stop the loop after its first successful observation.
        monkeypatch.setattr(devices.threading.Event, "wait", lambda self, _: self.set())
        state["threads"][0].target()
        state["parents"][300] = 1
        calls = []

        def release(_):
            calls.append(True)
            state["table"] = table(pids=(100, 200))

        monkeypatch.setattr(devices.time, "sleep", release)
    receipt = args["receipt"]
    assert calls and receipt["engine_children_released"]
    assert receipt["parent_context_pending_exit"] and not receipt["after_target_idle"]
    assert receipt["after_process_pids"] == {1: [100, 200]}
    assert receipt["baseline"] == {1: {200: [8, "2000"]}}
    assert not state["signals"]


def test_idle_rejects_background_on_selected_chip(guard_env):
    devices, repo, args, state = guard_env
    args.update(resource_mode="idle", min_free_mb=None)
    state["table"] = table(pids=(200,))
    with pytest.raises(ValueError, match="occupied"):
        with devices.native_hardware_lease(repo, **args):
            pytest.fail("occupied idle target accepted")


def test_source_audit_runs_after_workload_failure(guard_env):
    devices, repo, args, state = guard_env
    with pytest.raises(RuntimeError, match="workload failed"):
        with devices.native_hardware_lease(repo, **args):
            (repo / "python/new.py").write_text("pass\n")
            raise RuntimeError("workload failed")
    assert any("coverage" in error for error in args["receipt"]["after_errors"])
    assert (args["output_dir"] / "after_mapping.txt").exists()


@pytest.mark.parametrize("script", ["validation", "performance"])
@pytest.mark.parametrize("missing", ["--npu-id", "--chip-logic-ids", "--resource-mode"])
def test_cli_requires_all_device_behavior_arguments(monkeypatch, script, missing):
    module = __import__(f"scripts.run_native_agent_{script}", fromlist=["main"])
    selected = {"--npu-id": "5", "--chip-logic-ids": "1:11", "--resource-mode": "idle"}
    extra = [
        part
        for key, value in selected.items()
        if key != missing
        for part in (key, value)
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--config",
            "missing",
            "--workload" if script == "validation" else "--protocol",
            "missing",
            "--source-manifest",
            "missing",
            "--output-dir",
            "missing",
            "--engine-port",
            "18780",
            "--metrics-port",
            "39982",
        ]
        + extra,
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_collector_context_in_live_baseline_is_still_pending_own_exit(guard_env):
    devices, repo, args, state = guard_env
    state["table"] = table(pids=(100, 200))
    with devices.native_hardware_lease(repo, **args):
        pass
    assert args["receipt"]["parent_context_pending_exit"]
    assert args["receipt"]["baseline"] == {1: {200: [8, "2000"]}}


@pytest.mark.parametrize("script", ["validation", "performance"])
def test_cli_persists_guard_startup_failure_with_explicit_devices(
    monkeypatch, tmp_path, script
):
    from contextlib import contextmanager

    module = __import__(f"scripts.run_native_agent_{script}", fromlist=["main"])
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    cfg = {
        "device": "npu",
        "dtype": "bfloat16",
        "enforce_eager": False,
        "tp_size": 1,
        "prefix_caching": False,
        "disable_kvstore": True,
        "enable_metrics": True,
        "max_model_len": 4096,
        "model_id": "fake",
    }
    monkeypatch.setattr(module, "load_model_config", lambda path: cfg)
    monkeypatch.setattr(module, "_local_model_path", lambda config: tmp_path)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(module.socket, "socket", Socket)
    if script == "performance":
        monkeypatch.setattr(
            module.os, "sched_getaffinity", lambda pid: {0}, raising=False
        )
        (tmp_path / "trace.jsonl").write_text("{}\n")
        payload = {
            "traces": {
                "fake": {
                    "path": "trace.jsonl",
                    "sha256": hashlib.sha256(
                        (tmp_path / "trace.jsonl").read_bytes()
                    ).hexdigest(),
                }
            },
            "sessions": 2,
            "turns": 3,
            "max_new_tokens": 2,
        }
        monkeypatch.setattr(module, "validate_trace", lambda *args, **kwargs: [])
    else:
        payload = {}
    data = tmp_path / "data.json"
    data.write_text(json.dumps(payload))
    manifest = tmp_path / "sources.json"
    manifest.write_text("{}")
    output = tmp_path / "result"

    @contextmanager
    def lease(repo, **kwargs):
        assert kwargs["npu_id"] == 5 and kwargs["chip_logic_ids"] == {1: 11}
        kwargs["receipt"]["failure_table"] = "preserved"
        raise ValueError("guard startup failure")
        yield

    monkeypatch.setattr(module, "native_hardware_lease", lease)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--config",
            "unused",
            "--workload" if script == "validation" else "--protocol",
            str(data),
            "--source-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--engine-port",
            "18780",
            "--metrics-port",
            "39982",
            "--npu-id",
            "5",
            "--chip-logic-ids",
            "1:11",
            "--resource-mode",
            "shared",
            "--min-free-mb",
            "8192",
        ],
    )
    with pytest.raises(ValueError, match="guard startup failure"):
        module.main()
    saved = json.loads((output / "run.json").read_text())
    assert saved["status"] == "failed" and saved["error_type"] == "ValueError"
    assert saved["devices"] == {"npu_id": 5, "chip_logic_ids": {"1": 11}}
    assert (
        saved["resource_mode"] == "shared"
        and saved["device_guard"]["failure_table"] == "preserved"
    )


@pytest.mark.parametrize("proc_exists", [False, True])
def test_process_exit_between_table_and_identity_is_distinguished_from_unreadable_proc(
    guard_env, monkeypatch, proc_exists
):
    devices, repo, args, state = guard_env
    state["table"] = table(pids=(200,))
    original_identity = devices._identity
    original_exists = Path.exists

    def identity(pid):
        if pid == 200:
            raise FileNotFoundError("process disappeared")
        return original_identity(pid)

    monkeypatch.setattr(devices, "_identity", identity)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            proc_exists if path == Path("/proc/200") else original_exists(path)
        ),
    )
    if proc_exists:
        with pytest.raises(FileNotFoundError, match="disappeared"):
            with devices.native_hardware_lease(repo, **args):
                pytest.fail("unreadable live process accepted")
    else:
        with devices.native_hardware_lease(repo, **args):
            pass
        assert args["receipt"]["samples"][0]["exited_during_query"] == [200]
        assert args["receipt"]["after_target_idle"]
        assert not state["signals"]


def test_every_selected_chip_obeys_the_explicit_shared_memory_reserve(
    guard_env, monkeypatch
):
    devices, repo, args, state = guard_env
    args.update(tp_size=2, chip_logic_ids={0: 10, 1: 11}, min_free_mb=16384)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "10,11")
    state["table"] = table().replace(
        "| 1 11 |", "| 0 10 | 0000:02:00.0 | 0 / 0 50000 / 65536 |\n| 1 11 |"
    )
    with pytest.raises(ValueError, match="memory reserve"):
        with devices.native_hardware_lease(repo, **args):
            pytest.fail("insufficient reserve on sibling chip accepted")
    assert args["receipt"]["samples"][0]["memory"][0]["used_mb"] == 50000


def test_workspace_must_be_explicit_to_coordinate_nested_checkouts(
    guard_env, monkeypatch
):
    devices, repo, args, state = guard_env
    monkeypatch.delenv("TOKENSPEED_HW2_WORKSPACE")
    with pytest.raises(KeyError, match="TOKENSPEED_HW2_WORKSPACE"):
        with devices.native_hardware_lease(repo, **args):
            pytest.fail("implicit workspace would use a different coordination lock")


def test_violation_finishing_during_monitor_join_rejects_run(guard_env, monkeypatch):
    devices, repo, args, state = guard_env
    with pytest.raises(ValueError, match="protection|cleanup"):
        with devices.native_hardware_lease(repo, **args):
            monkeypatch.setattr(
                state["threads"][0],
                "join",
                lambda timeout: args["receipt"]["violations"].append(
                    {"type": "ValueError", "message": "late unknown process"}
                ),
            )
    assert args["receipt"]["violations"]
    assert args["receipt"]["sources_unchanged"]


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_failed_guard_query_preserves_stdout_stderr_and_metadata(
    guard_env, monkeypatch, failure
):
    devices, repo, args, state = guard_env
    query = devices._query

    def failed(arguments):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                ["npu-smi", "info"],
                15,
                output=b"PARTIAL TABLE",
                stderr=b"TIMEOUT DETAIL",
            )
        raise subprocess.CalledProcessError(
            1, ["npu-smi", "info"], output="FAILED TABLE", stderr="DRIVER DETAIL"
        )

    with pytest.raises(ValueError):
        with devices.native_hardware_lease(repo, **args):
            monkeypatch.setattr(devices, "_query", failed)
            state["threads"][0].target()
            monkeypatch.setattr(devices, "_query", query)
    item = args["receipt"]["samples"][1]
    assert (args["output_dir"] / item["raw_table"]).read_text() == (
        "PARTIAL TABLE" if failure == "timeout" else "FAILED TABLE"
    )
    assert (args["output_dir"] / item["raw_stderr"]).read_text() == (
        "TIMEOUT DETAIL" if failure == "timeout" else "DRIVER DETAIL"
    )
    assert (
        item["query_timeout_seconds"] == 15
        if failure == "timeout"
        else item["query_returncode"] == 1
    )
    json.dumps(args["receipt"])


@pytest.mark.parametrize("script", ["validation", "performance"])
@pytest.mark.parametrize("failure", [None, "workload", "constructor"])
def test_runner_stops_monitor_after_workload_before_engine_shutdown(
    monkeypatch, tmp_path, script, failure
):
    from contextlib import contextmanager

    module = __import__(f"scripts.run_native_agent_{script}", fromlist=["main"])
    events = []
    stopped = False
    cfg = {
        "device": "npu",
        "dtype": "bfloat16",
        "enforce_eager": False,
        "tp_size": 1,
        "prefix_caching": False,
        "disable_kvstore": True,
        "enable_metrics": True,
        "max_model_len": 4096,
        "model_id": "fake",
        "model_revision": "local",
    }
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_model_config", lambda path: cfg)
    monkeypatch.setattr(module, "_local_model_path", lambda config: tmp_path)
    monkeypatch.setattr(module, "build_serve_command", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_engine_cli_args", lambda args: args)
    monkeypatch.setattr(module.asyncio, "run", lambda value: value)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(module.socket, "socket", Socket)

    def stop_monitor():
        nonlocal stopped
        if not stopped:
            events.append("monitor_stopped")
            stopped = True

    @contextmanager
    def lease(repo, **kwargs):
        try:
            yield stop_monitor
        finally:
            stop_monitor()
            events.append("after_engine_exit")

    monkeypatch.setattr(module, "native_hardware_lease", lease)

    class FakeEngine:
        def __init__(self, **kwargs):
            assert not stopped
            events.append("engine_constructed")
            if failure == "constructor":
                raise RuntimeError("constructor failed")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert stopped, "Engine shutdown would kill a live guard query"
            events.append("engine_shutdown")

    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.entrypoints.engine",
        SimpleNamespace(Engine=FakeEngine),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.utils.server_args",
        SimpleNamespace(prepare_server_args=lambda args: args),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            )
        ),
    )

    def workload(*args, **kwargs):
        assert not stopped, "monitor stopped before the workload finished"
        events.append("workload")
        if failure == "workload":
            raise RuntimeError("workload failed")

    monkeypatch.setattr(
        module,
        "run_native_validation" if script == "validation" else "measure_performance",
        workload,
    )
    if script == "performance":
        monkeypatch.setattr(
            module.os, "sched_getaffinity", lambda pid: {0}, raising=False
        )
        (tmp_path / "trace.jsonl").write_text("{}\n")
        data = {
            "traces": {
                "fake": {
                    "path": "trace.jsonl",
                    "sha256": hashlib.sha256(
                        (tmp_path / "trace.jsonl").read_bytes()
                    ).hexdigest(),
                }
            },
            "sessions": 2,
            "turns": 3,
            "max_new_tokens": 2,
        }
        monkeypatch.setattr(module, "validate_trace", lambda *args, **kwargs: [])
    else:
        data = {"control_schedule": "concurrent", "tool_metrics_interval_seconds": 0.1}
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "sources.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--config",
            "unused",
            "--workload" if script == "validation" else "--protocol",
            str(tmp_path / "data.json"),
            "--source-manifest",
            str(tmp_path / "sources.json"),
            "--output-dir",
            str(tmp_path / "result"),
            "--engine-port",
            "18780",
            "--metrics-port",
            "39982",
            "--npu-id",
            "5",
            "--chip-logic-ids",
            "1:11",
            "--resource-mode",
            "shared",
            "--min-free-mb",
            "8192",
        ],
    )
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            module.main()
    else:
        module.main()
    if failure != "constructor":
        assert events == [
            "engine_constructed",
            "workload",
            "monitor_stopped",
            "engine_shutdown",
            "after_engine_exit",
        ]
    else:
        assert events == ["engine_constructed", "monitor_stopped", "after_engine_exit"]


def test_interrupt_during_monitor_join_still_audits_sources_and_devices(
    guard_env, monkeypatch
):
    devices, repo, args, state = guard_env
    with pytest.raises(KeyboardInterrupt):
        with devices.native_hardware_lease(repo, **args):

            def interrupted(timeout):
                raise KeyboardInterrupt("guard interrupted during join")

            monkeypatch.setattr(state["threads"][0], "join", interrupted)
    assert args["receipt"]["sources_unchanged"]
    assert args["receipt"]["after_mapping"] == "after_mapping.txt"


def test_ignored_sigint_is_rejected_before_guard_queries_or_engine_start(
    guard_env, monkeypatch
):
    devices, repo, args, state = guard_env
    monkeypatch.setattr(devices.signal, "getsignal", lambda sig: signal.SIG_IGN)

    def forbidden_query(arguments):
        pytest.fail("unsafe signal handling reached hardware queries")

    monkeypatch.setattr(devices, "_query", forbidden_query)
    with pytest.raises(ValueError, match="SIGINT"):
        with devices.native_hardware_lease(repo, **args):
            pytest.fail("Engine would start without interrupt protection")
