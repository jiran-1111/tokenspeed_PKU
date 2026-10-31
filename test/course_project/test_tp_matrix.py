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

import copy
import hashlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def protocol():
    return {
        "input_lengths": [4, 8],
        "concurrency": [1, 2],
        "max_new_tokens": 3,
        "warmup_requests": 2,
        "requests_per_cell": 3,
        "repetitions": 2,
        "seed_text": "matrix seed",
        "hbm_interval_seconds": 0.001,
        "engine_port": 18880,
    }


class FakeEngine:
    def __init__(self, failure):
        self.failure = failure
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.calls = []

    def generate(self, *, input_ids, sampling_params, session_id, stream):
        assert stream and session_id is None
        assert sampling_params == {
            "temperature": 0,
            "ignore_eos": True,
            "stream_interval": 1,
            "max_new_tokens": 3,
        }
        with self.lock:
            self.calls.append(list(input_ids))
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(0.003)
            yield {"output_ids": [10], "meta_info": {}}
            if self.failure == "partial":
                raise RuntimeError("fake stream failed")
            if self.failure == "malformed":
                yield {"output_ids": [True], "meta_info": {}}
            yield {"output_ids": [11, 12], "meta_info": {}}
        finally:
            with self.lock:
                self.inflight -= 1


def hbm_table():
    return "\n".join(
        [
            "| 0 Ascend910 | OK | ignored |",
            "| 0 0 | 0000:01:00.0 | 999 / 65536 |",
            "| 1 Ascend910 | OK | ignored |",
            "| 0 2 | 0000:02:00.0 | 123 / 65536 |",
            "| 1 3 | 0000:03:00.0 | 456 / 65536 |",
        ]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_lengths", [4, 4]),
        ("input_lengths", []),
        ("concurrency", []),
        ("concurrency", [1, True]),
        ("requests_per_cell", 1),
        ("hbm_interval_seconds", float("nan")),
        ("warmup_requests", 0),
        ("max_new_tokens", 100),
    ],
)
def test_protocol_rejects_invalid_conditions_before_execution(field, value):
    from benchmarks.course_project.tp_matrix import validate_protocol

    candidate = protocol()
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_protocol(candidate, max_model_len=32)


def test_exact_inputs_are_shared_across_cells_and_do_not_mutate_seed():
    from benchmarks.course_project.tp_matrix import build_inputs

    seed = [3, 7, 5]
    assert build_inputs(seed, lengths=[4, 8]) == {
        "4": [3, 7, 5, 3],
        "8": [3, 7, 5, 3, 7, 5, 3, 7],
    }
    assert seed == [3, 7, 5]
    with pytest.raises(ValueError):
        build_inputs([], lengths=[4, 8])


def test_strategy_protocol_accepts_single_explicit_length_and_concurrency():
    from benchmarks.course_project.tp_matrix import validate_protocol

    candidate = {**protocol(), "input_lengths": [8], "concurrency": [2]}
    validate_protocol(candidate, max_model_len=32)


@pytest.mark.parametrize("failure", ["partial", "malformed"])
def test_failed_stream_preserves_partial_tokens_and_drains_submitted_request(failure):
    from benchmarks.course_project.tp_matrix import measure_request

    engine = FakeEngine(failure)
    result = measure_request(engine, input_ids=[1] * 4, max_new_tokens=3)
    assert result["status"] == "failed"
    assert result["output_ids"][0] == 10
    assert result["error"]
    assert engine.inflight == 0
    assert result["chunks"][0]["output_tokens"] == 1
    if failure == "malformed":
        assert result["output_ids"] == [10, 11, 12]


def test_matrix_separates_warmup_uses_batch_wall_and_samples_target_chips(tmp_path):
    from benchmarks.course_project.tp_matrix import build_inputs, run_matrix

    engine = FakeEngine(None)
    inputs = build_inputs([1, 2, 3], lengths=[4, 8])
    saved = copy.deepcopy(inputs)
    record = {}
    run_matrix(
        engine,
        protocol=protocol(),
        inputs=inputs,
        output_dir=tmp_path,
        npu_id=1,
        chip_logic_ids={0: 2, 1: 3},
        query_hbm=hbm_table,
        record=record,
    )
    assert inputs == saved and engine.inflight == 0 and engine.peak == 2
    assert len(record["batches"]) == 12
    formal = [row for row in record["batches"] if row["phase"] == "formal"]
    assert len(formal) == 8
    for row in formal:
        assert row["status"] == "passed" and row["requests"] == 3
        assert row["output_tokens"] == 9
        assert row["wall_ns"] == row["finished_ns"] - row["started_ns"]
        assert row["output_tokens_per_second"] == pytest.approx(9e9 / row["wall_ns"])
    requests = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 32
    assert sum(row["phase"] == "formal" for row in requests) == 24
    assert all(row["output_ids"] == [10, 11, 12] for row in requests)
    assert all(row["input_ids"] == inputs[str(row["input_length"])] for row in requests)
    assert all(
        row["ttft_ms"] > 0 and row["e2e_ms"] >= row["ttft_ms"] for row in requests
    )
    assert record["hbm_samples"][0]["chips"][0]["used_mb"] == 123
    assert record["hbm_samples"][0]["chips"][1]["used_mb"] == 456
    assert record["sampler_started_ns"] <= record["sampler_finished_ns"]
    assert not record["sampler_errors"]
    assert (tmp_path / "npu_smi_00000.txt").read_text() == hbm_table()


def test_sampler_failure_rejects_measurement_and_preserves_error(tmp_path):
    from benchmarks.course_project.tp_matrix import build_inputs, run_matrix

    record = {}
    with pytest.raises(ValueError, match="HBM"):
        run_matrix(
            FakeEngine(None),
            protocol=protocol(),
            inputs=build_inputs([1], lengths=[4, 8]),
            output_dir=tmp_path,
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
            query_hbm=lambda: "invalid table",
            record=record,
        )
    assert record["sampler_errors"]
    assert (tmp_path / "npu_smi_00000.txt").read_text() == "invalid table"


def test_failed_batch_retains_every_submitted_partial_request(tmp_path):
    from benchmarks.course_project.tp_matrix import build_inputs, run_matrix

    engine = FakeEngine("partial")
    record = {}
    with pytest.raises(ValueError, match="request"):
        run_matrix(
            engine,
            protocol=protocol(),
            inputs=build_inputs([1], lengths=[4, 8]),
            output_dir=tmp_path,
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
            query_hbm=hbm_table,
            record=record,
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and all(row["output_ids"] == [10] for row in rows)
    assert record["batches"][0]["status"] == "failed" and engine.inflight == 0


@pytest.mark.parametrize("value", ["0:2,0:3", "0:2,1:2", "-1:2", "0", "0:2:3", ""])
def test_cli_rejects_ambiguous_device_mapping(value):
    import argparse

    from scripts.run_tp_matrix import parse_chip_logic_ids

    with pytest.raises(argparse.ArgumentTypeError):
        parse_chip_logic_ids(value)


@pytest.mark.parametrize(
    "stat,parent_pid",
    [
        ("2468 (worker name ) odd) S 1234 1 1", 1234),
        ("2468 (worker) S 0 1 1", 0),
        ("2468 worker S 1234", None),
        ("2468 (worker)", None),
        ("2468 (worker) S invalid", None),
    ],
)
def test_linux_parent_pid_parses_process_names_and_rejects_unknown_stat(
    monkeypatch, stat, parent_pid
):
    from scripts import run_tp_matrix

    def read_text(path, **kwargs):
        assert path.name == "stat" and path.parent.name == "2468"
        return stat

    monkeypatch.setattr(Path, "read_text", read_text)
    if parent_pid is None:
        with pytest.raises(ValueError):
            run_tp_matrix._read_parent_pid(2468)
    else:
        assert run_tp_matrix._read_parent_pid(2468) == parent_pid


@pytest.mark.parametrize(
    "profile_mode,failure",
    [
        ("off", None),
        ("all-batches", None),
        ("all-batches", "body"),
        ("all-batches", "missing_rank"),
        ("off", "old_import"),
        ("off", "missing_dependencies"),
        ("off", "foreign_pid"),
        ("off", "unknown_pid"),
        ("off", "dependencies_changed"),
        ("off", "module_changed"),
        ("off", "extension_changed"),
        ("off", "old_import_inside"),
    ],
)
def test_cli_holds_guard_through_engine_teardown_and_preserves_source_identity(
    tmp_path, monkeypatch, profile_mode, failure
):
    from scripts import run_tp_matrix

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/final/qwen35_4b.yaml").read_text())
    if profile_mode == "all-batches":
        config["tp_size"] = 2
    chip_logic_ids = {0: 2} if config["tp_size"] == 1 else {0: 2, 1: 3}
    config_path, protocol_path = tmp_path / "config.json", tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config))
    protocol_path.write_text(json.dumps(protocol()))
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"config.json": hashlib.sha256(config_path.read_bytes()).hexdigest()}
        )
    )
    dependencies_path = (
        tmp_path / "docs/evidence/homework2_20261023/runtime_dependencies.json"
    )
    dependencies_path.parent.mkdir(parents=True)
    dependencies_path.write_text(json.dumps({"external_extensions": {}}))
    dependencies_sha = hashlib.sha256(dependencies_path.read_bytes()).hexdigest()
    if failure == "missing_dependencies":
        dependencies_path.unlink()
    events = []

    @contextmanager
    def lease(repo_root, **kwargs):
        events.append("guard-enter")
        assert repo_root == tmp_path and kwargs["npu_id"] == 1
        assert kwargs["chip_logic_ids"] == chip_logic_ids
        state = {}
        try:
            yield state
        finally:
            events.append("guard-exit")
            state.update({"after_target_idle": True, "sources_unchanged": True})

    class Engine(FakeEngine):
        def __init__(self, *, server_args):
            assert events == ["guard-enter"]
            assert "--chunked-prefill-size" in server_args
            super().__init__("partial" if failure == "body" else None)
            self.profiling = False
            self.profile_dir = None
            self.tokenizer_manager = SimpleNamespace(
                start_profile=self.start_profile, stop_profile=self.stop_profile
            )
            self.llm = SimpleNamespace(run=lambda value: value)

        def __enter__(self):
            events.append("engine-enter")
            return self

        def __exit__(self, *_args):
            assert not self.profiling
            events.append("engine-exit")
            if failure == "dependencies_changed":
                dependencies_path.write_text('{"external_extensions": {}}\n')
            if failure == "module_changed":
                Path(runtime_modules["tokenspeed_kernel"].__file__).write_text(
                    "changed module"
                )
            if failure == "extension_changed":
                Path(
                    runtime_modules[
                        "tokenspeed_scheduler.tokenspeed_scheduler_ext"
                    ].__file__
                ).write_bytes(b"changed binary")

        def start_profile(self, **kwargs):
            events.append("profile-start")
            self.profile_dir = Path(kwargs["output_dir"])
            self.profiling = True
            return SimpleNamespace(success=True, message="started")

        def stop_profile(self):
            events.append("profile-stop")
            self.profiling = False
            self.profile_dir.mkdir()
            ranks = (0,) if failure == "missing_rank" else (0, 1)
            for rank in ranks:
                (self.profile_dir / f"hw2-TP{rank}.trace.json").write_text(
                    '{"traceEvents": []}'
                )
            return SimpleNamespace(success=True, message="stopped")

        def generate(self, **kwargs):
            assert self.profiling is (profile_mode == "all-batches")
            yield from super().generate(**kwargs)

    monkeypatch.setattr(run_tp_matrix, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_tp_matrix, "_local_model_path", lambda _config: tmp_path)
    process_table = "\n".join(
        [
            "+---+---+---+---+",
            "| NPU Chip | Process id | Process name | Process memory(MB) |",
            "+===+===+===+===+",
            "| 1 0 | 2468 | python | 112 |",
            "+===+===+===+===+",
        ]
    )
    monkeypatch.setattr(
        run_tp_matrix.subprocess,
        "check_output",
        lambda *_args, **_kwargs: hbm_table() + "\n" + process_table,
    )
    from benchmarks.course_project import tp_devices

    monkeypatch.setattr(tp_devices, "hardware_lease", lease)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                    encode=lambda *_args, **_kwargs: [1, 2]
                )
            )
        ),
    )
    runtime_modules = {}
    for name in (
        "tokenspeed.runtime.entrypoints.engine",
        "tokenspeed_kernel",
        "tokenspeed_kernel_npu",
        "tokenspeed_scheduler",
        "tokenspeed_scheduler.tokenspeed_scheduler_ext",
    ):
        module_file = tmp_path / "loaded_modules" / f"{name}.py"
        if failure == "old_import" and name == "tokenspeed_kernel":
            module_file = (
                tmp_path.parent / f"{tmp_path.name}-old-editable" / "kernel.py"
            )
        if failure == "old_import_inside" and name == "tokenspeed_kernel":
            module_file = tmp_path / ".tmp/old-editable/kernel.py"
        if (
            failure == "extension_changed"
            and name == "tokenspeed_scheduler.tokenspeed_scheduler_ext"
        ):
            module_file = (
                tmp_path.parent
                / f"{tmp_path.name}-pinned-extension"
                / "tokenspeed_scheduler_ext.fake.so"
            )
        module_file.parent.mkdir(parents=True, exist_ok=True)
        module_file.write_text("fake runtime module")
        runtime_modules[name] = (
            SimpleNamespace(__file__=str(module_file), Engine=Engine)
            if name == "tokenspeed.runtime.entrypoints.engine"
            else SimpleNamespace(__file__=str(module_file))
        )
        monkeypatch.setitem(sys.modules, name, runtime_modules[name])
        if (
            failure == "extension_changed"
            and name == "tokenspeed_scheduler.tokenspeed_scheduler_ext"
        ):
            dependencies_path.write_text(
                json.dumps(
                    {
                        "external_extensions": {
                            module_file.name: hashlib.sha256(
                                module_file.read_bytes()
                            ).hexdigest()
                        },
                    }
                )
            )
            dependencies_sha = hashlib.sha256(
                dependencies_path.read_bytes()
            ).hexdigest()
    sources = json.loads(manifest_path.read_text())
    for name, module in runtime_modules.items():
        path = Path(module.__file__)
        if path.is_relative_to(tmp_path) and not (
            failure == "old_import_inside" and name == "tokenspeed_kernel"
        ):
            sources[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest_path.write_text(json.dumps(sources))
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.utils.server_args",
        SimpleNamespace(prepare_server_args=lambda args: args),
    )

    def read_parent(pid):
        if pid == 1:
            return 0
        assert pid == 2468
        if failure == "unknown_pid":
            raise FileNotFoundError("fake process disappeared")
        return 1 if failure == "foreign_pid" else run_tp_matrix.os.getpid()

    monkeypatch.setattr(run_tp_matrix, "_read_parent_pid", read_parent, raising=False)
    monkeypatch.setenv("TOKENSPEED_NPU_PROFILER_LEVEL", "Level1")
    output = tmp_path / "measurement"
    argv = [
        "run_tp_matrix.py",
        "--config",
        str(config_path),
        "--protocol",
        str(protocol_path),
        "--source-manifest",
        str(manifest_path),
        "--npu-id",
        "1",
        "--chip-logic-ids",
        ",".join(f"{chip}:{logic}" for chip, logic in chip_logic_ids.items()),
        "--output-dir",
        str(output),
        "--profile-mode",
        profile_mode,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    if failure == "missing_dependencies":
        with pytest.raises(FileNotFoundError):
            run_tp_matrix.main()
    elif failure:
        messages = {
            "body": "request",
            "missing_rank": "profile traces",
            "old_import": "outside",
            "foreign_pid": "HBM",
            "unknown_pid": "HBM",
            "dependencies_changed": "runtime",
            "module_changed": "runtime",
            "extension_changed": "origin",
            "old_import_inside": "source manifest",
        }
        with pytest.raises(ValueError, match=messages[failure]):
            run_tp_matrix.main()
    else:
        run_tp_matrix.main()
    result = json.loads((output / "run.json").read_text())
    assert result["status"] == ("failed" if failure else "passed")
    assert result["source_after_mismatches"] == (
        ["loaded_modules/tokenspeed_kernel.py"] if failure == "module_changed" else []
    )
    assert result["guard"]["after_target_idle"] is True
    assert result["profile_mode"] == profile_mode
    assert result["normal_performance_eligible"] is (
        profile_mode == "off" and failure is None
    )
    if failure != "missing_dependencies":
        assert result["runtime_dependencies_sha256"] == dependencies_sha
        assert (
            hashlib.sha256(
                (output / "runtime_dependencies.json").read_bytes()
            ).hexdigest()
            == dependencies_sha
        )
    rejected_before_engine = failure in (
        "old_import",
        "old_import_inside",
        "missing_dependencies",
    )
    if not rejected_before_engine:
        assert set(result["runtime_origins"]) == set(runtime_modules)
        assert result["runtime_origins_unchanged"] is (
            failure
            not in ("module_changed", "dependencies_changed", "extension_changed")
        )
        sample = result["process_samples"][0]
        assert sample["root_pid"] == run_tp_matrix.os.getpid()
        assert sample["chips"]["0"]["pids"] == [2468]
        assert (
            output / sample["raw_table"]
        ).read_text() == hbm_table() + "\n" + process_table
        if failure in ("foreign_pid", "unknown_pid"):
            status = "foreign" if failure == "foreign_pid" else "unknown"
            assert result["foreign_processes"][0]["status"] == status
            assert result["foreign_processes"][0]["pid"] == 2468
            assert sample["chips"]["0"][f"{status}_pids"] == [2468]
        else:
            assert result["foreign_processes"] == []
            assert sample["chips"]["0"]["parents"] == {
                "2468": run_tp_matrix.os.getpid()
            }
    if profile_mode == "all-batches":
        assert [event["action"] for event in result["profile"]["controls"]] == [
            "start",
            "stop",
        ]
        assert all(event["acknowledged"] for event in result["profile"]["controls"])
        assert len(result["profile"]["traces"]) == (
            1 if failure == "missing_rank" else 2
        )
        for trace in result["profile"]["traces"]:
            assert (
                trace["sha256"]
                == hashlib.sha256((output / trace["path"]).read_bytes()).hexdigest()
            )
    for name in ("config", "protocol", "inputs"):
        saved = (output / f"{name}.json").read_bytes()
        assert result[f"{name}_sha256"] == hashlib.sha256(saved).hexdigest()
        assert result[name] == json.loads(saved)
    expected_events = ["guard-enter"]
    if not rejected_before_engine:
        expected_events += ["engine-enter"]
    if profile_mode == "all-batches":
        expected_events += ["profile-start", "profile-stop"]
    if not rejected_before_engine:
        expected_events += ["engine-exit"]
    expected_events += ["guard-exit"]
    assert events == expected_events
    with pytest.raises(FileExistsError):
        run_tp_matrix.main()
    assert events == expected_events


@pytest.mark.parametrize("case", ["missing_mode", "missing_level", "concurrency"])
def test_cli_rejects_missing_profile_choices_or_unsupported_concurrency_before_hardware(
    tmp_path, monkeypatch, case
):
    from scripts import run_tp_matrix

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/final/qwen35_4b.yaml").read_text())
    if case == "concurrency":
        config["max_num_seqs"] = 1
    config_path, protocol_path = tmp_path / "config.json", tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config))
    protocol_path.write_text(json.dumps(protocol()))
    argv = [
        "run_tp_matrix.py",
        "--config",
        str(config_path),
        "--protocol",
        str(protocol_path),
        "--source-manifest",
        str(tmp_path / "unused_manifest"),
        "--npu-id",
        "1",
        "--chip-logic-ids",
        "0:2",
        "--output-dir",
        str(tmp_path / "measurement"),
    ]
    if case != "missing_mode":
        argv.extend(
            ["--profile-mode", "all-batches" if case == "missing_level" else "off"]
        )
    monkeypatch.delenv("TOKENSPEED_NPU_PROFILER_LEVEL", raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit if case == "missing_mode" else ValueError):
        run_tp_matrix.main()
    assert not (tmp_path / "measurement").exists()


@pytest.mark.parametrize(
    "failure", [None, "body", "start_none", "stop_none", "start_raise", "stop_raise"]
)
def test_profile_capture_checks_acknowledgements_stops_on_failure_and_hashes_rank_traces(
    tmp_path, failure
):
    from benchmarks.course_project.tp_matrix import capture_profile

    calls, record = [], {}

    def start_profile(**kwargs):
        calls.append("start")
        assert kwargs == {
            "output_dir": str(tmp_path / "profile"),
            "activities": ["CPU", "NPU", "MEM"],
            "with_stack": False,
            "record_shapes": True,
            "profile_id": "hw2",
        }
        if failure == "start_raise":
            raise RuntimeError("fake profile start failure")
        return (
            None
            if failure == "start_none"
            else SimpleNamespace(success=True, message="ok")
        )

    def stop_profile():
        calls.append("stop")
        target = tmp_path / "profile"
        target.mkdir(exist_ok=True)
        for rank in (0, 1):
            (target / f"hw2-TP{rank}.trace.json").write_text('{"traceEvents": []}')
        if failure == "stop_raise":
            raise RuntimeError("fake profile stop failure")
        return (
            None
            if failure == "stop_none"
            else SimpleNamespace(success=True, message="ok")
        )

    engine = SimpleNamespace(
        llm=SimpleNamespace(run=lambda value: value),
        tokenizer_manager=SimpleNamespace(
            start_profile=start_profile, stop_profile=stop_profile
        ),
    )

    def execute():
        with capture_profile(engine, output_dir=tmp_path, tp_size=2, record=record):
            calls.append("body")
            if failure == "body":
                raise ValueError("fake workload failure")

    if failure == "body":
        with pytest.raises(ValueError, match="workload"):
            execute()
    elif failure in ("start_raise", "stop_raise"):
        with pytest.raises(RuntimeError, match="fake profile"):
            execute()
    elif failure:
        with pytest.raises(RuntimeError, match="acknowledg"):
            execute()
    else:
        execute()
    assert calls[-1] == "stop" and calls.count("stop") == 1
    assert ("body" in calls) is (failure not in ("start_none", "start_raise"))
    capture = record["profile"]
    assert capture["clock_alignment"] == "unverified"
    assert len(capture["controls"]) == 2
    for control in capture["controls"]:
        assert control["before_perf_counter_ns"] <= control["after_perf_counter_ns"]
        assert control["before_epoch_ns"] <= control["after_epoch_ns"]
    assert len(capture["traces"]) == 2
    for trace in capture["traces"]:
        assert (
            trace["sha256"]
            == hashlib.sha256((tmp_path / trace["path"]).read_bytes()).hexdigest()
        )
