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

import asyncio
import copy
import time
from types import SimpleNamespace

import pytest


def trace():
    rows = []
    for index in range(2):
        history = []
        for turn in range(1, 4):
            suffix = [index + 10] if turn == 1 else [20 + turn]
            full = history + suffix
            output = [sum(full) % 73 + 1] * 2
            rows.append(
                {
                    "session_index": index,
                    "turn_index": turn,
                    "suffix_ids": suffix,
                    "full_input_ids": full,
                    "output_ids": output,
                }
            )
            history = full + output
    return rows


def protocol():
    return {
        "sessions": 2,
        "turns": 3,
        "max_new_tokens": 2,
        "tool_session_index": 0,
        "tool_before_turn": 2,
        "tool_delay_seconds": 0.01,
        "tool_range_stop": 31,
        "frontier_policy": "current-frontier",
    }


class Engine:
    def __init__(self, failure):
        self.histories = {}
        self.closed = []
        self.calls = []
        self.inflight = 0
        self.failure = failure
        self.counter = 0

    def open_session(self):
        name = f"native-{self.counter}"
        self.counter += 1
        self.histories[name] = []
        return name

    def close_session(self, name):
        assert self.inflight == 0
        self.closed.append(name)
        del self.histories[name]

    def get_server_info(self):
        return {"internal_states": [{}]}

    def generate(self, *, input_ids, sampling_params, session_id, stream):
        assert stream and sampling_params["stream_interval"] == 1
        self.inflight += 1
        try:
            prior = self.histories[session_id] if session_id is not None else []
            full = prior + input_ids
            out = [sum(full) % 73 + 1] * sampling_params["max_new_tokens"]
            meta = {}
            if session_id is not None:
                reused = max(len(prior) - 1, 0)
                meta["session"] = {
                    "total_context_tokens": len(full),
                    "newly_prefilled_tokens": len(full) - reused,
                    "reused_context_tokens": reused,
                }
                self.histories[session_id] = full + out
                if self.failure == "context":
                    meta["session"]["reused_context_tokens"] += 1
            if self.failure == "output":
                out[0] += 1
            self.calls.append({"input_ids": input_ids, "session_id": session_id})
            time.sleep(0.01)
            for token in out:
                yield {"output_ids": [token], "meta_info": meta}
        finally:
            self.inflight -= 1


def test_counterbalanced_order_and_negative_index():
    from benchmarks.course_project.native_performance import arm_order

    assert [arm_order(i) for i in range(3)] == [
        ("session", "stateless"),
        ("stateless", "session"),
        ("session", "stateless"),
    ]
    with pytest.raises(ValueError):
        arm_order(-1)


@pytest.mark.parametrize(
    "mutation",
    ["prefix", "duplicate", "missing", "bool_id", "output_count", "overflow"],
)
def test_trace_rejects_invalid_histories_before_execution(mutation):
    from benchmarks.course_project.native_performance import validate_trace

    rows = copy.deepcopy(trace())
    if mutation == "prefix":
        rows[1]["full_input_ids"][0] += 1
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "bool_id":
        rows[0]["suffix_ids"][0] = True
    elif mutation == "output_count":
        rows[0]["output_ids"].pop()
    elif mutation == "overflow":
        rows[0]["full_input_ids"] = [1] * 4096
    with pytest.raises(ValueError):
        validate_trace(rows, sessions=2, turns=3, max_new_tokens=2, max_model_len=4096)


def test_trace_is_ordered_and_preserves_the_original_ids():
    from benchmarks.course_project.native_performance import validate_trace

    rows = trace()
    saved = copy.deepcopy(rows)
    grouped = validate_trace(
        list(reversed(rows)), sessions=2, turns=3, max_new_tokens=2, max_model_len=4096
    )
    assert grouped == [rows[:3], rows[3:]] and rows == saved


@pytest.mark.parametrize("mode", ["session", "stateless"])
def test_replay_uses_only_its_selected_path_and_checks_real_outputs(mode):
    from scripts.run_native_agent_performance import run_replay_arm

    engine = Engine(failure=None)
    record = {}
    asyncio.run(
        run_replay_arm(
            engine,
            trace=trace(),
            protocol=protocol(),
            mode=mode,
            max_model_len=4096,
            record=record,
        )
    )
    assert len(engine.calls) == len(record["requests"]) == 6 and all(
        r["output_ids_match"] for r in record["requests"]
    )
    assert engine.histories == {} and len(engine.closed) == (
        2 if mode == "session" else 0
    )
    expected = sorted(trace(), key=lambda r: (r["session_index"], r["turn_index"]))
    observed = sorted(
        record["requests"], key=lambda r: (r["session_index"], r["turn_index"])
    )
    assert [r["submitted_ids"] for r in observed] == [
        r["suffix_ids" if mode == "session" else "full_input_ids"] for r in expected
    ]
    assert record["tools"][0]["result"] == 465
    assert record["started_ns"] < record["finished_ns"]
    assert all(r["result"]["ended_ns"] <= record["finished_ns"] for r in observed)


@pytest.mark.parametrize("failure", ["output", "context"])
def test_failed_quality_drains_submitted_streams_and_closes_native_sessions(failure):
    from scripts.run_native_agent_performance import run_replay_arm

    engine = Engine(failure=failure)
    record = {}
    with pytest.raises(ValueError):
        asyncio.run(
            run_replay_arm(
                engine,
                trace=trace(),
                protocol=protocol(),
                mode="session",
                max_model_len=4096,
                record=record,
            )
        )
    assert engine.inflight == 0 and engine.histories == {} and len(engine.closed) == 2
    assert record["requests"]


def test_unsupported_mode_is_rejected_without_opening_sessions():
    from scripts.run_native_agent_performance import run_replay_arm

    engine = Engine(failure=None)
    with pytest.raises(ValueError, match="mode"):
        asyncio.run(
            run_replay_arm(
                engine,
                trace=trace(),
                protocol=protocol(),
                mode="unsupported",
                max_model_len=4096,
                record={},
            )
        )
    assert engine.counter == 0 and not engine.calls


def test_hbm_parser_keeps_both_target_chips_and_ignores_process_rows():
    from benchmarks.course_project.native_performance import parse_target_hbm

    text = """| 0 Ascend910 | OK | irrelevant |
| 0 0 | 0000:01:00.0 | 0 / 0 123 / 65536 |
| 1 Ascend910 | OK | irrelevant |
| 0 2 | 0000:03:00.0 | 0 / 0 1234 / 65536 |
| 1 3 | 0000:04:00.0 | 0 / 0 5678 / 65536 |
| 1 0 | 99999 | python 9999 |
| 1 1 | 88888 | python 8888 |"""
    assert parse_target_hbm(text, npu_id=1, chip_logic_ids={0: 2, 1: 3}) == {
        0: {"used_mb": 1234, "capacity_mb": 65536},
        1: {"used_mb": 5678, "capacity_mb": 65536},
    }
    with pytest.raises(ValueError):
        parse_target_hbm(text, npu_id=2, chip_logic_ids={0: 4})
    with pytest.raises(ValueError):
        parse_target_hbm(
            text + "\n| 0 2 | 0000:03:00.0 | 0 / 0 1234 / 65536 |",
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
        )


@pytest.mark.parametrize("missing_metrics", [False, True])
def test_measurement_joins_sampling_before_return_and_keeps_failures(
    monkeypatch, tmp_path, missing_metrics
):
    import scripts.run_native_agent_performance as runner

    engine = Engine(failure=None)
    queries = {"active": 0, "count": 0}

    def query(*args, **kwargs):
        queries["active"] += 1
        queries["count"] += 1
        try:
            time.sleep(0.005)
            return SimpleNamespace(
                returncode=0,
                stdout="| 1 Ascend910 | OK | irrelevant |\n| 0 2 | 0000:03:00.0 | 0 / 0 1234 / 65536 |",
                stderr="",
            )
        finally:
            queries["active"] -= 1

    def metrics(*args, **kwargs):
        if missing_metrics:
            return {}
        return {
            "requests_running": engine.inflight,
            "kv_lcm_blocks_total": 100,
            "kv_lcm_blocks_active": len(engine.histories) * 3,
            "kv_lcm_blocks_cached": 0,
            "kv_lcm_blocks_empty": 100 - len(engine.histories) * 3,
            "kv_lcm_blocks_allocated": len(engine.histories) * 3,
            "tuple_key_metric": {("not-json",): 1},
        }

    monkeypatch.setattr(runner.subprocess, "run", query)
    monkeypatch.setattr(runner, "fetch_server_metrics", metrics)
    p = {
        **protocol(),
        "repetitions": 3,
        "warmup_pairs": 1,
        "metrics_interval_seconds": 0.01,
        "hbm_interval_seconds": 0.01,
    }
    record = {}

    async def execute():
        await runner.measure_performance(
            engine,
            trace=trace(),
            protocol=p,
            max_model_len=4096,
            metrics_url="http://unused-local-test",
            output_dir=tmp_path,
            chip_logic_ids={0: 2},
            record=record,
        )

    if missing_metrics:
        with pytest.raises(ValueError, match="sampling|metrics"):
            asyncio.run(execute())
        assert record["sampler_errors"]
    else:
        asyncio.run(execute())
        formal = [a for a in record["arms"] if a["phase"] == "formal"]
        warmup = [a for a in record["arms"] if a["phase"] == "warmup"]
        assert len(formal) == 6 and len(warmup) == 2
        assert [a["mode"] for a in formal] == [
            "session",
            "stateless",
            "stateless",
            "session",
            "session",
            "stateless",
        ]
        assert all(
            a["idle_after_close"]["kv_lcm_blocks_active"] == 0 for a in record["arms"]
        )
        assert record["hbm_samples"] and record["metric_samples"]
        import json

        json.dumps(record)
    assert queries["active"] == 0 and engine.histories == {}
    count = queries["count"]
    time.sleep(0.03)
    assert queries["count"] == count


@pytest.mark.parametrize("outside", [True, False])
def test_performance_output_guards_precede_configuration_and_hardware(
    monkeypatch, tmp_path, outside
):
    import scripts.run_native_agent_performance as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    output = (
        tmp_path.parent / f"outside-perf-{tmp_path.name}"
        if outside
        else tmp_path / "existing"
    )
    if not outside:
        output.mkdir()
        (output / "marker").write_text("preserve")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_agent_performance.py",
            "--config",
            "missing",
            "--protocol",
            "missing",
            "--source-manifest",
            "missing",
            "--output-dir",
            str(output),
            "--engine-port",
            "18780",
            "--metrics-port",
            "39982",
        ],
    )
    with pytest.raises(
        ValueError if outside else FileExistsError, match="output directory"
    ):
        runner.main()
    if not outside:
        assert (output / "marker").read_text() == "preserve"


@pytest.mark.parametrize("bad_field", ["meta_info", "output_ids"])
def test_malformed_stream_is_consumed_before_native_session_close(bad_field):
    from scripts.run_native_agent_performance import run_replay_arm

    class LiveStreamEngine(Engine):
        def __init__(self):
            super().__init__(failure=None)
            self.streams = []
            self.fully_consumed = 0

        def generate(self, **kwargs):
            def chunks():
                self.inflight += 1
                try:
                    chunk = {"output_ids": [1], "meta_info": {}}
                    chunk[bad_field] = None
                    yield chunk
                    yield {"output_ids": [2], "meta_info": {}}
                    self.fully_consumed += 1
                finally:
                    self.inflight -= 1

            stream = chunks()
            self.streams.append(stream)
            return stream

    engine = LiveStreamEngine()
    record = {}
    try:
        with pytest.raises(ValueError, match=bad_field):
            asyncio.run(
                run_replay_arm(
                    engine,
                    trace=trace(),
                    protocol=protocol(),
                    mode="session",
                    max_model_len=4096,
                    record=record,
                )
            )
        assert engine.inflight == 0 and engine.fully_consumed == len(engine.streams)
        assert (
            len(engine.closed) == 2
            and engine.histories == {}
            and not record["close_errors"]
        )
    finally:
        for stream in engine.streams:
            for _ in stream:
                pass


def test_partial_session_creation_closes_the_first_session():
    from scripts.run_native_agent_performance import run_replay_arm

    class PartialOpenEngine(Engine):
        def open_session(self):
            if self.counter == 1:
                raise RuntimeError("second open failed")
            return super().open_session()

    engine = PartialOpenEngine(failure=None)
    record = {}
    with pytest.raises(RuntimeError, match="second open failed"):
        asyncio.run(
            run_replay_arm(
                engine,
                trace=trace(),
                protocol=protocol(),
                mode="session",
                max_model_len=4096,
                record=record,
            )
        )
    assert len(engine.closed) == len(record["closed_session_ids"]) == 1
    assert engine.histories == {}
