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
import json
import threading
import time

import pytest

from benchmarks.course_project.native_agent import (
    append_qwen_user_suffix,
    initial_qwen_context,
)
from benchmarks.course_project.telemetry import parse_prometheus_metrics


class Tokenizer:
    eos_token_id = 0

    def __init__(self, prefix, return_mode):
        self.prefix = prefix
        self.return_mode = return_mode

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text.replace("<|im_end|>", "\0")]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += self.prefix
        if not tokenize:
            return text
        ids = self.encode(text, add_special_tokens=False)
        if self.return_mode == "mapping":
            return {"input_ids": ids}
        if self.return_mode == "nested":
            return [ids]
        return ids


@pytest.mark.parametrize("return_mode", ["list", "mapping", "nested"])
@pytest.mark.parametrize("thinking", ["<think>\n", "<think>\n\n</think>\n\n"])
def test_initial_context_uses_exact_actual_generation_prefix(return_mode, thinking):
    prefix = "<|im_start|>assistant\n" + thinking
    tokenizer = Tokenizer(prefix, return_mode)
    ids, actual_prefix = initial_qwen_context(
        tokenizer, system_prompt="system", user_text="first"
    )
    assert actual_prefix == prefix
    assert ids == tokenizer.encode(
        "<|im_start|>system\nsystem<|im_end|>\n"
        "<|im_start|>user\nfirst<|im_end|>\n" + prefix,
        add_special_tokens=False,
    )


@pytest.mark.parametrize("eos", [False, True])
@pytest.mark.parametrize("tail", ["42", "reasoning</think>\n42", "answer   "])
def test_append_turn_retains_generated_ids_and_adds_only_a_suffix(eos, tail):
    prefix = "<|im_start|>assistant\n<think>\n"
    tokenizer = Tokenizer(prefix, "list")
    previous = tokenizer.encode(tail, add_special_tokens=False) + ([0] if eos else [])
    saved = previous.copy()
    suffix = append_qwen_user_suffix(
        tokenizer,
        generation_prefix=prefix,
        previous_output_ids=previous,
        user_text="next",
    )
    boundary = "\n" if eos else "<|im_end|>\n"
    assert suffix == tokenizer.encode(
        boundary + "<|im_start|>user\nnext<|im_end|>\n" + prefix,
        add_special_tokens=False,
    )
    assert previous == saved


def test_sixteen_turn_history_is_append_only_with_trailing_space_and_closed_thinking():
    tokenizer = Tokenizer("<|im_start|>assistant\n<think>\n", "list")
    history, prefix = initial_qwen_context(
        tokenizer, system_prompt="system", user_text="first"
    )
    for turn in range(1, 16):
        output = tokenizer.encode(f"</think>\n{turn}   ", add_special_tokens=False)
        output += [0] if turn % 2 else []
        before = history + output
        suffix = append_qwen_user_suffix(
            tokenizer,
            generation_prefix=prefix,
            previous_output_ids=output,
            user_text=f"turn {turn + 1}",
        )
        history = before + suffix
        assert history[: len(before)] == before
        assert history[-len(suffix) :] == suffix


def test_unsupported_template_is_rejected():
    tokenizer = Tokenizer("unsupported assistant prefix", "list")
    with pytest.raises(ValueError, match="Qwen"):
        initial_qwen_context(tokenizer, system_prompt="system", user_text="first")


def test_suffix_requires_previous_output():
    tokenizer = Tokenizer("<|im_start|>assistant\n<think>\n", "list")
    with pytest.raises(ValueError, match="output"):
        append_qwen_user_suffix(
            tokenizer,
            generation_prefix=tokenizer.prefix,
            previous_output_ids=[],
            user_text="next",
        )


class Engine:
    def __init__(self, failure):
        self.histories = {}
        self.next_session = 0
        self.closed = []
        self.failure = failure

    def open_session(self):
        session_id = f"native-{self.next_session}"
        self.next_session += 1
        self.histories[session_id] = []
        return session_id

    def close_session(self, session_id):
        del self.histories[session_id]
        self.closed.append(session_id)

    def get_server_info(self):
        return {"internal_states": [{}]}

    def generate(self, *, input_ids, sampling_params, session_id, stream):
        assert stream is True
        prior = self.histories[session_id] if session_id is not None else []
        context = prior + input_ids
        outputs = [sum(context) % 73 + 1] * sampling_params["max_new_tokens"]
        meta = {}
        if session_id is not None:
            self.histories[session_id] = context + outputs
            reused = max(len(prior) - 1, 0)
            meta["session"] = {
                "total_context_tokens": len(context),
                "newly_prefilled_tokens": len(context) - reused,
                "reused_context_tokens": reused,
            }
            if self.failure == "context":
                meta["session"]["newly_prefilled_tokens"] += 1
        elif self.failure == "output":
            outputs[0] += 1
        time.sleep(0.01)
        for token in outputs:
            yield {"output_ids": [token], "meta_info": meta}


def workload():
    return {
        "system_prompt": "system",
        "initial_user_prefix": "context ",
        "initial_filler": "x ",
        "initial_filler_repetitions": 1,
        "sessions": 2,
        "turns": 3,
        "max_new_tokens": 2,
        "next_user_text": "continue",
        "tool_session_index": 0,
        "tool_before_turn": 2,
        "tool_delay_seconds": 0.1,
        "tool_range_stop": 31,
        "tool_result_prefix": "Scripted result: ",
        "frontier_policy": "current-frontier",
    }


def metric_reader(engine):
    def read(url, *, timeout_seconds):
        active = sum(bool(history) for history in engine.histories.values())
        return {
            "requests_running": 0,
            "kv_lcm_blocks_total": 10,
            "kv_lcm_blocks_active": active,
            "kv_lcm_blocks_cached": 0,
            "kv_lcm_blocks_allocated": active,
            "kv_lcm_blocks_empty": 10 - active,
            **parse_prometheus_metrics(
                'tokenspeed:inter_token_latency_seconds_bucket{model="fixture",le="+Inf"} 1\n'
                'tokenspeed:inter_token_latency_seconds_count{model="fixture"} 1\n'
                'tokenspeed:inter_token_latency_seconds_sum{model="fixture"} 0.01\n'
            ),
        }

    return read


def test_native_runner_pairs_outputs_reuse_and_tool_progress_then_releases(monkeypatch):
    import scripts.run_native_agent_validation as runner

    record = {"turns": [], "tools": [], "metric_snapshots": []}
    tool_started = threading.Event()

    class CoordinatedEngine(Engine):
        def __init__(self, started):
            super().__init__(failure=None)
            self.tool_started = started

        def generate(self, **kwargs):
            if kwargs["session_id"] == "native-1" and self.histories["native-1"]:
                assert self.tool_started.wait(timeout=5)
            yield from super().generate(**kwargs)

    def coordinated_tool(*, delay_seconds, range_stop, event):
        assert delay_seconds == workload()["tool_delay_seconds"]
        event["started_ns"] = time.monotonic_ns()
        tool_started.set()
        deadline = time.monotonic() + 5
        while True:
            assert time.monotonic() < deadline, "other Session must publish progress"
            metrics = [
                s["monotonic_ns"]
                for s in record["metric_snapshots"]
                if s["label"] == "during_tool"
                and s["monotonic_ns"] > event["started_ns"]
            ]
            completions = [
                t["session_result"]["ended_ns"]
                for t in record["turns"]
                if t["session_index"] == 1
                and t["session_result"]["ended_ns"] > event["started_ns"]
            ]
            if metrics and completions:
                last_observation = max(metrics + completions)
                break
            time.sleep(0.001)
        # Windows monotonic can have 15.6ms ticks. Complete on a later tick
        # instead of making strict inside-window assertions on equal timestamps.
        while time.monotonic_ns() <= last_observation:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        event["result"] = sum(range(range_stop))
        event["ended_ns"] = time.monotonic_ns()

    engine = CoordinatedEngine(tool_started)
    monkeypatch.setattr(runner, "fetch_server_metrics", metric_reader(engine))
    monkeypatch.setattr(runner, "_scripted_tool", coordinated_tool)
    asyncio.run(
        runner.run_native_validation(
            engine,
            Tokenizer("<|im_start|>assistant\n<think>\n", "list"),
            workload=workload(),
            max_model_len=4096,
            metrics_url="http://unused-local-test",
            record=record,
        )
    )
    assert len(record["turns"]) == 6
    assert all(
        t["output_ids_match"] and t["context_validated"] for t in record["turns"]
    )
    assert record["tools"][0]["result"] == 465
    assert record["other_session_progress_during_tool"]
    assert record["metric_snapshots"][-1]["metrics"]["kv_lcm_blocks_active"] == 0
    assert len(engine.closed) == 2 and engine.histories == {}
    json.dumps(record)


@pytest.mark.parametrize("failure", ["output", "context"])
def test_native_runner_retains_failed_pair_and_closes_sessions(monkeypatch, failure):
    import scripts.run_native_agent_validation as runner

    engine = Engine(failure=failure)
    monkeypatch.setattr(runner, "fetch_server_metrics", metric_reader(engine))
    record = {"turns": [], "tools": [], "metric_snapshots": []}
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(
            runner.run_native_validation(
                engine,
                Tokenizer("<|im_start|>assistant\n<think>\n", "list"),
                workload=workload(),
                max_model_len=4096,
                metrics_url="http://unused-local-test",
                record=record,
            )
        )
    assert record["turns"]
    assert len(engine.closed) == 2 and engine.histories == {}


def test_missing_device_metrics_fail_instead_of_becoming_zero(monkeypatch):
    import scripts.run_native_agent_validation as runner

    engine = Engine(failure=None)
    monkeypatch.setattr(
        runner, "fetch_server_metrics", lambda url, *, timeout_seconds: {}
    )
    record = {"turns": [], "tools": [], "metric_snapshots": []}
    with pytest.raises(ValueError, match="metrics"):
        asyncio.run(
            runner.run_native_validation(
                engine,
                Tokenizer("<|im_start|>assistant\n<think>\n", "list"),
                workload=workload(),
                max_model_len=4096,
                metrics_url="http://unused-local-test",
                record=record,
            )
        )
    assert len(engine.closed) == 2 and engine.histories == {}


@pytest.mark.parametrize("invalid_output", ["outside", "existing"])
def test_output_guards_run_before_config_dependency_or_hardware_access(
    monkeypatch, tmp_path, invalid_output
):
    import scripts.run_native_agent_validation as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    if invalid_output == "outside":
        output = runner.REPO_ROOT.parent / f"outside-native-agent-{tmp_path.name}"
        error = ValueError
    else:
        output = tmp_path / "existing"
        output.mkdir()
        (output / "marker").write_text("preserve")
        error = FileExistsError
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_agent_validation.py",
            "--config",
            "missing-config",
            "--workload",
            "missing-workload",
            "--source-manifest",
            "missing-manifest",
            "--output-dir",
            str(output),
            "--engine-port",
            "18780",
            "--metrics-port",
            "39982",
        ],
    )
    with pytest.raises(error, match="output directory"):
        runner.main()
    if invalid_output == "existing":
        assert (output / "marker").read_text() == "preserve"
