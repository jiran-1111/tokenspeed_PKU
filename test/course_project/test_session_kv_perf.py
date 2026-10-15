"""Local contracts for the paired Session KV performance sweep."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "run_session_kv_perf.py"


@pytest.fixture(scope="module")
def perf_runner():
    if not RUNNER_PATH.is_file():
        pytest.skip("Session KV performance runner is not implemented yet")
    spec = importlib.util.spec_from_file_location("run_session_kv_perf", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_kv_performance_runner_exists():
    assert RUNNER_PATH.is_file(), "Session KV performance runner is missing"


class _QwenTemplateTokenizer:
    eos_token_id = 2
    special_tokens = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3}

    def _encode(self, text: str) -> list[int]:
        token_ids = []
        while text:
            special = next(
                (token for token in self.special_tokens if text.startswith(token)),
                None,
            )
            if special is None:
                token_ids.append(ord(text[0]) + 1000)
                text = text[1:]
            else:
                token_ids.append(self.special_tokens[special])
                text = text[len(special) :]
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        reverse_tokens = {value: key for key, value in self.special_tokens.items()}
        return "".join(
            (
                reverse_tokens[token_id]
                if token_id in reverse_tokens
                else chr(token_id - 1000)
            )
            for token_id in token_ids
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        rendered = ""
        if messages and messages[0]["role"] == "system":
            rendered += (
                "<|im_start|>system\n" + messages[0]["content"].strip() + "<|im_end|>\n"
            )
        for message in messages:
            if message["role"] == "system":
                continue
            rendered += (
                f"<|im_start|>{message['role']}\n"
                f"{message['content'].strip()}<|im_end|>\n"
            )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>\n"
        return self._encode(rendered)


class _SerialFakeEngine:
    def __init__(self, runner):
        self.runner = runner
        self.history = {}
        self.calls = []
        self.opened = []
        self.closed = []
        self.active = 0
        self.max_active = 0
        self.mismatch_control = False
        self.bad_session_metadata = False

    def open_session(self):
        session_id = f"session-{len(self.opened)}"
        self.opened.append(session_id)
        self.history[session_id] = []
        return session_id

    def close_session(self, session_id):
        self.closed.append(session_id)

    def generate(self, *, input_ids, sampling_params, session_id):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        assert self.active == 1
        path = "session" if session_id is not None else "stateless"
        self.calls.append((session_id, path, list(input_ids)))
        try:
            output_ids = [1065]
            if self.mismatch_control and session_id is None:
                output_ids = [1066]
            meta_info = {"prompt_tokens": len(input_ids)}
            if session_id is not None:
                prior_history_tokens = len(self.history[session_id])
                meta_info["session"] = self.runner.expected_session_context_counts(
                    policy="current-frontier",
                    prior_history_tokens=prior_history_tokens,
                    suffix_tokens=len(input_ids),
                )
                if self.bad_session_metadata:
                    meta_info["session"]["reused_context_tokens"] += 1
                self.history[session_id].extend(input_ids + output_ids)
            return {"output_ids": output_ids, "meta_info": meta_info}
        finally:
            self.active -= 1


def test_context_targets_produce_chat_template_inputs_at_or_above_target(perf_runner):
    tokenizer = _QwenTemplateTokenizer()

    first_user_text, input_ids = perf_runner.build_context_prompt(
        tokenizer, context_length=80
    )

    assert first_user_text
    assert len(input_ids) >= 80
    assert input_ids == perf_runner._token_ids(
        tokenizer,
        [
            {"role": "system", "content": perf_runner.SYSTEM_PROMPT},
            {"role": "user", "content": first_user_text},
        ],
        add_generation_prompt=True,
    )


def test_turn_suffix_is_exact_chat_template_delta_and_frontier_counts(perf_runner):
    tokenizer = _QwenTemplateTokenizer()
    prior_user_text = "Use the provided context."
    prior_input_ids = perf_runner._token_ids(
        tokenizer,
        [
            {"role": "system", "content": perf_runner.SYSTEM_PROMPT},
            {"role": "user", "content": prior_user_text},
        ],
        add_generation_prompt=True,
    )
    prior_output_ids = [1065]
    history_ids = prior_input_ids + prior_output_ids

    suffix_ids, full_input_ids = perf_runner.build_next_turn_inputs(
        tokenizer,
        history_ids,
        prior_user_text=prior_user_text,
        previous_output_ids=prior_output_ids,
        next_user_text=perf_runner.NEXT_USER_TEXT,
    )
    counts = perf_runner.expected_session_context_counts(
        policy="current-frontier",
        prior_history_tokens=len(history_ids),
        suffix_tokens=len(suffix_ids),
    )

    assert full_input_ids == history_ids + suffix_ids
    assert counts["reused_context_tokens"] == len(history_ids) - 1
    assert counts["newly_prefilled_tokens"] == len(suffix_ids) + 1


@pytest.mark.parametrize(
    ("repetition_index", "expected"),
    ((0, ("session", "stateless")), (1, ("stateless", "session"))),
)
def test_pair_order_alternates_by_repetition(perf_runner, repetition_index, expected):
    assert perf_runner.execution_order(repetition_index) == expected


def test_sweep_rebuilds_fresh_pairs_serially_and_records_raw_measurements(
    perf_runner,
):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    run_record = {"measurements": []}

    asyncio.run(
        perf_runner._run_sweep(
            engine,
            tokenizer,
            context_lengths=[80],
            repetitions=2,
            frontier_policy="current-frontier",
            max_new_tokens=4,
            max_model_len=4096,
            run_record=run_record,
        )
    )

    assert engine.opened == ["session-0", "session-1"]
    assert engine.closed == engine.opened
    assert engine.max_active == 1
    assert [tuple(item["execution_order"]) for item in run_record["measurements"]] == [
        ("session", "stateless"),
        ("stateless", "session"),
    ]
    for item in run_record["measurements"]:
        assert item["context_length_target"] == 80
        assert item["initial_context_tokens"] >= 80
        assert item["prime"]["output_ids_match"] is True
        assert item["measurement"]["output_ids_match"] is True
        assert item["measurement"]["session"]["input_token_count"] == len(
            item["measurement"]["session"]["input_ids"]
        )
        assert item["measurement"]["stateless"]["input_token_count"] == len(
            item["measurement"]["stateless"]["input_ids"]
        )
        assert item["measurement"]["session"]["output_token_count"] == len(
            item["measurement"]["session"]["output_ids"]
        )
        assert item["measurement"]["stateless"]["output_token_count"] == len(
            item["measurement"]["stateless"]["output_ids"]
        )
        assert (
            item["measurement"]["session"]["session_metadata"]["newly_prefilled_tokens"]
            == len(item["measurement"]["session"]["input_ids"]) + 1
        )


def test_sweep_checkpoints_prime_and_completed_measurements(perf_runner):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    run_record = {"measurements": []}
    checkpoints = []

    asyncio.run(
        perf_runner._run_sweep(
            engine,
            tokenizer,
            context_lengths=[80],
            repetitions=1,
            frontier_policy="current-frontier",
            max_new_tokens=4,
            max_model_len=4096,
            run_record=run_record,
            checkpoint=lambda: checkpoints.append(copy.deepcopy(run_record)),
        )
    )

    assert len(checkpoints) == 2
    assert checkpoints[0]["measurements"][0]["measurement"] is None
    assert checkpoints[1]["measurements"][0]["measurement"] is not None


def test_run_record_checkpoint_replaces_json_without_leaving_temp_file(
    perf_runner, tmp_path
):
    output_path = tmp_path / "run.json"
    run_record = {"status": "running", "measurements": [{"repetition": 1}]}

    perf_runner._write_run_record(output_path, run_record)

    assert json.loads(output_path.read_text(encoding="utf-8")) == run_record
    assert list(tmp_path.iterdir()) == [output_path]


def test_disk_checkpoint_preserves_completed_pair_after_later_failure(
    perf_runner, tmp_path
):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    output_path = tmp_path / "run.json"
    run_record = {"status": "running", "measurements": []}
    original_generate = engine.generate
    generate_calls = 0

    def fail_at_next_pair(**kwargs):
        nonlocal generate_calls
        if generate_calls == 4:
            raise RuntimeError("simulated engine failure")
        generate_calls += 1
        return original_generate(**kwargs)

    engine.generate = fail_at_next_pair

    with pytest.raises(RuntimeError, match="simulated engine failure"):
        asyncio.run(
            perf_runner._run_sweep(
                engine,
                tokenizer,
                context_lengths=[80],
                repetitions=2,
                frontier_policy="current-frontier",
                max_new_tokens=4,
                max_model_len=4096,
                run_record=run_record,
                checkpoint=lambda: perf_runner._write_run_record(
                    output_path, run_record
                ),
            )
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert len(persisted["measurements"]) == 1
    completed_pair = persisted["measurements"][0]["measurement"]
    assert completed_pair["output_ids_match"] is True


def test_sweep_stops_on_prime_output_mismatch_and_closes_session(perf_runner):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    engine.mismatch_control = True

    with pytest.raises(ValueError, match="prime output token IDs differ"):
        asyncio.run(
            perf_runner._run_sweep(
                engine,
                tokenizer,
                context_lengths=[80],
                repetitions=1,
                frontier_policy="current-frontier",
                max_new_tokens=4,
                max_model_len=4096,
                run_record={"measurements": []},
            )
        )

    assert engine.closed == engine.opened


def test_sweep_rejects_impossible_context_before_open_or_tokenization(
    perf_runner, monkeypatch
):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    build_calls = []
    original_build_context_prompt = perf_runner.build_context_prompt

    def observe_build_context_prompt(*args, **kwargs):
        build_calls.append((args, kwargs))
        return original_build_context_prompt(*args, **kwargs)

    monkeypatch.setattr(
        perf_runner, "build_context_prompt", observe_build_context_prompt
    )

    with pytest.raises(ValueError, match="target and generation reserve"):
        asyncio.run(
            perf_runner._run_sweep(
                engine,
                tokenizer,
                context_lengths=[4096],
                repetitions=1,
                frontier_policy="current-frontier",
                max_new_tokens=1,
                max_model_len=4096,
                run_record={"measurements": []},
            )
        )

    assert engine.opened == []
    assert build_calls == []


def test_sweep_keeps_raw_prime_record_when_frontier_metadata_fails(perf_runner):
    tokenizer = _QwenTemplateTokenizer()
    engine = _SerialFakeEngine(perf_runner)
    engine.bad_session_metadata = True
    run_record = {"measurements": []}

    with pytest.raises(ValueError, match="session context metadata mismatch"):
        asyncio.run(
            perf_runner._run_sweep(
                engine,
                tokenizer,
                context_lengths=[80],
                repetitions=1,
                frontier_policy="current-frontier",
                max_new_tokens=4,
                max_model_len=4096,
                run_record=run_record,
            )
        )

    assert len(run_record["measurements"]) == 1
    measurement = run_record["measurements"][0]
    assert (
        measurement["prime"]["session"]["session_metadata"]["reused_context_tokens"]
        == 1
    )
    assert engine.closed == engine.opened


@pytest.mark.parametrize(
    ("value", "expected"),
    (("64,128", [64, 128]), (" 64, 128 ", [64, 128])),
)
def test_context_length_parser_accepts_positive_comma_separated_values(
    perf_runner, value, expected
):
    assert perf_runner.parse_context_lengths(value) == expected


@pytest.mark.parametrize("value", ("", "0,64", "64,-1", "64,64", "64,,128"))
def test_context_length_parser_rejects_invalid_values(perf_runner, value):
    with pytest.raises(ValueError, match="context lengths"):
        perf_runner.parse_context_lengths(value)


def test_npu_snapshot_records_available_status_and_project_local_path(
    perf_runner, monkeypatch
):
    assert hasattr(perf_runner, "_capture_hardware_snapshot")
    calls = []

    def capture(output_path):
        calls.append(output_path)
        return True

    monkeypatch.setattr(perf_runner, "capture_npu_smi", capture)
    hardware_dir = (
        perf_runner.REPO_ROOT / "results" / "raw" / "session_kv_perf" / "hardware"
    )

    snapshot = perf_runner._capture_hardware_snapshot(hardware_dir, "run-id", "before")

    expected_path = hardware_dir / "run-id-before.txt"
    assert calls == [expected_path]
    assert snapshot == {"available": True, "path": str(expected_path)}
    assert expected_path.is_relative_to(perf_runner.REPO_ROOT)


def test_capture_npu_smi_is_scoped_to_npu_zero(monkeypatch):
    from benchmarks.course_project import telemetry

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 1, "stdout": ""})()

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)

    assert telemetry.capture_npu_smi(Path("unused-npu-snapshot.txt")) is False
    assert calls[0][0] == ["npu-smi", "info", "-t", "usages", "-i", "0"]


def test_performance_wrapper_sources_project_environment_and_execs_runner():
    wrapper_path = ROOT / "scripts" / "run_session_kv_perf.sh"
    assert wrapper_path.is_file(), "Session KV performance wrapper is missing"

    wrapper = wrapper_path.read_text(encoding="utf-8")
    assert 'source "${REPO_ROOT}/scripts/use_project_env.sh"' in wrapper
    assert 'exec python "${REPO_ROOT}/scripts/run_session_kv_perf.py" "$@"' in wrapper
