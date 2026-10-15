from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "run_session_kv_ttft.py"
REPORT_PATH = ROOT / "benchmarks" / "course_project" / "session_kv_ttft_report.py"


@pytest.fixture(scope="module")
def ttft_runner():
    if not RUNNER_PATH.is_file():
        pytest.skip("Session KV TTFT runner is not implemented yet")
    spec = importlib.util.spec_from_file_location("run_session_kv_ttft", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ttft_report():
    if not REPORT_PATH.is_file():
        pytest.skip("Session KV TTFT report is not implemented yet")
    spec = importlib.util.spec_from_file_location("session_kv_ttft_report", REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyChatTokenizer:
    eos_token_id = 2
    special_tokens = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3}

    def __init__(self):
        self.alter_user_one = False

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
        user_index = 0
        for message in messages:
            if message["role"] == "system":
                continue
            content = message["content"].strip()
            if message["role"] == "user":
                user_index += 1
                if user_index == 1 and self.alter_user_one:
                    content += " changed"
            rendered += f"<|im_start|>{message['role']}\n" f"{content}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>\n"
        return self._encode(rendered)


class _PrefixChangingTokenizer(_TinyChatTokenizer):
    def __init__(self):
        super().__init__()
        self.template_calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if self.template_calls:
            self.alter_user_one = True
        self.template_calls += 1
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


def test_ttft_runner_exists():
    assert RUNNER_PATH.is_file(), "Session KV TTFT runner is missing"


def test_context_prompt_meets_target_with_chat_template(ttft_runner):
    tokenizer = _TinyChatTokenizer()

    user_text, input_ids = ttft_runner.build_context_prompt(tokenizer, 80)

    assert user_text
    assert len(input_ids) >= 80
    assert input_ids == tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ttft_runner.SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_assistant_content_strips_only_terminal_eos(ttft_runner):
    tokenizer = _TinyChatTokenizer()

    content = ttft_runner.assistant_content_from_output_ids(
        tokenizer,
        [
            ord("A") + 1000,
            tokenizer.eos_token_id,
            ord("B") + 1000,
            tokenizer.eos_token_id,
        ],
    )

    assert content == "<think>\nA<|im_end|>B"


def test_sixteen_turn_history_reconstructs_exact_session_suffix(ttft_runner):
    tokenizer = _TinyChatTokenizer()
    initial_user, initial_ids = ttft_runner.build_context_prompt(tokenizer, 80)
    messages = [
        {"role": "system", "content": ttft_runner.SYSTEM_PROMPT},
        {"role": "user", "content": initial_user},
    ]
    output_for_turn = [ord("A") + 1000, tokenizer.eos_token_id]
    history_ids = initial_ids + output_for_turn

    for turn in range(2, 17):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": ttft_runner.assistant_content_from_output_ids(
                        tokenizer, output_for_turn
                    ),
                },
                {"role": "user", "content": ttft_runner.NEXT_USER_TEXT},
            ]
        )
        suffix_ids, full_input_ids = ttft_runner.build_multiturn_inputs(
            tokenizer, messages, history_ids
        )
        assert full_input_ids == history_ids + suffix_ids
        assert full_input_ids[: len(history_ids)] == history_ids

        output_for_turn = [ord("A") + turn + 1000, tokenizer.eos_token_id]
        history_ids = full_input_ids + output_for_turn

    assert sum(message["role"] == "user" for message in messages) == 16
    assert sum(message["role"] == "assistant" for message in messages) == 15


def test_full_history_builder_rejects_changed_cached_prefix(ttft_runner):
    tokenizer = _TinyChatTokenizer()
    initial_user, initial_ids = ttft_runner.build_context_prompt(tokenizer, 80)
    output_ids = [ord("A") + 1000, tokenizer.eos_token_id]
    messages = [
        {"role": "system", "content": ttft_runner.SYSTEM_PROMPT},
        {"role": "user", "content": initial_user},
        {
            "role": "assistant",
            "content": ttft_runner.assistant_content_from_output_ids(
                tokenizer, output_ids
            ),
        },
        {"role": "user", "content": ttft_runner.NEXT_USER_TEXT},
    ]
    tokenizer.alter_user_one = True

    with pytest.raises(ValueError, match="prefix"):
        ttft_runner.build_multiturn_inputs(
            tokenizer, messages, initial_ids + output_ids
        )


class _FakeStreamingEngine:
    def __init__(self, *, empty: bool = False):
        self.empty = empty
        self.request = None
        self.drained = False

    def generate(self, **kwargs):
        self.request = kwargs

        def chunks():
            if self.empty:
                self.drained = True
                return
            yield {"output_ids": [], "meta_info": {"phase": "waiting"}}
            yield {"output_ids": [21], "meta_info": {"phase": "first-token"}}
            yield {
                "output_ids": [22, 23],
                "meta_info": {"session": {"total_context_tokens": 12}},
            }
            self.drained = True

        return chunks()


@pytest.mark.parametrize("session_id", [None, "session-test"])
def test_stream_ttft_uses_first_token_and_drains_all_chunks(
    ttft_runner, monkeypatch, session_id
):
    clock_values = iter([10.0, 10.003])
    monkeypatch.setattr(ttft_runner.time, "perf_counter", lambda: next(clock_values))
    engine = _FakeStreamingEngine()
    sampling_params = {"max_new_tokens": 8, "temperature": 0}

    result = ttft_runner.measure_stream_ttft(
        engine, [1, 2, 3], sampling_params, session_id=session_id
    )

    assert result["ttft_ms"] == pytest.approx(3.0)
    assert result["output_ids"] == [21, 22, 23]
    assert result["meta_info"]["session"]["total_context_tokens"] == 12
    assert engine.drained is True
    assert engine.request["input_ids"] == [1, 2, 3]
    assert engine.request["session_id"] == session_id
    assert engine.request["stream"] is True
    assert engine.request["sampling_params"] == {
        "max_new_tokens": 8,
        "temperature": 0,
        "stream_interval": 1,
    }
    assert sampling_params == {"max_new_tokens": 8, "temperature": 0}


def test_stream_ttft_rejects_stream_without_output_ids(ttft_runner, monkeypatch):
    monkeypatch.setattr(ttft_runner.time, "perf_counter", lambda: 10.0)
    engine = _FakeStreamingEngine(empty=True)

    with pytest.raises(ValueError, match="without a non-empty output_ids"):
        ttft_runner.measure_stream_ttft(
            engine, [1], {"max_new_tokens": 8, "temperature": 0}, session_id=None
        )
    assert engine.drained is True


class _FakeSweepEngine:
    def __init__(
        self,
        *,
        mismatch_call_index: int | None = None,
        bad_metadata_call_index: int | None = None,
    ):
        self.mismatch_call_index = mismatch_call_index
        self.bad_metadata_call_index = bad_metadata_call_index
        self.calls = []
        self.opened = []
        self.closed = []
        self.histories = {}
        self.active = 0
        self.max_active = 0

    def open_session(self):
        session_id = f"session-{len(self.opened)}"
        self.opened.append(session_id)
        self.histories[session_id] = []
        return session_id

    def close_session(self, session_id):
        self.closed.append(session_id)

    def generate(self, *, input_ids, sampling_params, session_id, stream):
        call_index = len(self.calls)
        path = "session" if session_id is not None else "stateless"
        self.calls.append(
            {
                "session_id": session_id,
                "path": path,
                "input_ids": list(input_ids),
                "sampling_params": dict(sampling_params),
                "stream": stream,
            }
        )

        def chunks():
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            assert self.active == 1
            try:
                output_ids = [1065]
                if call_index == self.mismatch_call_index:
                    output_ids = [1066]
                meta_info = {}
                if session_id is not None:
                    history_ids = self.histories[session_id]
                    reused = max(len(history_ids) - 1, 0)
                    total = len(history_ids) + len(input_ids)
                    metadata = {
                        "total_context_tokens": total,
                        "newly_prefilled_tokens": total - reused,
                        "reused_context_tokens": reused,
                    }
                    if call_index == self.bad_metadata_call_index:
                        metadata["reused_context_tokens"] += 1
                    meta_info["session"] = metadata
                    history_ids.extend(input_ids + output_ids)
                yield {"output_ids": output_ids, "meta_info": meta_info}
            finally:
                self.active -= 1

        return chunks()


def test_sweep_records_four_sixteen_turn_pairs_with_balanced_order(ttft_runner):
    tokenizer = _TinyChatTokenizer()
    engine = _FakeSweepEngine()
    run_record = {"measurements": []}
    checkpoints = []

    ttft_runner._run_sweep(
        engine,
        tokenizer,
        repetitions=4,
        initial_context_target=80,
        max_new_tokens=8,
        max_model_len=4096,
        run_record=run_record,
        checkpoint=lambda: checkpoints.append(len(run_record["measurements"])),
    )

    assert len(run_record["measurements"]) == 64
    assert len(checkpoints) == 64
    assert [
        tuple(run_record["measurements"][index * 16]["execution_order"])
        for index in range(4)
    ] == [
        ("session", "stateless"),
        ("stateless", "session"),
        ("session", "stateless"),
        ("stateless", "session"),
    ]
    assert len(engine.opened) == 5
    assert engine.closed == engine.opened
    assert engine.max_active == 1
    assert len(engine.calls) == 130
    assert len(run_record["warmup"]["requests"]) == 2
    assert all(pair["output_ids_match"] is True for pair in run_record["measurements"])
    first_turn = run_record["measurements"][0]
    second_turn = run_record["measurements"][1]
    assert first_turn["session"]["input_ids"] == first_turn["stateless"]["input_ids"]
    assert first_turn["session"]["session_metadata"] == {
        "total_context_tokens": first_turn["full_context_tokens"],
        "newly_prefilled_tokens": first_turn["full_context_tokens"],
        "reused_context_tokens": 0,
    }
    assert len(second_turn["session"]["input_ids"]) < len(
        second_turn["stateless"]["input_ids"]
    )
    prior_history_tokens = (
        second_turn["full_context_tokens"] - second_turn["session_suffix_tokens"]
    )
    assert second_turn["session"]["session_metadata"] == {
        "total_context_tokens": second_turn["full_context_tokens"],
        "newly_prefilled_tokens": second_turn["full_context_tokens"]
        - (prior_history_tokens - 1),
        "reused_context_tokens": prior_history_tokens - 1,
    }
    assert all(call["sampling_params"]["stream_interval"] == 1 for call in engine.calls)


def test_sweep_stops_on_first_invalid_pair_and_keeps_prior_checkpoint(
    ttft_runner,
    tmp_path,
):
    engine = _FakeSweepEngine(bad_metadata_call_index=4)
    run_record = {"measurements": []}
    checkpoints = []
    checkpoint_path = tmp_path / "run.json"

    def checkpoint():
        checkpoints.append((len(run_record["measurements"]), run_record.get("status")))
        ttft_runner._write_run_record(checkpoint_path, run_record)

    with pytest.raises(ValueError, match="metadata mismatch"):
        ttft_runner._run_sweep(
            engine,
            _TinyChatTokenizer(),
            repetitions=4,
            initial_context_target=80,
            max_new_tokens=8,
            max_model_len=4096,
            run_record=run_record,
            checkpoint=checkpoint,
        )

    assert len(run_record["measurements"]) == 1
    assert run_record["status"] == "failed"
    assert run_record["failure"]["turn"] == 2
    assert engine.closed == engine.opened
    assert len(engine.calls) == 5
    assert checkpoints[-1] == (1, "failed")
    saved_record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert saved_record["status"] == "failed"
    assert len(saved_record["measurements"]) == 1
    assert saved_record["failure"]["turn"] == 2


def test_sweep_marks_prefix_reconstruction_error_failed_and_checkpoints(
    ttft_runner,
):
    engine = _FakeSweepEngine()
    run_record = {"measurements": []}
    checkpoints = []

    with pytest.raises(ValueError, match="prefix"):
        ttft_runner._run_sweep(
            engine,
            _PrefixChangingTokenizer(),
            repetitions=4,
            initial_context_target=80,
            max_new_tokens=8,
            max_model_len=4096,
            run_record=run_record,
            checkpoint=lambda: checkpoints.append(
                (len(run_record["measurements"]), run_record.get("status"))
            ),
        )

    assert len(run_record["measurements"]) == 1
    assert run_record["status"] == "failed"
    assert run_record["failure"]["turn"] == 2
    assert run_record["failure"]["phase"] == "measurement"
    assert engine.closed == engine.opened
    assert checkpoints[-1] == (1, "failed")


def test_sweep_stops_on_output_id_mismatch_without_partial_pair(ttft_runner):
    engine = _FakeSweepEngine(mismatch_call_index=5)
    run_record = {"measurements": []}
    checkpoints = []

    with pytest.raises(ValueError, match="output token IDs differ"):
        ttft_runner._run_sweep(
            engine,
            _TinyChatTokenizer(),
            repetitions=4,
            initial_context_target=80,
            max_new_tokens=8,
            max_model_len=4096,
            run_record=run_record,
            checkpoint=lambda: checkpoints.append(
                (len(run_record["measurements"]), run_record.get("status"))
            ),
        )

    assert len(run_record["measurements"]) == 1
    assert run_record["status"] == "failed"
    assert run_record["failure"]["turn"] == 2
    assert (
        run_record["failed_pair"]["completed_requests"]["session"]["output_ids"]
        != run_record["failed_pair"]["completed_requests"]["stateless"]["output_ids"]
    )
    assert engine.closed == engine.opened
    assert len(engine.calls) == 6
    assert checkpoints[-1] == (1, "failed")


def test_sweep_rejects_context_limit_other_than_pinned_4096(ttft_runner):
    engine = _FakeSweepEngine()

    with pytest.raises(ValueError, match="requires max_model_len=4096"):
        ttft_runner._run_sweep(
            engine,
            _TinyChatTokenizer(),
            repetitions=4,
            initial_context_target=80,
            max_new_tokens=8,
            max_model_len=8192,
            run_record={"measurements": []},
        )

    assert engine.opened == []


def test_run_record_checkpoint_atomically_replaces_json(ttft_runner, tmp_path):
    output_path = tmp_path / "run.json"
    output_path.write_text('{"status": "old"}\n', encoding="utf-8")

    ttft_runner._write_run_record(output_path, {"status": "running", "count": 2})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "status": "running",
        "count": 2,
    }
    assert not output_path.with_suffix(".json.tmp").exists()


def _synthetic_completed_run():
    measurements = []
    for repetition in range(1, 5):
        prior_history_ids = None
        for turn in range(1, 17):
            full_context_tokens = 512 + (turn - 1) * 30
            session_suffix_tokens = full_context_tokens if turn == 1 else 28
            if turn == 1:
                session_input_ids = list(range(full_context_tokens))
                stateless_input_ids = list(session_input_ids)
            else:
                assert prior_history_ids is not None
                session_input_ids = [5000 + turn * 100 + index for index in range(28)]
                stateless_input_ids = prior_history_ids + session_input_ids
            prior_history_tokens = full_context_tokens - session_suffix_tokens
            reused = max(prior_history_tokens - 1, 0)
            newly_prefilled = full_context_tokens - reused
            session_ttft = 10.0 + turn + (repetition**2) / 10
            stateless_ttft = 30.0 + turn * 2 + (repetition**2) / 5
            session_output = [100 + turn, 2]
            stateless_output = list(session_output)
            measurements.append(
                {
                    "repetition": repetition,
                    "turn": turn,
                    "execution_order": (
                        ["session", "stateless"]
                        if repetition % 2
                        else ["stateless", "session"]
                    ),
                    "full_context_tokens": full_context_tokens,
                    "session_suffix_tokens": session_suffix_tokens,
                    "session": {
                        "path": "session",
                        "input_ids": session_input_ids,
                        "input_token_count": session_suffix_tokens,
                        "output_ids": session_output,
                        "output_token_count": len(session_output),
                        "ttft_ms": session_ttft,
                        "session_metadata": {
                            "total_context_tokens": full_context_tokens,
                            "newly_prefilled_tokens": newly_prefilled,
                            "reused_context_tokens": reused,
                        },
                    },
                    "stateless": {
                        "path": "stateless",
                        "input_ids": stateless_input_ids,
                        "input_token_count": full_context_tokens,
                        "output_ids": stateless_output,
                        "output_token_count": len(stateless_output),
                        "ttft_ms": stateless_ttft,
                    },
                    "output_ids_match": True,
                    "ttft_reduction_pct": 100.0 * (1.0 - session_ttft / stateless_ttft),
                }
            )
            prior_history_ids = stateless_input_ids + session_output
    return {
        "status": "completed",
        "run_id": "synthetic-run",
        "measurements": measurements,
    }


def test_ttft_report_module_exists():
    assert REPORT_PATH.is_file(), "Session KV TTFT report module is missing"


def test_report_writes_64_valid_rows_and_svg_observations(ttft_report, tmp_path):
    raw_path = tmp_path / "run.json"
    raw_path.write_text(json.dumps(_synthetic_completed_run()), encoding="utf-8")
    analysis_dir = tmp_path / "analysis"

    csv_path, svg_path = ttft_report.generate_report(raw_path, analysis_dir)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 64
    assert set(rows[0]) == {
        "repetition",
        "turn",
        "full_context_tokens",
        "session_suffix_tokens",
        "session_ttft_ms",
        "stateless_ttft_ms",
        "reused_context_tokens",
        "newly_prefilled_tokens",
        "output_ids_match",
        "ttft_reduction_pct",
        "session_turn_median_ttft_ms",
        "session_turn_min_ttft_ms",
        "session_turn_max_ttft_ms",
        "stateless_turn_median_ttft_ms",
        "stateless_turn_min_ttft_ms",
        "stateless_turn_max_ttft_ms",
    }
    expected = _synthetic_completed_run()["measurements"]
    first_turn = [item for item in expected if item["turn"] == 1]
    assert float(rows[0]["ttft_reduction_pct"]) == pytest.approx(
        expected[0]["ttft_reduction_pct"]
    )
    assert float(rows[0]["session_turn_median_ttft_ms"]) == pytest.approx(
        median(item["session"]["ttft_ms"] for item in first_turn)
    )
    assert float(rows[0]["session_turn_min_ttft_ms"]) == pytest.approx(
        min(item["session"]["ttft_ms"] for item in first_turn)
    )
    assert float(rows[0]["session_turn_max_ttft_ms"]) == pytest.approx(
        max(item["session"]["ttft_ms"] for item in first_turn)
    )
    assert float(rows[0]["stateless_turn_median_ttft_ms"]) == pytest.approx(
        median(item["stateless"]["ttft_ms"] for item in first_turn)
    )
    assert float(rows[0]["stateless_turn_min_ttft_ms"]) == pytest.approx(
        min(item["stateless"]["ttft_ms"] for item in first_turn)
    )
    assert float(rows[0]["stateless_turn_max_ttft_ms"]) == pytest.approx(
        max(item["stateless"]["ttft_ms"] for item in first_turn)
    )

    root = ElementTree.parse(svg_path).getroot()
    elements = list(root.iter())
    assert root.tag.endswith("svg")
    medians = {
        element.get("id"): element
        for element in elements
        if element.get("id") in {"session-median", "stateless-median"}
    }
    assert set(medians) == {"session-median", "stateless-median"}
    assert all(
        len(element.get("points", "").split()) == 16 for element in medians.values()
    )
    assert sum("observation" in element.get("class", "") for element in elements) == 128
    assert sum("range-marker" in element.get("class", "") for element in elements) == 32
    assert sum(element.get("class") == "turn-label" for element in elements) == 16
    actual_values = sorted(
        float(element.get("data-value-ms"))
        for element in elements
        if "observation" in element.get("class", "")
    )
    expected_values = sorted(
        item[path]["ttft_ms"] for item in expected for path in ("session", "stateless")
    )
    assert actual_values == pytest.approx(expected_values)
    assert (
        "native engine ttft"
        in " ".join(element.text or "" for element in elements).lower()
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(status="running"), "completed"),
        (
            lambda record: record["measurements"].pop(),
            "64",
        ),
        (
            lambda record: record["measurements"][0].update(output_ids_match=False),
            "output",
        ),
        (
            lambda record: record["measurements"][0]["stateless"].update(
                output_ids=[999, 2]
            ),
            "output",
        ),
        (
            lambda record: record["measurements"][0]["session"][
                "session_metadata"
            ].update(reused_context_tokens=999),
            "metadata",
        ),
        (
            lambda record: record["measurements"][0]["stateless"].update(
                ttft_ms=float("nan")
            ),
            "finite",
        ),
        (
            lambda record: record["measurements"][0]["stateless"].update(ttft_ms=0),
            "positive",
        ),
        (
            lambda record: record["measurements"][0]["session"].update(
                ttft_ms=float("nan")
            ),
            "finite",
        ),
        (
            lambda record: record["measurements"][0]["session"].update(ttft_ms=0),
            "positive",
        ),
        (
            lambda record: record["measurements"][0].update(
                full_context_tokens=4090, session_suffix_tokens=4090
            ),
            "4096-token limit",
        ),
        (
            lambda record: record["measurements"][0]["session"][
                "input_ids"
            ].__setitem__(0, 9999),
            "first-turn",
        ),
        (
            lambda record: record["measurements"][1]["session"][
                "input_ids"
            ].__setitem__(0, 9999),
            "suffix",
        ),
        (
            lambda record: [
                pair[path].update(output_ids=list(range(9)), output_token_count=9)
                for pair in record["measurements"][:2]
                for path in ("session", "stateless")
            ],
            "8-token output limit",
        ),
        (
            lambda record: record["measurements"][0].update(ttft_reduction_pct=10**400),
            "reduction",
        ),
    ],
)
def test_invalid_raw_run_is_rejected_without_report_files(
    ttft_report, tmp_path, mutate, message
):
    record = _synthetic_completed_run()
    mutate(record)
    raw_path = tmp_path / "run.json"
    raw_path.write_text(json.dumps(record, allow_nan=True), encoding="utf-8")
    analysis_dir = tmp_path / "analysis"

    with pytest.raises(ValueError, match=message):
        ttft_report.generate_report(raw_path, analysis_dir)

    assert not (analysis_dir / "session_kv_multiturn_ttft.csv").exists()
    assert not (analysis_dir / "session_kv_multiturn_ttft.svg").exists()


def test_report_csv_preserves_ttft_float_round_trip(ttft_report, tmp_path):
    record = _synthetic_completed_run()
    exact_session_ttft = 1.23456789012345
    exact_stateless_ttft = 2.34567890123456
    pair = record["measurements"][0]
    pair["session"]["ttft_ms"] = exact_session_ttft
    pair["stateless"]["ttft_ms"] = exact_stateless_ttft
    pair["ttft_reduction_pct"] = 100.0 * (
        1.0 - exact_session_ttft / exact_stateless_ttft
    )
    raw_path = tmp_path / "run.json"
    raw_path.write_text(json.dumps(record), encoding="utf-8")

    csv_path, _ = ttft_report.generate_report(raw_path, tmp_path / "analysis")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        first_row = next(csv.DictReader(csv_file))
    assert float(first_row["session_ttft_ms"]) == exact_session_ttft
    assert float(first_row["stateless_ttft_ms"]) == exact_stateless_ttft


def test_report_write_failure_does_not_publish_one_file(
    ttft_report, tmp_path, monkeypatch
):
    raw_path = tmp_path / "run.json"
    raw_path.write_text(json.dumps(_synthetic_completed_run()), encoding="utf-8")
    analysis_dir = tmp_path / "analysis"
    original_write_text = Path.write_text

    def fail_svg_write(path, *args, **kwargs):
        if path.name == "session_kv_multiturn_ttft.svg":
            raise OSError("injected SVG write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_svg_write)

    with pytest.raises(OSError, match="injected SVG write failure"):
        ttft_report.generate_report(raw_path, analysis_dir)

    assert not analysis_dir.exists()


def test_cli_defaults_output_dir_and_help_documents_it(ttft_runner):
    parser = ttft_runner.build_parser()

    args = parser.parse_args([])

    assert args.output_dir == Path("results/raw/session_kv_ttft")
    help_text = parser.format_help().replace("\\", "/")
    assert "--output-dir" in help_text
    assert "results/raw/session_kv_ttft" in help_text


def test_output_directory_is_confined_to_project(ttft_runner, tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)

    assert ttft_runner.resolve_output_dir(Path("results/raw/session_kv_ttft")) == (
        project_root / "results" / "raw" / "session_kv_ttft"
    )
    with pytest.raises(ValueError, match="inside the project"):
        ttft_runner.resolve_output_dir(tmp_path / "outside")


def test_local_model_path_rejects_external_directory(
    ttft_runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    external_model = tmp_path / "external-model"
    external_model.mkdir()
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)

    with pytest.raises(ValueError, match="inside the project"):
        ttft_runner._local_model_path({"local_model_path": str(external_model)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "another/model"),
        ("model_revision", "different-revision"),
        ("tp_size", 2),
        ("max_model_len", 8192),
        ("overlap_schedule", False),
    ],
)
def test_pinned_config_rejects_unapproved_model_settings(ttft_runner, field, value):
    config = json.loads(
        (ROOT / "configs" / "bringup" / "qwen35_4b.yaml").read_text(encoding="utf-8")
    )
    config[field] = value

    with pytest.raises(ValueError, match="pinned"):
        ttft_runner._validate_pinned_config(config)


def test_launch_configuration_disables_prefix_cache_and_kvstore_but_keeps_overlap(
    ttft_runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    model_path = project_root / "models" / "Qwen3.5-4B"
    model_path.mkdir(parents=True)
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)
    config = json.loads(
        (ROOT / "configs" / "bringup" / "qwen35_4b.yaml").read_text(encoding="utf-8")
    )
    config["local_model_path"] = str(model_path)

    resolved_model_path = ttft_runner._validate_pinned_config(config)
    engine_config, cli_args = ttft_runner._build_engine_config_and_cli_args(
        config, resolved_model_path
    )

    assert engine_config["prefix_caching"] is False
    assert engine_config["disable_kvstore"] is True
    assert engine_config["overlap_schedule"] is True
    assert "--disable-prefix-caching" in cli_args
    assert "--enable-prefix-caching" not in cli_args
    assert "--disable-kvstore" in cli_args
    assert "--disable-overlap-schedule" not in cli_args
    assert cli_args[0] == str(resolved_model_path)


def test_run_paths_skip_existing_raw_and_analysis_names(
    ttft_runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)
    ids = iter(["analysis-occupied", "raw-occupied", "fresh"])
    monkeypatch.setattr(ttft_runner, "_new_run_id", lambda: next(ids))

    raw_dir = project_root / "results" / "raw" / "session_kv_ttft"
    analysis_root = project_root / "results" / "analysis"
    raw_dir.mkdir(parents=True)
    analysis_root.mkdir(parents=True)
    occupied_raw = raw_dir / "qwen35-4b-session-kv-ttft-raw-occupied.json"
    occupied_raw.write_text("preserve", encoding="utf-8")
    (analysis_root / "session_kv_ttft_analysis-occupied").mkdir()

    run_id, raw_path, analysis_dir = ttft_runner.create_run_paths(
        Path("results/raw/session_kv_ttft")
    )

    assert run_id == "fresh"
    assert raw_path.name == "qwen35-4b-session-kv-ttft-fresh.json"
    assert analysis_dir == analysis_root / "session_kv_ttft_fresh"
    assert occupied_raw.read_text(encoding="utf-8") == "preserve"


def test_ttft_wrapper_sources_project_environment_and_execs_runner():
    wrapper_path = ROOT / "scripts" / "run_session_kv_ttft.sh"
    assert wrapper_path.is_file(), "Session KV TTFT wrapper is missing"

    wrapper = wrapper_path.read_text(encoding="utf-8")
    assert 'source "${REPO_ROOT}/scripts/use_project_env.sh"' in wrapper
    assert 'exec python "${REPO_ROOT}/scripts/run_session_kv_ttft.py" "$@"' in wrapper


def test_report_is_loaded_and_called_only_for_completed_run(
    ttft_runner, tmp_path, monkeypatch
):
    calls = []
    fake_report = ModuleType("benchmarks.course_project.session_kv_ttft_report")

    def generate_report(raw_path, analysis_dir):
        assert not analysis_dir.exists()
        calls.append((raw_path, analysis_dir))
        analysis_dir.mkdir()
        return analysis_dir / "out.csv", analysis_dir / "out.svg"

    fake_report.generate_report = generate_report
    monkeypatch.setitem(
        sys.modules, "benchmarks.course_project.session_kv_ttft_report", fake_report
    )
    raw_path = tmp_path / "run.json"
    failed_analysis = tmp_path / "failed-analysis"

    assert (
        ttft_runner._generate_report_if_completed(
            {"status": "failed"}, raw_path, failed_analysis
        )
        is None
    )
    assert calls == []
    assert not failed_analysis.exists()

    completed_analysis = tmp_path / "completed-analysis"
    report_paths = ttft_runner._generate_report_if_completed(
        {"status": "completed"}, raw_path, completed_analysis
    )

    assert report_paths == (
        completed_analysis / "out.csv",
        completed_analysis / "out.svg",
    )
    assert calls == [(raw_path, completed_analysis)]
    assert completed_analysis.is_dir()
    assert not list(tmp_path.glob(".session_kv_ttft_report_*"))


def test_runner_report_hook_publishes_real_csv_and_svg(
    ttft_runner, ttft_report, tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules, "benchmarks.course_project.session_kv_ttft_report", ttft_report
    )
    raw_path = tmp_path / "synthetic-run.json"
    raw_path.write_text(json.dumps(_synthetic_completed_run()), encoding="utf-8")
    analysis_dir = tmp_path / "session_kv_ttft_synthetic-run"

    report_paths = ttft_runner._generate_report_if_completed(
        {"status": "completed"}, raw_path, analysis_dir
    )

    assert report_paths is not None
    with report_paths[0].open(newline="", encoding="utf-8") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 64
    assert ElementTree.parse(report_paths[1]).getroot().tag.endswith("svg")


def test_main_failure_finalizes_raw_provenance_without_runtime_or_npu(
    ttft_runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_session_kv_ttft.py",
            "--output-dir",
            "results/raw/session_kv_ttft",
        ],
    )
    snapshot_calls = []
    monkeypatch.setattr(
        ttft_runner,
        "capture_npu_smi",
        lambda path: snapshot_calls.append(path) or False,
    )
    environment_calls = []
    monkeypatch.setattr(
        ttft_runner, "_set_offline_environment", lambda: environment_calls.append(True)
    )

    def fail_config_load(path):
        raise RuntimeError("expected test failure before runtime import")

    monkeypatch.setattr(ttft_runner, "load_model_config", fail_config_load)

    with pytest.raises(RuntimeError, match="expected test failure"):
        ttft_runner.main()

    raw_paths = list(
        (project_root / "results" / "raw" / "session_kv_ttft").glob("*.json")
    )
    assert len(raw_paths) == 1
    run_record = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    timestamps = {}
    for field in ("started_at_utc", "finished_at_utc"):
        timestamp = datetime.fromisoformat(run_record[field])
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset() == timedelta(0)
        timestamps[field] = timestamp
    assert timestamps["finished_at_utc"] >= timestamps["started_at_utc"]
    assert run_record["stream_interval"] == 1
    assert run_record["concurrency"] == 1
    assert len(snapshot_calls) == 2
    assert environment_calls == [True]
    assert not list((project_root / "results" / "analysis").glob("session_kv_ttft_*"))
