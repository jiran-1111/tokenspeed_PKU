from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import socket
import sys
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.course_project.core import StreamChunkEvent

try:
    from benchmarks.course_project import long_prefill_decode
except ImportError:
    long_prefill_decode = None

try:
    from scripts import run_long_prefill_decode
except ImportError:
    run_long_prefill_decode = None


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bringup" / "qwen35_4b.yaml"


def test_port_probe_rejects_live_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert not run_long_prefill_decode.check_port_available(
            "127.0.0.1", listener.getsockname()[1]
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX TIME_WAIT reuse semantics")
def test_port_probe_accepts_closed_listener_with_time_wait():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            connection.close()
            assert client.recv(1) == b""
    assert run_long_prefill_decode.check_port_available("127.0.0.1", port)


class _TinyChatTokenizer:
    def __init__(self):
        self.template_calls = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.template_calls += 1
        content = messages[-1]["content"]
        return self.encode(f"<user>{content}</user><assistant>")


class _UnachievableTokenizer(_TinyChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls += 1
        return [0] * 4097


class _NonMonotoneTokenizer(_TinyChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls += 1
        content_length = len(messages[-1]["content"])
        if content_length == 128:
            observed = 250
        elif content_length == 190:
            observed = 256
        else:
            observed = 255
        return [0] * observed


@pytest.fixture
def base_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def module():
    return long_prefill_decode


def test_successful_http_status_is_written_to_request_jsonl(module):
    assert module is not None
    assert run_long_prefill_decode is not None
    result = SimpleNamespace(
        stream_completed=True,
        input_tokens=256,
        output_tokens=128,
        request_send_epoch_ns=10,
        request_send_monotonic_ns=10,
        first_chunk_epoch_ns=20,
        first_chunk_monotonic_ns=20,
        first_chunk_field="content",
        stream_drain_epoch_ns=30,
        stream_drain_monotonic_ns=30,
        ttft_ms=1.0,
        e2e_ms=2.0,
        http_status=206,
    )
    request = module._request_record(
        request_role="decode",
        result=result,
        expected_input_tokens=256,
        expected_output_tokens=128,
        max_tokens=128,
        ignore_eos=True,
        chunk_events=[StreamChunkEvent(1, "content", 20, 20, 2)],
    )
    spec = module.TrialSpec(
        trial_id="decode-only-r1-c4096",
        condition="decode-only",
        prefill_tokens=0,
        repetition=1,
        chunk_size=4096,
        execution_order=1,
    )
    requests_output = io.StringIO()
    trials_output = io.StringIO()

    run_long_prefill_decode._record_trial(
        trial={
            "requests": [request],
            "status": "valid",
            "invalid_reason": None,
            "requests_overlap": None,
            "chunk_gap_events": [],
        },
        trial_spec=spec,
        run_id="synthetic-run",
        pair_id="decode-only-r1",
        round_number=1,
        config={
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "synthetic-revision",
            "served_model_name": "synthetic-model",
        },
        config_sha256="a" * 64,
        prompt_hashes={"decode": "b" * 64},
        requests_output=requests_output,
        trials_output=trials_output,
    )

    written_request = json.loads(requests_output.getvalue())
    assert written_request["http_status"] == 206


def test_record_trial_persists_prompt_provenance_without_prompt_text(module, tmp_path):
    assert module is not None
    assert run_long_prefill_decode is not None
    prompt_text = "PROMPT_SENTINEL_long_prefill_task2_7f31"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_hash = module.prompt_sha256(messages)
    spec = module.TrialSpec(
        trial_id="prefill-only-256-r1-c4096",
        condition="prefill-only-256",
        prefill_tokens=256,
        repetition=1,
        chunk_size=4096,
        execution_order=1,
    )
    requests_path = tmp_path / "requests.jsonl"
    trials_path = tmp_path / "trials.jsonl"

    with requests_path.open("w", encoding="utf-8") as requests_output, trials_path.open(
        "w", encoding="utf-8"
    ) as trials_output:
        run_long_prefill_decode._record_trial(
            trial={
                "requests": [{"request_role": "prefill", "status": "valid"}],
                "status": "valid",
                "invalid_reason": None,
                "requests_overlap": None,
                "chunk_gap_events": [],
            },
            trial_spec=spec,
            run_id="synthetic-run",
            pair_id="synthetic-pair",
            round_number=1,
            config={
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": "synthetic-revision",
                "served_model_name": "synthetic-model",
            },
            config_sha256="a" * 64,
            prompt_hashes={"prefill": prompt_hash},
            requests_output=requests_output,
            trials_output=trials_output,
        )

    request_record = json.loads(requests_path.read_text(encoding="utf-8"))
    trial_record = json.loads(trials_path.read_text(encoding="utf-8"))
    assert request_record["prompt_recipe_version"] == module.PROMPT_RECIPE_VERSION
    assert request_record["prompt_sha256"] == prompt_hash
    assert trial_record["prompt_recipe_version"] == module.PROMPT_RECIPE_VERSION
    assert trial_record["prompt_hashes"]["prefill"] == prompt_hash
    assert prompt_text not in requests_path.read_text(encoding="utf-8")
    assert prompt_text not in trials_path.read_text(encoding="utf-8")


def test_http_error_status_is_recorded_without_response_body(module):
    assert module is not None
    response_body = b"synthetic-response-body"
    error = urllib.error.HTTPError(
        "https://example.invalid/chat/completions",
        429,
        "Too Many Requests",
        {},
        io.BytesIO(response_body),
    )

    class _FailingClient:
        def chat(self, **_kwargs):
            raise error

    trial = module.run_single_trial(
        client=_FailingClient(),
        trial_spec=module.TrialSpec(
            trial_id="prefill-only-2048-r1-c4096",
            condition="prefill-only-2048",
            prefill_tokens=2048,
            repetition=1,
            chunk_size=4096,
            execution_order=1,
        ),
        request_role="prefill",
        model="synthetic-model",
        messages=[{"role": "user", "content": "synthetic prompt"}],
        expected_input_tokens=2048,
        expected_output_tokens=1,
    )
    request_json = json.dumps(trial["requests"][0])

    assert trial["requests"][0]["http_status"] == 429
    assert response_body.decode() not in request_json
    assert error.fp.read() == response_body


def test_network_error_has_null_http_status(module):
    assert module is not None

    class _FailingClient:
        def chat(self, **_kwargs):
            raise urllib.error.URLError("synthetic connection failure")

    trial = module.run_single_trial(
        client=_FailingClient(),
        trial_spec=module.TrialSpec(
            trial_id="prefill-only-2048-r1-c4096",
            condition="prefill-only-2048",
            prefill_tokens=2048,
            repetition=1,
            chunk_size=4096,
            execution_order=1,
        ),
        request_role="prefill",
        model="synthetic-model",
        messages=[{"role": "user", "content": "synthetic prompt"}],
        expected_input_tokens=2048,
        expected_output_tokens=1,
    )

    assert trial["requests"][0]["http_status"] is None


def test_build_config_variants_disable_kvstore_with_prefix_cache(module, base_config):
    assert module is not None, "long prefill decode module is missing"
    original = copy.deepcopy(base_config)

    variants = module.build_config_variants(base_config)

    assert set(variants) == {4096, 512}
    assert base_config == original
    for chunk_size, variant in variants.items():
        expected = copy.deepcopy(original)
        expected["chunked_prefill_size"] = chunk_size
        expected["prefix_caching"] = False
        expected["disable_kvstore"] = True
        assert variant == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "Qwen/Qwen3.5-9B"),
        ("tp_size", 2),
        ("dtype", "float16"),
        ("max_model_len", 8192),
        ("chunked_prefill_size", 1024),
    ],
)
def test_build_config_variants_reject_unpinned_config(
    module, base_config, field, value
):
    assert module is not None, "long prefill decode module is missing"
    base_config[field] = value

    with pytest.raises(ValueError):
        module.build_config_variants(base_config)


@pytest.mark.parametrize("target_tokens", [256, 2048, 3584])
def test_build_exact_messages_matches_complete_chat_template(module, target_tokens):
    assert module is not None, "long prefill decode module is missing"
    tokenizer = _TinyChatTokenizer()

    messages = module.build_exact_messages(tokenizer, target_tokens, seed=17)
    token_ids = module.chat_template_token_ids(tokenizer, messages)

    assert messages == module.build_exact_messages(
        _TinyChatTokenizer(), target_tokens, seed=17
    )
    assert len(token_ids) == target_tokens
    assert len(token_ids) <= 4096
    assert module.PROMPT_RECIPE_VERSION
    assert len(module.prompt_sha256(messages)) == 64


@pytest.mark.parametrize("target_tokens", [256, 2048, 3584])
def test_build_exact_messages_handles_seed_boundary_token_merges(module, target_tokens):
    class BoundaryMergeTokenizer(_TinyChatTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            if text.startswith("Long prefill workload seed"):
                copies = text.count(
                    "Summarize the sequence and preserve every stated detail."
                )
                return [ord("x")] * (28 if copies == 1 else 24 * copies)
            return super().encode(text, add_special_tokens=add_special_tokens)

    tokenizer = BoundaryMergeTokenizer()
    messages = module.build_exact_messages(tokenizer, target_tokens, seed=17)
    assert len(module.chat_template_token_ids(tokenizer, messages)) == target_tokens
    assert tokenizer.template_calls <= 33


def test_build_exact_messages_fails_with_observed_length_after_bounded_search(module):
    assert module is not None, "long prefill decode module is missing"
    tokenizer = _UnachievableTokenizer()

    with pytest.raises(ValueError, match=r"target 256.*observed length 4097"):
        module.build_exact_messages(tokenizer, 256, seed=17)

    assert tokenizer.template_calls <= 32


def test_build_exact_messages_finds_target_near_nonmonotone_binary_branch(module):
    assert module is not None, "long prefill decode module is missing"
    tokenizer = _NonMonotoneTokenizer()

    messages = module.build_exact_messages(tokenizer, 256, seed=17)

    assert len(module.chat_template_token_ids(tokenizer, messages)) == 256
    assert len(messages[0]["content"]) == 190
    assert tokenizer.template_calls <= 32


def test_build_exact_messages_rejects_target_above_context_limit(module):
    assert module is not None, "long prefill decode module is missing"
    tokenizer = _TinyChatTokenizer()

    with pytest.raises(ValueError, match="4096"):
        module.build_exact_messages(tokenizer, 4097, seed=17)

    assert tokenizer.template_calls == 0


def test_local_model_path_requires_existing_project_internal_path(
    module, tmp_path, monkeypatch
):
    assert module is not None, "long prefill decode module is missing"
    project_root = tmp_path / "project"
    model_path = project_root / "models" / "Qwen3.5-4B"
    model_path.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", project_root)

    assert (
        module.local_model_path(
            {"local_model_path": "models/Qwen3.5-4B", "model_id": "Qwen/remote"}
        )
        == model_path.resolve()
    )

    outside_path = tmp_path / "external-model"
    outside_path.mkdir()
    with pytest.raises(ValueError, match="project"):
        module.local_model_path(
            {"local_model_path": str(outside_path), "model_id": "Qwen/remote"}
        )
    with pytest.raises(ValueError, match="local_model_path"):
        module.local_model_path({"model_id": "Qwen/remote"})


def test_load_local_tokenizer_forces_local_files_only(module, tmp_path, monkeypatch):
    assert module is not None, "long prefill decode module is missing"
    project_root = tmp_path / "project"
    model_path = project_root / "models" / "Qwen3.5-4B"
    model_path.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", project_root)
    calls = []

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            calls.append((path, local_files_only))
            return "tokenizer"

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_AutoTokenizer)
    )

    tokenizer = module.load_local_tokenizer(
        {"local_model_path": "models/Qwen3.5-4B", "model_id": "Qwen/remote"}
    )

    assert tokenizer == "tokenizer"
    assert calls == [(str(model_path.resolve()), True)]


def _result(*, input_tokens=256, output_tokens=128, stream_completed=True):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stream_completed=stream_completed,
    )


def test_validate_result_accepts_expected_complete_stream(module):
    assert module is not None, "long prefill decode module is missing"
    assert module.validate_result(
        _result(), expected_input_tokens=256, expected_output_tokens=128
    ) == (True, None)


@pytest.mark.parametrize(
    ("result", "expected_input", "expected_output", "reason"),
    [
        (_result(input_tokens=255), 256, 128, "input token mismatch"),
        (_result(output_tokens=127), 256, 128, "output token mismatch"),
        (_result(stream_completed=False), 256, 128, "stream did not complete"),
    ],
)
def test_validate_result_rejects_invalid_stream(
    module, result, expected_input, expected_output, reason
):
    assert module is not None, "long prefill decode module is missing"
    valid, invalid_reason = module.validate_result(
        result,
        expected_input_tokens=expected_input,
        expected_output_tokens=expected_output,
    )

    assert valid is False
    assert reason in invalid_reason


def _timed_result(
    *,
    input_tokens=256,
    output_tokens=128,
    stream_completed=True,
    request_send_monotonic_ns=10,
    first_chunk_monotonic_ns=100,
    stream_drain_monotonic_ns=800,
    chunk_events=None,
):
    if chunk_events is None:
        chunk_events = (
            StreamChunkEvent(1, "content", 110, 100, 1),
            StreamChunkEvent(2, "content", 210, 200, 1),
            StreamChunkEvent(3, "reasoning_content", 410, 400, 2),
            StreamChunkEvent(4, "content", 610, 600, 1),
        )
    return SimpleNamespace(
        text="private generated output",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=0.09,
        e2e_ms=0.79,
        request_send_epoch_ns=request_send_monotonic_ns + 1000,
        request_send_monotonic_ns=request_send_monotonic_ns,
        first_chunk_epoch_ns=(
            None
            if first_chunk_monotonic_ns is None
            else first_chunk_monotonic_ns + 1000
        ),
        first_chunk_monotonic_ns=first_chunk_monotonic_ns,
        stream_drain_epoch_ns=stream_drain_monotonic_ns + 1000,
        stream_drain_monotonic_ns=stream_drain_monotonic_ns,
        first_chunk_field="content" if first_chunk_monotonic_ns is not None else None,
        stream_completed=stream_completed,
        chunk_events=tuple(chunk_events),
        http_status=None,
    )


class _StagedFakeClient:
    def __init__(self, *, decode_result=None, prefill_result=None, fail_role=None):
        self.decode_result = decode_result or _timed_result()
        self.prefill_result = prefill_result or _timed_result(
            input_tokens=2048,
            output_tokens=1,
            request_send_monotonic_ns=250,
            first_chunk_monotonic_ns=300,
            stream_drain_monotonic_ns=700,
            chunk_events=(StreamChunkEvent(1, "content", 1300, 300, 1),),
        )
        self.fail_role = fail_role
        self.calls = []
        self.events = []
        self.decode_finished = threading.Event()
        self.prefill_thread_name = None

    def chat(
        self,
        *,
        model,
        messages,
        max_tokens,
        on_first_chunk=None,
        on_chunk=None,
        on_request_send=None,
        ignore_eos=False,
    ):
        role = messages[0]["content"]
        self.calls.append(
            {
                "role": role,
                "model": model,
                "max_tokens": max_tokens,
                "ignore_eos": ignore_eos,
            }
        )
        if role == "decode":
            result = self.decode_result
            if on_request_send is not None:
                on_request_send(
                    result.request_send_epoch_ns,
                    result.request_send_monotonic_ns,
                )
            if self.fail_role == "decode_http":
                raise urllib.error.HTTPError(
                    "http://example.invalid", 503, "unavailable", {}, None
                )
            if self.fail_role == "decode":
                raise RuntimeError("fake decode failure")
            if (
                result.first_chunk_monotonic_ns is not None
                and on_first_chunk is not None
            ):
                on_first_chunk(
                    result.first_chunk_field,
                    result.first_chunk_epoch_ns,
                    result.first_chunk_monotonic_ns,
                )
                on_first_chunk(
                    result.first_chunk_field,
                    result.first_chunk_epoch_ns,
                    result.first_chunk_monotonic_ns,
                )
            if on_chunk is not None:
                for event in result.chunk_events:
                    on_chunk(event)
            self.events.append("decode_drained")
            self.decode_finished.set()
            if self.fail_role == "decode_after_chunks":
                raise RuntimeError("fake decode failure after chunks")
            return result

        if on_request_send is not None:
            on_request_send(
                self.prefill_result.request_send_epoch_ns,
                self.prefill_result.request_send_monotonic_ns,
            )
        if self.fail_role == "prefill":
            raise RuntimeError("fake prefill failure")
        self.prefill_thread_name = threading.current_thread().name
        if not self.decode_finished.wait(timeout=1.0):
            raise TimeoutError("decode drain waited for prefill")
        if on_chunk is not None:
            for event in self.prefill_result.chunk_events:
                on_chunk(event)
        self.events.append("prefill_finished")
        return self.prefill_result


def _trial_spec(module, *, chunk_size=4096, condition="staged-2048", repetition=1):
    return module.TrialSpec(
        trial_id=f"{condition}-r{repetition}-c{chunk_size}",
        condition=condition,
        prefill_tokens=2048,
        repetition=repetition,
        chunk_size=chunk_size,
        execution_order=repetition,
    )


def test_run_staged_trial_submits_once_without_blocking_decode(module):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient()

    trial = module.run_staged_trial(
        client=client,
        trial_spec=_trial_spec(module),
        model="fake-model",
        decode_messages=[{"role": "user", "content": "decode"}],
        prefill_messages=[{"role": "user", "content": "prefill"}],
        expected_decode_input_tokens=256,
        expected_decode_output_tokens=128,
        expected_prefill_input_tokens=2048,
        expected_prefill_output_tokens=1,
    )

    assert trial["status"] == "valid"
    assert [call["role"] for call in client.calls].count("prefill") == 1
    assert [call["role"] for call in client.calls][0] == "decode"
    assert client.events.index("decode_drained") < client.events.index(
        "prefill_finished"
    )
    assert client.prefill_thread_name.startswith("ThreadPoolExecutor")
    decode_call = next(call for call in client.calls if call["role"] == "decode")
    assert decode_call["ignore_eos"] is True
    assert [gap["classification"] for gap in trial["chunk_gap_events"]] == [
        "before",
        "crosses",
        "after",
    ]
    assert all("client_sse_chunk_gap_ms" in gap for gap in trial["chunk_gap_events"])
    assert "private generated output" not in json.dumps(trial)
    assert all("text" not in request for request in trial["requests"])


def test_run_staged_trial_without_nonempty_decode_chunk_does_not_submit_prefill(module):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient(
        decode_result=_timed_result(
            output_tokens=0,
            first_chunk_monotonic_ns=None,
            stream_drain_monotonic_ns=500,
            chunk_events=(),
        )
    )

    trial = module.run_staged_trial(
        client=client,
        trial_spec=_trial_spec(module),
        model="fake-model",
        decode_messages=[{"role": "user", "content": "decode"}],
        prefill_messages=[{"role": "user", "content": "prefill"}],
        expected_decode_input_tokens=256,
        expected_decode_output_tokens=128,
        expected_prefill_input_tokens=2048,
        expected_prefill_output_tokens=1,
    )

    assert [call["role"] for call in client.calls] == ["decode"]
    assert trial["status"] == "invalid"
    assert "decode_stream_has_no_nonempty_chunk" in trial["invalid_reason"]


def test_run_staged_trial_does_not_submit_prefill_for_empty_decode_chunk(module):
    assert module is not None, "long prefill decode module is missing"
    empty_chunk = StreamChunkEvent(1, "content", 110, 100, 0)
    client = _StagedFakeClient(
        decode_result=_timed_result(
            output_tokens=128,
            first_chunk_monotonic_ns=None,
            chunk_events=(empty_chunk,),
        )
    )

    trial = module.run_staged_trial(
        client=client,
        trial_spec=_trial_spec(module),
        model="fake-model",
        decode_messages=[{"role": "user", "content": "decode"}],
        prefill_messages=[{"role": "user", "content": "prefill"}],
        expected_decode_input_tokens=256,
        expected_decode_output_tokens=128,
        expected_prefill_input_tokens=2048,
        expected_prefill_output_tokens=1,
    )

    assert [call["role"] for call in client.calls] == ["decode"]
    assert trial["status"] == "invalid"
    assert trial["requests"][0]["status"] == "invalid"
    assert trial["requests"][0]["invalid_reason"] == (
        "request_stream_has_no_nonempty_chunk"
    )


def test_run_staged_trial_classifies_chunks_when_decode_fails_after_emitting(module):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient(fail_role="decode_after_chunks")

    trial = module.run_staged_trial(
        client=client,
        trial_spec=_trial_spec(module),
        model="fake-model",
        decode_messages=[{"role": "user", "content": "decode"}],
        prefill_messages=[{"role": "user", "content": "prefill"}],
        expected_decode_input_tokens=256,
        expected_decode_output_tokens=128,
        expected_prefill_input_tokens=2048,
        expected_prefill_output_tokens=1,
    )

    assert trial["status"] == "invalid"
    assert "decode_request_failed" in trial["invalid_reason"]
    assert [gap["classification"] for gap in trial["chunk_gap_events"]] == [
        "before",
        "crosses",
        "after",
    ]
    request = trial["requests"][0]
    assert (
        request["request_send_epoch_ns"] == client.decode_result.request_send_epoch_ns
    )
    assert (
        request["request_send_monotonic_ns"]
        == client.decode_result.request_send_monotonic_ns
    )
    assert request["failure_observed_epoch_ns"] > request["first_chunk_epoch_ns"]
    assert (
        request["failure_observed_monotonic_ns"] > request["first_chunk_monotonic_ns"]
    )
    assert request["input_tokens"] is None
    assert request["output_tokens"] is None
    assert request["stream_drain_epoch_ns"] is None
    assert request["stream_drain_monotonic_ns"] is None
    assert len(request["chunk_events"]) == 4


@pytest.mark.parametrize("fail_role", ["decode", "prefill"])
def test_run_staged_trial_marks_either_request_failure_invalid(module, fail_role):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient(fail_role=fail_role)

    trial = module.run_staged_trial(
        client=client,
        trial_spec=_trial_spec(module),
        model="fake-model",
        decode_messages=[{"role": "user", "content": "decode"}],
        prefill_messages=[{"role": "user", "content": "prefill"}],
        expected_decode_input_tokens=256,
        expected_decode_output_tokens=128,
        expected_prefill_input_tokens=2048,
        expected_prefill_output_tokens=1,
    )

    assert trial["status"] == "invalid"
    assert f"{fail_role}_request_failed" in trial["invalid_reason"]


def test_run_single_trial_records_lifecycle_and_rejects_mismatched_usage(module):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient(decode_result=_timed_result(output_tokens=127))

    trial = module.run_single_trial(
        client=client,
        trial_spec=_trial_spec(module, condition="decode-only"),
        request_role="decode",
        model="fake-model",
        messages=[{"role": "user", "content": "decode"}],
        expected_input_tokens=256,
        expected_output_tokens=128,
    )

    request = trial["requests"][0]
    assert trial["status"] == "invalid"
    assert "output token mismatch" in trial["invalid_reason"]
    assert request["request_send_monotonic_ns"] == 10
    assert request["first_chunk_monotonic_ns"] == 100
    assert request["stream_drain_monotonic_ns"] == 800
    assert request["output_tokens"] == 127
    assert len(request["chunk_events"]) == 4
    assert client.calls[0]["ignore_eos"] is True
    assert "private generated output" not in json.dumps(trial)


def test_run_single_trial_records_send_and_failure_time_for_http_error(module):
    assert module is not None, "long prefill decode module is missing"
    client = _StagedFakeClient(fail_role="decode_http")

    trial = module.run_single_trial(
        client=client,
        trial_spec=_trial_spec(module, condition="decode-only"),
        request_role="decode",
        model="fake-model",
        messages=[{"role": "user", "content": "decode"}],
        expected_input_tokens=256,
        expected_output_tokens=128,
    )

    request = trial["requests"][0]
    assert (
        request["request_send_epoch_ns"] == client.decode_result.request_send_epoch_ns
    )
    assert (
        request["request_send_monotonic_ns"]
        == client.decode_result.request_send_monotonic_ns
    )
    assert request["failure_observed_epoch_ns"] > request["request_send_epoch_ns"]
    assert (
        request["failure_observed_monotonic_ns"] > request["request_send_monotonic_ns"]
    )
    assert request["first_chunk_epoch_ns"] is None
    assert request["first_chunk_monotonic_ns"] is None
    assert request["stream_drain_epoch_ns"] is None
    assert request["stream_drain_monotonic_ns"] is None
    assert request["input_tokens"] is None
    assert request["output_tokens"] is None
    assert request["http_status"] == 503
    assert request["status"] == "invalid"


def test_run_single_trial_rejects_stream_with_only_empty_chunks(module):
    assert module is not None, "long prefill decode module is missing"
    empty_chunk = StreamChunkEvent(1, "content", 110, 100, 0)
    client = _StagedFakeClient(decode_result=_timed_result(chunk_events=(empty_chunk,)))

    trial = module.run_single_trial(
        client=client,
        trial_spec=_trial_spec(module, condition="decode-only"),
        request_role="decode",
        model="fake-model",
        messages=[{"role": "user", "content": "decode"}],
        expected_input_tokens=256,
        expected_output_tokens=128,
    )

    assert trial["status"] == "invalid"
    assert trial["requests"][0]["status"] == "invalid"
    assert trial["requests"][0]["invalid_reason"] == (
        "request_stream_has_no_nonempty_chunk"
    )


def _pair_trial(*, chunk_size, repetition=1, status="valid", ttft_ms=10.0):
    return {
        "trial_id": f"staged-r{repetition}-c{chunk_size}",
        "condition": "staged-2048",
        "prefill_tokens": 2048,
        "repetition": repetition,
        "chunk_size": chunk_size,
        "status": status,
        "invalid_reason": None if status == "valid" else "decode_request_failed",
        "requests": [{"request_role": "decode", "ttft_ms": ttft_ms, "e2e_ms": 100.0}],
        "chunk_gap_events": [
            {"client_sse_chunk_gap_ms": 2.0, "classification": "before"}
        ],
    }


def test_compare_paired_trials_reports_raw_values_differences_and_range(module):
    assert module is not None, "long prefill decode module is missing"
    trials = [
        _pair_trial(chunk_size=4096, repetition=1, ttft_ms=10.0),
        _pair_trial(chunk_size=512, repetition=1, ttft_ms=14.0),
        _pair_trial(chunk_size=4096, repetition=2, ttft_ms=20.0),
        _pair_trial(chunk_size=512, repetition=2, ttft_ms=21.0),
    ]

    comparison = module.compare_paired_trials(trials)

    first_pair = comparison["pairs"][0]
    assert first_pair["status"] == "valid"
    assert first_pair["raw_values"]["chunk_size_4096"]["decode_ttft_ms"] == 10.0
    assert first_pair["raw_values"]["chunk_size_512"]["decode_ttft_ms"] == 14.0
    assert first_pair["paired_differences"]["decode_ttft_ms"] == 4.0
    summary = next(
        item for item in comparison["summaries"] if item["metric"] == "decode_ttft_ms"
    )
    assert summary["median"] == 2.5
    assert summary["range"] == [1.0, 4.0]


@pytest.mark.parametrize(
    ("trials", "expected_reason"),
    [
        ([_pair_trial(chunk_size=4096)], "missing_chunk_size_512"),
        (
            [
                _pair_trial(chunk_size=4096, status="invalid"),
                _pair_trial(chunk_size=512),
            ],
            "invalid_trial",
        ),
    ],
)
def test_compare_paired_trials_excludes_missing_or_invalid_pairs(
    module, trials, expected_reason
):
    assert module is not None, "long prefill decode module is missing"

    comparison = module.compare_paired_trials(trials)

    assert comparison["pairs"][0]["status"] == "invalid"
    assert expected_reason in comparison["pairs"][0]["invalid_reason"]
    assert comparison["pairs"][0]["paired_differences"] == {}
    assert all(summary["n"] == 0 for summary in comparison["summaries"])


@pytest.fixture
def runner():
    if run_long_prefill_decode is None:

        class _MissingRunner:
            def __getattr__(self, _name):
                raise AssertionError("long prefill runner is missing")

        return _MissingRunner()
    return run_long_prefill_decode


def _npu0_snapshot(**updates):
    values = {
        "NPU Utilization(%)": "0",
        "Aicore Usage Rate(%)": "0",
        "Aivector Usage Rate(%)": "0",
        "HBM Bandwidth Usage Rate(%)": "0",
    }
    values.update(updates)
    return "\n".join(
        ["NPU ID : 0", "HBM Capacity(MB) : 65536", "HBM Usage Rate(%) : 99"]
        + [f"{name} : {value}" for name, value in values.items()]
    )


def _runner_project(tmp_path, runner, monkeypatch, *, model_exists=True):
    project_root = tmp_path / "project"
    project_root.mkdir()
    model_path = project_root / "models" / "Qwen3.5-4B"
    if model_exists:
        model_path.mkdir(parents=True)
    config_path = project_root / "configs" / "qwen35_4b.json"
    config_path.parent.mkdir()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["local_model_path"] = "models/Qwen3.5-4B"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", project_root)
    monkeypatch.setattr(long_prefill_decode, "REPO_ROOT", project_root)
    return project_root, config_path


def _runner_args(config_path, output_root, **updates):
    values = {
        "config": config_path,
        "host": "127.0.0.1",
        "port": 18400,
        "startup_timeout": 0.01,
        "output_root": output_root,
        "dry_run": False,
        "confirm_npu0_idle": True,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_dry_run_prints_exact_alternating_schedule_without_side_effects(
    runner, tmp_path, monkeypatch, capsys
):
    project_root, config_path = _runner_project(
        tmp_path, runner, monkeypatch, model_exists=False
    )
    output_root = project_root / "results" / "raw" / "benchmark" / "long_prefill_decode"
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called")
    )
    monkeypatch.setattr(
        runner, "capture_npu_smi", lambda *_: pytest.fail("npu-smi called")
    )
    monkeypatch.setattr(
        runner, "check_port_available", lambda *_: pytest.fail("port checked")
    )
    monkeypatch.setattr(
        runner, "wait_for_readiness", lambda *_: pytest.fail("HTTP called")
    )
    monkeypatch.setattr(
        runner, "load_local_tokenizer", lambda *_: pytest.fail("tokenizer loaded")
    )

    runner.run(_runner_args(config_path, output_root, dry_run=True))

    output = capsys.readouterr().out
    launch_lines = [line for line in output.splitlines() if line.startswith("round=")]
    assert [line.split("chunk_size=")[1].split()[0] for line in launch_lines] == [
        "4096",
        "512",
        "512",
        "4096",
        "4096",
        "512",
    ]
    conditions = (
        "decode-only,prefill-only-2048,prefill-only-3584,staged-2048,staged-3584"
    )
    assert all(conditions in line for line in launch_lines)
    assert not output_root.exists()


def test_npu0_parser_ignores_hbm_allocation_but_requires_known_zero_metrics(runner):
    parsed = runner.parse_npu0_utilization(_npu0_snapshot())

    assert parsed == {
        "npu_utilization": 0.0,
        "aicore_utilization": 0.0,
        "aivector_utilization": 0.0,
        "hbm_bandwidth_utilization": 0.0,
    }


@pytest.mark.parametrize(
    "snapshot",
    [
        _npu0_snapshot(**{"NPU Utilization(%)": "1"}),
        _npu0_snapshot(**{"Aicore Usage Rate(%)": "1"}),
        _npu0_snapshot(**{"Aivector Usage Rate(%)": "1"}),
        _npu0_snapshot(**{"HBM Bandwidth Usage Rate(%)": "1"}),
    ],
)
def test_npu0_parser_rejects_any_nonzero_compute_or_bandwidth_metric(runner, snapshot):
    with pytest.raises(ValueError, match="busy"):
        runner.require_npu0_idle(snapshot)


@pytest.mark.parametrize(
    "snapshot",
    [
        "NPU ID : 0\nHBM Usage Rate(%) : 4",
        _npu0_snapshot(**{"Aicore Usage Rate(%)": "unknown"}),
        _npu0_snapshot() + "\nMystery Usage Rate(%) : 0",
        _npu0_snapshot().replace("NPU ID : 0", "NPU ID : 1"),
    ],
)
def test_npu0_parser_fails_closed_on_unknown_or_unparseable_snapshot(runner, snapshot):
    with pytest.raises(ValueError):
        runner.parse_npu0_utilization(snapshot)


def test_existing_run_directory_is_rejected_before_hardware_or_service(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    existing = output_root / "fixed-run-id"
    existing.mkdir(parents=True)
    monkeypatch.setattr(runner, "new_run_id", lambda: "fixed-run-id")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner, "capture_npu_smi", lambda *_: pytest.fail("npu-smi called")
    )
    monkeypatch.setattr(
        runner, "check_port_available", lambda *_: pytest.fail("port checked")
    )
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called")
    )

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run(_runner_args(config_path, output_root))


def test_missing_idle_confirmation_stops_before_npu_or_output_creation(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "new-output"
    monkeypatch.setattr(
        runner, "capture_npu_smi", lambda *_: pytest.fail("npu-smi called")
    )
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called")
    )

    with pytest.raises(ValueError, match="confirm-npu0-idle"):
        runner.run(_runner_args(config_path, output_root, confirm_npu0_idle=False))

    assert not output_root.exists()


def test_runner_requests_exclusive_npu_snapshot(runner, tmp_path, monkeypatch):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "new_run_id", lambda: "exclusive-snapshot-run")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    exclusive_values = []

    def capture(_path, *, exclusive=False):
        exclusive_values.append(exclusive)
        return False

    monkeypatch.setattr(runner, "capture_npu_smi", capture)

    with pytest.raises(ValueError, match="NPU0 usage snapshot"):
        runner.run(_runner_args(config_path, output_root))

    assert exclusive_values == [True]


def test_occupied_port_stops_before_launch_or_measurement(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    monkeypatch.setattr(runner, "check_port_available", lambda *_: False)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called")
    )
    monkeypatch.setattr(
        runner, "OpenAIChatClient", lambda *_: pytest.fail("HTTP client created")
    )

    with pytest.raises(OSError, match="port"):
        runner.run(_runner_args(config_path, output_root))


@pytest.mark.parametrize(
    ("busy_listener", "expected_checks", "error_pattern"),
    [
        (
            ("127.0.0.1", 18400),
            [("127.0.0.1", 18400)],
            r"API service.*18400",
        ),
        (
            ("0.0.0.0", 39141),
            [("127.0.0.1", 18400), ("0.0.0.0", 39141)],
            r"Prometheus exporter.*39141",
        ),
        (
            ("127.0.0.1", 39142),
            [
                ("127.0.0.1", 18400),
                ("0.0.0.0", 39141),
                ("127.0.0.1", 39142),
            ],
            r"TokenSpeed metrics.*39142",
        ),
    ],
)
def test_busy_listener_stops_before_service_launch_or_measurement(
    runner, tmp_path, monkeypatch, busy_listener, expected_checks, error_pattern
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    checked = []

    def check_port(host, port):
        checked.append((host, port))
        return (host, port) != busy_listener

    monkeypatch.setattr(runner, "check_port_available", check_port)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen called"),
    )
    monkeypatch.setattr(
        runner,
        "OpenAIChatClient",
        lambda *_: pytest.fail("HTTP client created"),
    )

    with pytest.raises(OSError, match=error_pattern):
        runner.run(_runner_args(config_path, output_root))

    assert checked == expected_checks


@pytest.mark.parametrize(
    ("collision", "api_port", "prometheus_port", "metrics_port"),
    [
        ("API service.*Prometheus exporter", 39141, 39141, 39142),
        ("API service.*TokenSpeed metrics", 39142, 39141, 39142),
        ("Prometheus exporter.*TokenSpeed metrics", 18400, 39141, 39141),
    ],
)
def test_listener_port_collisions_stop_before_service_launch(
    runner,
    tmp_path,
    monkeypatch,
    collision,
    api_port,
    prometheus_port,
    metrics_port,
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["prometheus_port"] = prometheus_port
    config["tokenspeed_metrics_port"] = metrics_port
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    checked = []
    monkeypatch.setattr(
        runner,
        "check_port_available",
        lambda host, port: (checked.append((host, port)), True)[1],
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen called"),
    )
    monkeypatch.setattr(
        runner,
        "OpenAIChatClient",
        lambda *_: pytest.fail("HTTP client created"),
    )

    with pytest.raises(OSError, match=collision):
        runner.run(_runner_args(config_path, output_root, port=api_port))

    assert checked == []


def test_disabled_metrics_listener_is_not_preflighted(runner, tmp_path, monkeypatch):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["enable_metrics"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    checked = []
    monkeypatch.setattr(
        runner,
        "check_port_available",
        lambda host, port: (checked.append((host, port)), True)[1],
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after service launch")
        ),
    )

    with pytest.raises(RuntimeError, match="stop after service launch"):
        runner.run(_runner_args(config_path, output_root))

    assert checked == [("127.0.0.1", 18400), ("0.0.0.0", 39141)]


@pytest.mark.parametrize(
    ("capture_ok", "snapshot"),
    [
        (False, ""),
        (True, _npu0_snapshot(**{"NPU Utilization(%)": "2"})),
        (True, "NPU ID : 0\nno utilization fields"),
    ],
)
def test_unreadable_busy_or_unknown_npu_snapshot_stops_before_service(
    runner, tmp_path, monkeypatch, capture_ok, snapshot
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())

    def capture(path, **_kwargs):
        if capture_ok:
            path.write_text(snapshot, encoding="utf-8")
        return capture_ok

    monkeypatch.setattr(runner, "capture_npu_smi", capture)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called")
    )
    monkeypatch.setattr(
        runner, "OpenAIChatClient", lambda *_: pytest.fail("HTTP client created")
    )

    with pytest.raises(ValueError):
        runner.run(_runner_args(config_path, output_root))


def test_startup_timeout_cleans_up_only_owned_process_before_measurements(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())

    class _Process:
        returncode = None

        def __init__(self):
            self.terminated = 0
            self.killed = 0
            self.waited = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1

        def wait(self, timeout=None):
            self.waited += 1
            if timeout is not None:
                raise runner.subprocess.TimeoutExpired("service", timeout)

    owned = _Process()
    foreign = _Process()
    monkeypatch.setattr(runner, "new_run_id", lambda: "timeout-run")
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    monkeypatch.setattr(runner, "check_port_available", lambda *_: True)
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: owned)
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no readiness")),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        runner, "OpenAIChatClient", lambda *_: pytest.fail("measurement client created")
    )

    with pytest.raises(TimeoutError):
        runner.run(_runner_args(config_path, output_root, startup_timeout=0.01))

    assert (owned.terminated, owned.killed, owned.waited) == (1, 1, 2)
    assert (foreign.terminated, foreign.killed, foreign.waited) == (0, 0, 0)


def test_service_log_suppresses_child_output_and_records_safe_status(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    process_streams = []

    class _Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            self.returncode = -15

    process = _Process()

    def popen(_command, **kwargs):
        stdout = kwargs["stdout"]
        stderr = kwargs["stderr"]
        process_streams.append((stdout, stderr))
        if hasattr(stdout, "write"):
            stdout.write("synthetic-stdout-secret\n")
        if stderr == runner.subprocess.STDOUT and hasattr(stdout, "write"):
            stdout.write("synthetic-stderr-secret\n")
        elif hasattr(stderr, "write"):
            stderr.write("synthetic-stderr-secret\n")
        return process

    monkeypatch.setattr(runner, "new_run_id", lambda: "safe-service-log-run")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "capture_npu_smi",
        lambda path, **_kwargs: (path.write_text(_npu0_snapshot()), True)[1],
    )
    monkeypatch.setattr(runner, "check_port_available", lambda *_: True)
    monkeypatch.setattr(runner, "build_serve_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "wait_for_readiness", lambda *_args: None)
    monkeypatch.setattr(runner, "OpenAIChatClient", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "_run_warmup",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after startup")),
    )

    with pytest.raises(RuntimeError, match="stop after startup"):
        runner.run(_runner_args(config_path, output_root))

    service_log = (
        output_root / "safe-service-log-run" / "service-round-01-chunk-4096.log"
    ).read_text(encoding="utf-8")

    assert process_streams == [(runner.subprocess.DEVNULL, runner.subprocess.DEVNULL)]
    assert service_log.splitlines() == [
        "startup_status=ready",
        "exit_status=-15",
    ]
    assert "synthetic-stdout-secret" not in service_log
    assert "synthetic-stderr-secret" not in service_log


@pytest.mark.parametrize(
    ("host", "expected_client_host", "expected_authority"),
    [
        ("192.0.2.15", "192.0.2.15", "192.0.2.15:18400"),
        ("2001:db8::15", "2001:db8::15", "[2001:db8::15]:18400"),
        ("0.0.0.0", "127.0.0.1", "127.0.0.1:18400"),
        ("::", "::1", "[::1]:18400"),
    ],
)
def test_service_urls_use_custom_host_and_format_ipv6(
    runner, host, expected_client_host, expected_authority
):
    assert runner.client_host(host) == expected_client_host
    assert runner.service_base_url(host, 18400) == f"http://{expected_authority}/v1"
    assert runner.readiness_url(host, 18400) == f"http://{expected_authority}/readiness"


def test_readiness_http_receives_bracketed_custom_ipv6_url(runner, monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    urls = []
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda url, **_kwargs: (urls.append(url), _Response())[1],
    )
    process = SimpleNamespace(poll=lambda: None, returncode=None)

    runner.wait_for_readiness(process, "2001:db8::15", 18400, 1)

    assert urls == ["http://[2001:db8::15]:18400/readiness"]


@pytest.mark.parametrize(
    ("host", "expected_url"),
    [
        ("192.0.2.15", "http://192.0.2.15:18400/v1"),
        ("2001:db8::15", "http://[2001:db8::15]:18400/v1"),
    ],
)
def test_measurement_client_uses_selected_host(
    runner, tmp_path, monkeypatch, host, expected_url
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"
    monkeypatch.setattr(runner, "new_run_id", lambda: "client-host-run")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())

    def capture(path, **_kwargs):
        path.write_text(_npu0_snapshot(), encoding="utf-8")
        return True

    class _Process:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    monkeypatch.setattr(runner, "capture_npu_smi", capture)
    monkeypatch.setattr(runner, "check_port_available", lambda *_: True)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _Process())
    monkeypatch.setattr(runner, "wait_for_readiness", lambda *_: None)
    client_urls = []
    monkeypatch.setattr(
        runner,
        "OpenAIChatClient",
        lambda url: (client_urls.append(url), object())[1],
    )
    monkeypatch.setattr(
        runner,
        "_run_warmup",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after client")),
    )

    with pytest.raises(RuntimeError, match="stop after client"):
        runner.run(_runner_args(config_path, output_root, host=host))

    assert client_urls == [expected_url]


def test_run_stops_before_formal_trials_when_decode_warmup_is_not_128(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    output_root = project_root / "results"

    class _Process:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def capture(path, **_kwargs):
        path.write_text(_npu0_snapshot(), encoding="utf-8")
        return True

    monkeypatch.setattr(runner, "new_run_id", lambda: "short-warmup-run")
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda *_: object())
    monkeypatch.setattr(runner, "capture_npu_smi", capture)
    monkeypatch.setattr(runner, "check_port_available", lambda *_: True)
    monkeypatch.setattr(runner, "build_serve_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *_args, **_kwargs: _Process()
    )
    monkeypatch.setattr(runner, "wait_for_readiness", lambda *_args: None)
    monkeypatch.setattr(runner, "OpenAIChatClient", lambda *_: object())
    warmup_conditions = []
    monkeypatch.setattr(
        runner,
        "_run_warmup",
        lambda **kwargs: (
            warmup_conditions.append(kwargs["condition"]),
            {"status": "valid", "requests": [{"output_tokens": 127}]},
        )[1],
    )
    monkeypatch.setattr(
        runner,
        "_run_one_trial",
        lambda **_kwargs: pytest.fail("formal trial started after a short warmup"),
    )

    with pytest.raises(RuntimeError, match="exactly 128 output tokens"):
        runner.run(_runner_args(config_path, output_root))

    assert warmup_conditions == ["warmup-decode"]


def test_dry_run_rejects_config_path_outside_project(runner, tmp_path, monkeypatch):
    project_root, _config_path = _runner_project(tmp_path, runner, monkeypatch)
    outside_config = tmp_path / "outside.json"
    outside_config.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="config path.*inside"):
        runner.run(
            _runner_args(
                outside_config,
                project_root / "results",
                dry_run=True,
            )
        )


def test_dry_run_rejects_model_path_outside_project(runner, tmp_path, monkeypatch):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["local_model_path"] = str(tmp_path / "outside-model")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="local_model_path.*inside"):
        runner.run(
            _runner_args(
                config_path,
                project_root / "results",
                dry_run=True,
            )
        )


def test_dry_run_rejects_output_root_outside_project(runner, tmp_path, monkeypatch):
    project_root, config_path = _runner_project(tmp_path, runner, monkeypatch)

    with pytest.raises(ValueError, match="output root.*inside"):
        runner.run(
            _runner_args(
                config_path,
                tmp_path / "outside-output",
                dry_run=True,
            )
        )


def test_model_symlink_setup_does_not_skip_missing_parent(
    runner, tmp_path, monkeypatch
):
    def missing_parent(self, target, target_is_directory=False):
        raise FileNotFoundError(errno.ENOENT, "synthetic missing symlink parent")

    monkeypatch.setattr(Path, "symlink_to", missing_parent)
    with pytest.raises(FileNotFoundError):
        try:
            test_dry_run_rejects_model_symlink_escape_when_supported(
                runner, tmp_path, monkeypatch
            )
        except pytest.skip.Exception:
            pytest.fail("missing symlink parent must fail rather than skip")


def test_dry_run_rejects_model_symlink_escape_when_supported(
    runner, tmp_path, monkeypatch
):
    project_root, config_path = _runner_project(
        tmp_path, runner, monkeypatch, model_exists=False
    )
    outside_model = tmp_path / "outside-model"
    outside_model.mkdir()
    model_link = project_root / "models" / "outside-link"
    model_link.parent.mkdir()
    try:
        model_link.symlink_to(outside_model, target_is_directory=True)
    except NotImplementedError as error:
        pytest.skip(f"directory symlinks are unavailable in this environment: {error}")
    except OSError as error:
        if (
            error.errno not in (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP)
            and getattr(error, "winerror", None) != 1314
        ):
            raise
        pytest.skip(f"directory symlinks are unavailable in this environment: {error}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["local_model_path"] = "models/outside-link"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="local_model_path.*inside"):
        runner.run(
            _runner_args(
                config_path,
                project_root / "results",
                dry_run=True,
            )
        )


def test_config_sanitizer_redacts_custom_key_fields_but_keeps_known_config(
    runner,
):
    sanitized = runner._sanitize_persisted_config(
        {
            "key": "synthetic-key-value",
            "server_key": "synthetic-server-key-value",
            "rapid_key": "synthetic-rapid-key-value",
            "max_tokens": 64,
            "tokenizer": "local-tokenizer",
            "model_id": "Qwen/Qwen3.5-4B",
        }
    )

    assert sanitized == {
        "key": "<REDACTED>",
        "server_key": "<REDACTED>",
        "rapid_key": "<REDACTED>",
        "max_tokens": 64,
        "tokenizer": "local-tokenizer",
        "model_id": "Qwen/Qwen3.5-4B",
    }


def test_config_snapshots_redact_nested_credentials_but_keep_service_config(
    runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "synthetic-project"
    model_path = project_root / "models" / "synthetic-model"
    config_path = project_root / "configs" / "synthetic.json"
    model_path.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("synthetic test config", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", project_root)

    secret_values = [f"FAKE_CREDENTIAL_SENTINEL_{index}" for index in range(34)]
    synthetic_config = {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "synthetic-revision",
        "served_model_name": "synthetic-model",
        "local_model_path": "models/synthetic-model",
        "enable_metrics": True,
        "prometheus_port": 39141,
        "tokenspeed_metrics_port": 39142,
        "dtype": "bfloat16",
        "tp_size": 1,
        "max_model_len": 4096,
        "chunked_prefill_size": 4096,
        "max_tokens": 64,
        "max_new_tokens": 96,
        "output_tokens": 128,
        "prompt_tokens": 192,
        "num_tokens": 320,
        "max_num_batched_tokens": 512,
        "tokenizer": "synthetic-tokenizer-config",
        "extension_options": {
            "key": secret_values[31],
            "server_key": secret_values[32],
            "rapid_key": secret_values[33],
            "auth": secret_values[0],
            "Authorization": secret_values[1],
            "API-Key": secret_values[2],
            "api_key": secret_values[3],
            "nested": [
                {
                    "accessToken": secret_values[4],
                    "refresh_token": secret_values[5],
                    "Bearer Token": secret_values[6],
                },
                {
                    "dbPassword": secret_values[7],
                    "clientSecret": secret_values[8],
                    "credential_json": secret_values[9],
                    "sshPrivateKey": secret_values[10],
                },
                {
                    "apikey": secret_values[11],
                    "accesstoken": secret_values[12],
                    "refreshtoken": secret_values[13],
                    "bearertoken": secret_values[14],
                    "privatekey": secret_values[15],
                    "dbpassword": secret_values[16],
                    "credentialblob": secret_values[17],
                    "authorizationheader": secret_values[18],
                },
                {
                    "authtoken": secret_values[19],
                    "authheader": secret_values[20],
                    "apitoken": secret_values[21],
                    "hf_token": secret_values[22],
                    "huggingFaceToken": secret_values[23],
                    "token": secret_values[24],
                    "sessiontoken": secret_values[25],
                    "clienttoken": secret_values[26],
                    "idtoken": secret_values[27],
                    "session_tokens": secret_values[28],
                    "authtokens": secret_values[29],
                    "api_tokens": secret_values[30],
                },
            ],
        },
    }
    monkeypatch.setattr(runner, "load_model_config", lambda _path: synthetic_config)
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda _config: object())
    monkeypatch.setattr(runner, "new_run_id", lambda: "synthetic-redaction-run")

    def capture_npu_snapshot(path, *, exclusive=False):
        if not exclusive:
            pytest.fail("NPU snapshot was not requested exclusively")
        path.write_text(_npu0_snapshot(), encoding="utf-8")
        return True

    class _Process:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    service_configs = []

    def build_command(config, **_kwargs):
        service_configs.append(config)
        return ["synthetic-mock-command"]

    monkeypatch.setattr(runner, "capture_npu_smi", capture_npu_snapshot)
    monkeypatch.setattr(runner, "check_port_available", lambda *_: True)
    monkeypatch.setattr(runner, "build_serve_command", build_command)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _Process())
    monkeypatch.setattr(runner, "wait_for_readiness", lambda *_: None)
    monkeypatch.setattr(runner, "OpenAIChatClient", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "_prompt",
        lambda *_args: [{"role": "user", "content": "synthetic test prompt"}],
    )
    monkeypatch.setattr(
        runner,
        "_run_warmup",
        lambda **kwargs: {
            "status": "valid",
            "requests": [
                {"output_tokens": 128 if kwargs["condition"] == "warmup-decode" else 1}
            ],
        },
    )

    def make_trial(trial_spec, requests):
        return {
            "trial_id": trial_spec.trial_id,
            "condition": trial_spec.condition,
            "status": "valid",
            "invalid_reason": None,
            "requests": requests,
            "chunk_gap_events": [],
            "requests_overlap": False,
        }

    def mock_single_trial(
        *,
        trial_spec,
        request_role,
        expected_input_tokens,
        expected_output_tokens,
        **_kwargs,
    ):
        return make_trial(
            trial_spec,
            [
                {
                    "request_role": request_role,
                    "expected_input_tokens": expected_input_tokens,
                    "expected_output_tokens": expected_output_tokens,
                    "input_tokens": expected_input_tokens,
                    "output_tokens": expected_output_tokens,
                    "status": "valid",
                }
            ],
        )

    def mock_staged_trial(
        *,
        trial_spec,
        expected_decode_input_tokens,
        expected_decode_output_tokens,
        expected_prefill_input_tokens,
        expected_prefill_output_tokens,
        **_kwargs,
    ):
        return make_trial(
            trial_spec,
            [
                {
                    "request_role": "decode",
                    "expected_input_tokens": expected_decode_input_tokens,
                    "expected_output_tokens": expected_decode_output_tokens,
                    "input_tokens": expected_decode_input_tokens,
                    "output_tokens": expected_decode_output_tokens,
                    "status": "valid",
                },
                {
                    "request_role": "prefill",
                    "expected_input_tokens": expected_prefill_input_tokens,
                    "expected_output_tokens": expected_prefill_output_tokens,
                    "input_tokens": expected_prefill_input_tokens,
                    "output_tokens": expected_prefill_output_tokens,
                    "status": "valid",
                },
            ],
        )

    monkeypatch.setattr(runner.workload, "run_single_trial", mock_single_trial)
    monkeypatch.setattr(runner.workload, "run_staged_trial", mock_staged_trial)

    run_dir = runner.run(_runner_args(config_path, project_root / "results"))

    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    source_root = Path(runner.__file__).resolve().parents[1]
    source_paths = (
        "scripts/run_long_prefill_decode.py",
        "benchmarks/course_project/long_prefill_decode.py",
        "benchmarks/course_project/core.py",
        "benchmarks/course_project/runner.py",
        "benchmarks/course_project/telemetry.py",
    )
    assert run_record["benchmark_source_sha256"] == {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in source_paths
    }

    snapshot_paths = sorted(run_dir.glob("config-*.json"))
    if len(snapshot_paths) != 6:
        pytest.fail("expected one config snapshot for each scheduled launch")
    request_records = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    trial_records = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not request_records or not trial_records:
        pytest.fail("synthetic run did not write non-empty request and trial records")
    persisted_files = [
        path.read_text(encoding="utf-8")
        for path in run_dir.rglob("*")
        if path.is_file()
    ]
    if any(
        secret in content for secret in secret_values for content in persisted_files
    ):
        pytest.fail("synthetic credential value was persisted to raw output")
    if any(
        content.count("<REDACTED>") < len(secret_values)
        for content in persisted_files
        if '"config"' in content
    ):
        pytest.fail("config snapshot did not redact all nested synthetic credentials")
    if any(
        '"max_tokens": 64' not in content
        or '"max_new_tokens": 96' not in content
        or '"output_tokens": 128' not in content
        or '"prompt_tokens": 192' not in content
        or '"num_tokens": 320' not in content
        or '"max_num_batched_tokens": 512' not in content
        or '"tokenizer": "synthetic-tokenizer-config"' not in content
        or '"key": "<REDACTED>"' not in content
        or '"server_key": "<REDACTED>"' not in content
        or '"rapid_key": "<REDACTED>"' not in content
        for content in persisted_files
        if '"config"' in content
    ):
        pytest.fail(
            "snapshot did not preserve known config fields and redact key values"
        )
    serialized_service_configs = [json.dumps(config) for config in service_configs]
    if len(serialized_service_configs) != 6 or any(
        any(secret not in content for secret in secret_values)
        for content in serialized_service_configs
    ):
        pytest.fail("service config was altered instead of snapshot-only redaction")
