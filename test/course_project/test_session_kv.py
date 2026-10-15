"""Focused Session KV API contracts that run without the model runtime deps."""

from __future__ import annotations

import ast
import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


class _AbortReq(SimpleNamespace):
    pass


def _load_method(
    relative_path: str, class_name: str | None, method_name: str, **globals_
):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    if class_name is None:
        nodes = tree.body
    else:
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = cls.body
    method = copy.deepcopy(
        next(
            node
            for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
    )
    method.decorator_list = []
    method.returns = None
    for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    isolated = ast.Module(body=[method], type_ignores=[])
    namespace = dict(globals_)
    exec(compile(ast.fix_missing_locations(isolated), relative_path, "exec"), namespace)
    return namespace[method_name]


def _find_method(relative_path: str, class_name: str, method_name: str):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def test_request_spec_and_scheduler_bind_session_lifecycle():
    binding = (ROOT / "tokenspeed-scheduler/bindings/python_module.cpp").read_text(
        encoding="utf-8"
    )
    assert '.def_rw("session_id", &tokenspeed::RequestSpec::session_id)' in binding
    for python_name, cpp_name in (
        ("open_session", "OpenSession"),
        ("close_session", "CloseSession"),
        ("has_session", "HasSession"),
        ("can_submit_session_turn", "CanSubmitSessionTurn"),
        ("can_defer_session_turn", "CanDeferSessionTurn"),
        ("session_info_for_session", "SessionInfoForSession"),
    ):
        assert f'.def("{python_name}", &tokenspeed::Scheduler::{cpp_name}' in binding


def test_async_open_registers_future_before_send_and_returns_acknowledged_id():
    class OpenInput:
        def __init__(self, *, capacity_of_str_len, session_id):
            self.capacity_of_str_len = capacity_of_str_len
            self.session_id = session_id

    class OpenOutput:
        def __init__(self, session_id, success):
            self.session_id = session_id
            self.success = success

    open_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "open_session",
        asyncio=asyncio,
        uuid=SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="stable-session")),
        OpenSessionReqInput=OpenInput,
    )
    output_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "_handle_open_session_req_output",
    )

    async def run():
        loop = asyncio.get_running_loop()
        client = SimpleNamespace()
        frontend = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            session_futures={},
            close_session_futures={},
            engine_core_client=client,
        )
        frontend._handle_open_session_req_output = lambda result: output_method(
            frontend, result
        )

        class Sender:
            def send_pyobj(self, request):
                assert request.session_id in frontend.session_futures
                frontend._handle_open_session_req_output(
                    OpenOutput(request.session_id, True)
                )

        client.send_to_scheduler = Sender()
        result = await open_method(
            frontend,
            OpenInput(capacity_of_str_len=13, session_id=None),
        )
        assert result == "stable-session"
        assert frontend.session_futures == {}

    asyncio.run(run())


def test_async_open_raises_on_scheduler_rejection_and_clears_pending_future():
    class OpenInput:
        def __init__(self, *, capacity_of_str_len, session_id):
            self.capacity_of_str_len = capacity_of_str_len
            self.session_id = session_id

    class OpenOutput:
        session_id = "duplicate"
        success = False
        message = "session already exists"

    open_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "open_session",
        asyncio=asyncio,
        uuid=SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="unused")),
        OpenSessionReqInput=OpenInput,
    )
    output_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "_handle_open_session_req_output",
    )

    async def run():
        client = SimpleNamespace()
        frontend = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            session_futures={},
            close_session_futures={},
            engine_core_client=client,
        )
        frontend._handle_open_session_req_output = lambda result: output_method(
            frontend, result
        )

        class Sender:
            def send_pyobj(self, request):
                frontend._handle_open_session_req_output(OpenOutput())

        client.send_to_scheduler = Sender()
        with pytest.raises(ValueError, match="session already exists"):
            await open_method(
                frontend,
                OpenInput(capacity_of_str_len=9, session_id="duplicate"),
            )
        assert frontend.session_futures == {}

    asyncio.run(run())


def test_async_close_waits_for_scheduler_ack_and_reports_rejection():
    class CloseInput:
        def __init__(self, *, session_id):
            self.session_id = session_id

    class CloseOutput:
        def __init__(self, session_id, success, message=""):
            self.session_id = session_id
            self.success = success
            self.message = message

    close_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "close_session",
        asyncio=asyncio,
        CloseSessionReqInput=CloseInput,
    )
    output_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "_handle_close_session_req_output",
    )

    async def run():
        client = SimpleNamespace()
        frontend = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            session_futures={},
            close_session_futures={},
            engine_core_client=client,
        )
        frontend._handle_close_session_req_output = lambda result: output_method(
            frontend, result
        )

        class Sender:
            def send_pyobj(self, request):
                pending = frontend.close_session_futures.get(request.session_id)
                assert pending is not None
                asyncio.get_running_loop().call_soon(
                    frontend._handle_close_session_req_output,
                    CloseOutput(request.session_id, True),
                )

        client.send_to_scheduler = Sender()
        result = await close_method(frontend, CloseInput(session_id="stable-session"))
        assert result is None
        assert frontend.close_session_futures == {}

        class RejectingSender:
            def send_pyobj(self, request):
                pending = frontend.close_session_futures.get(request.session_id)
                assert pending is not None
                asyncio.get_running_loop().call_soon(
                    frontend._handle_close_session_req_output,
                    CloseOutput(
                        request.session_id,
                        False,
                        "session is active",
                    ),
                )

        client.send_to_scheduler = RejectingSender()
        with pytest.raises(ValueError, match="session is active"):
            await close_method(frontend, CloseInput(session_id="stable-session"))

    asyncio.run(run())


def test_async_close_cancel_keeps_pending_mapping_until_ack_before_retry():
    class CloseInput:
        def __init__(self, *, session_id):
            self.session_id = session_id

    class CloseOutput:
        def __init__(self, session_id, success, message=""):
            self.session_id = session_id
            self.success = success
            self.message = message

    close_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "close_session",
        asyncio=asyncio,
        CloseSessionReqInput=CloseInput,
    )
    output_method = _load_method(
        "python/tokenspeed/runtime/engine/async_llm.py",
        "AsyncLLM",
        "_handle_close_session_req_output",
    )

    async def run():
        requests = []
        frontend = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            session_futures={},
            close_session_futures={},
            engine_core_client=SimpleNamespace(
                send_to_scheduler=SimpleNamespace(send_pyobj=requests.append)
            ),
        )
        frontend._handle_close_session_req_output = lambda result: output_method(
            frontend, result
        )
        first_attempt = asyncio.create_task(
            close_method(frontend, CloseInput(session_id="stable-session"))
        )
        await asyncio.sleep(0)
        first_future = frontend.close_session_futures["stable-session"]
        first_attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_attempt

        assert frontend.close_session_futures["stable-session"] is first_future
        assert not first_future.cancelled()
        with pytest.raises(ValueError, match="pending operation"):
            await close_method(frontend, CloseInput(session_id="stable-session"))
        assert len(requests) == 1

        frontend._handle_close_session_req_output(CloseOutput("stable-session", True))
        assert frontend.close_session_futures == {}

        retry = asyncio.create_task(
            close_method(frontend, CloseInput(session_id="stable-session"))
        )
        await asyncio.sleep(0)
        assert len(requests) == 2
        frontend._handle_close_session_req_output(CloseOutput("stable-session", True))
        await retry
        assert frontend.close_session_futures == {}

    asyncio.run(run())


def test_close_session_ack_handler_is_registered():
    async_llm = (ROOT / "python/tokenspeed/runtime/engine/async_llm.py").read_text(
        encoding="utf-8"
    )
    assert "self._handle_close_session_req_output" in async_llm


def test_engine_generate_passes_suffix_and_session_id_without_changing_stateless_path():
    class GenerateInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    generate = _load_method(
        "python/tokenspeed/runtime/entrypoints/engine.py",
        "Engine",
        "generate",
        GenerateReqInput=GenerateInput,
    )

    class SyncLLM:
        def generate(self, obj):
            return obj

        def generate_stream(self, obj):
            return obj

    engine = SimpleNamespace(llm=SyncLLM())
    turn = generate(engine, input_ids=[8, 9], session_id="stable-session")
    assert turn.input_ids == [8, 9]
    assert turn.session_params == {"id": "stable-session"}

    ordinary = generate(engine, input_ids=[1, 2])
    assert ordinary.input_ids == [1, 2]
    assert ordinary.session_params is None


def test_engine_async_generate_passes_suffix_and_session_id():
    class GenerateInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async_generate = _load_method(
        "python/tokenspeed/runtime/entrypoints/engine.py",
        "Engine",
        "async_generate",
        GenerateReqInput=GenerateInput,
        asyncio=asyncio,
    )

    class TokenizerManager:
        async def generate_request(self, obj):
            yield obj

    async def run():
        engine = SimpleNamespace(tokenizer_manager=TokenizerManager())
        result = await async_generate(
            engine,
            input_ids=[8, 9],
            session_id="stable-session",
        )
        assert result.input_ids == [8, 9]
        assert result.session_params == {"id": "stable-session"}

    asyncio.run(run())


def test_sync_engine_open_and_close_session_use_acknowledgement_facade():
    open_session = _load_method(
        "python/tokenspeed/runtime/entrypoints/engine.py", "Engine", "open_session"
    )
    close_session = _load_method(
        "python/tokenspeed/runtime/entrypoints/engine.py", "Engine", "close_session"
    )
    calls = []
    llm = SimpleNamespace(
        open_session=lambda session_id: calls.append(("open", session_id))
        or "stable-session",
        close_session=lambda session_id: calls.append(("close", session_id)),
    )
    engine = SimpleNamespace(llm=llm)
    assert open_session(engine) == "stable-session"
    assert close_session(engine, "stable-session") is None
    assert calls == [("open", None), ("close", "stable-session")]


def test_scheduler_spec_receives_session_id_and_unmodified_suffix():
    class RequestSpec:
        def __init__(self):
            self.request_id = ""
            self.tokens = []
            self.max_new_tokens = 0
            self.session_id = ""

    make_spec = _load_method(
        "python/tokenspeed/runtime/engine/scheduler_utils.py",
        None,
        "make_spec",
        RequestSpec=RequestSpec,
    )

    spec = make_spec("turn-2", [8, 9], 4, session_id="stable-session")
    assert spec.tokens == [8, 9]
    assert spec.session_id == "stable-session"


@pytest.mark.parametrize(
    "session_params",
    ({"id": "stable-session"}, [{"id": "stable-session"}]),
)
def test_generate_request_item_preserves_session_params(session_params):
    class SubRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    get_item = _load_method(
        "python/tokenspeed/runtime/engine/io_struct.py",
        "GenerateReqInput",
        "__getitem__",
        GenerateReqInput=SubRequest,
    )
    request = SimpleNamespace(
        text=["prompt"],
        input_ids=[[8, 9]],
        precomputed_multimodal_inputs=None,
        input_multi_ids=None,
        input_embeds=None,
        input_extra_infos=None,
        sampling_params=[{}],
        user_rid=[None],
        return_logprob=[False],
        logprob_start_len=[-1],
        top_logprobs_num=[0],
        token_ids_logprob=[None],
        return_text_in_logprobs=False,
        logprob_format=[None],
        stream=False,
        log_metrics=True,
        session_params=session_params,
        custom_logit_processor=None,
        return_hidden_states=False,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        data_parallel_rank=None,
        rid=["turn-rid"],
    )

    subrequest = get_item(request, 0)

    assert subrequest.session_params == {"id": "stable-session"}


def _request_handler_method():
    class RequestState:
        @classmethod
        def from_recv_req(cls, recv_req, tokenizer, eos_token_ids):
            return SimpleNamespace(
                finished_reason=None,
                sampling_params=recv_req.sampling_params,
                prompt_input_ids=recv_req.input_ids or [],
                session_id=(
                    recv_req.session_params.id
                    if recv_req.session_params is not None
                    else None
                ),
            )

    class BootstrapInfo:
        def __init__(self, host, port, room):
            self.host = host
            self.port = port
            self.room = room

    return _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "handle_generate_request",
        RequestState=RequestState,
        BootstrapInfo=BootstrapInfo,
        make_spec=lambda **kwargs: SimpleNamespace(
            **{
                **kwargs,
                "request_id": kwargs["rid"],
                "session_id": kwargs.get("session_id") or "",
            }
        ),
        FINISH_ABORT=lambda message: f"ABORT: {message}",
    )


def _tokenized_request(*, session_id=None, sampling_params=None, **overrides):
    values = {
        "bootstrap_port": None,
        "bootstrap_host": None,
        "bootstrap_room": None,
        "rid": "turn-rid",
        "input_ids": [8, 9],
        "input_embeds": None,
        "input_multi_ids": None,
        "multimodal_inputs": None,
        "validation_error": None,
        "session_params": (
            SimpleNamespace(id=session_id, rid=None, offset=None, replace=None)
            if session_id is not None
            else None
        ),
        "sampling_params": sampling_params
        or SimpleNamespace(
            max_new_tokens=3,
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_request_handler_routes_session_suffix_and_preserves_stateless_requests():
    handle = _request_handler_method()
    scheduler = SimpleNamespace(
        has_session=lambda session_id: session_id == "stable-session",
        can_submit_session_turn=lambda session_id: True,
    )
    handler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=scheduler,
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )
    pending = set()
    spec, state, _ = handle(
        handler,
        _tokenized_request(session_id="stable-session"),
        pending_session_ids=pending,
    )
    assert spec.tokens == [8, 9]
    assert spec.session_id == "stable-session"
    assert state.session_id == "stable-session"
    assert state.finished_reason is None

    stateless_spec, stateless_state, _ = handle(
        handler,
        _tokenized_request(),
        pending_session_ids=set(),
    )
    assert stateless_spec.session_id == ""
    assert stateless_state.finished_reason is None


def test_request_handler_accepts_session_turn_during_overlap_drain():
    handle = _request_handler_method()
    scheduler = SimpleNamespace(
        has_session=lambda session_id: session_id == "stable-session",
        can_submit_session_turn=lambda _: False,
        can_defer_session_turn=lambda _: True,
    )
    handler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=scheduler,
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )

    spec, state, _ = handle(
        handler,
        _tokenized_request(session_id="stable-session"),
        pending_session_ids=set(),
    )

    assert spec.session_id == "stable-session"
    assert state.session_id == "stable-session"
    assert state.finished_reason is None


def test_request_handler_buffers_quiescing_session_turn_until_submittable():
    class TokenizedGenerateReqInput(SimpleNamespace):
        pass

    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
        TokenizedGenerateReqInput=TokenizedGenerateReqInput,
        AbortReq=_AbortReq,
    )
    handle_generate = _request_handler_method()
    scheduler_state = {"can_submit": False}
    scheduler = SimpleNamespace(
        has_session=lambda _: True,
        can_submit_session_turn=lambda _: scheduler_state["can_submit"],
        can_defer_session_turn=lambda _: not scheduler_state["can_submit"],
    )
    handler = SimpleNamespace(
        pending_session_turns={},
        pause_controller=SimpleNamespace(buffered_specs=[]),
        handle_generate_request=lambda recv_req, pending_session_ids: handle_generate(
            handler,
            recv_req,
            pending_session_ids=pending_session_ids,
        ),
        _rendezvous_replica_flush=lambda **kwargs: True,
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=scheduler,
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )
    recv_req = TokenizedGenerateReqInput(
        **_tokenized_request(session_id="stable-session").__dict__
    )

    first_specs, first_states, _, _ = process_requests(handler, [recv_req])
    assert first_specs == []
    assert first_states == []
    assert list(handler.pending_session_turns) == ["stable-session"]

    scheduler_state["can_submit"] = True
    ready_specs, ready_states, _, _ = process_requests(handler, [])
    assert [spec.session_id for spec in ready_specs] == ["stable-session"]
    assert ready_states[0].finished_reason is None
    assert handler.pending_session_turns == {}


@pytest.mark.parametrize("can_submit", [False, True])
def test_abort_discards_deferred_session_turn(can_submit):
    class TokenizedGenerateReqInput(SimpleNamespace):
        pass

    class ProfileReq:
        pass

    class OpenSessionReqInput:
        pass

    class CloseSessionReqInput:
        pass

    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
        TokenizedGenerateReqInput=TokenizedGenerateReqInput,
        ProfileReq=ProfileReq,
        OpenSessionReqInput=OpenSessionReqInput,
        CloseSessionReqInput=CloseSessionReqInput,
        AbortReq=_AbortReq,
        logger=SimpleNamespace(debug=lambda *_: None),
    )
    scheduler = SimpleNamespace(
        can_submit_session_turn=lambda _: can_submit,
    )
    spec = SimpleNamespace(
        request_id="queued-rid",
        session_id="stable-session",
    )
    handler = SimpleNamespace(
        pending_session_turns={
            "stable-session": (spec, SimpleNamespace(), SimpleNamespace())
        },
        pause_controller=SimpleNamespace(buffered_specs=[]),
        session_scheduler=scheduler,
        _rendezvous_replica_flush=lambda **kwargs: True,
    )

    specs, states, bootstraps, abort_rids = process_requests(
        handler, [_AbortReq(rid="queued-rid")]
    )

    assert specs == []
    assert states == []
    assert bootstraps == []
    assert abort_rids == []
    assert handler.pending_session_turns == {}


def test_ready_deferred_session_turn_stays_queued_while_paused():
    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
    )
    spec = SimpleNamespace(request_id="queued-rid", session_id="stable-session")
    pending = (spec, SimpleNamespace(), SimpleNamespace())
    handler = SimpleNamespace(
        pending_session_turns={"stable-session": pending},
        pause_controller=SimpleNamespace(buffered_specs=[], admit_blocked=True),
        session_scheduler=SimpleNamespace(can_submit_session_turn=lambda _: True),
        _rendezvous_replica_flush=lambda **kwargs: True,
    )

    specs, states, bootstraps, abort_rids = process_requests(handler, [])

    assert specs == []
    assert states == []
    assert bootstraps == []
    assert abort_rids == []
    assert handler.pending_session_turns == {"stable-session": pending}


@pytest.mark.parametrize("abort_first", [False, True])
def test_same_batch_abort_discards_new_session_turn(abort_first):
    class TokenizedGenerateReqInput(SimpleNamespace):
        pass

    class ProfileReq:
        pass

    class OpenSessionReqInput:
        pass

    class CloseSessionReqInput:
        pass

    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
        TokenizedGenerateReqInput=TokenizedGenerateReqInput,
        ProfileReq=ProfileReq,
        OpenSessionReqInput=OpenSessionReqInput,
        CloseSessionReqInput=CloseSessionReqInput,
        AbortReq=_AbortReq,
        logger=SimpleNamespace(debug=lambda *_: None),
    )
    handle_generate = _request_handler_method()
    scheduler = SimpleNamespace(
        has_session=lambda _: True,
        can_submit_session_turn=lambda _: False,
        can_defer_session_turn=lambda _: True,
    )
    handler = SimpleNamespace(
        pending_session_turns={},
        pause_controller=SimpleNamespace(buffered_specs=[]),
        handle_generate_request=lambda recv_req, pending_session_ids: handle_generate(
            handler,
            recv_req,
            pending_session_ids=pending_session_ids,
        ),
        _rendezvous_replica_flush=lambda **kwargs: True,
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=scheduler,
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )
    generate = TokenizedGenerateReqInput(
        **_tokenized_request(rid="same-batch-rid", session_id="stable-session").__dict__
    )
    abort = _AbortReq(rid="same-batch-rid")
    recv_reqs = [abort, generate] if abort_first else [generate, abort]

    specs, states, bootstraps, abort_rids = process_requests(handler, recv_reqs)

    assert specs == []
    assert states == []
    assert bootstraps == []
    assert abort_rids == []
    assert handler.pending_session_turns == {}


def test_delayed_forward_without_request_state_notifies_scheduler():
    class _Values(list):
        def tolist(self):
            return list(self)

    class _ForwardOp:
        request_ids = ["finished-session-turn"]
        input_lengths = [1]
        prefill_lengths = []
        extend_prefix_lens = []
        extend_replay_lens = []

        def num_extends(self):
            return 0

    class _ExecutionResult:
        output_tokens = _Values([99])
        output_lengths = _Values([1])
        output_logprobs = None
        output_nan_flags = None
        grammar_completion = None

    post_process = _load_method(
        "python/tokenspeed/runtime/engine/generation_output_processor.py",
        "OutputProcesser",
        "post_process_forward_op",
        make_extend_result_event=lambda rid: SimpleNamespace(request_id=rid, tokens=[]),
    )
    processor = SimpleNamespace(
        add_cached_tokens=lambda *args: None,
        _emit_spec_decode_metrics=lambda *args: None,
        log_cnt=0,
        add_computed_length=lambda *args: None,
        enable_log_request_stats=False,
        spec_algorithm=None,
        spec_num_tokens=None,
        rid_to_state={},
        attn_tp_rank=0,
        stream_output=lambda *args: None,
    )

    events = post_process(
        processor, _ForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    assert len(events) == 1
    assert events[0].request_id == "finished-session-turn"
    assert events[0].tokens == []


@pytest.mark.parametrize(
    ("scheduler", "recv_req", "pending", "error"),
    [
        (
            SimpleNamespace(
                has_session=lambda _: False,
                can_submit_session_turn=lambda _: False,
            ),
            _tokenized_request(session_id="closed-session"),
            set(),
            "unknown or closed",
        ),
        (
            SimpleNamespace(
                has_session=lambda _: True,
                can_submit_session_turn=lambda _: False,
            ),
            _tokenized_request(session_id="busy-session"),
            set(),
            "active turn",
        ),
        (
            SimpleNamespace(
                has_session=lambda _: True,
                can_submit_session_turn=lambda _: True,
            ),
            _tokenized_request(
                session_id="grammar-session",
                sampling_params=SimpleNamespace(
                    max_new_tokens=3,
                    json_schema=None,
                    regex="[0-9]+",
                    ebnf=None,
                    structural_tag=None,
                ),
            ),
            set(),
            "grammar-constrained",
        ),
        (
            SimpleNamespace(
                has_session=lambda _: True,
                can_submit_session_turn=lambda _: True,
            ),
            _tokenized_request(
                session_id="multimodal-session",
                multimodal_inputs=object(),
            ),
            set(),
            "text token IDs",
        ),
    ],
)
def test_request_handler_rejects_invalid_session_turns(
    scheduler, recv_req, pending, error
):
    handle = _request_handler_method()
    handler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=scheduler,
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )
    _, state, _ = handle(handler, recv_req, pending_session_ids=pending)
    assert error in state.finished_reason
    assert state.session_id is None


def test_request_handler_rejects_two_session_writes_in_one_batch():
    handle = _request_handler_method()
    handler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        session_scheduler=SimpleNamespace(
            has_session=lambda _: True,
            can_submit_session_turn=lambda _: True,
        ),
        tokenizer=None,
        hf_eos_token_id=[],
        max_req_len=64,
    )
    pending = set()
    _, first_state, _ = handle(
        handler,
        _tokenized_request(session_id="same-session"),
        pending_session_ids=pending,
    )
    _, second_state, _ = handle(
        handler,
        _tokenized_request(session_id="same-session"),
        pending_session_ids=pending,
    )
    assert first_state.finished_reason is None
    assert "active turn" in second_state.finished_reason
    assert second_state.session_id is None


def test_request_handler_rejects_session_already_buffered_during_pause():
    class TokenizedGenerateReqInput(SimpleNamespace):
        pass

    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
        TokenizedGenerateReqInput=TokenizedGenerateReqInput,
        AbortReq=_AbortReq,
    )
    handle_generate = _request_handler_method()
    handler = SimpleNamespace(
        pending_session_turns={},
        pause_controller=SimpleNamespace(
            buffered_specs=[SimpleNamespace(session_id="same-session")]
        ),
        handle_generate_request=lambda recv_req, pending_session_ids: handle_generate(
            handler,
            recv_req,
            pending_session_ids=pending_session_ids,
        ),
        _rendezvous_replica_flush=lambda **kwargs: True,
    )
    scheduler = SimpleNamespace(
        has_session=lambda _: True,
        can_submit_session_turn=lambda _: True,
    )
    handler.server_args = SimpleNamespace(disaggregation_bootstrap_port=0)
    handler.session_scheduler = scheduler
    handler.tokenizer = None
    handler.hf_eos_token_id = []
    handler.max_req_len = 64
    recv_req = TokenizedGenerateReqInput(
        **_tokenized_request(session_id="same-session").__dict__
    )

    _, states, _, _ = process_requests(handler, [recv_req])

    assert "active turn" in states[0].finished_reason
    assert states[0].session_id is None


def test_request_handler_rejects_close_for_session_buffered_during_pause():
    class TokenizedGenerateReqInput:
        pass

    class ProfileReq:
        pass

    class OpenSessionReqInput:
        pass

    class CloseSessionReqInput:
        session_id = "same-session"

    class CloseSessionReqOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    process_requests = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "process_requests",
        TokenizedGenerateReqInput=TokenizedGenerateReqInput,
        ProfileReq=ProfileReq,
        OpenSessionReqInput=OpenSessionReqInput,
        CloseSessionReqInput=CloseSessionReqInput,
        AbortReq=_AbortReq,
    )
    close_method = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "_handle_close_session_req",
        CloseSessionReqOutput=CloseSessionReqOutput,
    )
    close_calls = []
    outputs = []
    handler = SimpleNamespace(
        pending_session_turns={},
        pause_controller=SimpleNamespace(
            buffered_specs=[SimpleNamespace(session_id="same-session")]
        ),
        session_scheduler=SimpleNamespace(
            close_session=lambda session_id: close_calls.append(session_id) or True
        ),
        send_func=SimpleNamespace(send_pyobj=outputs.append),
        _handle_close_session_req=lambda recv_req, pending_session_ids: close_method(
            handler,
            recv_req,
            pending_session_ids=pending_session_ids,
        ),
        _rendezvous_replica_flush=lambda **kwargs: True,
    )

    process_requests(handler, [CloseSessionReqInput()])

    assert close_calls == []
    assert len(outputs) == 1
    assert outputs[0].session_id == "same-session"
    assert outputs[0].success is False


def test_request_handler_rejects_close_after_same_batch_session_turn():
    class CloseInput:
        session_id = "same-session"

    class CloseOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    close_method = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "_handle_close_session_req",
        CloseSessionReqOutput=CloseOutput,
    )
    close_calls = []
    outputs = []
    handler = SimpleNamespace(
        session_scheduler=SimpleNamespace(
            close_session=lambda session_id: close_calls.append(session_id) or True
        ),
        send_func=SimpleNamespace(send_pyobj=outputs.append),
    )
    source = (ROOT / "python/tokenspeed/runtime/engine/request_handler.py").read_text(
        encoding="utf-8"
    )
    close_method(
        handler,
        CloseInput(),
        pending_session_ids={"same-session"},
    )
    assert "pending_session_ids=pending_session_ids" in source
    assert close_calls == []
    assert outputs[0].success is False
    assert "active" in outputs[0].message


def test_request_handler_acknowledges_open_and_close_control_messages():
    class OpenInput:
        session_id = "stable-session"

    class CloseInput:
        session_id = "stable-session"

    class OpenOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class CloseOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    open_method = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "_handle_open_session_req",
        OpenSessionReqOutput=OpenOutput,
    )
    close_method = _load_method(
        "python/tokenspeed/runtime/engine/request_handler.py",
        "RequestHandler",
        "_handle_close_session_req",
        CloseSessionReqOutput=CloseOutput,
    )
    outputs = []
    handler = SimpleNamespace(
        session_scheduler=SimpleNamespace(
            open_session=lambda _: True,
            close_session=lambda _: False,
        ),
        send_func=SimpleNamespace(send_pyobj=outputs.append),
    )
    open_method(handler, OpenInput())
    close_method(handler, CloseInput())
    assert outputs[0].success is True
    assert outputs[0].session_id == "stable-session"
    assert outputs[1].success is False
    assert outputs[1].session_id == "stable-session"
    assert outputs[1].message


def test_output_metadata_uses_scheduler_session_counts_not_prefix_cache_counters():
    source = (ROOT / "python/tokenspeed/runtime/engine/output_processor.py").read_text(
        encoding="utf-8"
    )
    assert 'meta_info["session"]' in source
    generation_source = (
        ROOT / "python/tokenspeed/runtime/engine/generation_output_processor.py"
    ).read_text(encoding="utf-8")
    for field in (
        "session_id",
        "total_context_tokens",
        "newly_prefilled_tokens",
        "reused_context_tokens",
    ):
        assert field in generation_source
    metadata_helper = _load_method(
        "python/tokenspeed/runtime/engine/output_processor.py",
        None,
        "_attach_session_metadata",
    )
    meta = {}
    original = {"decode_prefix_len": 2, "session": {"reused_context_tokens": 7}}
    remaining = metadata_helper(meta, original)
    assert meta == {"session": {"reused_context_tokens": 7}}
    assert remaining == {"decode_prefix_len": 2}
    assert "session" in original


def test_output_processor_moves_session_metadata_out_of_raw_extra_info():
    class BatchStrOut:
        pass

    class BatchTokenIDOut:
        pass

    class BatchEmbeddingOut:
        pass

    output_method = _load_method(
        "python/tokenspeed/runtime/engine/output_processor.py",
        "OutputProcessor",
        "handle_batch_output",
        BatchStrOut=BatchStrOut,
        BatchTokenIDOut=BatchTokenIDOut,
        BatchEmbeddingOut=BatchEmbeddingOut,
        _attach_session_metadata=_load_method(
            "python/tokenspeed/runtime/engine/output_processor.py",
            None,
            "_attach_session_metadata",
        ),
    )

    def render(output_type, session_info):
        recv_obj = output_type()
        recv_obj.rids = ["rid"]
        recv_obj.finished_reasons = [None]
        recv_obj.prompt_tokens = [2]
        recv_obj.completion_tokens = [0]
        recv_obj.cached_tokens = [0]
        recv_obj.output_extra_infos = [
            {
                "decode_prefix_len": 2,
                **({"session": session_info} if session_info is not None else {}),
            }
        ]
        recv_obj.batch_accept_draft_tokens = []
        recv_obj.output_ids = [[7]]
        recv_obj.output_multi_ids = None
        if output_type is BatchStrOut:
            recv_obj.output_strs = ["text"]

        outputs = []
        state = SimpleNamespace(
            obj=SimpleNamespace(
                stream=False,
                sampling_params={},
                return_logprob=False,
                log_metrics=True,
            ),
            logprobs_info={},
            text="",
            output_ids=[],
            last_output_offset=0,
            inline_detokenizer=None,
            collector=SimpleNamespace(
                put=lambda output, stream: outputs.append(output)
            ),
            event=SimpleNamespace(set=lambda: None),
            created_time=0,
            finished=False,
        )
        engine = SimpleNamespace(
            rid_to_state={"rid": state},
            server_args=SimpleNamespace(
                weight_version="test",
                enable_inline_detokenizer=False,
                stream_output=False,
                speculative_algorithm=None,
            ),
            tokenizer=None,
            enable_metrics=False,
            dump_requests_folder=None,
        )
        output_method(SimpleNamespace(engine=engine), recv_obj)
        return outputs[0]

    session_info = {
        "session_id": "stable-session",
        "reused_context_tokens": 7,
    }
    for output_type in (BatchStrOut, BatchTokenIDOut):
        with_session = render(output_type, session_info)
        assert with_session["meta_info"]["session"] == session_info
        assert with_session["output_extra_info"] == {"decode_prefix_len": 2}

        stateless = render(output_type, None)
        assert "session" not in stateless["meta_info"]
        assert stateless["output_extra_info"] == {"decode_prefix_len": 2}


def test_final_session_metadata_is_read_from_scheduler_session_info():
    source = (
        ROOT / "python/tokenspeed/runtime/engine/generation_output_processor.py"
    ).read_text(encoding="utf-8")
    assert "self.session_info_fn(rs.session_id)" in source
    assert "session_info.total_context_tokens" in source
    assert "session_info.newly_prefilled_tokens" in source
    assert "session_info.reused_context_tokens" in source
    assert "session_info_fn=self.scheduler.session_info_for_session" in (
        ROOT / "python/tokenspeed/runtime/engine/event_loop.py"
    ).read_text(encoding="utf-8")


def test_session_frontier_metadata_validation_obeys_explicit_policy():
    runner_path = ROOT / "scripts/run_session_kv.py"
    assert runner_path.is_file(), "Session KV runner is missing"

    import importlib.util

    spec = importlib.util.spec_from_file_location("run_session_kv", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    current = {
        "total_context_tokens": 19,
        "newly_prefilled_tokens": 5,
        "reused_context_tokens": 14,
    }
    assert (
        runner.expected_session_context_counts(
            policy="current-frontier", prior_history_tokens=15, suffix_tokens=4
        )
        == current
    )
    with pytest.raises(ValueError, match="materialize-final-token is not implemented"):
        runner.expected_session_context_counts(
            policy="materialize-final-token", prior_history_tokens=15, suffix_tokens=4
        )
    initial = {
        "total_context_tokens": 7,
        "newly_prefilled_tokens": 7,
        "reused_context_tokens": 0,
    }
    assert (
        runner.expected_session_context_counts(
            policy="current-frontier", prior_history_tokens=0, suffix_tokens=7
        )
        == initial
    )
    runner.validate_session_context_metadata(
        initial,
        policy="current-frontier",
        prior_history_tokens=0,
        suffix_tokens=7,
    )
    runner.validate_session_context_metadata(
        current,
        policy="current-frontier",
        prior_history_tokens=15,
        suffix_tokens=4,
    )
    with pytest.raises(ValueError, match="materialize-final-token is not implemented"):
        runner.validate_session_context_metadata(
            current,
            policy="materialize-final-token",
            prior_history_tokens=15,
            suffix_tokens=4,
        )


@pytest.fixture(scope="module")
def session_kv_runner():
    import importlib.util

    runner_path = ROOT / "scripts/run_session_kv.py"
    spec = importlib.util.spec_from_file_location("run_session_kv", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_engine_args_remove_gateway_only_prometheus_port(session_kv_runner):
    command = [
        "python",
        "-m",
        "tokenspeed.cli",
        "serve",
        "/models/qwen35-4b",
        "--host",
        "127.0.0.1",
        "--prometheus-port",
        "39141",
        "--metrics-port",
        "39142",
    ]

    assert session_kv_runner._engine_cli_args(command) == [
        "/models/qwen35-4b",
        "--host",
        "127.0.0.1",
        "--metrics-port",
        "39142",
    ]


def test_engine_session_kv_disables_conflicting_cache_backends(session_kv_runner):
    config = session_kv_runner.load_model_config(session_kv_runner.CONFIG_PATH)
    engine_config, cli_args = session_kv_runner._build_engine_config_and_cli_args(
        config
    )

    assert engine_config["prefix_caching"] is False
    assert engine_config["disable_kvstore"] is True
    assert engine_config["overlap_schedule"] is False
    assert "--disable-prefix-caching" in cli_args
    assert "--disable-kvstore" in cli_args
    assert "--disable-overlap-schedule" in cli_args
    assert "--prometheus-port" not in cli_args


def test_engine_session_kv_can_enable_overlap(session_kv_runner):
    config = session_kv_runner.load_model_config(session_kv_runner.CONFIG_PATH)
    engine_config, cli_args = session_kv_runner._build_engine_config_and_cli_args(
        config, overlap_schedule=True
    )

    assert engine_config["overlap_schedule"] is True
    assert "--disable-overlap-schedule" not in cli_args


def test_generate_reports_request_start_and_completion(session_kv_runner, capsys):
    class FakeEngine:
        def __init__(self):
            self.thread_id = None

        def generate(self, **kwargs):
            assert kwargs["session_id"] == "session-0"
            self.thread_id = threading.get_ident()
            return {"output_ids": [42], "meta_info": {}}

    engine = FakeEngine()
    caller_thread_id = threading.get_ident()
    result = asyncio.run(
        session_kv_runner._generate(
            engine,
            [1, 2],
            {"max_new_tokens": 1, "temperature": 0},
            request_label="session-0-turn-1-session",
            session_id="session-0",
        )
    )

    assert result["output_ids"] == [42]
    assert engine.thread_id != caller_thread_id
    output = capsys.readouterr().out
    assert "request_started label=session-0-turn-1-session" in output
    assert "request_finished label=session-0-turn-1-session" in output


@pytest.mark.parametrize(
    ("finish_reason", "should_pass"),
    (
        (
            {
                "type": "abort",
                "message": "unknown or closed session id 'closed-session'",
            },
            True,
        ),
        ({"type": "length", "length": 1}, False),
    ),
)
def test_closed_session_probe_uses_sync_generate_and_requires_expected_abort(
    session_kv_runner, monkeypatch, finish_reason, should_pass
):
    class FakeEngine:
        def __init__(self):
            self.generate_calls = []
            self.async_generate_calls = 0
            self.close_session_calls = []

        def open_session(self):
            return "closed-session"

        def close_session(self, session_id):
            self.close_session_calls.append(session_id)

        def generate(self, **kwargs):
            assert kwargs["input_ids"] == [0]
            assert kwargs["session_id"] == "closed-session"
            self.generate_calls.append(threading.get_ident())
            return {
                "output_ids": [],
                "meta_info": {"finish_reason": finish_reason},
            }

        async def async_generate(self, **kwargs):
            self.async_generate_calls += 1
            return {
                "output_ids": [],
                "meta_info": {"finish_reason": finish_reason},
            }

    monkeypatch.setattr(session_kv_runner, "SESSION_TURNS", ((),))
    engine = FakeEngine()
    caller_thread_id = threading.get_ident()
    run_record = {"turns": [], "closed_session_rejections": []}

    async def run():
        if should_pass:
            await session_kv_runner._run_sessions(
                engine,
                None,
                frontier_policy="current-frontier",
                max_new_tokens=1,
                run_record=run_record,
            )
        else:
            with pytest.raises(RuntimeError, match="expected closed-session abort"):
                await session_kv_runner._run_sessions(
                    engine,
                    None,
                    frontier_policy="current-frontier",
                    max_new_tokens=1,
                    run_record=run_record,
                )

    asyncio.run(run())

    assert len(engine.generate_calls) == 1
    assert engine.generate_calls[0] != caller_thread_id
    assert engine.async_generate_calls == 0
    assert engine.close_session_calls == ["closed-session"]
    if should_pass:
        assert run_record["closed_session_rejections"] == [
            {"session_id": "closed-session", "finish_reason": finish_reason}
        ]
    else:
        assert run_record["closed_session_rejections"] == []


@pytest.mark.parametrize("output_has_eos", (False, True))
def test_next_suffix_preserves_qwen_history_and_appends_only_missing_boundary(
    session_kv_runner, output_has_eos
):
    class QwenTemplateTokenizer:
        eos_token_id = 2
        special_tokens = {
            "<|im_start|>": 1,
            "<|im_end|>": eos_token_id,
            "<think>": 3,
        }

        def _encode(self, text):
            token_ids = []
            while text:
                special_token = next(
                    (token for token in self.special_tokens if text.startswith(token)),
                    None,
                )
                if special_token is None:
                    token_ids.append(ord(text[0]) + 1000)
                    text = text[1:]
                else:
                    token_ids.append(self.special_tokens[special_token])
                    text = text[len(special_token) :]
            return token_ids

        def decode(self, token_ids, *, skip_special_tokens):
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
            if not messages or not any(
                message["role"] == "user" for message in messages
            ):
                raise ValueError("Qwen template requires a user query")

            rendered = ""
            if messages[0]["role"] == "system":
                rendered += (
                    "<|im_start|>system\n"
                    + messages[0]["content"].strip()
                    + "<|im_end|>\n"
                )
            for message in messages:
                if message["role"] == "system":
                    continue
                role = message["role"]
                rendered += (
                    f"<|im_start|>{role}\n" f"{message['content'].strip()}<|im_end|>\n"
                )
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n<think>\n"
            return self._encode(rendered)

    tokenizer = QwenTemplateTokenizer()
    prior_user_text = "What is 17 plus 25? Answer briefly."
    next_user_text = "Now answer with just the number."
    output_ids = tokenizer._encode("42")
    if output_has_eos:
        output_ids.append(tokenizer.eos_token_id)
    history_ids = (
        session_kv_runner._initial_suffix(tokenizer, prior_user_text) + output_ids
    )

    suffix_ids = session_kv_runner._next_suffix(
        tokenizer,
        next_user_text,
        prior_user_text=prior_user_text,
        previous_output_ids=output_ids,
        history_ids=history_ids,
    )
    full_context_ids = session_kv_runner._token_ids(
        tokenizer,
        [
            {"role": "system", "content": session_kv_runner.SYSTEM_PROMPT},
            {"role": "user", "content": prior_user_text},
            {"role": "assistant", "content": "<think>\n42"},
            {"role": "user", "content": next_user_text},
        ],
        add_generation_prompt=True,
    )

    assert full_context_ids[: len(history_ids)] == history_ids
    assert suffix_ids == full_context_ids[len(history_ids) :]
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=False)
    if output_has_eos:
        assert suffix_text.startswith("\n<|im_start|>user\n")
    else:
        assert suffix_text.startswith("<|im_end|>\n<|im_start|>user\n")
    assert "42" not in suffix_text

    counts = session_kv_runner.expected_session_context_counts(
        policy="current-frontier",
        prior_history_tokens=len(history_ids),
        suffix_tokens=len(suffix_ids),
    )
    assert counts["reused_context_tokens"] == len(history_ids) - 1
    assert counts["newly_prefilled_tokens"] == len(suffix_ids) + 1

    class NonRoundTripTokenizer(QwenTemplateTokenizer):
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return "a different continuation"

    with pytest.raises(
        ValueError, match="does not preserve the cached Qwen session history"
    ):
        session_kv_runner._next_suffix(
            NonRoundTripTokenizer(),
            next_user_text,
            prior_user_text=prior_user_text,
            previous_output_ids=output_ids,
            history_ids=history_ids,
        )


def test_frontier_policy_cli_rejects_unimplemented_materialization(
    session_kv_runner,
):
    assert (
        session_kv_runner.build_parser()
        .parse_args(["--frontier-policy", "current-frontier"])
        .frontier_policy
        == "current-frontier"
    )
    with pytest.raises(SystemExit):
        session_kv_runner.build_parser().parse_args(
            ["--frontier-policy", "materialize-final-token"]
        )


def test_closed_session_probe_accepts_expected_abort_result(session_kv_runner):
    result = {
        "output_ids": [],
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "message": (
                    "Invalid session request: unknown or closed session id "
                    "'closed-session'"
                ),
                "status_code": None,
            }
        },
    }
    assert session_kv_runner.is_expected_closed_session_rejection(
        result, "closed-session"
    )
    assert not session_kv_runner.is_expected_closed_session_rejection(
        result, "another-session"
    )


def test_closed_session_probe_rejects_normal_success_result(session_kv_runner):
    result = {
        "output_ids": [27],
        "meta_info": {"finish_reason": {"type": "length", "length": 1}},
    }
    assert not session_kv_runner.is_expected_closed_session_rejection(
        result, "closed-session"
    )


def test_closed_session_probe_does_not_count_unrelated_exception(session_kv_runner):
    error = RuntimeError("transport disconnected")
    assert not session_kv_runner.is_expected_closed_session_rejection(
        error, "closed-session"
    )


def test_output_directory_resolver_accepts_default_and_project_relative_paths(
    session_kv_runner,
):
    assert (
        session_kv_runner.resolve_output_dir(session_kv_runner.DEFAULT_OUTPUT_DIR)
        == session_kv_runner.DEFAULT_OUTPUT_DIR.resolve()
    )
    assert (
        session_kv_runner.resolve_output_dir(Path("results/raw/session_kv"))
        == (ROOT / "results" / "raw" / "session_kv").resolve()
    )


def test_output_directory_resolver_rejects_paths_outside_project(
    session_kv_runner,
):
    with pytest.raises(ValueError, match="must be inside the project"):
        session_kv_runner.resolve_output_dir(ROOT.parent / "outside-project-path")


def test_output_path_resolver_rejects_hardware_link_outside_project(
    session_kv_runner,
):
    import os
    import subprocess
    import tempfile

    outside_target = ROOT.parent
    with tempfile.TemporaryDirectory(
        prefix=".test-session-kv-output-", dir=ROOT
    ) as temporary_project_dir:
        output_dir = Path(temporary_project_dir) / "output"
        output_dir.mkdir()
        hardware_link = output_dir / "hardware"
        if os.name == "nt":
            junction = subprocess.run(
                [
                    "cmd.exe",
                    "/c",
                    "mklink",
                    "/J",
                    str(hardware_link),
                    str(outside_target),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert junction.returncode == 0, junction.stderr or junction.stdout
        else:
            hardware_link.symlink_to(outside_target, target_is_directory=True)

        with pytest.raises(ValueError, match="must be inside the project"):
            session_kv_runner.resolve_output_paths(output_dir)
