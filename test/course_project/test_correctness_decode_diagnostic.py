import builtins
import hashlib
import importlib.util
import json
import os
import struct
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from benchmarks.course_project import correctness
from benchmarks.course_project.core import TokenSpeedGenerateClient
from scripts import run_correctness_decode_diagnostic as diagnostic_script


class _DiagnosticHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.payloads.append(payload)
        params = payload["sampling_params"]
        generated_offset = len(payload["input_ids"]) - 2
        output_ids = [41, 42, 43][
            generated_offset : generated_offset + params["max_new_tokens"]
        ]
        body = {
            "text": "answer",
            "output_ids": output_ids,
            "meta_info": {"prompt_tokens": len(payload["input_ids"])},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format: str, *args) -> None:
        del format, args


class TestCorrectnessDecodeDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _DiagnosticHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_full_recompute_resubmits_prompt_and_generated_tokens_each_step(
        self,
    ) -> None:
        run_diagnostic = getattr(correctness, "run_decode_diagnostic", None)
        self.assertIsNotNone(run_diagnostic, "decode diagnostic runner is missing")
        _DiagnosticHandler.payloads.clear()
        port = self.server.server_address[1]
        client = TokenSpeedGenerateClient(f"http://127.0.0.1:{port - 1}/v1")

        continuous_ids, continuous_requests = run_diagnostic(
            client=client,
            input_token_ids=[1, 2],
            max_new_tokens=3,
            mode="continuous",
        )
        recomputed_ids, recompute_requests = run_diagnostic(
            client=client,
            input_token_ids=[1, 2],
            max_new_tokens=3,
            mode="full_recompute",
        )

        self.assertEqual(continuous_ids, [41, 42, 43])
        self.assertEqual(recomputed_ids, continuous_ids)
        self.assertEqual(continuous_requests, 1)
        self.assertEqual(recompute_requests, 3)
        self.assertEqual(
            [item["input_ids"] for item in _DiagnosticHandler.payloads[1:]],
            [[1, 2], [1, 2, 41], [1, 2, 41, 42]],
        )
        self.assertEqual(
            [
                item["sampling_params"]["max_new_tokens"]
                for item in _DiagnosticHandler.payloads
            ],
            [3, 1, 1, 1],
        )

    def test_selects_only_the_requested_decode_mode(self) -> None:
        select_modes = getattr(
            diagnostic_script, "select_decode_diagnostic_modes", None
        )
        self.assertIsNotNone(select_modes, "decode diagnostic mode selector is missing")
        self.assertEqual(select_modes("continuous"), ("continuous",))
        self.assertEqual(select_modes("full_recompute"), ("full_recompute",))
        self.assertEqual(select_modes("both"), ("continuous", "full_recompute"))

    def test_diag_state_snapshot_is_not_aliased_to_recurrent_page(self) -> None:
        module_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "diagnostics"
            / "sitecustomize.py"
        )
        spec = importlib.util.spec_from_file_location(
            "diag_capture_sitecustomize_test", module_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            os.environ,
            {
                "TOKENSPEED_DIAG_LOGITS_PATH": "",
                "TOKENSPEED_DIAG_STATES_PATH": "",
                "TOKENSPEED_DIAG_ARM_FILE": "",
            },
        ):
            spec.loader.exec_module(module)

        class FakeStatePage:
            def __init__(self, values: list[int]) -> None:
                self.values = values

            def detach(self):
                return self

            def clone(self):
                return FakeStatePage(self.values.copy())

        pages = [FakeStatePage([0, 0]), FakeStatePage([1, 2])]
        snapshot = module._clone_state_page(pages, 1)
        pages[1].values[:] = [3, 4]

        self.assertEqual(snapshot.values, [1, 2])


class _Tensor:
    """Small CPU tensor double for exercising the real optional capture hook."""

    dtype = "float32"
    ndim = 1

    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def __getitem__(self, index):
        if isinstance(index, _Tensor):
            return _Tensor(v for v, keep in zip(self.values, index.values) if keep)
        return _Tensor(
            self.values[index] if isinstance(index, slice) else [self.values[index]]
        )

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.values.copy())

    def contiguous(self):
        return self

    def to(self, **kwargs):
        return self

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return struct.pack(f"<{len(self.values)}f", *self.values)

    def float(self):
        return self

    def numel(self):
        return len(self.values)

    def item(self):
        return self.values[0]

    def min(self):
        return _Tensor([min(self.values)])

    def max(self):
        return _Tensor([max(self.values)])

    def sum(self):
        return _Tensor([sum(self.values)])


def _load_capture_hook():
    path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/sitecustomize.py"
    spec = importlib.util.spec_from_file_location("decode_capture_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_capture_imports_no_torch_and_patches_nothing(monkeypatch):
    for name in ("LOGITS_PATH", "STATES_PATH", "ARM_FILE"):
        monkeypatch.delenv(f"TOKENSPEED_DIAG_{name}", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "torch" and not name.startswith("tokenspeed.runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _load_capture_hook()


def _exercise_capture(
    monkeypatch,
    tmp_path,
    *,
    input_page,
    output_page,
    armed=True,
    position=1010,
    input_id=271,
    layer_id=0,
    operation="decode",
):
    class Backend:
        def __init__(self):
            self.conv = [_Tensor([1, 2]), _Tensor([10, 20])]
            self.ssm = [_Tensor([3, 4]), _Tensor([30, 40])]
            self.reads = 0

        def _layer_state(self, layer_id):
            self.reads += 1
            return _Tensor([input_page]), _Tensor([output_page]), self.conv, self.ssm

        def forward_decode(self, **kwargs):
            self.conv[output_page].values[:] = [5, 6]
            self.ssm[output_page].values[:] = [7, 8]
            return _Tensor([9])

        forward_extend = forward_decode

    backend = Backend()

    class ModelRunner:
        def forward(self, ctx, input_ids, positions):
            return getattr(backend, f"forward_{operation}")(layer_id=layer_id)

    class LogitsProcessor:
        def _get_logits(self, *args, **kwargs):
            raise AssertionError("logits are outside this state test")

    torch = ModuleType("torch")
    torch.Tensor = _Tensor
    torch.int64 = "int64"
    torch.uint8 = "uint8"
    torch.isfinite = lambda tensor: _Tensor([True] * tensor.numel())
    torch.linalg = SimpleNamespace(
        vector_norm=lambda tensor: _Tensor([sum(v * v for v in tensor.values) ** 0.5])
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name, cls in (
        ("tokenspeed.runtime.execution.model_runner", ModelRunner),
        ("tokenspeed.runtime.layers.logits_processor", LogitsProcessor),
        (
            "tokenspeed.runtime.layers.attention.backends.paged.mha",
            type(
                "MHAAttnBackend",
                (),
                {
                    "forward_decode": lambda self: None,
                    "forward_extend": lambda self: None,
                },
            ),
        ),
        ("tokenspeed.runtime.layers.attention.backends.state.mamba", Backend),
    ):
        module = ModuleType(name)
        setattr(module, cls.__name__, cls)
        if cls is Backend:
            module.MambaAttnBackend = cls
        monkeypatch.setitem(sys.modules, name, module)
    arm = tmp_path / "arm"
    if armed:
        arm.write_text("continuous", encoding="utf-8")
    states = tmp_path / "probe-states.jsonl"
    monkeypatch.setenv("TOKENSPEED_DIAG_ARM_FILE", str(arm))
    monkeypatch.setenv("TOKENSPEED_DIAG_STATES_PATH", str(states))
    monkeypatch.setenv("TOKENSPEED_DIAG_LOGITS_PATH", "")
    monkeypatch.setenv("TOKENSPEED_DIAG_TARGET_POSITION", "1010")
    monkeypatch.setenv("TOKENSPEED_DIAG_TARGET_INPUT_ID", "271")
    _load_capture_hook()
    result = ModelRunner().forward(
        SimpleNamespace(forward_mode="decode"), _Tensor([input_id]), _Tensor([position])
    )
    assert result.values == [9]
    records = (
        [json.loads(line) for line in states.read_text().splitlines()]
        if states.exists()
        else []
    )
    return backend, records, tmp_path / "probe-layer0-arrays"


def test_wrapper_captures_input_before_in_place_update_and_poststate(
    monkeypatch, tmp_path
):
    _, records, raw = _exercise_capture(
        monkeypatch, tmp_path, input_page=0, output_page=0
    )
    (record,) = records
    assert "capture_error" not in record
    for name, before, after in (
        ("conv_state", [1, 2], [5, 6]),
        ("ssm_state", [3, 4], [7, 8]),
    ):
        assert record[f"{name}_before"]["min"] == min(before)
        assert record[name]["min"] == min(after)
        for field, values in ((f"{name}_before", before), (name, after)):
            data = (raw / record["raw_arrays"][field]["file"]).read_bytes()
            assert data == struct.pack("<2f", *values)
            assert (
                record["raw_arrays"][field]["sha256"]
                == hashlib.sha256(data).hexdigest()
            )


def test_wrapper_before_reads_input_page_when_output_differs(monkeypatch, tmp_path):
    _, records, _ = _exercise_capture(
        monkeypatch, tmp_path, input_page=0, output_page=1, operation="extend"
    )
    (record,) = records
    assert record["input_page"] == 0 and record["output_page"] == 1
    assert record["conv_state_before"]["min"] == 1
    assert record["ssm_state_before"]["min"] == 3
    assert record["conv_state"]["min"] == 5


def test_wrapper_unarmed_and_wrong_target_do_not_read_state(monkeypatch, tmp_path):
    for index, guards in enumerate(
        ({"armed": False}, {"position": 1009}, {"input_id": 270})
    ):
        directory = tmp_path / str(index)
        directory.mkdir()
        backend, records, raw = _exercise_capture(
            monkeypatch, directory, input_page=0, output_page=0, **guards
        )
        assert backend.reads == 0
        assert records == [] and not raw.exists()


def test_wrapper_invalid_input_is_reported_without_negative_index_capture(
    monkeypatch, tmp_path
):
    backend, records, raw = _exercise_capture(
        monkeypatch, tmp_path, input_page=-1, output_page=0
    )
    (record,) = records
    assert "invalid input state page" in record["capture_error"]
    assert backend.conv[0].values == [5, 6]
    assert not raw.exists()


def test_nonzero_layer_keeps_poststate_without_before_clones(monkeypatch, tmp_path):
    def forbidden_clone(self):
        raise AssertionError("nonzero-layer state must not be cloned")

    monkeypatch.setattr(_Tensor, "clone", forbidden_clone)
    for index, input_page in enumerate((0, -1)):
        directory = tmp_path / str(index)
        directory.mkdir()
        backend, records, raw = _exercise_capture(
            monkeypatch, directory, input_page=input_page, output_page=0, layer_id=1
        )
        (record,) = records
        assert "capture_error" not in record
        assert record["conv_state"]["min"] == 5
        assert record["ssm_state"]["min"] == 7
        assert "conv_state_before" not in record
        assert "ssm_state_before" not in record
        assert backend.reads == 1
        assert not raw.exists()


if __name__ == "__main__":
    unittest.main()
