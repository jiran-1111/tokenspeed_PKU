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

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from benchmarks.course_project import core
except ImportError:
    core = None

try:
    from benchmarks.course_project import runner
except ImportError:
    runner = None

append_jsonl = getattr(core, "append_jsonl", None)
build_agent_conversation = getattr(core, "build_agent_conversation", None)
build_serve_command = getattr(core, "build_serve_command", None)
metrics_server_url = getattr(core, "metrics_server_url", None)
model_load_path = getattr(core, "model_load_path", None)
calculate_request_metrics = getattr(core, "calculate_request_metrics", None)
load_model_config = getattr(core, "load_model_config", None)
chat_template_token_ids = getattr(core, "chat_template_token_ids", None)
OpenAIChatClient = getattr(core, "OpenAIChatClient", None)
TokenSpeedGenerateClient = getattr(core, "TokenSpeedGenerateClient", None)
BenchmarkRequest = getattr(runner, "BenchmarkRequest", None)
run_concurrent_requests = getattr(runner, "run_concurrent_requests", None)
summarize_records = getattr(runner, "summarize_records", None)
make_token_sized_text = getattr(runner, "make_token_sized_text", None)
build_agent_turn_request = getattr(runner, "build_agent_turn_request", None)


class _StreamingHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.payloads.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.005)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args) -> None:
        del format, args


class _GenerateHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []
    paths: list[str] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.paths.append(self.path)
        self.payloads.append(json.loads(self.rfile.read(length)))
        body = {
            "text": "answer",
            "output_ids": [41, 42],
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format: str, *args) -> None:
        del format, args


class TestModelConfig(unittest.TestCase):
    def test_loads_complete_json_compatible_yaml(self) -> None:
        self.assertIsNotNone(load_model_config, "course benchmark core is missing")
        config = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "served_model_name": "qwen35-0.8b",
            "mm_attention_backend": "triton_attn",
            "device": "npu",
            "tp_size": 1,
            "max_model_len": 32768,
            "dtype": "bfloat16",
            "max_num_seqs": 16,
            "max_total_tokens": 65536,
            "chunked_prefill_size": 2048,
            "block_size": 128,
            "memory_utilization": 0.9,
            "graph_capture_sizes": [1, 2, 4, 8, 16],
            "prefix_caching": True,
            "enable_metrics": True,
            "prometheus_port": 39141,
            "tokenspeed_metrics_port": 39142,
            "mixed_batch": True,
            "overlap_schedule": True,
            "enforce_eager": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_model_config(path), config)

    def test_rejects_missing_required_field(self) -> None:
        self.assertIsNotNone(load_model_config, "course benchmark core is missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(json.dumps({"model_id": "Qwen/Qwen3.5-0.8B"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "served_model_name"):
                load_model_config(path)


class TestChatTemplateTokenIds(unittest.TestCase):
    def test_unwraps_transformers_batch_encoding(self) -> None:
        self.assertIsNotNone(chat_template_token_ids, "chat template token helper is missing")
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [[11, 22, 33]],
            "attention_mask": [[1, 1, 1]],
        }

        token_ids = chat_template_token_ids(
            tokenizer, [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(token_ids, [11, 22, 33])


class TestMetrics(unittest.TestCase):
    def test_builds_metrics_url_from_configured_port(self) -> None:
        self.assertIsNotNone(metrics_server_url, "metrics URL builder is missing")
        self.assertEqual(
            metrics_server_url(
                {"prometheus_port": 39141, "tokenspeed_metrics_port": 39142}
            ),
            "http://127.0.0.1:39142",
        )

    def test_calculates_streaming_latency_metrics(self) -> None:
        self.assertIsNotNone(calculate_request_metrics, "course benchmark core is missing")
        metrics = calculate_request_metrics(
            started_at=10.0,
            first_token_at=10.2,
            ended_at=11.0,
            input_tokens=100,
            output_tokens=5,
        )
        self.assertAlmostEqual(metrics["ttft_ms"], 200.0)
        self.assertAlmostEqual(metrics["e2e_ms"], 1000.0)
        self.assertAlmostEqual(metrics["tpot_ms"], 200.0)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 5.0)


class TestServeCommand(unittest.TestCase):
    def test_builds_command_from_existing_local_model_path(self) -> None:
        config = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "local_model_path": "benchmarks",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "served_model_name": "qwen35-0.8b",
            "mm_attention_backend": "triton_attn",
            "device": "npu",
            "tp_size": 1,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "max_num_seqs": 16,
            "max_total_tokens": 65536,
            "chunked_prefill_size": 4096,
            "block_size": 128,
            "memory_utilization": 0.9,
            "graph_capture_sizes": [1, 2, 4, 8, 16],
            "prefix_caching": True,
            "enable_metrics": True,
            "prometheus_port": 39141,
            "tokenspeed_metrics_port": 39142,
            "mixed_batch": True,
            "overlap_schedule": True,
            "enforce_eager": False,
        }

        command = build_serve_command(config, host="127.0.0.1", port=8000)

        repo_root = Path(core.__file__).resolve().parents[2]
        self.assertEqual(command[4], str(repo_root / "benchmarks"))

    def test_model_load_path_uses_local_checkpoint(self) -> None:
        self.assertIsNotNone(model_load_path, "model path resolver is missing")
        repo_root = Path(core.__file__).resolve().parents[2]
        self.assertEqual(
            model_load_path({"model_id": "Qwen/model", "local_model_path": "benchmarks"}),
            str(repo_root / "benchmarks"),
        )

    def test_builds_explicit_ascend_bringup_command(self) -> None:
        self.assertIsNotNone(build_serve_command, "serve command builder is missing")
        config = {
            "model_id": "Qwen/Qwen3.5-27B",
            "model_revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
            "served_model_name": "qwen35-27b",
            "mm_attention_backend": "triton_attn",
            "device": "npu",
            "tp_size": 2,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "max_num_seqs": 8,
            "max_total_tokens": 32768,
            "chunked_prefill_size": 2048,
            "block_size": 128,
            "memory_utilization": 0.9,
            "graph_capture_sizes": [1, 2, 4, 8],
            "prefix_caching": False,
            "disable_kvstore": True,
            "enable_metrics": True,
            "prometheus_port": 39141,
            "tokenspeed_metrics_port": 39142,
            "mixed_batch": True,
            "overlap_schedule": False,
            "enforce_eager": True,
        }
        command = build_serve_command(config, host="0.0.0.0", port=8000)
        self.assertEqual(command[:5], ["python", "-m", "tokenspeed.cli", "serve", config["model_id"]])
        expected_pairs = {
            "--revision": config["model_revision"],
            "--prometheus-port": str(config["prometheus_port"]),
            "--metrics-port": str(config["tokenspeed_metrics_port"]),
            "--mm-attention-backend": config["mm_attention_backend"],
            "--served-model-name": "qwen35-27b",
            "--device": "npu",
            "--dtype": "bfloat16",
            "--attention-backend": "mha",
            "--sampling-backend": "greedy",
            "--world-size": "2",
            "--prefix-granularity": "128",
            "--chunked-prefill-size": "2048",
            "--max-total-tokens": "32768",
        }
        for flag, value in expected_pairs.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--disable-prefill-graph", command)
        self.assertIn("--disable-pdl", command)
        self.assertIn("--disable-autotune", command)
        self.assertIn("--enable-metrics", command)
        self.assertIn("--disable-prefix-caching", command)
        self.assertIn("--disable-kvstore", command)
        self.assertIn("--enable-mixed-batch", command)
        self.assertIn("--disable-overlap-schedule", command)
        self.assertIn("--enforce-eager", command)
        self.assertNotIn("--block-size", command)
        graph_index = command.index("--cudagraph-capture-sizes")
        self.assertEqual(command[graph_index + 1 : graph_index + 5], ["1", "2", "4", "8"])

    def test_serve_script_runs_from_source_checkout(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "scripts/serve.py",
                "configs/bringup/qwen35_0.8b.yaml",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = load_model_config(repo_root / "configs/bringup/qwen35_0.8b.yaml")
        self.assertIn(model_load_path(config), result.stdout)
        self.assertIn("--enable-metrics", result.stdout)
        self.assertIn("--prometheus-port 39141", result.stdout)
        self.assertIn("--metrics-port 39142", result.stdout)
        self.assertIn("--device npu", result.stdout)

    def test_all_qwen35_npu_bringup_configs_disable_unsupported_mixed_batch(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for config_path in sorted((repo_root / "configs/bringup").glob("qwen35_*.yaml")):
            with self.subTest(config=config_path.name):
                config = load_model_config(config_path)
                command = build_serve_command(config, host="127.0.0.1", port=8000)

                self.assertFalse(config["mixed_batch"])
                self.assertNotIn("--enable-mixed-batch", command)

    def test_benchmark_script_help_runs_without_model_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for script in ("scripts/run_benchmark.py", "scripts/run_agent_benchmark.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--base-url", result.stdout)


class TestAgentWorkload(unittest.TestCase):
    def test_builds_policy_specific_turn_requests(self) -> None:
        self.assertIsNotNone(build_agent_turn_request, "agent turn request builder is missing")
        mock_result = "Mock tool result: Search returned two relevant documents. Continue with the task."
        requests = [
            build_agent_turn_request(
                request_id=f"all-{session_index}",
                session_id=f"session-{session_index}",
                turn_index=2,
                history=[{"role": "assistant", "content": "Searching."}],
                user_message="Continue.",
                tool_wait_policy="all",
                tool_wait_ms=100,
            )
            for session_index in range(2)
        ]
        self.assertEqual([request.tool_wait_ms for request in requests], [100, 100])
        self.assertEqual([request.mock_tool_result_injected for request in requests], [False, False])
        self.assertTrue(all(mock_result not in str(request.messages) for request in requests))

        requests = [
            build_agent_turn_request(
                request_id=f"one-{session_index}",
                session_id=f"session-{session_index}",
                turn_index=2,
                history=[{"role": "assistant", "content": "Searching."}],
                user_message="Continue.",
                tool_wait_policy="one-session",
                tool_wait_ms=100,
            )
            for session_index in range(2)
        ]
        self.assertEqual([request.tool_wait_ms for request in requests], [100, 0])
        self.assertEqual([request.mock_tool_result_injected for request in requests], [True, False])
        self.assertEqual(requests[0].messages[-1], {"role": "user", "content": f"Continue.\n\n{mock_result}"})
        self.assertNotIn(mock_result, requests[1].messages[-1]["content"])

        first_turn = build_agent_turn_request(
            request_id="first",
            session_id="session-0",
            turn_index=1,
            history=[],
            user_message="Start.",
            tool_wait_policy="one-session",
            tool_wait_ms=100,
        )
        self.assertEqual(first_turn.tool_wait_ms, 0)
        self.assertFalse(first_turn.mock_tool_result_injected)
        self.assertEqual(first_turn.messages[-1], {"role": "user", "content": "Start."})

    def test_other_session_enters_chat_while_one_session_waits(self) -> None:
        self.assertIsNotNone(build_agent_turn_request, "agent turn request builder is missing")
        wait_started = threading.Event()
        release_wait = threading.Event()
        other_chat_entered = threading.Event()

        def controlled_sleep(seconds: float) -> None:
            wait_started.set()
            self.assertTrue(release_wait.wait(timeout=2), "wait was not released")

        class FakeClient:
            def chat(self, *, model: str, messages: list, max_tokens: int) -> SimpleNamespace:
                if messages[-1]["content"] == "session-1":
                    other_chat_entered.set()
                return SimpleNamespace(
                    text="done", input_tokens=2, output_tokens=1, ttft_ms=1.0,
                    tpot_ms=1.0, e2e_ms=2.0, inter_chunk_latencies_ms=[],
                )

        requests = [
            build_agent_turn_request(
                request_id=f"request-{session_index}",
                session_id=f"session-{session_index}",
                turn_index=2,
                history=[],
                user_message=f"session-{session_index}",
                tool_wait_policy="one-session",
                tool_wait_ms=100,
            )
            for session_index in range(2)
        ]
        with patch("benchmarks.course_project.runner.time.sleep", controlled_sleep):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(
                    run_concurrent_requests,
                    client=FakeClient(), requests=requests, concurrency=2,
                    model="test", max_tokens=4,
                )
                self.assertTrue(wait_started.wait(timeout=2), "session-0 did not enter wait")
                self.assertTrue(other_chat_entered.wait(timeout=2), "session-1 did not enter chat")
                release_wait.set()
                records = future.result(timeout=2)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["session_id"]: record["mock_tool_result_injected"] for record in records},
            {"session-0": True, "session-1": False},
        )

    def test_session_contexts_share_prefix_within_run_but_not_across_runs(self) -> None:
        build_contexts = getattr(runner, "build_agent_session_contexts", None)
        self.assertIsNotNone(build_contexts, "agent session context builder is missing")
        kwargs = {
            "concurrency": 2,
            "total_tokens": 24,
            "shared_tokens": 16,
            "encode": lambda value: list(value),
            "decode": lambda values: "".join(values),
        }
        run0 = build_contexts(repetition=0, **kwargs)
        run1 = build_contexts(repetition=1, **kwargs)

        self.assertEqual(len(run0["session-0"]), 24)
        self.assertEqual(run0["session-0"][:16], run0["session-1"][:16])
        self.assertNotEqual(run0["session-0"][16:], run0["session-1"][16:])
        self.assertNotEqual(run0["session-0"][:16], run1["session-0"][:16])

    def test_builds_growing_multi_turn_histories(self) -> None:
        self.assertIsNotNone(build_agent_conversation, "course benchmark core is missing")
        requests = build_agent_conversation(
            session_id="session-0",
            system_prompt="You are a tool-using assistant.",
            tool_schema='{"name":"search"}',
            user_turns=["first", "second", "third"],
            assistant_turns=["answer-1", "answer-2"],
            tool_wait_ms=[0, 100, 500],
        )
        self.assertEqual([request.turn_index for request in requests], [1, 2, 3])
        self.assertEqual([request.tool_wait_ms for request in requests], [0, 100, 500])
        self.assertEqual([len(request.messages) for request in requests], [2, 4, 6])
        self.assertEqual(requests[0].messages[0]["role"], "system")
        self.assertIn("search", requests[0].messages[0]["content"])
        self.assertEqual(requests[2].messages[-1], {"role": "user", "content": "third"})


class TestOpenAIChatClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _StreamingHandler.payloads.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_streams_chat_and_collects_metrics(self) -> None:
        self.assertIsNotNone(OpenAIChatClient, "streaming client is missing")
        port = self.server.server_address[1]
        client = OpenAIChatClient(f"http://127.0.0.1:{port}/v1")
        result = client.chat(
            model="qwen35-0.8b",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
        )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 2)
        self.assertGreaterEqual(result.ttft_ms, 0)
        self.assertGreater(result.e2e_ms, result.ttft_ms)
        self.assertGreater(result.tpot_ms, 0)
        self.assertEqual(len(result.inter_chunk_latencies_ms), 1)
        self.assertGreater(result.inter_chunk_latencies_ms[0], 0)
        payload = _StreamingHandler.payloads[-1]
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["do_sample"])
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])

    def test_runs_concurrent_requests_without_losing_ids(self) -> None:
        self.assertIsNotNone(BenchmarkRequest, "benchmark runner is missing")
        self.assertIsNotNone(run_concurrent_requests, "benchmark runner is missing")
        port = self.server.server_address[1]
        client = OpenAIChatClient(f"http://127.0.0.1:{port}/v1")
        requests = [
            BenchmarkRequest(
                request_id=f"request-{index}",
                messages=[{"role": "user", "content": f"prompt-{index}"}],
                workload="random",
            )
            for index in range(4)
        ]
        records = run_concurrent_requests(
            client=client,
            requests=requests,
            concurrency=2,
            model="qwen35-0.8b",
            max_tokens=8,
        )
        self.assertEqual(
            {record["request_id"] for record in records},
            {"request-0", "request-1", "request-2", "request-3"},
        )
        self.assertTrue(all(record["output_tokens"] == 2 for record in records))
        self.assertTrue(all(record["concurrency"] == 2 for record in records))
        self.assertTrue(all(record["itl_p95_ms"] is None for record in records))
        self.assertTrue(all(record["chunk_latency_p95_ms"] > 0 for record in records))
        summary = summarize_records(records, wall_time_seconds=1.0)
        self.assertIsNone(summary["itl_p95_ms"])
        self.assertGreater(summary["chunk_latency_p95_ms"], 0)


class TestTokenSpeedGenerateClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _GenerateHandler.payloads.clear()
        _GenerateHandler.paths.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _GenerateHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_uses_native_generate_route_and_returns_engine_token_ids(self) -> None:
        self.assertIsNotNone(TokenSpeedGenerateClient, "native generation client is missing")
        port = self.server.server_address[1]
        client = TokenSpeedGenerateClient(f"http://127.0.0.1:{port - 1}/v1")
        result = client.generate(input_token_ids=[1, 2, 3], max_tokens=7)

        self.assertEqual(result.text, "answer")
        self.assertEqual(list(result.output_ids), [41, 42])
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(_GenerateHandler.paths[-1], "/generate")
        self.assertEqual(_GenerateHandler.payloads[-1]["input_ids"], [1, 2, 3])
        self.assertEqual(
            _GenerateHandler.payloads[-1]["sampling_params"],
            {"max_new_tokens": 7, "temperature": 0},
        )

class TestJsonl(unittest.TestCase):
    def test_appends_traceable_records(self) -> None:
        self.assertIsNotNone(append_jsonl, "JSONL writer is missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "run.jsonl"
            append_jsonl(path, {"request_id": "r0", "ttft_ms": 10.0})
            append_jsonl(path, {"request_id": "r1", "ttft_ms": 20.0})
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["request_id"] for record in records], ["r0", "r1"])


class TestSummary(unittest.TestCase):
    def test_fixed_length_benchmark_requests_are_unique_and_same_token_size(self) -> None:
        from scripts import run_benchmark

        requests = run_benchmark._make_requests(
            "c4-i128-r1",
            3,
            "x" * 128,
            encode=lambda value: list(value),
            decode=lambda values: "".join(values),
        )
        contents = [request.messages[0]["content"] for request in requests]

        self.assertEqual(len(set(contents)), 3)
        self.assertEqual([len(content) for content in contents], [128, 128, 128])

    def test_benchmark_rejects_chat_template_context_overflow_before_requests(self) -> None:
        from scripts import run_benchmark

        config_path = Path(__file__).resolve().parents[2] / "configs/bringup/qwen35_0.8b.yaml"
        args = SimpleNamespace(
            config=config_path, output_dir=Path("unused"), input_tokens=[4096],
            output_tokens=128, base_url="http://127.0.0.1:8000/v1",
        )
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [list(range(4096))],
            "attention_mask": [[1] * 4096],
        }
        transformers = SimpleNamespace(AutoTokenizer=Mock())
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        with (
            patch.object(run_benchmark, "parse_args", return_value=args),
            patch.object(run_benchmark, "make_token_sized_text", return_value="prompt"),
            patch.object(run_benchmark, "run_concurrent_requests") as run_requests,
            patch.dict(sys.modules, {"transformers": transformers}),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds max_model_len=4096"):
                run_benchmark.main()
        run_requests.assert_not_called()

    def test_summarizes_latency_and_throughput(self) -> None:
        self.assertIsNotNone(summarize_records, "benchmark summary is missing")
        records = [
            {"ttft_ms": 10.0, "tpot_ms": 2.0, "e2e_ms": 20.0, "input_tokens": 10, "output_tokens": 5},
            {"ttft_ms": 20.0, "tpot_ms": 4.0, "e2e_ms": 40.0, "input_tokens": 20, "output_tokens": 5},
            {"ttft_ms": 30.0, "tpot_ms": 6.0, "e2e_ms": 60.0, "input_tokens": 30, "output_tokens": 5},
            {"ttft_ms": 40.0, "tpot_ms": 8.0, "e2e_ms": 80.0, "input_tokens": 40, "output_tokens": 5},
        ]
        summary = summarize_records(records, wall_time_seconds=0.5)
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["ttft_p50_ms"], 25.0)
        self.assertEqual(summary["ttft_p95_ms"], 38.5)
        self.assertEqual(summary["ttft_mean_ms"], 25.0)
        self.assertEqual(summary["tpot_mean_ms"], 5.0)
        self.assertEqual(summary["e2e_mean_ms"], 50.0)
        self.assertEqual(summary["request_throughput"], 8.0)
        self.assertEqual(summary["token_throughput"], 240.0)

    def test_builds_exact_token_sized_text(self) -> None:
        self.assertIsNotNone(make_token_sized_text, "token-sized prompt builder is missing")
        text = make_token_sized_text(
            target_tokens=257,
            seed="abcdef",
            encode=lambda value: list(value),
            decode=lambda values: "".join(values),
        )
        self.assertEqual(len(text), 257)


if __name__ == "__main__":
    unittest.main()
