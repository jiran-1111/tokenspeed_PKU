from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_agent_benchmark as agent_script

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeDateTime:
    @staticmethod
    def now(_timezone):
        return SimpleNamespace(strftime=lambda _format: "20261010T120000Z")


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, token_ids):
        return "".join(token_ids)


class TestAgentProfilerCapture(unittest.TestCase):
    def test_profile_flags_are_opt_in(self) -> None:
        with patch.object(sys, "argv", ["run_agent_benchmark.py", "fake.yaml"]):
            default_args = agent_script.parse_args()
        self.assertFalse(default_args.profile)
        self.assertFalse(default_args.profile_single_request_ttft)
        self.assertEqual(
            default_args.profile_dir, agent_script.REPO_ROOT / "results" / "profiles"
        )

        with patch.object(
            sys,
            "argv",
            [
                "run_agent_benchmark.py",
                "fake.yaml",
                "--profile",
                "--profile-dir",
                "custom-profiles",
            ],
        ):
            profile_args = agent_script.parse_args()
        self.assertTrue(profile_args.profile)
        self.assertEqual(profile_args.profile_dir, Path("custom-profiles"))

        with patch.object(
            sys,
            "argv",
            [
                "run_agent_benchmark.py",
                "fake.yaml",
                "--profile-single-request-ttft",
            ],
        ):
            single_profile_args = agent_script.parse_args()
        self.assertTrue(single_profile_args.profile_single_request_ttft)

    def test_single_request_ttft_profile_rejects_non_pinned_workload(self) -> None:
        args = self._args(profile=False, concurrency=[4], ratios=[0.5], runs=1)
        args.prefix_tokens = 512
        args.turns = 8
        args.tool_wait_ms = 100
        args.tool_wait_policy = "one-session"
        args.output_tokens = 32
        args.profile_single_request_ttft = True

        with self.assertRaisesRegex(ValueError, "concurrency"):
            agent_script.validate_single_request_ttft_args(args)

    def test_invalid_single_request_profile_stops_before_loading_or_writing(
        self,
    ) -> None:
        args = self._args(profile=False, concurrency=[4], ratios=[0.5], runs=1)
        args.prefix_tokens = 512
        args.turns = 8
        args.tool_wait_ms = 100
        args.tool_wait_policy = "one-session"
        args.output_tokens = 32
        args.profile_single_request_ttft = True
        profiler_factory = Mock()

        with self._patches(
            args,
            profiler_factory=profiler_factory,
            run_requests=Mock(),
        ) as patched:
            with self.assertRaisesRegex(ValueError, "concurrency"):
                agent_script.main()

        patched["tokenizer_factory"].assert_not_called()
        patched["load_model_config"].assert_not_called()
        patched["mkdir"].assert_not_called()
        patched["append_jsonl"].assert_not_called()
        profiler_factory.assert_not_called()

    def test_single_request_profile_requires_the_approved_model_configuration(
        self,
    ) -> None:
        config = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "served_model_name": "qwen35-0.8b",
            "device": "npu",
            "dtype": "float16",
            "tp_size": 1,
        }

        with self.assertRaisesRegex(ValueError, "dtype"):
            agent_script.validate_single_request_ttft_config(config)

    def test_single_request_profile_paths_are_unique_and_refuse_existing_targets(
        self,
    ) -> None:
        paths = agent_script.single_request_ttft_paths("run-1")

        self.assertEqual(
            paths[0], REPO_ROOT / "results/raw/profiler/single_request_ttft/run-1"
        )
        self.assertEqual(
            paths[1],
            REPO_ROOT / "results/profiles/qwen35-0.8b/single_request_ttft/run-1",
        )
        self.assertEqual(
            paths[2], REPO_ROOT / "results/analysis/profiler_single_request_ttft_run-1"
        )
        self.assertTrue(all(path.is_relative_to(REPO_ROOT) for path in paths))

        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                agent_script.ensure_single_request_ttft_paths_available(paths)

    def test_single_request_profile_wraps_only_the_final_one_token_turn(self) -> None:
        result = self._run_single_request_profile()

        self.assertIsNone(result.exception_type)
        self.assertEqual(len(result.client.calls), 8)
        self.assertEqual(
            [call["max_tokens"] for call in result.client.calls], [32] * 7 + [1]
        )
        self.assertEqual(
            [call["has_first_chunk_callback"] for call in result.client.calls],
            [False] * 7 + [True],
        )
        self.assertEqual(result.profiler.start_calls, 1)
        self.assertEqual(result.profiler.stop_calls, 1)
        self.assertEqual(len(result.raw_records), 1)
        self.assertEqual(len(result.summaries), 1)
        record = result.raw_records[0]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["turn_index"], 8)
        self.assertEqual(record["output_tokens"], 1)
        self.assertTrue(record["stream_completed"])
        self.assertEqual(record["first_chunk_field"], "content")
        self.assertNotIn("output_text", record)
        self.assertNotIn("messages", record)
        self.assertNotIn("Authorization", record)
        self.assertNotIn("api_key", record)
        self.assertTrue(
            any(call.args[0].name == "request.jsonl" for call in result.append_calls)
        )
        profile_id = next(
            event[1] for event in result.events if event[0] == "profile_start"
        )
        self.assertIn("single-request-ttft", profile_id)

        final_wait = max(
            index
            for index, event in enumerate(result.events)
            if event[0] == "tool_wait"
        )
        before_snapshot = next(
            index
            for index, event in enumerate(result.events)
            if event == ("npu_snapshot", "npu0-before.txt")
        )
        profile_start = next(
            index
            for index, event in enumerate(result.events)
            if event[0] == "profile_start"
        )
        final_send = next(
            index
            for index, event in enumerate(result.events)
            if event[:3] == ("chat_send", 7, 1)
        )
        profile_stop = next(
            index
            for index, event in enumerate(result.events)
            if event[0] == "profile_stop"
        )
        final_drain = next(
            index
            for index, event in enumerate(result.events)
            if event == ("stream_drain", 7)
        )
        after_snapshot = next(
            index
            for index, event in enumerate(result.events)
            if event == ("npu_snapshot", "npu0-after.txt")
        )
        self.assertLess(final_wait, before_snapshot)
        self.assertLess(before_snapshot, profile_start)
        self.assertLess(profile_start, final_send)
        self.assertLess(final_send, profile_stop)
        self.assertLess(profile_stop, final_drain)
        self.assertLess(final_drain, after_snapshot)

    def test_invalid_single_request_profile_never_writes_success_summary(self) -> None:
        for mode in ("no-first-chunk", "wrong-output-count", "missing-done"):
            with self.subTest(mode=mode):
                result = self._run_single_request_profile(client_mode=mode)

                self.assertEqual(result.exception_type, "RuntimeError")
                self.assertEqual(len(result.raw_records), 1)
                self.assertEqual(result.raw_records[0]["status"], "invalid")
                self.assertFalse(result.summaries)

    def test_single_request_profile_records_errors_and_attempts_cleanup(self) -> None:
        for mode, expected_stop_calls in (
            ("start-error", 1),
            ("request-error", 1),
            ("stop-error-once", 2),
        ):
            with self.subTest(mode=mode):
                result = self._run_single_request_profile(client_mode=mode)

                self.assertEqual(result.exception_type, "RuntimeError")
                self.assertEqual(result.profiler.stop_calls, expected_stop_calls)
                self.assertFalse(result.summaries)
                self.assertEqual(len(result.raw_records), 1)
                self.assertEqual(result.raw_records[0]["status"], "failed")
                self.assertIn("failure_type", result.raw_records[0])
                if mode == "request-error":
                    self.assertEqual(result.raw_records[0]["failure_type"], "OSError")
                self.assertNotIn("do not serialize", str(result.raw_records[0]))

    def _run_single_request_profile(self, *, client_mode="success"):
        events = []
        args = self._args(profile=False, concurrency=[1], ratios=[0.5], runs=1)
        args.config = REPO_ROOT / "configs" / "bringup" / "qwen35_0.8b.yaml"
        args.prefix_tokens = 512
        args.output_tokens = 32
        args.profile_single_request_ttft = True
        args.warmup_requests = 0

        class FakeProfiler:
            def __init__(self):
                self.start_calls = 0
                self.stop_calls = 0

            def start(self, *, output_dir, profile_id):
                self.start_calls += 1
                events.append(("profile_start", profile_id, str(output_dir)))
                if client_mode == "start-error":
                    raise RuntimeError("do not serialize start failure")
                return {"success": True}

            def stop(self):
                self.stop_calls += 1
                events.append(("profile_stop", self.stop_calls))
                if client_mode == "stop-error-once" and self.stop_calls == 1:
                    raise RuntimeError("do not serialize stop failure")
                return {"success": True}

        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, *, model, messages, max_tokens, on_first_chunk=None):
                index = len(self.calls)
                self.calls.append(
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "has_first_chunk_callback": on_first_chunk is not None,
                    }
                )
                events.append(
                    ("chat_send", index, max_tokens, on_first_chunk is not None)
                )
                if index == 7 and client_mode == "request-error":
                    raise OSError("do not serialize request failure")
                base_epoch_ns = 1_800_000_000_000_000_000 + index * 10_000_000
                base_monotonic_ns = 500_000_000_000 + index * 10_000_000
                first_epoch_ns = base_epoch_ns + 1_000_000
                first_monotonic_ns = base_monotonic_ns + 1_000_000
                if (
                    index == 7
                    and on_first_chunk is not None
                    and client_mode != "no-first-chunk"
                ):
                    events.append(("first_chunk", index))
                    on_first_chunk("content", first_epoch_ns, first_monotonic_ns)
                events.append(("stream_drain", index))
                output_tokens = max_tokens
                stream_completed = True
                first_field = "content"
                if index == 7 and client_mode == "no-first-chunk":
                    output_tokens = 1
                    first_epoch_ns = None
                    first_monotonic_ns = None
                    first_field = None
                if index == 7 and client_mode == "wrong-output-count":
                    output_tokens = 2
                if index == 7 and client_mode == "missing-done":
                    stream_completed = False
                return SimpleNamespace(
                    text="answer",
                    input_tokens=2,
                    output_tokens=output_tokens,
                    ttft_ms=1.0,
                    tpot_ms=1.0,
                    e2e_ms=2.0,
                    inter_chunk_latencies_ms=(),
                    request_send_epoch_ns=base_epoch_ns,
                    request_send_monotonic_ns=base_monotonic_ns,
                    first_chunk_epoch_ns=first_epoch_ns,
                    first_chunk_monotonic_ns=first_monotonic_ns,
                    stream_drain_epoch_ns=base_epoch_ns + 2_000_000,
                    stream_drain_monotonic_ns=base_monotonic_ns + 2_000_000,
                    first_chunk_field=first_field,
                    stream_completed=stream_completed,
                )

        profiler = FakeProfiler()
        client = FakeClient()
        model_config = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "fixture-revision",
            "served_model_name": "qwen35-0.8b",
            "device": "npu",
            "dtype": "bfloat16",
            "tp_size": 1,
            "tokenspeed_metrics_port": 39142,
        }
        capture_npu = Mock(
            side_effect=lambda path: events.append(("npu_snapshot", Path(path).name))
            or True
        )

        with self._patches(
            args,
            profiler_factory=Mock(return_value=profiler),
            run_requests=None,
            client=client,
            model_config=model_config,
            capture_npu=capture_npu,
        ) as patched:
            with patch.object(
                agent_script.time,
                "sleep",
                side_effect=lambda seconds: events.append(("tool_wait", seconds)),
            ):
                try:
                    agent_script.main()
                except BaseException as error:
                    exception_type = type(error).__name__
                else:
                    exception_type = None

        append_calls = patched["append_jsonl"].call_args_list
        raw_records = [
            call.args[1]
            for call in append_calls
            if call.args[0].name == "request.jsonl"
        ]
        summaries = [
            call.args[1]
            for call in append_calls
            if call.args[0].name == "summary.jsonl"
        ]
        return SimpleNamespace(
            exception_type=exception_type,
            events=events,
            client=client,
            profiler=profiler,
            append_calls=append_calls,
            raw_records=raw_records,
            summaries=summaries,
        )

    def test_single_request_raw_preserves_send_and_drain_timestamps(self) -> None:
        result = self._run_single_request_profile()
        self.assertIsNone(result.exception_type)
        self.assertEqual(len(result.raw_records), 1)
        raw = result.raw_records[0]
        epoch = 1_800_000_000_000_000_000 + 7 * 10_000_000
        monotonic = 500_000_000_000 + 7 * 10_000_000
        self.assertEqual(raw["request_send_epoch_ns"], epoch)
        self.assertEqual(raw["request_send_monotonic_ns"], monotonic)
        self.assertEqual(raw["stream_drain_epoch_ns"], epoch + 2_000_000)
        self.assertEqual(raw["stream_drain_monotonic_ns"], monotonic + 2_000_000)
        self.assertEqual(raw["first_chunk_epoch_ns"], epoch + 1_000_000)

    def test_control_url_keeps_api_host_and_port_and_drops_v1_path(self) -> None:
        self.assertEqual(
            agent_script.profiler_control_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8001",
        )

    def test_control_url_preserves_custom_base_path(self) -> None:
        self.assertEqual(
            agent_script.profiler_control_url(
                "https://example.invalid:8443/gateway/v1/"
            ),
            "https://example.invalid:8444/gateway",
        )
        self.assertEqual(
            agent_script.profiler_control_url("https://example.invalid:8443/api"),
            "https://example.invalid:8444/api",
        )

    def test_control_url_handles_ipv6_and_control_port_boundary(self) -> None:
        self.assertEqual(
            agent_script.profiler_control_url("http://[::1]:8000/v1"),
            "http://[::1]:8001",
        )
        with self.assertRaisesRegex(ValueError, "control server port"):
            agent_script.profiler_control_url("http://127.0.0.1:65535/v1")

    def test_profile_wraps_each_full_eight_turn_repetition_and_records_linkage(
        self,
    ) -> None:
        args = self._args(profile=True, concurrency=[1, 2], ratios=[0.5, 1.0], runs=2)
        events: list[tuple] = []
        active_profile: dict[str, str | None] = {"id": None}

        def start_profile(*, output_dir, profile_id):
            self.assertIsNone(active_profile["id"])
            active_profile["id"] = profile_id
            events.append(("start", profile_id, str(output_dir)))

        def stop_profile():
            events.append(("stop", active_profile["id"]))
            active_profile["id"] = None
            return {}

        profiler = Mock()
        profiler.start.side_effect = start_profile
        profiler.stop.side_effect = stop_profile
        profiler_factory = Mock(return_value=profiler)

        def run_requests(*, requests, **_kwargs):
            turn_index = requests[0].turn_index
            events.append(("turn", active_profile["id"], turn_index))
            return [
                {
                    "session_id": request.session_id,
                    "output_text": f"answer-{turn_index}",
                }
                for request in requests
            ]

        with self._patches(
            args,
            profiler_factory=profiler_factory,
            run_requests=run_requests,
        ) as patched:
            agent_script.main()

        self.assertEqual(profiler_factory.call_count, 1)
        self.assertEqual(profiler.start.call_count, 8)
        self.assertEqual(profiler.stop.call_count, 8)
        self.assertEqual(
            len([event for event in events if event[0] == "turn"]),
            8 * 8,
        )

        profile_ids = [event[1] for event in events if event[0] == "start"]
        expected_profile_ids = {
            f"20261010T120000Z-c{concurrency}-p{ratio}"
            f"-t4-n{round(4 * ratio)}-r{repetition}"
            for concurrency in (1, 2)
            for ratio in (0.5, 1.0)
            for repetition in (1, 2)
        }
        self.assertEqual(set(profile_ids), expected_profile_ids)
        for index, event in enumerate(events):
            if event[0] != "start":
                continue
            profile_id = event[1]
            following = events[index + 1 :]
            end = next(i for i, item in enumerate(following) if item[0] == "stop")
            captured = following[:end]
            self.assertEqual(
                [item[2] for item in captured if item[0] == "turn"],
                list(range(1, 9)),
            )
            self.assertTrue(all(item[1] == profile_id for item in captured))
            self.assertEqual(following[end][1], profile_id)

        writes = patched["append_jsonl"].call_args_list
        raw_records = [
            call.args[1]
            for call in writes
            if not str(call.args[0]).endswith(".summary.jsonl")
        ]
        summaries = [
            call.args[1]
            for call in writes
            if str(call.args[0]).endswith(".summary.jsonl")
        ]
        self.assertEqual(len(summaries), 8)
        self.assertTrue(raw_records)
        for record in raw_records + summaries:
            self.assertIn("profile_id", record)
            self.assertIn("profile_dir", record)
            self.assertIn("20261010T120000Z", record["profile_id"])
            self.assertIn(record["profile_id"], Path(record["profile_dir"]).name)
            self.assertIn("-c", record["profile_id"])
            self.assertIn("-p", record["profile_id"])
            self.assertIn("-t4", record["profile_id"])
            self.assertIn("-n", record["profile_id"])
            self.assertIn("-r", record["profile_id"])

        self.assertEqual(
            {summary["profile_id"] for summary in summaries}, set(profile_ids)
        )
        self.assertEqual(
            {record["profile_id"] for record in raw_records}, set(profile_ids)
        )
        self.assertEqual(profiler_factory.call_args.args[0], "http://127.0.0.1:8001")

    def test_profiler_stops_if_a_turn_raises(self) -> None:
        args = self._args(profile=True, concurrency=[1], ratios=[0.5], runs=1)
        profiler = Mock()
        profiler.stop.side_effect = RuntimeError("stop failed")
        run_requests = Mock(side_effect=RuntimeError("workload failed"))

        with self._patches(
            args,
            profiler_factory=Mock(return_value=profiler),
            run_requests=run_requests,
        ):
            with self.assertRaisesRegex(RuntimeError, "workload failed"):
                agent_script.main()

        profiler.start.assert_called_once()
        profiler.stop.assert_called_once()

    def test_profiler_stop_is_attempted_when_start_raises(self) -> None:
        args = self._args(profile=True, concurrency=[1], ratios=[0.5], runs=1)
        profiler = Mock()
        profiler.start.side_effect = RuntimeError("start timed out")
        profiler.stop.side_effect = RuntimeError("stop failed")

        with self._patches(
            args,
            profiler_factory=Mock(return_value=profiler),
            run_requests=Mock(),
        ):
            with self.assertRaisesRegex(RuntimeError, "start timed out"):
                agent_script.main()

        profiler.start.assert_called_once()
        profiler.stop.assert_called_once()

    def test_default_run_does_not_start_profiler_or_add_profile_fields(self) -> None:
        args = self._args(profile=None, concurrency=[1], ratios=[0.5], runs=1)
        profiler_factory = Mock()

        with self._patches(
            args,
            profiler_factory=profiler_factory,
            run_requests=lambda *, requests, **_kwargs: [
                {"session_id": request.session_id, "output_text": "answer"}
                for request in requests
            ],
        ) as patched:
            agent_script.main()

        profiler_factory.assert_not_called()
        records = [call.args[1] for call in patched["append_jsonl"].call_args_list]
        self.assertTrue(records)
        self.assertTrue(all("profile_id" not in record for record in records))
        self.assertTrue(all("profile_dir" not in record for record in records))

    def _args(self, *, profile, concurrency, ratios, runs):
        args = SimpleNamespace(
            config=Path("fake.yaml"),
            output_dir=Path("unused-output"),
            concurrency=concurrency,
            shared_prefix_ratios=ratios,
            prefix_tokens=4,
            turns=8,
            tool_wait_ms=100,
            tool_wait_policy="one-session",
            output_tokens=1,
            warmup_requests=0,
            runs=runs,
            base_url="http://127.0.0.1:8000/v1",
            profile_single_request_ttft=False,
        )
        if profile is not None:
            args.profile = profile
            args.profile_dir = Path("unused-profiles")
        return args

    def _patches(
        self,
        args,
        *,
        profiler_factory,
        run_requests,
        client=None,
        model_config=None,
        capture_npu=None,
    ):
        class _PatchContext:
            def __enter__(self_inner):
                self_inner.stack = []
                self_inner.values = {}
                tokenizer_factory = Mock(return_value=_FakeTokenizer())
                transformers = SimpleNamespace(AutoTokenizer=tokenizer_factory)
                config = model_config or {
                    "served_model_name": "fake-model",
                    "model_id": "fake-id",
                    "model_revision": "fake-revision",
                    "tp_size": 1,
                }
                patches = [
                    patch.dict(
                        sys.modules,
                        {"transformers": transformers},
                    ),
                    patch.object(agent_script, "parse_args", return_value=args),
                    patch.object(
                        agent_script,
                        "load_model_config",
                        return_value=config,
                    ),
                    patch.object(agent_script, "datetime", _FakeDateTime),
                    patch.object(
                        agent_script,
                        "model_load_path",
                        return_value=str(REPO_ROOT / "configs" / "bringup"),
                    ),
                    patch.object(
                        agent_script,
                        "OpenAIChatClient",
                        return_value=client if client is not None else object(),
                    ),
                    patch.object(
                        agent_script, "metrics_server_url", return_value="http://unused"
                    ),
                    patch.object(
                        agent_script,
                        "build_agent_session_contexts",
                        side_effect=lambda *, concurrency, **_kwargs: {
                            f"session-{index}": "context"
                            for index in range(concurrency)
                        },
                    ),
                    patch.object(
                        agent_script,
                        "ProfilerControlClient",
                        profiler_factory,
                        create=True,
                    ),
                    patch.object(
                        agent_script,
                        "ServerMetricsSampler",
                        return_value=SimpleNamespace(
                            start=Mock(), stop=Mock(return_value={})
                        ),
                    ),
                    patch.object(
                        agent_script, "snapshot_server_metrics", return_value={}
                    ),
                    patch.object(
                        agent_script,
                        "capture_npu_smi",
                        side_effect=capture_npu,
                        return_value=False,
                    ),
                    patch.object(
                        agent_script,
                        "summarize_records",
                        side_effect=lambda *_args, **_kwargs: {},
                    ),
                    patch.object(agent_script, "add_metric_delta"),
                    patch.object(agent_script, "append_jsonl"),
                    patch.object(Path, "mkdir"),
                    patch.object(Path, "exists", return_value=False),
                    patch.object(Path, "read_bytes", return_value=b"fixture-config"),
                    patch.object(Path, "write_text"),
                ]
                if run_requests is not None:
                    patches.append(
                        patch.object(
                            agent_script,
                            "run_concurrent_requests",
                            side_effect=run_requests,
                        )
                    )
                for item in patches:
                    self_inner.stack.append(item)
                    started = item.start()
                    if getattr(item, "attribute", None) == "append_jsonl":
                        self_inner.values["append_jsonl"] = started
                    if getattr(item, "attribute", None) == "load_model_config":
                        self_inner.values["load_model_config"] = started
                    if getattr(item, "attribute", None) == "mkdir":
                        self_inner.values["mkdir"] = started
                self_inner.values["tokenizer_factory"] = tokenizer_factory
                return self_inner.values

            def __exit__(self_inner, exc_type, exc_value, traceback):
                for item in reversed(self_inner.stack):
                    item.stop()

        return _PatchContext()


if __name__ == "__main__":
    unittest.main()
