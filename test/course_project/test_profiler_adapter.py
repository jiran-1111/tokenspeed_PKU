import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tokenspeed.runtime.engine.profiler_adapter import (
    create_profiler,
    default_profiler_activities,
    resolve_profiler_backend,
    trace_file_name,
)


class TestProfilerAdapter(unittest.TestCase):
    def test_npu_uses_cpu_and_npu_by_default(self) -> None:
        self.assertEqual(default_profiler_activities(is_npu=True), ["CPU", "NPU"])

    def test_npu_uses_torch_npu_profiler(self) -> None:
        torch_profiler = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="torch-cpu", CUDA="torch-cuda")
        )
        npu_profiler = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="npu-cpu", NPU="npu-device")
        )
        torch_module = SimpleNamespace(profiler=torch_profiler)
        torch_npu_module = SimpleNamespace(profiler=npu_profiler)

        backend = resolve_profiler_backend(
            is_npu=True,
            torch_module=torch_module,
            import_module=lambda name: torch_npu_module,
        )

        self.assertIs(backend.profiler, npu_profiler)
        self.assertEqual(backend.activity_map, {"CPU": "npu-cpu", "NPU": "npu-device"})
        self.assertTrue(backend.is_npu)

    def test_npu_trace_is_uncompressed_json(self) -> None:
        self.assertEqual(
            trace_file_name("profile-TP0", is_npu=True), "profile-TP0.trace.json"
        )
        self.assertEqual(
            trace_file_name("profile-TP0", is_npu=False),
            "profile-TP0.trace.json.gz",
        )

    def test_npu_memory_activity_enables_profiler_memory(self) -> None:
        profile = mock.Mock(return_value=object())
        backend = SimpleNamespace(
            profiler=SimpleNamespace(profile=profile),
            activity_map={"CPU": "cpu", "NPU": "npu"},
            is_npu=True,
        )

        result = create_profiler(
            backend,
            activities=["CPU", "NPU", "MEM"],
            with_stack=True,
            record_shapes=True,
        )

        self.assertIsNotNone(result)
        profile.assert_called_once_with(
            activities=["cpu", "npu"],
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
        )

    def test_npu_profiler_level_defaults_to_level0(self) -> None:
        profile = mock.Mock(return_value=object())
        profiler = SimpleNamespace(profile=profile)
        backend = SimpleNamespace(
            profiler=profiler,
            activity_map={"CPU": "cpu", "NPU": "npu"},
            is_npu=True,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            create_profiler(
                backend,
                activities=["CPU", "NPU"],
                with_stack=False,
                record_shapes=False,
            )

        profile.assert_called_once_with(
            activities=["cpu", "npu"],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
        )

    def test_npu_level1_uses_pipe_utilization_experimental_config(self) -> None:
        profile = mock.Mock(return_value=object())
        experimental_config = mock.Mock(return_value="level1-config")
        profiler = SimpleNamespace(
            profile=profile,
            _ExperimentalConfig=experimental_config,
            ProfilerLevel=SimpleNamespace(Level1="level1"),
            AiCMetrics=SimpleNamespace(PipeUtilization="pipe-utilization"),
        )
        backend = SimpleNamespace(
            profiler=profiler,
            activity_map={"CPU": "cpu", "NPU": "npu"},
            is_npu=True,
        )

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_NPU_PROFILER_LEVEL": "Level1"}, clear=True
        ):
            create_profiler(
                backend,
                activities=["CPU", "NPU"],
                with_stack=False,
                record_shapes=False,
            )

        experimental_config.assert_called_once_with(
            profiler_level="level1", aic_metrics="pipe-utilization"
        )
        profile.assert_called_once_with(
            activities=["cpu", "npu"],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            experimental_config="level1-config",
        )

    def test_npu_profiler_rejects_unknown_level(self) -> None:
        backend = SimpleNamespace(
            profiler=SimpleNamespace(profile=mock.Mock()),
            activity_map={"CPU": "cpu", "NPU": "npu"},
            is_npu=True,
        )

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_NPU_PROFILER_LEVEL": "Level2"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "Level2"):
                create_profiler(
                    backend,
                    activities=["CPU", "NPU"],
                    with_stack=False,
                    record_shapes=False,
                )

    def test_npu_level1_fails_when_experimental_config_is_unavailable(self) -> None:
        backend = SimpleNamespace(
            profiler=SimpleNamespace(profile=mock.Mock()),
            activity_map={"CPU": "cpu", "NPU": "npu"},
            is_npu=True,
        )

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_NPU_PROFILER_LEVEL": "Level1"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Level1.*unsupported"):
                create_profiler(
                    backend,
                    activities=["CPU", "NPU"],
                    with_stack=False,
                    record_shapes=False,
                )

    def test_npu_level1_is_rejected_for_non_npu_backend(self) -> None:
        backend = SimpleNamespace(
            profiler=SimpleNamespace(profile=mock.Mock()),
            activity_map={"CPU": "cpu", "GPU": "gpu"},
            is_npu=False,
        )

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_NPU_PROFILER_LEVEL": "Level1"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "only supported on NPU"):
                create_profiler(
                    backend,
                    activities=["CPU", "GPU"],
                    with_stack=False,
                    record_shapes=False,
                )


if __name__ == "__main__":
    unittest.main()
