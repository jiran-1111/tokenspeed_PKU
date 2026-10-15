import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "serve.py"
SPEC = importlib.util.spec_from_file_location("course_project_serve", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
serve_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_script)


class TestProfilerServiceLauncher(unittest.TestCase):
    def _run_launcher(self, argv: list[str]) -> dict[str, str]:
        command = ["python", "-m", "tokenspeed.cli", "serve"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(serve_script, "load_model_config", return_value={}),
            mock.patch.object(
                serve_script, "build_serve_command", return_value=command
            ),
            mock.patch.object(serve_script.subprocess, "run") as run,
            mock.patch.dict(
                os.environ,
                {
                    "PATH": "preserved-path",
                    "TOKENSPEED_NPU_PROFILER_LEVEL": "parent-value",
                    "UNRELATED_SETTING": "preserved-value",
                },
                clear=True,
            ),
        ):
            parent_env = os.environ.copy()
            try:
                serve_script.main()
            except SystemExit as exc:
                self.fail(f"serve.py rejected the profiler option (exit {exc.code})")

            run.assert_called_once()
            self.assertEqual(os.environ, parent_env)
            self.assertEqual(run.call_args.args[0], command)
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertIn("env", run.call_args.kwargs)
            return run.call_args.kwargs["env"]

    def test_level1_option_reaches_child_without_dropping_other_environment(
        self,
    ) -> None:
        child_env = self._run_launcher(
            ["serve.py", "config.yaml", "--npu-profiler-level", "Level1"]
        )

        self.assertEqual(child_env["TOKENSPEED_NPU_PROFILER_LEVEL"], "Level1")
        self.assertEqual(child_env["PATH"], "preserved-path")
        self.assertEqual(child_env["UNRELATED_SETTING"], "preserved-value")

    def test_default_level0_overrides_parent_value_only_in_child(self) -> None:
        child_env = self._run_launcher(["serve.py", "config.yaml"])

        self.assertEqual(child_env["TOKENSPEED_NPU_PROFILER_LEVEL"], "Level0")
        self.assertEqual(child_env["UNRELATED_SETTING"], "preserved-value")


if __name__ == "__main__":
    unittest.main()
