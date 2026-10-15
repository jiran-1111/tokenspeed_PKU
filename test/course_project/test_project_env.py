import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("bash"), "requires bash")
class TestProjectEnvironment(unittest.TestCase):
    def test_ascend_work_path_is_inside_project(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                'source scripts/use_project_env.sh >/dev/null 2>&1 && printf "%s" "$ASCEND_WORK_PATH"',
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        expected = (REPO_ROOT / ".tmp" / "ascend_work").resolve()
        self.assertEqual(Path(result.stdout).resolve(), expected)
