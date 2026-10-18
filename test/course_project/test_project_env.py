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

"""Exercise environment entry points without installing packages or using an NPU."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
if os.name == "nt" and shutil.which("git"):
    git_bash = Path(shutil.which("git")).resolve().parents[1] / "bin" / "bash.exe"
    if git_bash.is_file():
        BASH = str(git_bash)


@unittest.skipUnless(BASH, "requires Bash (Git Bash is supported)")
class TestBashProjectEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = REPO_ROOT / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "test" / "ci_system").mkdir(parents=True)
        for filename in ("setup_env.sh", "use_project_env.sh"):
            shutil.copyfile(
                REPO_ROOT / "scripts" / filename, self.root / "scripts" / filename
            )
        shutil.copyfile(
            REPO_ROOT / "test" / "ci_system" / "install_triton_ascend.sh",
            self.root / "test" / "ci_system" / "install_triton_ascend.sh",
        )
        self.bin = Path(self.temporary.name) / "bin"
        self.bin.mkdir()
        self.conda_base = Path(self.temporary.name) / "conda base"
        hook = self.conda_base / "etc" / "profile.d" / "conda.sh"
        hook.parent.mkdir(parents=True)
        hook.write_text(
            "conda() {\n"
            '  if [[ "$1" == activate ]]; then\n'
            '    export CONDA_PREFIX="$2"\n'
            "  else\n"
            '    command conda "$@"\n'
            "  fi\n"
            "}\n",
            encoding="utf-8",
        )
        conda = self.bin / "conda"
        conda.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1 $2" == "info --base" ]]; then\n'
            '  printf "%s\\n" "$STUB_CONDA_BASE"\n'
            'elif [[ "$1" == create && "$2" == --prefix ]]; then\n'
            '  printf "conda-create\\n" >> "$STUB_LOG"\n'
            '  mkdir -p "$3/conda-meta"\n'
            "else\n"
            "  exit 92\n"
            "fi\n",
            encoding="utf-8",
        )
        conda.chmod(0o755)
        python = self.bin / "python"
        python.write_text(
            "#!/usr/bin/env bash\n"
            'printf "python|%s|%s|%s|%s\\n" "$TOKENSPEED_CANN_ROOT" '
            '"$CANN_ENV" "$CONDA_PKGS_DIRS" "$*" >> "$STUB_LOG"\n',
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.cann = Path(self.temporary.name) / "cann with spaces"
        self.cann.mkdir()
        (self.cann / "set_env.sh").write_text(
            "export STUB_CANN_LOADED=yes\n", encoding="utf-8"
        )
        self.log = Path(self.temporary.name) / "calls.log"
        self.environment = dict(os.environ)
        for name in ("CANN_ENV", "TOKENSPEED_CANN_ROOT", "CONDA_PREFIX"):
            self.environment.pop(name, None)
        self.environment.update(
            STUB_CONDA_BASE=self.conda_base.as_posix(),
            STUB_LOG=self.log.as_posix(),
            STUB_BIN=self.bin.as_posix(),
        )

    def run_script(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                "-c",
                "if command -v cygpath >/dev/null 2>&1; then "
                'STUB_BIN="$(cygpath -u "$STUB_BIN")"; fi; '
                f'export PATH="$STUB_BIN:/usr/bin:/bin"; {command}',
            ],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def resolve_shell_path(self, value: str) -> Path:
        if os.name == "nt":
            converted = subprocess.run(
                [BASH, "--noprofile", "--norc", "-c", 'cygpath -w "$1"', "--", value],
                env=self.environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            value = converted.stdout.strip()
        return Path(value).resolve()

    def test_explicit_cann_script_reaches_install_subprocess(self) -> None:
        self.environment["CANN_ENV"] = (self.cann / "set_env.sh").as_posix()
        result = self.run_script("bash scripts/setup_env.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text(encoding="utf-8").splitlines()
        installers = [line for line in calls if "triton-ascend==" in line]
        self.assertEqual(len(installers), 1)
        parts = installers[0].split("|")
        self.assertEqual(self.resolve_shell_path(parts[1]), self.cann.resolve())
        self.assertEqual(
            self.resolve_shell_path(parts[2]), (self.cann / "set_env.sh").resolve()
        )
        self.assertEqual(
            self.resolve_shell_path(parts[3]), (self.root / ".conda-pkgs").resolve()
        )

    def test_missing_cann_fails_before_create_or_pip(self) -> None:
        self.environment["CANN_ENV"] = (self.cann / "missing.sh").as_posix()
        result = self.run_script("bash scripts/setup_env.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN", result.stderr)
        self.assertFalse(
            self.log.exists(), "invalid CANN must not create an environment or install"
        )

    def test_conflicting_cann_settings_fail_before_create_or_pip(self) -> None:
        self.environment["CANN_ENV"] = (self.cann / "set_env.sh").as_posix()
        other = self.cann.parent / "other cann"
        other.mkdir()
        (other / "set_env.sh").write_text("true\n", encoding="utf-8")
        self.environment["TOKENSPEED_CANN_ROOT"] = other.as_posix()
        result = self.run_script("bash scripts/setup_env.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN", result.stderr)
        self.assertFalse(self.log.exists())

    def test_activation_rejects_missing_local_environment(self) -> None:
        self.environment["CANN_ENV"] = (self.cann / "set_env.sh").as_posix()
        result = self.run_script("source scripts/use_project_env.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup_env.sh", result.stderr)
        self.assertFalse(self.log.exists())

    def test_activation_keeps_cache_and_temporary_paths_in_project(self) -> None:
        self.environment["TOKENSPEED_CANN_ROOT"] = self.cann.as_posix()
        (self.root / ".conda-env" / "conda-meta").mkdir(parents=True)
        result = self.run_script(
            'source scripts/use_project_env.sh && printf "%s\\n" '
            '"$CONDA_PREFIX" "$CONDA_PKGS_DIRS" "$TMPDIR" "$ASCEND_WORK_PATH"'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = [".conda-env", ".conda-pkgs", ".tmp", ".tmp/ascend_work"]
        self.assertEqual(
            [self.resolve_shell_path(line) for line in result.stdout.splitlines()],
            [(self.root / path).resolve() for path in expected],
        )

    def test_failed_cann_script_does_not_create_or_install(self) -> None:
        (self.cann / "set_env.sh").write_text("return 37\n", encoding="utf-8")
        self.environment["CANN_ENV"] = (self.cann / "set_env.sh").as_posix()
        result = self.run_script("bash scripts/setup_env.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_activation_preserves_the_callers_nounset_setting(self) -> None:
        self.environment["TOKENSPEED_CANN_ROOT"] = self.cann.as_posix()
        (self.root / ".conda-env" / "conda-meta").mkdir(parents=True)
        for option, expected in (("+u", "off"), ("-u", "on")):
            with self.subTest(option=option):
                result = self.run_script(
                    f"set {option}; source scripts/use_project_env.sh && "
                    "if [[ $- == *u* ]]; then printf on; else printf off; fi"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "requires Windows PowerShell")
class TestPowerShellProjectEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = REPO_ROOT / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        (self.root / "scripts").mkdir(parents=True)
        for filename in ("setup_env.ps1", "use_project_env.ps1"):
            shutil.copyfile(
                REPO_ROOT / "scripts" / filename, self.root / "scripts" / filename
            )
        self.environment = dict(os.environ)
        self.environment["STUB_ROOT"] = str(self.root)
        self.environment.pop("CONDA_PREFIX", None)

    def create_environment(self, prefix: Path) -> None:
        (prefix / "conda-meta").mkdir(parents=True)
        (prefix / "python.exe").touch()

    def run_script(self, body: str) -> subprocess.CompletedProcess:
        runner = Path(self.temporary.name) / "runner.ps1"
        runner.write_text(
            "$ErrorActionPreference = 'Stop'\n"
            "function conda {\n"
            "  if ($args[0] -ne 'shell.powershell' -or $args[1] -ne 'hook') { throw 'Unexpected conda command' }\n"
            "  @'\n"
            "function global:conda {\n"
            "  if ($args[0] -ne 'activate') { throw 'Unexpected conda command' }\n"
            "  $prefix = [System.IO.Path]::GetFullPath($args[1])\n"
            "  $root = [System.IO.Path]::GetPathRoot($prefix)\n"
            "  if ($prefix.Length -gt $root.Length) { $prefix = $prefix.TrimEnd([char[]]@('\\', '/')) }\n"
            "  $env:CONDA_PREFIX = $prefix\n"
            "  $global:LASTEXITCODE = 0\n"
            "}\n"
            "'@\n"
            "}\n"
            "try {\n"
            f"{body}\n"
            "} catch {\n"
            "  [Console]::Error.WriteLine($_.Exception.Message)\n"
            "  exit 1\n"
            "}\n",
            encoding="utf-8",
            newline="\n",
        )
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(runner)],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_default_activation_uses_checkout_environment_and_caches(self) -> None:
        self.create_environment(self.root / ".conda-env")
        result = self.run_script(
            ". (Join-Path $env:STUB_ROOT 'scripts/use_project_env.ps1')\n"
            "@{ prefix=$env:CONDA_PREFIX; pkgs=$env:CONDA_PKGS_DIRS; temp=$env:TEMP; "
            "ascend=$env:ASCEND_WORK_PATH } | ConvertTo-Json -Compress"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        expected = {
            "prefix": ".conda-env",
            "pkgs": ".conda-pkgs",
            "temp": ".tmp",
            "ascend": ".tmp/ascend_work",
        }
        self.assertEqual(
            {name: Path(value).resolve() for name, value in observed.items()},
            {name: (self.root / value).resolve() for name, value in expected.items()},
        )

    def test_explicit_existing_environment_keeps_caches_in_checkout(self) -> None:
        prefix = Path(self.temporary.name) / "existing environment"
        self.create_environment(prefix)
        self.environment["STUB_ENV_PREFIX"] = str(prefix)
        result = self.run_script(
            ". (Join-Path $env:STUB_ROOT 'scripts/use_project_env.ps1') -EnvironmentPrefix $env:STUB_ENV_PREFIX\n"
            "@{ prefix=$env:CONDA_PREFIX; pkgs=$env:CONDA_PKGS_DIRS } | ConvertTo-Json -Compress"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(Path(observed["prefix"]).resolve(), prefix.resolve())
        self.assertEqual(
            Path(observed["pkgs"]).resolve(), (self.root / ".conda-pkgs").resolve()
        )

    def test_missing_default_does_not_reuse_an_unrequested_environment(self) -> None:
        result = self.run_script(
            ". (Join-Path $env:STUB_ROOT 'scripts/use_project_env.ps1')"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup_env.ps1", result.stderr)

    def test_relative_activation_uses_powershell_location(self) -> None:
        prefix = self.root / "nested" / "existing environment"
        self.create_environment(prefix)
        self.environment["STUB_PROCESS_ROOT"] = self.temporary.name
        result = self.run_script(
            "[Environment]::CurrentDirectory = $env:STUB_PROCESS_ROOT\n"
            "Push-Location $env:STUB_ROOT\n"
            "try {\n"
            "  . (Join-Path $env:STUB_ROOT 'scripts/use_project_env.ps1') -EnvironmentPrefix './nested/existing environment'\n"
            "  @{ prefix=$env:CONDA_PREFIX; process=[Environment]::CurrentDirectory; location=$PWD.ProviderPath } | ConvertTo-Json -Compress\n"
            "} finally { Pop-Location }"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(Path(observed["prefix"]).resolve(), prefix.resolve())
        self.assertNotEqual(observed["process"], observed["location"])

    def test_trailing_separator_matches_condas_normalized_prefix(self) -> None:
        prefix = Path(self.temporary.name) / "existing environment"
        self.create_environment(prefix)
        self.environment["STUB_ENV_PREFIX"] = f"{prefix}{os.sep}"
        result = self.run_script(
            ". (Join-Path $env:STUB_ROOT 'scripts/use_project_env.ps1') -EnvironmentPrefix $env:STUB_ENV_PREFIX\n"
            "$env:CONDA_PREFIX"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(), prefix.resolve())

    def test_relative_setup_uses_powershell_location_before_create(self) -> None:
        requirements = self.root / "requirements"
        requirements.mkdir()
        (requirements / "course-test.txt").write_text(
            "pytest==8.4.2\n", encoding="utf-8", newline="\n"
        )
        self.environment["STUB_PROCESS_ROOT"] = self.temporary.name
        result = self.run_script(
            "function conda {\n"
            "  if ($args[0] -ne 'create') { throw 'Unexpected conda command' }\n"
            "  Write-Output $args[2]\n"
            "  $global:LASTEXITCODE = 19\n"
            "}\n"
            "[Environment]::CurrentDirectory = $env:STUB_PROCESS_ROOT\n"
            "Push-Location $env:STUB_ROOT\n"
            "try {\n"
            "  & (Join-Path $env:STUB_ROOT 'scripts/setup_env.ps1') -EnvironmentPrefix './nested/new environment/'\n"
            "} finally { Pop-Location }"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not create", result.stderr)
        self.assertEqual(
            result.stdout.strip(), str(self.root / "nested" / "new environment")
        )

    def test_setup_preserves_a_drive_root_separator(self) -> None:
        requirements = self.root / "requirements"
        requirements.mkdir()
        (requirements / "course-test.txt").write_text(
            "pytest==8.4.2\n", encoding="utf-8", newline="\n"
        )
        result = self.run_script(
            "function conda {\n"
            "  if ($args[0] -ne 'create') { throw 'Unexpected conda command' }\n"
            "  Write-Output $args[2]\n"
            "  $global:LASTEXITCODE = 19\n"
            "}\n"
            "& (Join-Path $env:STUB_ROOT 'scripts/setup_env.ps1') -EnvironmentPrefix ([System.IO.Path]::GetPathRoot($env:STUB_ROOT))"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not create", result.stderr)
        self.assertEqual(result.stdout.strip(), self.root.anchor)

    def test_failed_environment_creation_stops_before_installing(self) -> None:
        requirements = self.root / "requirements"
        requirements.mkdir()
        (requirements / "course-test.txt").write_text(
            "pytest==8.4.2\n", encoding="utf-8"
        )
        result = self.run_script(
            "function conda { $global:LASTEXITCODE = 19 }\n"
            "& (Join-Path $env:STUB_ROOT 'scripts/setup_env.ps1')"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not create", result.stderr)
        self.assertFalse((self.root / ".conda-env").exists())
