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

"""Prepare fresh software/model receipts without starting an NPU experiment.

Run with the activated project Python in a short, isolated source checkout.
The scheduler extension and fixed-revision model files must already be present.
This post-measurement helper is separately versioned from the measured tools.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def file_origin(module, repo):
    path = Path(module.__file__).resolve(strict=True)
    if not path.is_relative_to(repo):
        raise ValueError("import resolved outside the selected snapshot")
    return {"relative_path": path.relative_to(repo).as_posix(), "sha256": sha(path)}


def probe(repo, output):
    import tokenspeed_scheduler
    from tokenspeed_scheduler import tokenspeed_scheduler_ext

    import tokenspeed

    write_json(
        output,
        {
            name: file_origin(module, repo)
            for name, module in (
                ("tokenspeed", tokenspeed),
                ("tokenspeed_scheduler", tokenspeed_scheduler),
                ("scheduler_extension", tokenspeed_scheduler_ext),
            )
        },
    )


def model_state(repo, expected_models):
    state = {"models": {}, "checked_utc": datetime.now(timezone.utc).isoformat()}
    for model in expected_models:
        name = model["model_id"].split("/")[-1]
        root = (repo / "models" / name).resolve(strict=True)
        if not root.is_relative_to(repo):
            raise ValueError("model directory must remain inside this snapshot")
        files = {}
        for filename, expected in model["files"].items():
            path = (root / filename).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("model file escapes its directory")
            before = path.stat()
            actual_sha = sha(path)
            after = path.stat()
            stat = lambda value: [
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
            ]
            if (
                stat(before) != stat(after)
                or after.st_size != expected["bytes"]
                or actual_sha != expected["sha256"]
            ):
                raise ValueError(f"model identity mismatch: {name}/{filename}")
            files[filename] = {"stat": stat(after), "sha256": actual_sha}
        state["models"][model["model_id"]] = {
            "revision": model["model_revision"],
            "files": files,
        }
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "workspace", "task-dir", "environment-script"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    workspace = args.workspace.resolve(strict=True)
    task = args.task_dir.resolve()
    if not repo.is_relative_to(workspace) or task.parent != workspace / ".tmp":
        raise ValueError("source and fresh task must be inside the workspace")
    if task.exists() or not args.environment_script.is_file():
        raise ValueError("task already exists or environment script is missing")
    if len(str(repo / ".tmp")) > 58:
        raise ValueError("use a shorter isolated source path for forkserver sockets")
    evidence = repo / "docs/evidence/project_agent_20261031"
    runners = evidence / "runners"
    source_manifest = evidence / "source_manifest.json"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(runners))
    from controller import dependency_versions

    from benchmarks.course_project.tp_devices import check_sources

    software = json.loads((evidence / "software_current.json").read_text())
    if check_sources(repo, source_manifest) != software["source_manifest_sha256"]:
        raise ValueError("selected source differs from the measured source")
    if software["precommit"]["exit_code"] != 0:
        raise ValueError("measured-source precommit receipt failed")
    extension = repo / (
        "tokenspeed-scheduler/python/tokenspeed_scheduler/"
        "tokenspeed_scheduler_ext.cpython-311-aarch64-linux-gnu.so"
    )
    extension_sha = sha(extension)
    task.mkdir(parents=False, exist_ok=False)
    (repo / ".tmp").mkdir(exist_ok=True)
    test_temp = repo / ".tmp" / f"repro-{task.name}"
    if test_temp.exists():
        raise ValueError("test temporary directory already exists")
    bootstrap = repo / ".tmp/import_bootstrap/sitecustomize.py"
    bootstrap.parent.mkdir(exist_ok=True)
    template = runners / "import_bootstrap.txt"
    if bootstrap.exists() and sha(bootstrap) != sha(template):
        raise ValueError("refusing to replace an existing different import bootstrap")
    if not bootstrap.exists():
        shutil.copyfile(template, bootstrap)
    for name in (
        "controller.py",
        "owned_processes.py",
        "test_owned_processes.py",
        "jobs.json",
    ):
        shutil.copyfile(runners / name, task / name)
    env = {
        **os.environ,
        "TOKENSPEED_SOURCE_SNAPSHOT": str(repo),
        "TOKENSPEED_HW2_WORKSPACE": str(workspace),
        "PYTHONPATH": ":".join(
            str(repo / p)
            for p in (
                ".tmp/import_bootstrap",
                "python",
                "tokenspeed-kernel/python",
                "tokenspeed-kernel-npu/python",
                "tokenspeed-scheduler/python",
                ".",
            )
        ),
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(repo / ".tmp"),
        "TMP": str(repo / ".tmp"),
        "TEMP": str(repo / ".tmp"),
        "XDG_CACHE_HOME": str(repo / ".cache"),
        "HF_HOME": str(repo / ".cache/huggingface"),
        "TORCH_HOME": str(repo / ".cache/torch"),
        "TRITON_CACHE_DIR": str(repo / ".cache/triton"),
    }
    # The separate probe process starts with the chosen import environment.
    probe_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe",
        str(repo),
        str(task),
    ]
    tests = [
        [
            sys.executable,
            "-m",
            "pytest",
            "test/course_project",
            "-q",
            "-rs",
            "--basetemp",
            str(test_temp),
            "--junitxml",
            str(task / "cpu.xml"),
        ],
        [sys.executable, str(task / "test_owned_processes.py")],
        probe_command,
    ]
    for index, command in enumerate(tests):
        with (task / f"software_{index}.log").open("x") as log:
            result = subprocess.run(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"software check {index} failed; preserve and inspect its log"
            )
    state = model_state(
        repo, json.loads((evidence / "model_files.json").read_text())["models"]
    )
    write_json(task / "model_state.json", state)
    ready = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "server_cpu_exit": 0,
        "precommit_exit": 0,
        "precommit_provenance": "inherited measured-source receipt; NOT a newly executed precommit",
        "precommit_log_sha256": software["precommit"]["log_sha256"],
        "source_manifest_sha256": check_sources(repo, source_manifest),
        "controller_sha256": sha(task / "controller.py"),
        "owned_processes_sha256": sha(task / "owned_processes.py"),
        "jobs_sha256": sha(task / "jobs.json"),
        "model_state_sha256": sha(task / "model_state.json"),
        "bootstrap_sha256": sha(bootstrap),
        "dependency_versions": dependency_versions(),
        "scheduler_extension_sha256": extension_sha,
        "analysis_tools": {
            f"docs/evidence/project_agent_20261031/runners/{name}": sha(runners / name)
            for name in (
                "analyze.py",
                "audit_devices.py",
                "audit_native.py",
                "audit_phases.py",
            )
        },
        "python": str(Path(sys.executable).resolve()),
        "environment_script_sha256": sha(args.environment_script.resolve(strict=True)),
        "cpu_log_sha256": sha(task / "software_0.log"),
    }
    write_json(task / "ready_current.json", ready)
    print(
        json.dumps(
            {"ready": True, "npu_experiment_started": False, "source_files": 1535}
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        source, destination = (Path(p) for p in sys.argv[2:])
        probe(source, destination / "parent.json")
        for mode in ("spawn", "forkserver"):
            process = multiprocessing.get_context(mode).Process(
                target=probe, args=(source, destination / f"{mode}.json")
            )
            process.start()
            process.join(120)
            if process.is_alive():
                process.terminate()
                process.join(10)
                raise RuntimeError(f"{mode} import probe timed out")
            if process.exitcode:
                raise RuntimeError(f"{mode} import probe failed")
        if not all(
            json.loads((destination / f"{mode}.json").read_text())
            == json.loads((destination / "parent.json").read_text())
            for mode in ("spawn", "forkserver")
        ):
            raise ValueError("parent and worker import identities differ")
    else:
        main()
