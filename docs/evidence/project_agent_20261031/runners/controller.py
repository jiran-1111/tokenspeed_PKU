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

"""Run the approved finite Agent suite serially; preserve every attempt."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from owned_processes import terminate_owned_group


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    root = Path(f"/proc/{pid}")
    return [
        root.stat().st_uid,
        root.joinpath("stat").read_text().rpartition(")")[2].split()[19],
    ]


def dependency_versions():
    versions = {}
    for name in (
        "torch",
        "torch-npu",
        "transformers",
        "tokenspeed-scheduler",
        "triton-ascend",
        "tokenspeed-triton",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    # A background shell may inherit SIGINT ignored; our guarded children must
    # receive an interrupt that actually triggers their normal teardown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser()
    for name in (
        "repo",
        "task-dir",
        "model-state",
        "source-manifest",
        "python",
        "environment-script",
        "jobs",
        "ready",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--npu-id", type=int, required=True)
    parser.add_argument("--chip-logic-ids", required=True)
    args = parser.parse_args()
    repo, task = args.repo.resolve(), args.task_dir.resolve()
    sys.path.insert(0, str(repo))
    from benchmarks.course_project.native_devices import parse_chip_logic_ids
    from benchmarks.course_project.native_performance import parse_target_hbm
    from benchmarks.course_project.tp_devices import (
        check_sources,
        parse_mapping,
        target_process_pids,
        validate_target,
    )

    chips = parse_chip_logic_ids(args.chip_logic_ids)
    jobs = json.loads(args.jobs.read_text())
    models = json.loads(args.model_state.read_text())
    ready = json.loads(args.ready.read_text())
    assert ready["server_cpu_exit"] == ready["precommit_exit"] == 0
    assert ready["source_manifest_sha256"] == check_sources(repo, args.source_manifest)
    assert ready["controller_sha256"] == sha(Path(__file__))
    assert ready["owned_processes_sha256"] == sha(
        Path(__file__).with_name("owned_processes.py")
    )
    assert ready["jobs_sha256"] == sha(args.jobs)
    assert ready["model_state_sha256"] == sha(args.model_state)
    assert ready["bootstrap_sha256"] == sha(
        repo / ".tmp/import_bootstrap/sitecustomize.py"
    )
    assert ready["dependency_versions"] == dependency_versions()
    extension = (
        repo
        / "tokenspeed-scheduler/python/tokenspeed_scheduler/tokenspeed_scheduler_ext.cpython-311-aarch64-linux-gnu.so"
    )
    assert ready["scheduler_extension_sha256"] == sha(extension)

    def check_analysis_tools():
        for relative, expected in ready["analysis_tools"].items():
            path = (repo / relative).resolve(strict=True)
            assert path.is_relative_to(repo) and sha(path) == expected, relative

    check_analysis_tools()
    state = {
        "status": "starting",
        "finished": {},
        "current": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path = task / "suite_state.json"
    if state_path.exists():
        raise FileExistsError(
            "suite already started; inspect its process and state instead of duplicating it"
        )

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    def query(arguments):
        return subprocess.check_output(
            ["npu-smi", "info", *arguments], text=True, timeout=15
        )

    def check_model(config):
        entry = models["models"][config["model_id"]]
        assert entry["revision"] == config["model_revision"]
        for name, expected in entry["files"].items():
            value = (repo / config["local_model_path"] / name).stat()
            assert [
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
            ] == expected["stat"], name

    save()
    for job in jobs:
        if any(
            state["finished"].get(name, {}).get("status") != "passed"
            for name in job["after"]
        ):
            state["finished"][job["name"]] = {
                "status": "blocked_by_failed_quality",
                "dependencies": job["after"],
            }
            save()
            continue
        config = json.loads((repo / job["config"]).read_text())
        selected = chips if config["tp_size"] == 2 else {min(chips): chips[min(chips)]}
        source_sha = check_sources(repo, args.source_manifest)
        assert ready["dependency_versions"] == dependency_versions()
        assert ready["scheduler_extension_sha256"] == sha(extension)
        check_analysis_tools()
        check_model(config)
        while True:
            mapping = query(["-m"])
            validate_target(
                parse_mapping(mapping),
                npu_id=args.npu_id,
                chip_logic_ids=selected,
                tp_size=config["tp_size"],
            )
            before = query([])
            memory = parse_target_hbm(
                before, npu_id=args.npu_id, chip_logic_ids=selected
            )
            if all(
                value["capacity_mb"] - value["used_mb"] >= job["admission_free_mb"]
                for value in memory.values()
            ):
                break
            state.update(
                {
                    "status": "waiting_for_shared_memory",
                    "current": job["name"],
                    "memory": memory,
                }
            )
            save()
            time.sleep(30)
        out = repo / job["output"]
        assert not out.exists(), "refusing to overwrite an attempt"
        command = [
            str(args.python),
            str(repo / f"scripts/run_native_agent_{job['kind']}.py"),
            "--config",
            str(repo / job["config"]),
            "--workload" if job["kind"] == "validation" else "--protocol",
            str(repo / job["protocol"]),
            "--source-manifest",
            str(args.source_manifest),
            "--output-dir",
            str(out),
            "--engine-port",
            str(job["engine_port"]),
            "--metrics-port",
            str(job["metrics_port"]),
            "--npu-id",
            str(args.npu_id),
            "--chip-logic-ids",
            ",".join(f"{c}:{v}" for c, v in sorted(selected.items())),
            "--resource-mode",
            "shared",
            "--min-free-mb",
            "8192",
        ]
        env = {
            "TOKENSPEED_HW2_WORKSPACE": str(task.parent.parent),
            "TOKENSPEED_SOURCE_SNAPSHOT": str(repo),
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
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(
                str(selected[c]) for c in sorted(selected)
            ),
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": str(repo / ".tmp"),
            "TMP": str(repo / ".tmp"),
            "TEMP": str(repo / ".tmp"),
            "XDG_CACHE_HOME": str(repo / ".cache"),
            "HF_HOME": str(repo / ".cache/huggingface"),
            "TORCH_HOME": str(repo / ".cache/torch"),
            "TORCH_EXTENSIONS_DIR": str(repo / ".cache/torch-extensions"),
            "TRITON_CACHE_DIR": str(repo / ".cache/triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(repo / ".cache/torchinductor"),
            "ASCEND_WORK_PATH": str(repo / ".tmp/ascend_work"),
        }
        activation = f"source {shlex.quote(str(args.environment_script))}\n"
        activation += "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())
        activation += "\nunset TORCH_DEVICE_BACKEND_AUTOLOAD\nexec " + shlex.join(
            command
        )
        receipt = {
            "job": job,
            "devices": {"npu_id": args.npu_id, "chip_logic_ids": selected},
            "source_manifest_sha256": source_sha,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "resource_mode": "shared",
            "timed_out": False,
            "dependency_versions": dependency_versions(),
            "scheduler_extension_sha256": sha(extension),
        }
        state.update({"status": "running", "current": receipt})
        save()
        with (task / f"{job['name']}.log").open("x") as log:
            process = subprocess.Popen(
                ["bash", "--noprofile", "--norc", "-c", activation],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            uid, ticks = identity(process.pid)
            receipt.update({"parent_pid": process.pid, "parent_identity": [uid, ticks]})
            save()
            try:
                receipt["exit_code"] = process.wait(timeout=job["budget_seconds"])
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                terminate_owned_group(process, expected_start=ticks)
                receipt["exit_code"] = process.returncode
            terminate_owned_group(process, expected_start=ticks)
        check_model(config)
        assert ready["dependency_versions"] == dependency_versions()
        assert ready["scheduler_extension_sha256"] == sha(extension)
        receipt["model_files_unchanged"] = True
        receipt["source_unchanged"] = (
            check_sources(repo, args.source_manifest) == source_sha
        )
        out.mkdir(parents=True, exist_ok=True)
        run_path = out / "run.json"
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        guard = run.get("device_guard", {})
        baseline = {
            int(c): {int(pid): value for pid, value in values.items()}
            for c, values in guard.get("baseline", {}).items()
        }
        known_owned = {process.pid} | {
            pid
            for sample in guard.get("samples", [])
            for pid in sample.get("owned_pids", [])
        }
        release = {
            "run_sha256": sha(run_path) if run_path.is_file() else None,
            "devices": receipt["devices"],
            "exit_code": receipt["exit_code"],
            "timed_out": receipt["timed_out"],
            "parent_pid": process.pid,
            "parent_identity": [uid, ticks],
            "known_owned_pids": sorted(known_owned),
            "source_unchanged": receipt["source_unchanged"],
            "model_files_unchanged": True,
            "own_processes_released": False,
            "unknown_pids": [],
            "samples": [],
        }
        for attempt in range(15):
            mapping = query(["-m"])
            (out / "post_exit_mapping.txt").write_text(mapping)
            validate_target(
                parse_mapping(mapping),
                npu_id=args.npu_id,
                chip_logic_ids=selected,
                tp_size=config["tp_size"],
            )
            table = query([])
            filename = f"post_exit_devices_{attempt:02d}.txt"
            (out / filename).write_text(table)
            pids = target_process_pids(
                table, npu_id=args.npu_id, chip_ids=set(selected)
            )
            identities, exited, unknown = {}, [], []
            for chip, values in pids.items():
                for pid in values:
                    try:
                        identities[pid] = identity(pid)
                    except FileNotFoundError:
                        exited.append(pid)
                        continue
                    if (
                        pid not in known_owned
                        and baseline.get(chip, {}).get(pid) != identities[pid]
                    ):
                        unknown.append(pid)
            remaining = (set().union(*pids.values()) - set(exited)) & known_owned
            release["samples"].append(
                {
                    "raw_table": filename,
                    "pids": {c: sorted(v) for c, v in pids.items()},
                    "identities": identities,
                    "exited_during_query": exited,
                    "remaining_owned_pids": sorted(remaining),
                }
            )
            release["unknown_pids"] = sorted(set(unknown))
            if not remaining:
                release["own_processes_released"] = True
                break
            time.sleep(1)
        write_json(out / "post_exit_verification.json", release)
        receipt["post_exit_sha256"] = sha(out / "post_exit_verification.json")
        check_analysis_tools()
        audit_args = [str(args.python)] + [
            value.format(
                repo=repo,
                run=run_path,
                out=out / "analysis",
                manifest=args.source_manifest,
            )
            for value in job["audit_args"]
        ]
        with (out / "analysis.log").open("x") as log:
            audit = subprocess.run(
                audit_args,
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
                env={
                    **os.environ,
                    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                    "PYTHONPATH": str(repo),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        receipt["independent_audit_exit"] = audit.returncode
        receipt["status"] = (
            "passed"
            if (
                receipt["exit_code"] == 0
                and not receipt["timed_out"]
                and run.get("status") == "passed"
                and release["own_processes_released"]
                and not release["unknown_pids"]
                and receipt["independent_audit_exit"] == 0
            )
            else "failed"
        )
        receipt["run_status"] = run.get("status")
        receipt["run_error"] = run.get("error")
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(task / f"{job['name']}.receipt.json", receipt)
        state["finished"][job["name"]] = receipt
        state["current"] = None
        save()
        if not release["own_processes_released"] or release["unknown_pids"]:
            state["status"] = "stopped_device_release_or_identity_unverified"
            save()
            return
    state["status"] = "finished_pending_independent_audit"
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_json(
            Path(__file__).with_name(f"controller_failure_{time.time_ns()}.json"),
            {
                "type": type(error).__name__,
                "message": str(error),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
