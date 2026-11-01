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

"""Explicit device selection and observed process safety for native Agent runs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from benchmarks.course_project.native_performance import parse_target_hbm
from benchmarks.course_project.tp_devices import (
    check_sources,
    is_descendant,
    parse_mapping,
    target_process_pids,
    validate_target,
)


def parse_chip_logic_ids(value: str) -> dict[int, int]:
    """Parse explicit chip:logical-device pairs, rejecting duplicate selections."""
    try:
        pairs = [tuple(map(int, item.split(":"))) for item in value.split(",")]
        if (
            any(len(pair) != 2 or min(pair) < 0 for pair in pairs)
            or len({pair[0] for pair in pairs}) != len(pairs)
            or len({pair[1] for pair in pairs}) != len(pairs)
        ):
            raise ValueError("invalid or duplicate mapping")
        return dict(pairs)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected chip:logical pairs, e.g. 0:2,1:3"
        ) from error


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Require all choices that control device placement and sharing."""
    parser.add_argument("--npu-id", type=int, required=True)
    parser.add_argument("--chip-logic-ids", type=parse_chip_logic_ids, required=True)
    parser.add_argument("--resource-mode", choices=("idle", "shared"), required=True)
    parser.add_argument(
        "--min-free-mb",
        type=int,
        help="shared mode safety reserve per chip, at least 8192 MB",
    )


def validate_resource_policy(*, resource_mode: str, min_free_mb: int | None) -> None:
    if resource_mode not in {"idle", "shared"}:
        raise ValueError("unsupported resource mode")
    if resource_mode == "shared":
        if type(min_free_mb) is not int or min_free_mb < 8192:
            raise ValueError("shared mode requires explicit min-free-mb >= 8192")
    elif min_free_mb is not None:
        raise ValueError("min-free-mb applies only to shared resource mode")


def _identity(pid: int) -> tuple[int, str]:
    path = Path(f"/proc/{pid}")
    fields = (path / "stat").read_text().rpartition(")")[2].split()
    return path.stat().st_uid, fields[19]


def _parent(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[1])


def _query(arguments: list[str]) -> str:
    return subprocess.run(
        ["npu-smi", "info", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout


def _query_evidence(arguments: list[str], *, output_dir: Path, evidence: dict) -> str:
    """Preserve partial command output before propagating a failed query."""
    path = output_dir / evidence["raw_table"]
    try:
        text = _query(arguments)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        for value, target in (
            (error.stdout, path),
            (error.stderr, path.with_suffix(".stderr.txt")),
        ):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            target.write_text(value or "", encoding="utf-8")
        evidence["raw_stderr"] = path.with_suffix(".stderr.txt").name
        if isinstance(error, subprocess.TimeoutExpired):
            evidence["query_timeout_seconds"] = error.timeout
        else:
            evidence["query_returncode"] = error.returncode
        raise
    path.write_text(text, encoding="utf-8")
    return text


@contextmanager
def native_hardware_lease(
    repo_root: Path,
    *,
    npu_id: int,
    chip_logic_ids: dict[int, int],
    tp_size: int,
    resource_mode: str,
    min_free_mb: int | None,
    output_dir: Path,
    source_manifest: Path,
    receipt: dict,
) -> Iterator[Callable[[], None]]:
    """Observe before/during/after Engine use while holding our common task lock.

    Shared baselines are collected live, with UID and kernel start time. The lock
    coordinates our collectors; it cannot reserve devices against other users.
    Only this collector receives SIGINT on a safety violation. Engine teardown
    runs in the caller's normal finally path. Its own parent context may remain
    until process exit and must be checked by the external driver afterwards.
    Yield the monitor stop function; call it after the workload finishes and
    before Engine teardown, which terminates the collector's child processes.
    """
    import fcntl

    validate_resource_policy(resource_mode=resource_mode, min_free_mb=min_free_mb)
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise ValueError(
            "SIGINT is ignored; native device protection cannot interrupt the collector"
        )
    if type(npu_id) is not int or npu_id < 0:
        raise ValueError("invalid physical NPU ID")
    workspace = Path(os.environ["TOKENSPEED_HW2_WORKSPACE"]).resolve(strict=True)
    if not repo_root.resolve().is_relative_to(workspace):
        raise ValueError("experiment repository must be inside the server workspace")
    if not output_dir.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("experiment results must stay inside their source workspace")
    expected = ",".join(str(chip_logic_ids[chip]) for chip in sorted(chip_logic_ids))
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != expected:
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES differs from explicit target mapping"
        )
    lock_path = workspace / ".tmp/hw2-npu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    parent_pid = os.getpid()
    parent_identity = _identity(parent_pid)
    baseline = None
    owned_identities = {parent_pid: parent_identity}
    stopped = threading.Event()
    watcher = None
    receipt.update(
        {
            "resource_mode": resource_mode,
            "min_free_mb": min_free_mb,
            "parent_pid": parent_pid,
            "lock_path": ".tmp/hw2-npu.lock",
            "guard_started_ns": time.monotonic_ns(),
            "samples": [],
            "mapping_queries": {},
            "violations": [],
            "after_errors": [],
            "engine_children_released": False,
            "sources_unchanged": False,
        }
    )

    def mapping_snapshot(label: str) -> None:
        filename = f"{label}_mapping.txt"
        receipt[f"{label}_mapping"] = filename
        evidence = {"raw_table": filename}
        receipt["mapping_queries"][label] = evidence
        text = _query_evidence(["-m"], output_dir=output_dir, evidence=evidence)
        validate_target(
            parse_mapping(text),
            npu_id=npu_id,
            chip_logic_ids=chip_logic_ids,
            tp_size=tp_size,
        )

    def snapshot(label: str) -> set[int]:
        nonlocal baseline
        filename = f'guard_{len(receipt["samples"]):05d}.txt'
        item = {
            "label": label,
            "observed_ns": time.monotonic_ns(),
            "raw_table": filename,
        }
        receipt["samples"].append(item)
        try:
            text = _query_evidence([], output_dir=output_dir, evidence=item)
            current = target_process_pids(
                text, npu_id=npu_id, chip_ids=set(chip_logic_ids)
            )
            item["pids"] = {chip: sorted(pids) for chip, pids in current.items()}
            memory = parse_target_hbm(
                text, npu_id=npu_id, chip_logic_ids=chip_logic_ids
            )
            item["memory"] = memory
            identities, exited = {}, set()
            for pid in set().union(*current.values()):
                try:
                    identities[pid] = _identity(pid)
                except FileNotFoundError:
                    if Path(f"/proc/{pid}").exists():
                        raise
                    exited.add(pid)
            item["exited_during_query"] = sorted(exited)
            current = {chip: pids - exited for chip, pids in current.items()}
            item["live_pids"] = {chip: sorted(pids) for chip, pids in current.items()}
            item["identities"] = {pid: list(value) for pid, value in identities.items()}
            if baseline is None:
                baseline = {
                    chip: (
                        {pid: identities[pid] for pid in pids if pid != parent_pid}
                        if resource_mode == "shared"
                        else {}
                    )
                    for chip, pids in current.items()
                }
                receipt["baseline"] = {
                    chip: {pid: list(value) for pid, value in pids.items()}
                    for chip, pids in baseline.items()
                }
                if resource_mode == "idle" and any(current.values()):
                    raise ValueError("selected chip is occupied in idle resource mode")
            owned = set()
            for chip, pids in current.items():
                for pid in pids:
                    identity = identities[pid]
                    if pid in baseline[chip]:
                        if identity != baseline[chip][pid]:
                            raise ValueError(
                                "background process identity changed (PID reuse)"
                            )
                    elif pid in owned_identities:
                        if identity != owned_identities[pid]:
                            raise ValueError(
                                "owned process identity changed (PID reuse)"
                            )
                        owned.add(pid)
                    elif identity[0] == parent_identity[0] and is_descendant(
                        pid, root_pid=parent_pid, read_parent=_parent
                    ):
                        owned_identities[pid] = identity
                        owned.add(pid)
                    else:
                        raise ValueError("unknown process on selected chip")
            item["owned_pids"] = sorted(owned)
            if resource_mode == "shared" and any(
                value["capacity_mb"] - value["used_mb"] < min_free_mb
                for value in memory.values()
            ):
                raise ValueError("shared device memory reserve insufficient")
            return owned
        except BaseException as error:
            item["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            item["ended_ns"] = time.monotonic_ns()

    def monitor() -> None:
        while not stopped.is_set():
            try:
                snapshot("monitor")
            except Exception as error:
                receipt["violations"].append(
                    {"type": type(error).__name__, "message": str(error)}
                )
                # Never signal a listed device occupant or any background PID.
                os.kill(parent_pid, signal.SIGINT)
                return
            stopped.wait(1)

    def stop_monitor() -> None:
        stopped.set()
        if watcher is not None:
            watcher.join(timeout=20)
            if watcher.is_alive():
                raise RuntimeError("native device monitor did not stop")
        receipt.setdefault("monitor_stopped_ns", time.monotonic_ns())
        # A query already in flight may report a violation while join waits.
        if receipt["violations"]:
            raise ValueError("native device protection rejected this run")

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        body_failed = False
        try:
            receipt["sources_sha"] = check_sources(repo_root, source_manifest)
            mapping_snapshot("before")
            snapshot("before")
            watcher = threading.Thread(
                target=monitor, name="native-device-protection", daemon=True
            )
            watcher.start()
            yield stop_monitor
            if receipt["violations"]:
                raise ValueError("native device protection rejected this run")
        except BaseException as error:
            body_failed = True
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            stop_interrupt = None
            try:
                stop_monitor()
            except BaseException as error:
                receipt["after_errors"].append(f"{type(error).__name__}: {error}")
                if not isinstance(error, Exception):
                    stop_interrupt = error
            try:
                mapping_snapshot("after")
            except Exception as error:
                receipt["after_errors"].append(f"{type(error).__name__}: {error}")
            if baseline is not None and (watcher is None or not watcher.is_alive()):
                try:
                    deadline = time.monotonic() + 15
                    while True:
                        owned = snapshot("after_engine_exit")
                        if owned <= {parent_pid}:
                            break
                        if time.monotonic() >= deadline:
                            raise ValueError(
                                "own workers did not release within bounded grace"
                            )
                        time.sleep(0.5)
                    receipt["engine_children_released"] = True
                    receipt["parent_context_pending_exit"] = parent_pid in owned
                    receipt["after_process_pids"] = receipt["samples"][-1]["live_pids"]
                    receipt["after_target_idle"] = not any(
                        receipt["after_process_pids"].values()
                    )
                except Exception as error:
                    receipt["after_errors"].append(f"{type(error).__name__}: {error}")
            try:
                receipt["sources_after_sha"] = check_sources(repo_root, source_manifest)
                receipt["sources_unchanged"] = receipt[
                    "sources_after_sha"
                ] == receipt.get("sources_sha")
                if not receipt["sources_unchanged"]:
                    raise ValueError("source manifest changed during experiment")
            except Exception as error:
                receipt["after_errors"].append(f"{type(error).__name__}: {error}")
            receipt["guard_ended_ns"] = time.monotonic_ns()
            if not body_failed and stop_interrupt is not None:
                raise stop_interrupt
            if not body_failed and receipt["after_errors"]:
                raise ValueError(
                    "native device/source cleanup verification failed: "
                    + "; ".join(receipt["after_errors"])
                )
