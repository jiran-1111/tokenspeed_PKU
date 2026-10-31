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

"""Fail-closed device and source checks shared by isolated HW2 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

SOURCE_ROOTS = (
    "python",
    "tokenspeed-kernel",
    "tokenspeed-kernel-npu",
    "tokenspeed-scheduler",
    "benchmarks",
    "scripts",
    "configs",
    "test/course_project",
    "requirements",
)
SOURCE_SUFFIXES = {
    ".py",
    ".sh",
    ".ps1",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cu",
    ".cuh",
    ".cmake",
    ".in",
}
EXCLUDED_PARTS = {
    ".git",
    ".tmp",
    ".cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}


def source_inventory(repo_root: Path) -> list[str]:
    """List experiment code/configuration, including uncommitted new sources."""
    return sorted(
        {
            path.relative_to(repo_root).as_posix()
            for root in SOURCE_ROOTS
            for path in (repo_root / root).rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and not (set(path.relative_to(repo_root).parts) & EXCLUDED_PARTS)
        }
    )


def check_sources(repo_root: Path, manifest_path: Path) -> str:
    """Verify complete source coverage and bytes; return the manifest file SHA."""
    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(sources, dict) or set(sources) != set(
        source_inventory(repo_root)
    ):
        raise ValueError("source manifest coverage differs from experiment sources")
    for name, sha in sources.items():
        path = (repo_root / name).resolve()
        if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
            raise ValueError("source path escapes workspace or is missing")
        if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("invalid source SHA")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"source SHA differs: {name}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def verify_module_origins(
    repo_root: Path,
    module_files: dict[str, Path],
    *,
    external_extensions: dict[str, str],
) -> dict:
    """Verify loaded files belong to this source tree or a pinned scheduler binary.

    Return module-name records with a relative path (or external basename) and
    the actual file SHA. Only the scheduler extension may be outside the repo;
    its basename and SHA must match the explicit extension allowlist.
    """
    if not module_files:
        raise ValueError("module origins are missing")
    root = repo_root.resolve(strict=True)
    origins = {}
    for name, file in module_files.items():
        if not isinstance(name, str) or not name or not isinstance(file, Path):
            raise ValueError("module origin requires a module name and file path")
        try:
            path = file.resolve(strict=True)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"module origin is unreadable: {name}") from error
        if path.is_relative_to(root):
            origins[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha}
        elif (
            name == "tokenspeed_scheduler.tokenspeed_scheduler_ext"
            and path.name.startswith("tokenspeed_scheduler_ext.")
            and path.suffix in {".so", ".pyd"}
            and external_extensions.get(path.name) == sha
        ):
            origins[name] = {"basename": path.name, "sha256": sha}
        else:
            raise ValueError(f"module origin is outside verified sources: {name}")
    return origins


def target_process_pids(
    table: str, *, npu_id: int, chip_ids: set[int]
) -> dict[int, set[int]]:
    """Read physical-card/chip PIDs from a complete npu-smi info process table.

    Missing target rows mean no listed process only when a complete process
    table is present. Missing headers, truncated tables and unknown rows raise
    ValueError rather than implying empty devices.
    """
    if (
        type(npu_id) is not int
        or npu_id < 0
        or not chip_ids
        or any(type(chip) is not int or chip not in {0, 1} for chip in chip_ids)
    ):
        raise ValueError("invalid target physical card/chips")
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    header = ["NPU Chip", "Process id", "Process name", "Process memory(MB)"]
    headers = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|")
        and line.endswith("|")
        and [re.sub(r"\s+", " ", part.strip()) for part in line[1:-1].split("|")]
        == header
    ]
    if len(headers) != 1:
        raise ValueError("missing or ambiguous process table header")
    body = lines[headers[0] + 1 :]
    if len(body) < 3 or not all(
        re.fullmatch(r"\+[+=-]+\+", line) for line in (body[0], body[-1])
    ):
        raise ValueError("incomplete process table")
    pids = {chip: set() for chip in chip_ids}
    seen, empty, content = set(), False, False
    for line in body[1:-1]:
        if re.fullmatch(r"\+[+=-]+\+", line):
            continue
        if re.fullmatch(r"\|\s*No running processes found\.?\s*\|", line):
            if content:
                raise ValueError("ambiguous empty process table")
            empty, content = True, True
            continue
        if empty or not (line.startswith("|") and line.endswith("|")):
            raise ValueError("unknown process table row")
        columns = [part.strip() for part in line[1:-1].split("|")]
        location = re.fullmatch(r"(\d+)\s+(\d+)", columns[0])
        if (
            len(columns) != 4
            or location is None
            or re.fullmatch(r"[1-9]\d*", columns[1]) is None
            or not columns[2]
            or re.fullmatch(r"\d+", columns[3]) is None
        ):
            raise ValueError("invalid process table row")
        card, chip = map(int, location.groups())
        pid = int(columns[1])
        if (card, chip, pid) in seen:
            raise ValueError("duplicate process table row")
        seen.add((card, chip, pid))
        content = True
        if card == npu_id and chip in pids:
            pids[chip].add(pid)
    if not content:
        raise ValueError("process table has no process or explicit empty status")
    return pids


def is_descendant(
    pid: int, *, root_pid: int, read_parent: Callable[[int], int | None]
) -> bool:
    """Accept this run's root or a known ancestry chain; unknown/cycles fail closed."""
    if type(root_pid) is not int or root_pid < 1:
        return False
    seen = set()
    while type(pid) is int and pid > 0 and pid not in seen:
        if pid == root_pid:
            return True
        if pid == 1:
            return False
        seen.add(pid)
        try:
            pid = read_parent(pid)
        except (OSError, ValueError):
            return False
    return False


def parse_mapping(text: str) -> dict[tuple[int, int], int]:
    """Read physical-card/chip -> logical-device mapping from npu-smi info -m."""
    mapping = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+Ascend910\s*", line)
        if match is None:
            continue
        card, chip, logical = map(int, match.groups())
        if (card, chip) in mapping or logical in mapping.values():
            raise ValueError("duplicate physical or logical mapping")
        mapping[card, chip] = logical
    if not mapping:
        raise ValueError("no recognized device mapping")
    return mapping


def validate_target(
    mapping: dict[tuple[int, int], int],
    *,
    npu_id: int,
    chip_logic_ids: dict[int, int],
    tp_size: int,
) -> None:
    if tp_size not in (1, 2) or len(chip_logic_ids) != tp_size:
        raise ValueError("target device count must equal TP size")
    if set(chip_logic_ids) - {0, 1} or len(set(chip_logic_ids.values())) != tp_size:
        raise ValueError("invalid or duplicated chip selection")
    if any(
        mapping.get((npu_id, chip)) != logical
        for chip, logical in chip_logic_ids.items()
    ):
        raise ValueError("target mapping does not match selected same-card devices")


def require_idle(text: str, *, npu_id: int, chip_id: int) -> None:
    """Require an empty selected-chip body in an identified whole-card response.

    proc-mem without -c puts each chip's body before its Chip ID marker. The
    optional Chip Count metadata must agree with all complete chip sections;
    unselected chips may contain recognized process rows.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or re.fullmatch(rf"NPU ID\s*:\s*{npu_id}", lines[0]) is None:
        raise ValueError("process query card identity is missing or wrong")
    lines = lines[1:]
    count = None
    if lines and (metadata := re.fullmatch(r"Chip Count\s*:\s*(\d+)", lines[0])):
        count = int(metadata[1])
        if count not in (1, 2):
            raise ValueError("invalid chip count in process response")
        lines = lines[1:]
    sections, body = {}, []
    for line in lines:
        marker = re.fullmatch(r"Chip ID\s*:\s*(\d+)", line)
        if marker is None:
            body.append(line)
            continue
        chip = int(marker[1])
        if chip in sections or chip not in {0, 1} or not body:
            raise ValueError("duplicate or invalid process chip section")
        idle = body == ["No process in device."]
        if not idle and not all(
            re.fullmatch(
                r"Process id\s*:\s*[1-9]\d*\s+Process name\s*:\s*.+?\s+"
                r"Process memory\(MB\)\s*:\s*\d+",
                row,
            )
            for row in body
        ):
            raise ValueError("unknown or ambiguous process body")
        sections[chip] = idle
        body = []
    if (
        body
        or chip_id not in sections
        or (count is not None and set(sections) != set(range(count)))
    ):
        raise ValueError("incomplete process response or unknown trailing content")
    if not sections[chip_id]:
        raise ValueError("selected chip is occupied or its idle status is unknown")


def _query(arguments: list[str]) -> str:
    return subprocess.run(
        ["npu-smi", "info", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout


@contextmanager
def hardware_lease(
    repo_root: Path,
    *,
    npu_id: int,
    chip_logic_ids: dict[int, int],
    tp_size: int,
    output_dir: Path,
    source_manifest: Path,
):
    """Hold our common experiment lock and verify devices before/after Engine use.

    TOKENSPEED_HW2_WORKSPACE explicitly names the enclosing server workspace.
    This lock coordinates our experiments only; it cannot reserve shared hardware
    against another user. No process is ever stopped by this helper.
    A context owned only by this collector process may remain until process exit;
    the external driver must confirm final device release after that exit.
    """
    import fcntl

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
    lock_path = workspace / ".tmp" / "hw2-npu.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources_sha = check_sources(repo_root, source_manifest)
        mapping_text = _query(["-m"])
        validate_target(
            parse_mapping(mapping_text),
            npu_id=npu_id,
            chip_logic_ids=chip_logic_ids,
            tp_size=tp_size,
        )
        (output_dir / "before_mapping.txt").write_text(mapping_text, encoding="utf-8")
        before = {}
        for chip in sorted(chip_logic_ids):
            response = _query(["-t", "proc-mem", "-i", str(npu_id)])
            (output_dir / f"before_processes_chip{chip}.txt").write_text(
                response, encoding="utf-8"
            )
            require_idle(response, npu_id=npu_id, chip_id=chip)
            before[chip] = response
        receipt = {
            "sources_sha": sources_sha,
            "before_mapping": mapping_text,
            "before_processes": before,
            "guard_started_ns": time.monotonic_ns(),
            "parent_pid": os.getpid(),
            "engine_children_released": False,
            "after_target_idle": False,
        }
        try:
            yield receipt
        finally:
            receipt["sources_after_sha"] = check_sources(repo_root, source_manifest)
            if receipt["sources_after_sha"] != sources_sha:
                raise ValueError("source manifest changed during experiment")
            after_mapping = _query(["-m"])
            (output_dir / "after_mapping.txt").write_text(
                after_mapping, encoding="utf-8"
            )
            validate_target(
                parse_mapping(after_mapping),
                npu_id=npu_id,
                chip_logic_ids=chip_logic_ids,
                tp_size=tp_size,
            )
            after_processes = {chip: [] for chip in chip_logic_ids}
            for chip in sorted(chip_logic_ids):
                for attempt in range(10):
                    response = _query(["-t", "proc-mem", "-i", str(npu_id)])
                    (output_dir / f"after_processes_chip{chip}.txt").write_text(
                        response, encoding="utf-8"
                    )
                    try:
                        require_idle(response, npu_id=npu_id, chip_id=chip)
                        after_processes[chip] = []
                        break
                    except ValueError:
                        table = _query([])
                        receipt["after_npu_smi"] = "after_npu_smi.txt"
                        (output_dir / receipt["after_npu_smi"]).write_text(
                            table, encoding="utf-8"
                        )
                        processes = target_process_pids(
                            table, npu_id=npu_id, chip_ids=set(chip_logic_ids)
                        )
                        after_processes = {
                            target: sorted(pids) for target, pids in processes.items()
                        }
                        receipt["after_process_pids"] = after_processes
                        if all(
                            pids <= {receipt["parent_pid"]}
                            for pids in processes.values()
                        ):
                            break
                        if attempt == 9:
                            raise ValueError(
                                "engine children or foreign target processes remain"
                            )
                        time.sleep(1)
            receipt["guard_ended_ns"] = time.monotonic_ns()
            receipt["after_process_pids"] = after_processes
            receipt["engine_children_released"] = True
            receipt["parent_context_pending_exit"] = any(after_processes.values())
            receipt["after_target_idle"] = not receipt["parent_context_pending_exit"]
            receipt["sources_unchanged"] = True
