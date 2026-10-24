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

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.course_project.tp_devices as tp_devices
from benchmarks.course_project.tp_devices import (
    check_sources,
    parse_mapping,
    require_idle,
    validate_target,
)


def _proc_card(pids, *, npu_id):
    text = f"NPU ID : {npu_id}\nChip Count : 2\n"
    for chip in (0, 1):
        text += (
            "".join(
                f"Process id:{pid} Process name:python Process memory(MB):112\n"
                for pid in sorted(pids.get(chip, set()))
            )
            if pids.get(chip)
            else "No process in device.\n"
        )
        text += f"Chip ID : {chip}\n"
    return text


def test_target_requires_same_card_and_exact_mapping():
    mapping = parse_mapping("0 0 0 0 Ascend910\n1 0 2 2 Ascend910\n1 1 3 3 Ascend910\n")
    validate_target(mapping, npu_id=1, chip_logic_ids={0: 2, 1: 3}, tp_size=2)
    with pytest.raises(ValueError, match="mapping"):
        validate_target(mapping, npu_id=1, chip_logic_ids={0: 0, 1: 3}, tp_size=2)
    with pytest.raises(ValueError):
        validate_target(mapping, npu_id=1, chip_logic_ids={0: 2}, tp_size=2)


def test_duplicate_logical_or_physical_device_is_rejected():
    for text in (
        "1 0 2 2 Ascend910\n1 0 3 3 Ascend910",
        "1 0 2 2 Ascend910\n1 1 2 2 Ascend910",
    ):
        with pytest.raises(ValueError, match="duplicate"):
            parse_mapping(text)


def test_idle_requires_selected_chip_not_merely_some_empty_chip():
    idle = "NPU ID : 1\nNo process in device.\nChip ID : 0\n"
    require_idle(idle, npu_id=1, chip_id=0)
    for payload in (
        idle.replace("Chip ID : 0", "Chip ID : 1"),
        idle.replace("NPU ID : 1", "NPU ID : 0"),
        "query failed",
        idle + "| 1 0 | 421 | python | 100 |\n",
    ):
        with pytest.raises(ValueError):
            require_idle(payload, npu_id=1, chip_id=0)


def test_empty_chip_does_not_hide_busy_sibling():
    mixed = "NPU ID : 1\nNo process in device.\nChip ID : 0\n| Process id |\n| 231 | python |\nChip ID : 1\n"
    with pytest.raises(ValueError):
        require_idle(mixed, npu_id=1, chip_id=1)


@pytest.mark.parametrize(
    "pids,chip", [({}, 0), ({}, 1), ({0: {311}}, 1), ({1: {312}}, 0)]
)
def test_whole_card_process_query_accepts_selected_idle_chip(pids, chip):
    require_idle(_proc_card(pids, npu_id=1), npu_id=1, chip_id=chip)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_card",
        "duplicate_card",
        "duplicate_count",
        "duplicate_chip",
        "missing_chip",
        "count_mismatch",
        "zero_count",
        "unknown_tail",
        "unknown_body",
        "mixed_body",
        "busy_selected",
    ],
)
def test_whole_card_process_query_rejects_ambiguous_or_busy_selected_chip(fault):
    text = _proc_card({}, npu_id=1)
    if fault == "wrong_card":
        text = text.replace("NPU ID : 1", "NPU ID : 0")
    elif fault == "duplicate_card":
        text = "NPU ID : 1\n" + text
    elif fault == "duplicate_count":
        text = text.replace("Chip Count : 2\n", "Chip Count : 2\nChip Count : 2\n")
    elif fault == "duplicate_chip":
        text = text.replace("Chip ID : 1", "Chip ID : 0")
    elif fault == "missing_chip":
        text = text.replace("No process in device.\nChip ID : 1\n", "")
    elif fault == "count_mismatch":
        text = text.replace("Chip Count : 2", "Chip Count : 1")
    elif fault == "zero_count":
        text = text.replace("Chip Count : 2", "Chip Count : 0")
    elif fault == "unknown_tail":
        text += "query failed\n"
    elif fault == "unknown_body":
        text = text.replace("No process in device.", "unknown process status", 1)
    elif fault == "mixed_body":
        text = text.replace(
            "No process in device.",
            "No process in device.\nProcess id:311 Process name:python Process memory(MB):112",
            1,
        )
    else:
        text = _proc_card({0: {311}}, npu_id=1)
    with pytest.raises(ValueError):
        require_idle(text, npu_id=1, chip_id=0)


def test_manifest_rejects_changed_missing_and_escaping_source(tmp_path: Path):
    (tmp_path / "python").mkdir()
    source = tmp_path / "python/main.py"
    source.write_text("print(1)\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    payload = {"python/main.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert check_sources(tmp_path, manifest)
    source.write_text("print(2)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        check_sources(tmp_path, manifest)
    payload = {"../escape.py": "0" * 64}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        check_sources(tmp_path, manifest)


def test_manifest_cannot_omit_new_runtime_file(tmp_path: Path):
    (tmp_path / "python").mkdir()
    (tmp_path / "python/new.py").write_text("pass\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        check_sources(tmp_path, manifest)


def test_lease_rejects_added_source_even_if_manifest_is_updated(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "python").mkdir()
    (tmp_path / "python/main.py").write_text("pass\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    def write_manifest():
        manifest.write_text(
            json.dumps(
                {
                    name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                    for name in tp_devices.source_inventory(tmp_path)
                }
            ),
            encoding="utf-8",
        )

    write_manifest()
    output = tmp_path / ".tmp" / "result"
    output.mkdir(parents=True)
    monkeypatch.setenv("TOKENSPEED_HW2_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(
            LOCK_EX=1,
            LOCK_NB=2,
            flock=lambda *args: None,
        ),
    )
    monkeypatch.setattr(
        tp_devices,
        "_query",
        lambda args: (
            "1 0 2 2 Ascend910\n"
            if args == ["-m"]
            else "NPU ID : 1\nNo process in device.\nChip ID : 0\n"
        ),
    )
    with pytest.raises(ValueError, match="manifest changed"):
        with tp_devices.hardware_lease(
            tmp_path,
            npu_id=1,
            chip_logic_ids={0: 2},
            tp_size=1,
            output_dir=output,
            source_manifest=manifest,
        ):
            (tmp_path / "python/new.py").write_text("pass\n", encoding="utf-8")
            write_manifest()


def test_module_origins_record_only_workspace_relative_paths_and_pinned_extension(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    source = repo / "python" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n", encoding="utf-8")
    extension = tmp_path / "tokenspeed_scheduler_ext.cpython-311-x86_64-linux-gnu.so"
    extension.write_bytes(b"fixed scheduler extension")
    extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
    result = tp_devices.verify_module_origins(
        repo,
        {
            "tokenspeed.runtime": source,
            "tokenspeed_scheduler.tokenspeed_scheduler_ext": extension,
        },
        external_extensions={extension.name: extension_sha},
    )
    assert result == {
        "tokenspeed.runtime": {
            "path": "python/runtime.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "tokenspeed_scheduler.tokenspeed_scheduler_ext": {
            "basename": extension.name,
            "sha256": extension_sha,
        },
    }


@pytest.mark.parametrize(
    "fault",
    [
        "old_editable",
        "python_extension",
        "unlisted",
        "wrong_sha",
        "other_extension",
        "missing",
    ],
)
def test_module_origins_reject_unverified_external_code(tmp_path: Path, fault):
    repo = tmp_path / "repo"
    repo.mkdir()
    name = "tokenspeed_scheduler.tokenspeed_scheduler_ext"
    source = tmp_path / "tokenspeed_scheduler_ext.so"
    source.write_bytes(b"scheduler extension")
    allowed = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    if fault == "old_editable":
        source = tmp_path / "old_runtime.py"
        source.write_text("pass\n", encoding="utf-8")
        name = "tokenspeed.runtime"
        allowed = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    elif fault == "python_extension":
        source = tmp_path / "tokenspeed_scheduler_ext.py"
        source.write_text("pass\n", encoding="utf-8")
        allowed = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    elif fault == "unlisted":
        allowed = {}
    elif fault == "wrong_sha":
        allowed[source.name] = "0" * 64
    elif fault == "other_extension":
        name = "tokenspeed_kernel.other_ext"
    elif fault == "missing":
        source.unlink()
    with pytest.raises(ValueError):
        tp_devices.verify_module_origins(
            repo, {name: source}, external_extensions=allowed
        )


PROCESS_HEADER = "| NPU Chip | Process id | Process name | Process memory(MB) |\n"
PROCESS_BORDER = "+===+===+===+===+\n"


def test_process_table_selects_physical_card_and_chip_and_shared_pid():
    table = (
        "| 1 Ascend910 | OK | ignored |\n| 0 2 | 0000:99:00.0 | 1000 / 65536 |\n"
        + PROCESS_BORDER
        + PROCESS_HEADER
        + PROCESS_BORDER
        + "| 0 0 | 111 | python | 112 |\n"
        + PROCESS_BORDER
        + "| 1 0 | 222 | python | 112 |\n"
        + "| 1 0 | 333 | scheduler | 2000 |\n"
        + "| 1 1 | 333 | scheduler | 2000 |\n"
        + PROCESS_BORDER
    )
    assert tp_devices.target_process_pids(table, npu_id=1, chip_ids={0, 1}) == {
        0: {222, 333},
        1: {333},
    }
    assert tp_devices.target_process_pids(table, npu_id=2, chip_ids={0}) == {0: set()}


def test_explicit_empty_process_table_is_accepted():
    table = (
        PROCESS_HEADER
        + PROCESS_BORDER
        + "| No running processes found |\n"
        + PROCESS_BORDER
    )
    assert tp_devices.target_process_pids(table, npu_id=1, chip_ids={0, 1}) == {
        0: set(),
        1: set(),
    }


@pytest.mark.parametrize(
    "body",
    [
        "",
        "| 1 0 | unknown | python | 112 |\n" + PROCESS_BORDER,
        "| 1 0 | 0 | python | 112 |\n" + PROCESS_BORDER,
        "| 1 0 | 222 | python | 112 |\n",
        "| 1 0 | 222 | python | 112 |\n" + PROCESS_BORDER + "query failed\n",
        "| No running processes found |\n| 1 0 | 222 | python | 112 |\n"
        + PROCESS_BORDER,
        "| 1 0 | 222 | python | 112 |\n| 1 0 | 222 | python | 112 |\n" + PROCESS_BORDER,
    ],
)
def test_process_table_rejects_incomplete_or_ambiguous_rows(body):
    with pytest.raises(ValueError):
        tp_devices.target_process_pids(
            PROCESS_HEADER + PROCESS_BORDER + body, npu_id=1, chip_ids={0, 1}
        )


def test_missing_process_table_cannot_be_mistaken_for_idle():
    with pytest.raises(ValueError):
        tp_devices.target_process_pids(
            "| 1 0 | 0000:99:00.0 | 1000 / 65536 |\n", npu_id=1, chip_ids={0}
        )


@pytest.mark.parametrize(
    "pid, parents, expected",
    [
        (10, {}, True),
        (30, {30: 20, 20: 10}, True),
        (30, {30: 20, 20: 1}, False),
        (30, {30: 20, 20: 30}, False),
        (30, {30: None}, False),
        (30, {30: "10"}, False),
        (0, {}, False),
        (True, {}, False),
    ],
)
def test_process_ancestry_requires_a_known_chain_to_this_run(pid, parents, expected):
    assert (
        tp_devices.is_descendant(pid, root_pid=10, read_parent=parents.get) is expected
    )


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_process_ancestry_rejects_unreadable_parent(failure):
    def read_parent(pid):
        raise failure("process is unavailable")

    assert tp_devices.is_descendant(30, root_pid=10, read_parent=read_parent) is False


def test_process_ancestry_stops_at_system_root_without_reading_it():
    def read_parent(pid):
        assert pid == 30
        return 1

    assert tp_devices.is_descendant(30, root_pid=10, read_parent=read_parent) is False


@pytest.fixture
def lease_workspace(tmp_path, monkeypatch):
    (tmp_path / "python").mkdir()
    source = tmp_path / "python/main.py"
    source.write_text("pass\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "python/main.py": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / ".tmp" / "result"
    output.mkdir(parents=True)
    monkeypatch.setenv("TOKENSPEED_HW2_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(
            LOCK_EX=1,
            LOCK_NB=2,
            flock=lambda *args: None,
        ),
    )
    monkeypatch.setattr(tp_devices.time, "sleep", lambda seconds: None)
    state = {"entered": False, "pids": {}, "full_queries": 0}

    def query(args):
        if args == ["-m"]:
            return "1 0 2 2 Ascend910\n1 1 3 3 Ascend910\n"
        if args == []:
            state["full_queries"] += 1
            rows = "".join(
                f"| 1 {chip} | {pid} | python | 112 |\n"
                for chip, pids in state["pids"].items()
                for pid in sorted(pids)
            )
            table = (
                PROCESS_HEADER
                + PROCESS_BORDER
                + (rows or "| No running processes found |\n")
                + PROCESS_BORDER
            )
            if state.get("release_child"):
                state["pids"] = {0: {tp_devices.os.getpid()}}
                state["release_child"] = False
            return state.get("full_override", table)
        assert args == ["-t", "proc-mem", "-i", "1"]
        pids = state["pids"] if state["entered"] or state.get("before_busy") else {}
        return _proc_card(pids, npu_id=1)

    monkeypatch.setattr(tp_devices, "_query", query)
    return tmp_path, output, manifest, state


@pytest.mark.parametrize("parent_chips", [set(), {0}, {1}, {0, 1}])
def test_lease_allows_only_parent_context_until_driver_process_exit(
    lease_workspace, parent_chips
):
    repo, output, manifest, state = lease_workspace
    parent = tp_devices.os.getpid()
    state["pids"] = {chip: {parent} for chip in parent_chips}
    with tp_devices.hardware_lease(
        repo,
        npu_id=1,
        chip_logic_ids={0: 2, 1: 3},
        tp_size=2,
        output_dir=output,
        source_manifest=manifest,
    ) as receipt:
        state["entered"] = True
    expected = {chip: [parent] if chip in parent_chips else [] for chip in (0, 1)}
    assert receipt["engine_children_released"] is True
    assert receipt["parent_context_pending_exit"] is bool(parent_chips)
    assert receipt["after_target_idle"] is (not parent_chips)
    assert receipt["after_process_pids"] == expected
    assert receipt["sources_unchanged"] is True
    if parent_chips:
        assert receipt["parent_pid"] == parent
        raw = (output / receipt["after_npu_smi"]).read_text(encoding="utf-8")
        assert tp_devices.target_process_pids(raw, npu_id=1, chip_ids={0, 1}) == {
            chip: set(pids) for chip, pids in expected.items()
        }


@pytest.mark.parametrize(
    "residual", ["child", "foreign", "parent_and_child", "foreign_sibling"]
)
def test_lease_does_not_allow_child_or_foreign_residuals(lease_workspace, residual):
    repo, output, manifest, state = lease_workspace
    parent = tp_devices.os.getpid()
    if residual == "child":
        state["pids"] = {0: {parent + 1}}
    elif residual == "foreign":
        state["pids"] = {0: {parent + 1000}}
    elif residual == "parent_and_child":
        state["pids"] = {0: {parent, parent + 1}}
    else:
        state["pids"] = {0: {parent}, 1: {parent + 1000}}
    with pytest.raises(ValueError, match="process"):
        with tp_devices.hardware_lease(
            repo,
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
            tp_size=2,
            output_dir=output,
            source_manifest=manifest,
        ) as receipt:
            state["entered"] = True
    assert state["full_queries"] == 10
    assert receipt["engine_children_released"] is False
    assert receipt["after_target_idle"] is False
    assert (output / "after_npu_smi.txt").is_file()


def test_lease_grace_waits_for_children_to_release_before_allowing_parent(
    lease_workspace,
):
    repo, output, manifest, state = lease_workspace
    parent = tp_devices.os.getpid()
    state.update(pids={0: {parent, parent + 1}}, release_child=True)
    with tp_devices.hardware_lease(
        repo,
        npu_id=1,
        chip_logic_ids={0: 2, 1: 3},
        tp_size=2,
        output_dir=output,
        source_manifest=manifest,
    ) as receipt:
        state["entered"] = True
    assert state["full_queries"] == 2
    assert receipt["engine_children_released"] is True
    assert receipt["after_process_pids"] == {0: [parent], 1: []}


def test_lease_still_rejects_parent_pid_before_experiment(lease_workspace):
    repo, output, manifest, state = lease_workspace
    state.update(pids={0: {tp_devices.os.getpid()}}, before_busy=True)
    with pytest.raises(ValueError, match="occupied"):
        with tp_devices.hardware_lease(
            repo,
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
            tp_size=2,
            output_dir=output,
            source_manifest=manifest,
        ):
            pytest.fail("a preoccupied target must never enter the experiment")
    assert state["full_queries"] == 0


def test_lease_does_not_allow_parent_on_an_unknown_full_process_table(lease_workspace):
    repo, output, manifest, state = lease_workspace
    state.update(pids={0: {tp_devices.os.getpid()}}, full_override="query failed")
    with pytest.raises(ValueError, match="process table"):
        with tp_devices.hardware_lease(
            repo,
            npu_id=1,
            chip_logic_ids={0: 2, 1: 3},
            tp_size=2,
            output_dir=output,
            source_manifest=manifest,
        ) as receipt:
            state["entered"] = True
    assert receipt["engine_children_released"] is False


def test_lease_clears_stale_child_pids_when_grace_query_becomes_idle(
    lease_workspace, monkeypatch
):
    repo, output, manifest, state = lease_workspace
    state["pids"] = {0: {tp_devices.os.getpid() + 1}}
    monkeypatch.setattr(tp_devices.time, "sleep", lambda seconds: state["pids"].clear())
    with tp_devices.hardware_lease(
        repo,
        npu_id=1,
        chip_logic_ids={0: 2, 1: 3},
        tp_size=2,
        output_dir=output,
        source_manifest=manifest,
    ) as receipt:
        state["entered"] = True
    assert state["full_queries"] == 1
    assert receipt["engine_children_released"] is True
    assert receipt["after_process_pids"] == {0: [], 1: []}
    assert receipt["parent_context_pending_exit"] is False
    assert receipt["after_target_idle"] is True
