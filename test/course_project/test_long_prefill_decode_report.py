from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from benchmarks.course_project import long_prefill_decode

try:
    from scripts import analyze_long_prefill_decode
except ImportError:
    analyze_long_prefill_decode = None


CONDITIONS = (
    "decode-only",
    "prefill-only-2048",
    "prefill-only-3584",
    "staged-2048",
    "staged-3584",
)
CHUNK_ORDER_BY_ROUND = ((4096, 512), (512, 4096), (4096, 512))
_EPOCH_BASE_NS = 1_700_000_000_000_000_000


def _chunk_event(chunk_index: int, monotonic_ns: int) -> dict:
    return {
        "chunk_index": chunk_index,
        "field": "content",
        "epoch_ns": _EPOCH_BASE_NS + monotonic_ns,
        "monotonic_ns": monotonic_ns,
        "text_length": 1,
    }


def _request(role: str, condition: str, *, chunk_size: int, repetition: int) -> dict:
    if role == "decode":
        expected_input_tokens, expected_output_tokens = 256, 128
        ttft_ms = 10.0 + repetition + chunk_size / 4096
    else:
        expected_input_tokens = int(condition.rsplit("-", 1)[1])
        expected_output_tokens = 1
        ttft_ms = expected_input_tokens / 100 + repetition + chunk_size / 4096
    send_ns = 50_000_000
    first_chunk_ns = send_ns + round(ttft_ms * 1_000_000)
    e2e_ms = ttft_ms + 20
    drain_ns = send_ns + round(e2e_ms * 1_000_000)
    return {
        "request_role": role,
        "expected_input_tokens": expected_input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "input_tokens": expected_input_tokens,
        "output_tokens": expected_output_tokens,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "request_send_epoch_ns": _EPOCH_BASE_NS + send_ns,
        "request_send_monotonic_ns": send_ns,
        "first_chunk_epoch_ns": _EPOCH_BASE_NS + first_chunk_ns,
        "first_chunk_monotonic_ns": first_chunk_ns,
        "first_chunk_field": "content",
        "stream_drain_epoch_ns": _EPOCH_BASE_NS + drain_ns,
        "stream_drain_monotonic_ns": drain_ns,
        "chunk_events": [_chunk_event(1, first_chunk_ns)],
        "stream_completed": True,
        "status": "valid",
        "invalid_reason": None,
    }


def _trial(condition: str, repetition: int, chunk_size: int) -> dict:
    prefill_tokens = (
        0 if condition == "decode-only" else int(condition.rsplit("-", 1)[1])
    )
    if condition == "decode-only":
        roles = ("decode",)
    elif condition.startswith("prefill-only-"):
        roles = ("prefill",)
    else:
        roles = ("decode", "prefill")
    requests = [
        _request(role, condition, chunk_size=chunk_size, repetition=repetition)
        for role in roles
    ]
    staged = condition.startswith("staged-")
    gap_events = []
    if staged:
        send_ns = 100_000_000
        requests[1]["request_send_monotonic_ns"] = send_ns
        requests[1]["request_send_epoch_ns"] = _EPOCH_BASE_NS + send_ns
        requests[1]["first_chunk_monotonic_ns"] = 120_000_000
        requests[1]["first_chunk_epoch_ns"] = _EPOCH_BASE_NS + 120_000_000
        requests[1]["stream_drain_monotonic_ns"] = 140_000_000
        requests[1]["stream_drain_epoch_ns"] = _EPOCH_BASE_NS + 140_000_000
        requests[1]["chunk_events"] = [_chunk_event(1, 120_000_000)]
        base = 10 if chunk_size == 4096 else 20
        before_ms = base + repetition
        after_ms = (100 if chunk_size == 4096 else 200) + repetition
        crossing_ms = 9999
        before_from_ns = send_ns - (before_ms + 1) * 1_000_000
        before_to_ns = send_ns - 1_000_000
        crossing_to_ns = before_to_ns + crossing_ms * 1_000_000
        after_to_ns = crossing_to_ns + after_ms * 1_000_000
        decode_chunk_times = (
            before_from_ns,
            before_to_ns,
            crossing_to_ns,
            after_to_ns,
        )
        requests[0]["first_chunk_monotonic_ns"] = decode_chunk_times[0]
        requests[0]["first_chunk_epoch_ns"] = _EPOCH_BASE_NS + decode_chunk_times[0]
        requests[0]["chunk_events"] = [
            _chunk_event(index, monotonic_ns)
            for index, monotonic_ns in enumerate(decode_chunk_times, start=1)
        ]
        requests[0]["stream_drain_monotonic_ns"] = after_to_ns + 1_000_000
        requests[0]["stream_drain_epoch_ns"] = _EPOCH_BASE_NS + after_to_ns + 1_000_000
        for request in requests:
            request["ttft_ms"] = (
                request["first_chunk_monotonic_ns"]
                - request["request_send_monotonic_ns"]
            ) / 1_000_000
            request["e2e_ms"] = (
                request["stream_drain_monotonic_ns"]
                - request["request_send_monotonic_ns"]
            ) / 1_000_000
        gap_events = [
            {
                "classification": "before",
                "from_chunk_index": 1,
                "to_chunk_index": 2,
                "from_monotonic_ns": before_from_ns,
                "to_monotonic_ns": before_to_ns,
                "client_sse_chunk_gap_ms": float(before_ms),
            },
            {
                "classification": "crosses",
                "from_chunk_index": 2,
                "to_chunk_index": 3,
                "from_monotonic_ns": before_to_ns,
                "to_monotonic_ns": crossing_to_ns,
                "client_sse_chunk_gap_ms": float(crossing_ms),
            },
            {
                "classification": "after",
                "from_chunk_index": 3,
                "to_chunk_index": 4,
                "from_monotonic_ns": crossing_to_ns,
                "to_monotonic_ns": after_to_ns,
                "client_sse_chunk_gap_ms": float(after_ms),
            },
        ]
    config_sha = f"{chunk_size:064x}"
    return {
        "schema_version": 1,
        "trial_id": f"{condition}-r{repetition}-c{chunk_size}",
        "condition": condition,
        "prefill_tokens": prefill_tokens,
        "round": repetition,
        "repetition": repetition,
        "chunk_size": chunk_size,
        "execution_order": 1,
        "status": "valid",
        "invalid_reason": None,
        "requests_overlap": True if staged else None,
        "requests": requests,
        "chunk_gap_events": gap_events,
        "run_id": "synthetic-run",
        "pair_id": f"{condition}-r{repetition}",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "synthetic-revision",
        "served_model_name": "synthetic-model",
        "config_sha256": config_sha,
        "prompt_recipe_version": "long-prefill-v1",
        "prompt_hashes": {role: "a" * 64 for role in roles},
    }


def _write_synthetic_run(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    schedule = []
    for repetition, chunk_order in enumerate(CHUNK_ORDER_BY_ROUND, start=1):
        for order, chunk_size in enumerate(chunk_order, start=1):
            schedule.append(
                {"round": repetition, "order": order, "chunk_size": chunk_size}
            )
            config = {
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": "synthetic-revision",
                "served_model_name": "synthetic-model",
                "tp_size": 1,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "chunked_prefill_size": chunk_size,
                "prefix_caching": False,
                "synthetic_setting": "stable",
            }
            snapshot = {
                "schema_version": 1,
                "run_id": "synthetic-run",
                "round": repetition,
                "launch_order": order,
                "config_path": "configs/bringup/synthetic.json",
                "config_file_sha256": "b" * 64,
                "config_sha256": f"{chunk_size:064x}",
                "config": config,
            }
            (
                raw_dir / f"config-round-{repetition:02d}-chunk-{chunk_size}.json"
            ).write_text(json.dumps(snapshot), encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": "synthetic-run",
        "config_path": "configs/bringup/synthetic.json",
        "config_file_sha256": "b" * 64,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "synthetic-revision",
        "host": "127.0.0.1",
        "port": 8000,
        "schedule": schedule,
        "conditions": list(CONDITIONS),
    }
    (raw_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")

    trials = [
        _trial(condition, repetition, chunk_size)
        for condition in CONDITIONS
        for repetition in range(1, 4)
        for chunk_size in (4096, 512)
    ]
    (raw_dir / "trials.jsonl").write_text(
        "".join(json.dumps(trial) + "\n" for trial in trials), encoding="utf-8"
    )


def test_report_rejects_global_order_in_round_local_launch_field(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    path = raw_dir / "config-round-02-chunk-512.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["launch_order"] = 3
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="launch_order is invalid"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)


def _read_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_trials(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


@pytest.fixture
def synthetic_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "synthetic-project"
    raw_dir = (
        project_root
        / "results"
        / "raw"
        / "benchmark"
        / "long_prefill_decode"
        / "synthetic-run"
    )
    _write_synthetic_run(raw_dir)
    monkeypatch.setattr(long_prefill_decode, "REPO_ROOT", project_root)
    if analyze_long_prefill_decode is not None:
        monkeypatch.setattr(analyze_long_prefill_decode, "REPO_ROOT", project_root)
    return (
        project_root,
        raw_dir,
        project_root / "results" / "analysis" / "synthetic-run",
    )


def test_analyze_run_writes_paired_csv_summary_and_gap_svg(synthetic_paths):
    project_root, raw_dir, output_dir = synthetic_paths
    staged_trials = [
        trial
        for trial in _read_trials(raw_dir / "trials.jsonl")
        if trial["condition"].startswith("staged-")
    ]
    assert all(
        len(trial["chunk_gap_events"]) == len(trial["requests"][0]["chunk_events"]) - 1
        for trial in staged_trials
    )

    outputs = long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert outputs == [
        output_dir / "paired_trials.csv",
        output_dir / "summary.csv",
        output_dir / "decode_chunk_gaps.svg",
    ]
    with outputs[0].open(newline="", encoding="utf-8") as csv_file:
        paired_rows = list(csv.DictReader(csv_file))
    with outputs[1].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    assert len(paired_rows) == len(CONDITIONS) * 3
    assert set(row["condition"] for row in paired_rows) == set(CONDITIONS)

    staged_round = next(
        row
        for row in paired_rows
        if row["condition"] == "staged-2048" and row["repetition"] == "1"
    )
    assert staged_round["status"] == "valid"
    assert (
        float(staged_round["chunk_size_4096_client_sse_chunk_gap_before_median_ms"])
        == 11
    )
    assert (
        float(staged_round["chunk_size_4096_client_sse_chunk_gap_after_median_ms"])
        == 101
    )
    assert staged_round["chunk_size_4096_client_sse_chunk_gap_crosses_count"] == "1"
    assert (
        float(
            staged_round[
                "paired_difference_client_sse_chunk_gap_after_median_ms_512_minus_4096"
            ]
        )
        == 100
    )
    assert "9999" not in json.dumps(staged_round)

    after_summary = next(
        row
        for row in summary_rows
        if row["condition"] == "staged-2048"
        and row["metric"] == "client_sse_chunk_gap_after_median_ms"
    )
    assert after_summary["n_pairs"] == "3"
    assert float(after_summary["paired_difference_median"]) == 100
    assert "客户端 SSE 块间隔" in after_summary["conclusion_boundary"]
    assert "算子" in after_summary["conclusion_boundary"]

    svg_root = ElementTree.parse(outputs[2]).getroot()
    elements = list(svg_root.iter())
    assert svg_root.tag.endswith("svg")
    observations = [
        element for element in elements if element.get("class") == "observation"
    ]
    medians = [element for element in elements if element.get("class") == "median"]
    assert len(observations) == 24
    assert len(medians) == 8
    assert all(float(element.get("data-value-ms")) != 9999 for element in observations)
    assert (
        "confidence" not in " ".join(element.text or "" for element in elements).lower()
    )
    assert all(path.is_relative_to(project_root) for path in outputs)


def test_analyze_run_refuses_existing_output_without_overwriting(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    output_dir.mkdir(parents=True)
    marker = output_dir / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (output_dir / "paired_trials.csv").exists()


def test_analyze_run_rejects_output_nested_in_raw_run(synthetic_paths):
    _, raw_dir, _ = synthetic_paths
    nested_output = raw_dir / "analysis"

    with pytest.raises(ValueError, match="output directory.*inside the raw run"):
        long_prefill_decode.analyze_run(raw_dir, nested_output)

    assert not nested_output.exists()


def test_analyze_run_rejects_trial_served_model_name_mismatch(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    records[0]["served_model_name"] = "different-model-alias"
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="served_model_name.*config snapshot"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


def test_analyze_run_cleans_staging_directory_after_write_failure(
    synthetic_paths, monkeypatch
):
    _, raw_dir, output_dir = synthetic_paths
    original_open = Path.open

    def fail_on_summary(path, *args, **kwargs):
        if path.name == "summary.csv":
            raise OSError("synthetic output write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_on_summary)

    with pytest.raises(OSError, match="synthetic output write failure"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()
    assert list(output_dir.parent.glob(".long-prefill-decode-*")) == []


@pytest.mark.parametrize("mismatch", ["model_revision", "chunked_prefill_size"])
def test_analyze_run_rejects_config_inconsistency(synthetic_paths, mismatch):
    _, raw_dir, output_dir = synthetic_paths
    snapshot_path = raw_dir / "config-round-02-chunk-512.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if mismatch == "model_revision":
        snapshot["config"]["model_revision"] = "different-revision"
    else:
        snapshot["config"]["chunked_prefill_size"] = 4096
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="config|revision|chunk"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


def test_analyze_run_reports_insufficient_pairs_with_raw_observations(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
    ]
    records = [
        trial
        for trial in records
        if not (
            trial["condition"] == "staged-2048"
            and trial["repetition"] == 3
            and trial["chunk_size"] == 512
        )
    ]
    trials_path.write_text(
        "".join(json.dumps(trial) + "\n" for trial in records), encoding="utf-8"
    )

    outputs = long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert outputs == [
        output_dir / "paired_trials.csv",
        output_dir / "summary.csv",
        output_dir / "decode_chunk_gaps.svg",
    ]
    with outputs[0].open(newline="", encoding="utf-8") as csv_file:
        paired_rows = list(csv.DictReader(csv_file))
    with outputs[1].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))

    missing_trial_pair = next(
        row
        for row in paired_rows
        if row["condition"] == "staged-2048" and row["repetition"] == "3"
    )
    assert missing_trial_pair["status"] == "invalid"
    assert "chunk_size_512: missing sample" in missing_trial_pair["invalid_reason"]
    assert missing_trial_pair["chunk_size_4096_decode_ttft_ms"]
    assert missing_trial_pair["chunk_size_512_decode_ttft_ms"] == ""

    insufficient_summary = [
        row for row in summary_rows if row["condition"] == "staged-2048"
    ]
    assert len(insufficient_summary) == 6
    assert all(row["n_pairs"] == "2" for row in insufficient_summary)
    for row in insufficient_summary:
        assert "数据不足" in row["conclusion_boundary"]
        for field in (
            "median_chunk_size_4096",
            "min_chunk_size_4096",
            "max_chunk_size_4096",
            "median_chunk_size_512",
            "min_chunk_size_512",
            "max_chunk_size_512",
            "paired_difference_median",
            "paired_difference_min",
            "paired_difference_max",
        ):
            assert row[field] == ""

    elements = list(ElementTree.parse(outputs[2]).getroot().iter())
    assert not any(
        element.get("class") == "median"
        and element.get("data-condition") == "staged-2048"
        for element in elements
    )
    insufficient_labels = [
        element
        for element in elements
        if element.get("class") == "data-insufficient"
        and element.get("data-condition") == "staged-2048"
    ]
    assert len(insufficient_labels) == 2
    assert {element.get("data-n") for element in insufficient_labels} == {"2"}


@pytest.mark.parametrize(
    "filename",
    [
        "run.json",
        "config-round-01-chunk-4096.json",
        "trials.jsonl",
    ],
)
def test_analyze_run_rejects_input_symlink_escape(synthetic_paths, filename):
    project_root, raw_dir, output_dir = synthetic_paths
    input_path = raw_dir / filename
    outside_path = project_root.parent / f"outside-{filename}"
    outside_path.write_bytes(input_path.read_bytes())
    input_path.unlink()
    try:
        input_path.symlink_to(outside_path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="inside the project"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


def test_analyze_run_rejects_input_symlink_to_sibling_raw_run(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    sibling_raw_dir = raw_dir.parent / "sibling-run"
    _write_synthetic_run(sibling_raw_dir)
    sibling_run_path = sibling_raw_dir / "run.json"
    sibling_run = json.loads(sibling_run_path.read_text(encoding="utf-8"))
    sibling_run["sibling_only_marker"] = True
    sibling_run_path.write_text(json.dumps(sibling_run), encoding="utf-8")

    input_path = raw_dir / "run.json"
    input_path.unlink()
    try:
        input_path.symlink_to(sibling_run_path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="raw run"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


def test_report_input_path_rejects_resolved_path_outside_raw_run(synthetic_paths):
    _, raw_dir, _ = synthetic_paths
    sibling_raw_dir = raw_dir.parent / "sibling-run"
    _write_synthetic_run(sibling_raw_dir)

    with pytest.raises(ValueError, match="raw run"):
        long_prefill_decode._report_input_path(sibling_raw_dir / "run.json", raw_dir)


def test_analyze_run_rejects_repetition_four(synthetic_paths):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    extra = _trial("staged-2048", 4, 4096)
    records.append(extra)
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="between 1 and 3"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("bad_round", [2.0, True, 1])
def test_analyze_run_requires_integer_round_matching_repetition(
    synthetic_paths, bad_round
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "prefill-only-2048"
        and item["repetition"] == 2
        and item["chunk_size"] == 4096
    )
    trial["round"] = bad_round
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="round"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prefill_tokens", 2048.0),
        ("prefill_tokens", True),
        ("expected_input_tokens", 2048.0),
        ("expected_input_tokens", True),
        ("expected_output_tokens", 1.0),
        ("expected_output_tokens", True),
        ("input_tokens", 2048.0),
        ("input_tokens", True),
        ("output_tokens", 1.0),
        ("output_tokens", True),
    ],
)
def test_analyze_run_requires_integer_token_counts(synthetic_paths, field, bad_value):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "prefill-only-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    if field == "prefill_tokens":
        trial[field] = bad_value
    else:
        trial["requests"][0][field] = bad_value
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="integer"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("decode_drain_ns", [90_000_000, 100_000_000])
def test_report_recomputes_strict_staged_overlap_from_request_times(decode_drain_ns):
    trial = _trial("staged-2048", 1, 4096)
    decode, prefill = trial["requests"]
    decode["stream_drain_monotonic_ns"] = decode_drain_ns
    decode["stream_drain_epoch_ns"] = _EPOCH_BASE_NS + decode_drain_ns
    decode_times = (60_000_000, 70_000_000)
    decode["first_chunk_monotonic_ns"] = decode_times[0]
    decode["first_chunk_epoch_ns"] = _EPOCH_BASE_NS + decode_times[0]
    decode["chunk_events"] = [
        _chunk_event(index, timestamp)
        for index, timestamp in enumerate(decode_times, start=1)
    ]
    decode["ttft_ms"] = (
        decode["first_chunk_monotonic_ns"] - decode["request_send_monotonic_ns"]
    ) / 1_000_000
    decode["e2e_ms"] = (
        decode["stream_drain_monotonic_ns"] - decode["request_send_monotonic_ns"]
    ) / 1_000_000
    prefill["request_send_monotonic_ns"] = 100_000_000
    prefill["stream_drain_monotonic_ns"] = 150_000_000
    prefill["e2e_ms"] = (
        prefill["stream_drain_monotonic_ns"] - prefill["request_send_monotonic_ns"]
    ) / 1_000_000
    trial["requests_overlap"] = True

    with pytest.raises(ValueError, match="overlap"):
        long_prefill_decode._report_request_metrics(
            trial, condition="staged-2048", prefill_tokens=2048
        )


def test_report_requires_prefill_send_after_first_decode_chunk():
    trial = _trial("staged-2048", 1, 4096)
    decode, prefill = trial["requests"]
    decode_times = (110_000_000, 120_000_000, 130_000_000, 140_000_000)
    decode["first_chunk_monotonic_ns"] = decode_times[0]
    decode["first_chunk_epoch_ns"] = _EPOCH_BASE_NS + decode_times[0]
    decode["chunk_events"] = [
        _chunk_event(index, timestamp)
        for index, timestamp in enumerate(decode_times, start=1)
    ]
    decode["ttft_ms"] = (
        decode["first_chunk_monotonic_ns"] - decode["request_send_monotonic_ns"]
    ) / 1_000_000
    prefill["request_send_monotonic_ns"] = 100_000_000
    trial["requests_overlap"] = True

    with pytest.raises(ValueError, match="first.*chunk|chunk.*first"):
        long_prefill_decode._report_request_metrics(
            trial, condition="staged-2048", prefill_tokens=2048
        )


def test_report_rejects_valid_staged_decode_without_first_chunk():
    trial = _trial("staged-2048", 1, 4096)
    trial["requests"][0]["chunk_events"] = []

    with pytest.raises(ValueError, match="decode.*chunk"):
        long_prefill_decode._report_request_metrics(
            trial, condition="staged-2048", prefill_tokens=2048
        )


@pytest.mark.parametrize("role", ["decode", "prefill"])
@pytest.mark.parametrize(
    "tamper",
    ["sequence", "time", "before_send", "after_drain", "field", "length", "structure"],
)
def test_report_validates_each_valid_request_chunk_event(role, tamper):
    trial = _trial("staged-2048", 1, 4096)
    request = next(item for item in trial["requests"] if item["request_role"] == role)
    events = request["chunk_events"]
    if len(events) < 2:
        next_time = events[-1]["monotonic_ns"] + 1_000_000
        events.append(_chunk_event(2, next_time))
    if tamper == "sequence":
        events[1]["chunk_index"] = events[0]["chunk_index"]
    elif tamper == "time":
        events[1]["monotonic_ns"] = events[0]["monotonic_ns"]
    elif tamper == "before_send":
        events[0]["monotonic_ns"] = request["request_send_monotonic_ns"] - 1
        events[0]["epoch_ns"] = _EPOCH_BASE_NS + events[0]["monotonic_ns"]
    elif tamper == "after_drain":
        events[-1]["monotonic_ns"] = request["stream_drain_monotonic_ns"] + 1
        events[-1]["epoch_ns"] = _EPOCH_BASE_NS + events[-1]["monotonic_ns"]
    elif tamper == "field":
        events[0]["field"] = "other"
    elif tamper == "length":
        events[0]["text_length"] = 0
    else:
        del events[0]["epoch_ns"]

    with pytest.raises(ValueError, match="chunk"):
        long_prefill_decode._report_request_metrics(
            trial, condition="staged-2048", prefill_tokens=2048
        )


@pytest.mark.parametrize("tamper", ["classification", "duration", "send_time"])
def test_analyze_run_recomputes_gap_classification_and_duration(
    synthetic_paths, tamper
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    crossing = trial["chunk_gap_events"][1]
    if tamper == "classification":
        crossing["classification"] = "after"
    elif tamper == "duration":
        crossing["client_sse_chunk_gap_ms"] = 9998.0
    else:
        prefill = trial["requests"][1]
        prefill["request_send_monotonic_ns"] = 90_000_000
        prefill["ttft_ms"] = (
            prefill["first_chunk_monotonic_ns"] - prefill["request_send_monotonic_ns"]
        ) / 1_000_000
        prefill["e2e_ms"] = (
            prefill["stream_drain_monotonic_ns"] - prefill["request_send_monotonic_ns"]
        ) / 1_000_000
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="classification|duration"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("metric", ["ttft_ms", "e2e_ms"])
def test_analyze_run_rejects_metrics_that_disagree_with_monotonic_timestamps(
    synthetic_paths, metric
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "prefill-only-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    trial["requests"][0][metric] += 10
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match=f"{metric}.*monotonic timestamps"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("tamper", ["missing", "before_send", "after_drain"])
def test_analyze_run_rejects_successful_requests_without_ordered_first_chunk_time(
    synthetic_paths, tamper
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "prefill-only-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    request = trial["requests"][0]
    if tamper == "missing":
        request["first_chunk_monotonic_ns"] = None
    elif tamper == "before_send":
        request["first_chunk_monotonic_ns"] = request["request_send_monotonic_ns"] - 1
    else:
        request["first_chunk_monotonic_ns"] = request["stream_drain_monotonic_ns"] + 1
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="first.chunk|request interval"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("tamper", ["epoch", "field"])
def test_analyze_run_requires_first_chunk_audit_metadata_to_match_event(
    synthetic_paths, tamper
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "prefill-only-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    request = trial["requests"][0]
    if tamper == "epoch":
        request["first_chunk_epoch_ns"] += 1
    else:
        request["first_chunk_field"] = "reasoning_content"
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="first.chunk.*first chunk event"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("tamper", ["endpoint", "non_adjacent_indices"])
def test_analyze_run_rejects_self_consistent_gaps_not_backed_by_decode_chunks(
    synthetic_paths, tamper
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    decode_events = trial["requests"][0]["chunk_events"]
    gap = trial["chunk_gap_events"][0]
    if tamper == "endpoint":
        gap["from_monotonic_ns"] += 1_000_000
    else:
        gap["to_chunk_index"] = 3
        gap["to_monotonic_ns"] = decode_events[2]["monotonic_ns"]
        gap["classification"] = "crosses"
    gap["client_sse_chunk_gap_ms"] = (
        gap["to_monotonic_ns"] - gap["from_monotonic_ns"]
    ) / 1_000_000
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="chunk gap.*decode chunk|decode chunk.*gap"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("tamper", ["deleted", "duplicated"])
def test_analyze_run_requires_one_gap_record_per_decode_chunk_pair(
    synthetic_paths, tamper
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    if tamper == "deleted":
        trial["chunk_gap_events"].pop()
    else:
        trial["chunk_gap_events"].append(dict(trial["chunk_gap_events"][0]))
    _write_trials(trials_path, records)

    with pytest.raises(ValueError, match="gap records.*Decode chunk pair"):
        long_prefill_decode.analyze_run(raw_dir, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("boundary", ["before_end", "after_start"])
def test_gap_at_prefill_send_boundary_uses_half_open_classification(
    synthetic_paths, boundary
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 1
        and item["chunk_size"] == 4096
    )
    send_ns = trial["requests"][1]["request_send_monotonic_ns"]
    decode_request = trial["requests"][0]
    decode_events = decode_request["chunk_events"]
    if boundary == "before_end":
        event = trial["chunk_gap_events"][0]
        duration_ns = round(event["client_sse_chunk_gap_ms"] * 1_000_000)
        decode_events[0]["monotonic_ns"] = send_ns - duration_ns
        decode_events[1]["monotonic_ns"] = send_ns
        expected_metric = "client_sse_chunk_gap_before_median_ms"
    else:
        event = trial["chunk_gap_events"][2]
        duration_ns = round(event["client_sse_chunk_gap_ms"] * 1_000_000)
        decode_events[2]["monotonic_ns"] = send_ns
        decode_events[3]["monotonic_ns"] = send_ns + duration_ns
        expected_metric = "client_sse_chunk_gap_after_median_ms"
    for chunk_event in decode_events:
        chunk_event["epoch_ns"] = _EPOCH_BASE_NS + chunk_event["monotonic_ns"]
    decode_request["first_chunk_monotonic_ns"] = decode_events[0]["monotonic_ns"]
    decode_request["first_chunk_epoch_ns"] = decode_events[0]["epoch_ns"]
    decode_request["ttft_ms"] = (
        decode_request["first_chunk_monotonic_ns"]
        - decode_request["request_send_monotonic_ns"]
    ) / 1_000_000
    for gap, previous, current in zip(
        trial["chunk_gap_events"], decode_events, decode_events[1:]
    ):
        start_ns = previous["monotonic_ns"]
        end_ns = current["monotonic_ns"]
        gap["from_monotonic_ns"] = start_ns
        gap["to_monotonic_ns"] = end_ns
        gap["client_sse_chunk_gap_ms"] = (end_ns - start_ns) / 1_000_000
        gap["classification"] = (
            "before"
            if end_ns <= send_ns
            else "after" if start_ns >= send_ns else "crosses"
        )
    _write_trials(trials_path, records)

    outputs = long_prefill_decode.analyze_run(raw_dir, output_dir)

    with outputs[0].open(newline="", encoding="utf-8") as csv_file:
        paired_rows = list(csv.DictReader(csv_file))
    staged_round = next(
        row
        for row in paired_rows
        if row["condition"] == "staged-2048" and row["repetition"] == "1"
    )
    assert staged_round[f"chunk_size_4096_{expected_metric}"]
    assert staged_round["chunk_size_4096_client_sse_chunk_gap_crosses_count"] == "0"


def test_invalid_staged_trial_without_prefill_send_stays_out_of_statistics(
    synthetic_paths,
):
    _, raw_dir, _ = synthetic_paths
    run, hashes_by_chunk, served_model_name = long_prefill_decode._report_validate_run(
        raw_dir
    )
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    invalid_trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 2
        and item["chunk_size"] == 4096
    )
    invalid_trial["status"] = "invalid"
    invalid_trial["invalid_reason"] = "prefill request failed before send"
    prefill_request = invalid_trial["requests"][1]
    prefill_request["status"] = "invalid"
    prefill_request["invalid_reason"] = "request failed before send"
    prefill_request["request_send_monotonic_ns"] = None
    invalid_trial["requests_overlap"] = None
    invalid_trial["chunk_gap_events"] = []
    prefill_request["input_tokens"] = None
    prefill_request["output_tokens"] = None
    _write_trials(trials_path, records)

    trials = long_prefill_decode._report_load_trials(
        raw_dir, run, hashes_by_chunk, served_model_name
    )
    pairs = long_prefill_decode._report_pairs(trials)
    pair = next(
        item
        for item in pairs
        if item["condition"] == "staged-2048" and item["repetition"] == 2
    )

    assert pair["status"] == "invalid"
    assert "request failed before send" in pair["invalid_reason"]
    assert pair["values"][4096]["client_sse_chunk_gap_before_median_ms"] is None


def test_gap_metric_with_fewer_than_three_pairs_has_no_summary_or_differences(
    synthetic_paths,
):
    _, raw_dir, output_dir = synthetic_paths
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 3
        and item["chunk_size"] == 512
    )
    prefill = trial["requests"][1]
    prefill["request_send_monotonic_ns"] = 90_000_000
    prefill["request_send_epoch_ns"] = _EPOCH_BASE_NS + 90_000_000
    prefill["ttft_ms"] = (
        prefill["first_chunk_monotonic_ns"] - prefill["request_send_monotonic_ns"]
    ) / 1_000_000
    prefill["e2e_ms"] = (
        prefill["stream_drain_monotonic_ns"] - prefill["request_send_monotonic_ns"]
    ) / 1_000_000
    trial["chunk_gap_events"][0]["classification"] = "crosses"
    trial["chunk_gap_events"][1]["classification"] = "after"
    _write_trials(trials_path, records)

    outputs = long_prefill_decode.analyze_run(raw_dir, output_dir)

    with outputs[0].open(newline="", encoding="utf-8") as csv_file:
        paired_rows = list(csv.DictReader(csv_file))
    with outputs[1].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    paired_row = next(
        row
        for row in paired_rows
        if row["condition"] == "staged-2048" and row["repetition"] == "1"
    )
    assert paired_row["chunk_size_4096_client_sse_chunk_gap_before_median_ms"]
    assert (
        paired_row[
            "paired_difference_client_sse_chunk_gap_before_median_ms_512_minus_4096"
        ]
        == ""
    )
    summary = next(
        row
        for row in summary_rows
        if row["condition"] == "staged-2048"
        and row["metric"] == "client_sse_chunk_gap_before_median_ms"
    )
    assert summary["n_pairs"] == "2"
    for field in (
        "median_chunk_size_4096",
        "min_chunk_size_4096",
        "max_chunk_size_4096",
        "median_chunk_size_512",
        "min_chunk_size_512",
        "max_chunk_size_512",
        "paired_difference_median",
        "paired_difference_min",
        "paired_difference_max",
    ):
        assert summary[field] == ""
    assert "数据不足" in summary["conclusion_boundary"]

    elements = list(ElementTree.parse(outputs[2]).getroot().iter())
    before_medians = [
        element
        for element in elements
        if element.get("class") == "median"
        and element.get("data-condition") == "staged-2048"
        and element.get("data-phase") == "before"
    ]
    insufficient_labels = [
        element
        for element in elements
        if element.get("class") == "data-insufficient"
        and element.get("data-condition") == "staged-2048"
        and element.get("data-phase") == "before"
    ]
    assert before_medians == []
    assert len(insufficient_labels) == 1


def test_invalid_pair_with_early_eos_keeps_side_reason_and_excludes_values_from_summary(
    synthetic_paths,
):
    _, raw_dir, _ = synthetic_paths
    run, hashes_by_chunk, served_model_name = long_prefill_decode._report_validate_run(
        raw_dir
    )
    trials_path = raw_dir / "trials.jsonl"
    records = _read_trials(trials_path)
    invalid_trial = next(
        item
        for item in records
        if item["condition"] == "staged-2048"
        and item["repetition"] == 2
        and item["chunk_size"] == 4096
    )
    invalid_trial["status"] = "invalid"
    invalid_trial["invalid_reason"] = "decode: early EOS (output token mismatch)"
    invalid_trial["requests"][0]["status"] = "invalid"
    invalid_trial["requests"][0]["output_tokens"] = 127
    invalid_trial["requests"][0]["invalid_reason"] = "early EOS (output token mismatch)"
    _write_trials(trials_path, records)

    trials = long_prefill_decode._report_load_trials(
        raw_dir, run, hashes_by_chunk, served_model_name
    )
    pairs = long_prefill_decode._report_pairs(trials)
    paired_rows, _ = long_prefill_decode._report_paired_rows(pairs)
    summary_rows, _ = long_prefill_decode._report_summary_rows(pairs)
    invalid_pair = next(
        pair
        for pair in pairs
        if pair["condition"] == "staged-2048" and pair["repetition"] == 2
    )
    invalid_csv_row = next(
        row
        for row in paired_rows
        if row["condition"] == "staged-2048" and row["repetition"] == 2
    )
    after_summary = next(
        row
        for row in summary_rows
        if row["condition"] == "staged-2048"
        and row["metric"] == "client_sse_chunk_gap_after_median_ms"
    )

    assert invalid_pair["status"] == "invalid"
    assert "chunk_size_4096" in invalid_pair["invalid_reason"]
    assert "early EOS" in invalid_pair["invalid_reason"]
    assert "output token mismatch" in invalid_pair["invalid_reason"]
    assert "invalid_trial" not in invalid_pair["invalid_reason"]
    assert invalid_csv_row["chunk_size_4096_decode_ttft_ms"] is None
    assert invalid_csv_row["chunk_size_512_decode_ttft_ms"] is not None
    assert invalid_csv_row["paired_difference_decode_ttft_ms_512_minus_4096"] is None
    assert after_summary["n_pairs"] == 2
    assert after_summary["paired_difference_median"] is None


def test_analyze_cli_rejects_external_paths_and_prints_project_relative_outputs(
    synthetic_paths, capsys
):
    assert analyze_long_prefill_decode is not None
    project_root, raw_dir, output_dir = synthetic_paths
    outside_raw = project_root.parent / "outside-raw"
    outside_raw.mkdir()

    with pytest.raises(SystemExit) as error:
        analyze_long_prefill_decode.main(
            ["--raw-run-dir", str(outside_raw), "--output-dir", str(output_dir)]
        )
    assert error.value.code == 2
    assert not output_dir.exists()

    outside_output = project_root.parent / "outside-analysis"
    with pytest.raises(SystemExit) as error:
        analyze_long_prefill_decode.main(
            ["--raw-run-dir", str(raw_dir), "--output-dir", str(outside_output)]
        )
    assert error.value.code == 2
    assert not outside_output.exists()

    analyze_long_prefill_decode.main(
        ["--raw-run-dir", str(raw_dir), "--output-dir", str(output_dir)]
    )
    printed_paths = capsys.readouterr().out.splitlines()
    assert printed_paths == [
        path.relative_to(project_root).as_posix()
        for path in (
            output_dir / "paired_trials.csv",
            output_dir / "summary.csv",
            output_dir / "decode_chunk_gaps.svg",
        )
    ]


def test_analyze_cli_rejects_existing_output(synthetic_paths, capsys):
    assert analyze_long_prefill_decode is not None
    project_root, raw_dir, output_dir = synthetic_paths
    output_dir.mkdir(parents=True)
    marker = output_dir / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        analyze_long_prefill_decode.main(
            ["--raw-run-dir", str(raw_dir), "--output-dir", str(output_dir)]
        )

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "already exists" in stderr
    assert str(project_root) not in stderr
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_analyze_cli_sanitizes_output_error_paths(synthetic_paths, monkeypatch, capsys):
    assert analyze_long_prefill_decode is not None
    project_root, raw_dir, output_dir = synthetic_paths

    def fail_write(raw_path, target_path):
        raise OSError(f"synthetic write failure at {project_root}")

    monkeypatch.setattr(analyze_long_prefill_decode, "analyze_run", fail_write)

    with pytest.raises(SystemExit) as error:
        analyze_long_prefill_decode.main(
            ["--raw-run-dir", str(raw_dir), "--output-dir", str(output_dir)]
        )

    assert error.value.code == 2
    assert str(project_root) not in capsys.readouterr().err
