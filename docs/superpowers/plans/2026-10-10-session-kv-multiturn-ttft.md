# Session KV Multi-turn TTFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved 4B/TP1/NPU0 experiment that records 64 serial, matched Session KV vs stateless TTFT pairs across four 16-turn conversations, then emits a provenance-preserving CSV and SVG only for a fully valid run.

**Architecture:** Keep experiment orchestration, all-history prompt construction, streaming measurement, atomic raw checkpoints, and CLI in `scripts/run_session_kv_ttft.py`. Put strict raw-run validation and CSV/SVG rendering in `benchmarks/course_project/session_kv_ttft_report.py`. Add a shell entry point that activates the existing project Conda environment. Exercise pure helpers with fake tokenizer/engine/stream fixtures before any remote or NPU operation.

**Tech Stack:** Existing project Conda Python, pytest, TokenSpeed `Engine.generate(stream=True)`, and Python standard library (`asyncio`, `csv`, `json`, `math`, `statistics`, `time`, `xml.etree.ElementTree`). Do not install dependencies.

---

## Files and responsibilities

- Add `test/course_project/test_session_kv_ttft.py`: fake chat-template tokenizer, fake streaming engine, fixed raw-run fixtures, and focused tests for transcript fidelity, TTFT, checkpointing, validation, CSV, and SVG.
- Add `scripts/run_session_kv_ttft.py`: pinned config loading, transcript/token helpers, full-stream TTFT measurement, serial four-repeat sweep, project-local raw paths, hardware snapshots, atomic checkpoints, and CLI.
- Add `benchmarks/course_project/session_kv_ttft_report.py`: validate completed runs and write one CSV row per matched turn pair plus a deterministic SVG containing all observations and per-turn medians.
- Add `scripts/run_session_kv_ttft.sh`: follow `scripts/run_session_kv_perf.sh` and activate only the existing project Conda environment.
- Update `docs/superpowers/specs/2026-10-10-session-kv-multiturn-ttft-design.md`, `task_plan.md`, and `notes.md` only to keep their experiment status synchronized with implementation and verification.
- Raw artifacts go only under `results/raw/session_kv_ttft/`; a successful run's unique derived directory is `results/analysis/session_kv_ttft_<run_id>/`. Never overwrite prior run artifacts.

## Task 1: Add failing pure-helper tests

**Files:**
- Add: `test/course_project/test_session_kv_ttft.py`

- [x] Add a module loader for `scripts/run_session_kv_ttft.py` and a tiny deterministic tokenizer fixture based on `_QwenTemplateTokenizer` in `test/course_project/test_session_kv_perf.py`.
- [x] Add a transcript test that creates the first 512-token-target prompt, then appends 15 assistant outputs and the fixed follow-up. For each of the 15 continuations, assert `full_ids == prior_history_ids + suffix_ids`; assert the final request represents 16 user turns and every prior assistant response.
- [x] Add an EOS test proving the full-history helper strips only a terminal tokenizer EOS before decoding assistant content, decodes with `skip_special_tokens=False`, and uses the approved `<think>\n` prefix.
- [x] Add a mismatch test that makes the tokenizer alter an earlier message and expects the helper to reject a non-prefix transcript rather than send an unsafe Session suffix.
- [x] Run the focused test command and confirm it fails because the runner module does not exist yet.

```powershell
& ..\.conda-env\python.exe -m pytest test/course_project/test_session_kv_ttft.py -q
```

Expected before implementation: test collection fails at the missing runner module. Do not treat that initial failure as a completed verification.

## Task 2: Implement transcript construction and streaming TTFT helper

**Files:**
- Add: `scripts/run_session_kv_ttft.py`
- Modify: `test/course_project/test_session_kv_ttft.py`

- [x] Implement constants from the approved spec by reusing the values in `scripts/run_session_kv_perf.py`: `SYSTEM_PROMPT`, `NEXT_USER_TEXT`, 512-token initial target, 8-token output cap, 4096-token model limit, and `current-frontier` policy.
- [x] Implement `assistant_content_from_output_ids(tokenizer, output_ids)` using the approved terminal-EOS/decode/`<think>\n` rule, then `build_multiturn_inputs(tokenizer, messages, history_ids)` to chat-template the complete current transcript with `add_generation_prompt=True`, normalize list/tensor token output, compare the prefix with `history_ids`, and return only the suffix plus full IDs.
- [x] Implement `measure_stream_ttft(engine, input_ids, sampling_params, session_id)` around the blocking `Engine.generate(..., stream=True)` iterator. Start `perf_counter()` immediately before the call; set TTFT at the first non-empty `output_ids`; continue to iterator exhaustion; concatenate all delta IDs; retain final `meta_info`; reject empty output, malformed chunks, or missing final Session metadata.
- [x] Add fake-stream tests with an empty chunk before the first token, multiple later chunks, and a fake clock. Assert TTFT stops at the first non-empty chunk, all chunks are drained, and output IDs are concatenated exactly once.
- [x] Add tests asserting session and stateless calls receive the expected suffix/full IDs and the same sampling parameters (`max_new_tokens=8`, `temperature=0`, `stream_interval=1`).
- [x] Run focused tests and require all transcript and stream tests to pass.

```python
started = time.perf_counter()
ttft_ms = None
output_ids = []
final_meta_info = {}
stream = engine.generate(
    input_ids=input_ids,
    sampling_params={**sampling_params, "stream_interval": 1},
    session_id=session_id,
    stream=True,
)
for chunk in stream:
    chunk_ids = chunk.get("output_ids")
    if chunk_ids and ttft_ms is None:
        ttft_ms = (time.perf_counter() - started) * 1000
    output_ids.extend(chunk_ids or [])
    final_meta_info = chunk.get("meta_info", final_meta_info)
```

The implementation must validate chunk/metadata types and finite positive TTFT rather than silently accepting malformed stream data.

## Task 3: Implement the 64-pair serial runner and atomic raw checkpoints

**Files:**
- Modify: `scripts/run_session_kv_ttft.py`
- Modify: `test/course_project/test_session_kv_ttft.py`

- [x] Add a sweep helper for four fresh Sessions, each with exactly 16 turns. Use order Session→stateless on repetitions 1 and 3, stateless→Session on repetitions 2 and 4; never overlap requests.
- [x] Before measured repeats, warm one separate temporary Session and one stateless request using the same 512-token-target input and stream settings. Exclude both warmup requests from the 64 pairs; verify warmup output-ID parity and close the Session in `finally`.
- [x] At turn 1, send the same full initial IDs to both paths. For turns 2–16, send only the validated suffix to Session and the full-history IDs to stateless. After each pair, require exact output-ID equality; use the Session result to extend the shared transcript and cache history.
- [x] Validate each Session's `current-frontier` metadata against prior cached history and suffix lengths. Before every request, require full input length plus the 8-token generation reserve to be at most the pinned 4096-token context limit.
- [x] Checkpoint only after a complete valid pair, using write-to-`.tmp` then atomic replace. On the first failed pair, preserve all earlier pairs, store `status="failed"` and a concise failure type/message, stop the sweep, and close the active Session. Do not represent a partial pair as a completed observation.
- [x] Record per pair: repetition, turn, execution order, full/session input IDs and lengths, both output ID arrays and counts, both TTFT values, parity, full context length, Session metadata, and pair reduction `100 * (1 - session_ttft_ms / stateless_ttft_ms)`.
- [x] Record pinned model/revision, config and launch arguments, runtime offline variables, `ASCEND_RT_VISIBLE_DEVICES=0`, `stream_interval=1`, concurrency 1, frontier policy, UTC timestamps, and before/after NPU0 snapshots. Disable generic prefix caching and KVStore; retain the configured overlap schedule.
- [x] Add fake-engine tests for all 64 calls, repeat order balance, fresh/closed Sessions, strictly serial execution, metadata correctness, mismatched IDs stopping at the first pair, and atomic checkpoints surviving a later failure.
- [x] Run focused tests and require all runner tests to pass without importing or initializing NPU libraries in unit tests.

## Task 4: Add strict CSV/SVG generation

**Files:**
- Add: `benchmarks/course_project/session_kv_ttft_report.py`
- Modify: `test/course_project/test_session_kv_ttft.py`
- Modify: `scripts/run_session_kv_ttft.py`

- [x] Implement a report function that accepts a raw JSON path and a fresh analysis directory. Refuse non-completed runs, anything other than 64 pairs, mismatched IDs, missing/non-finite/non-positive TTFT, invalid metadata, or a stateless denominator of zero.
- [x] Write `session_kv_multiturn_ttft.csv` with exactly 64 data rows and columns for repetition, turn, full-context tokens, Session suffix tokens, Session/stateless TTFT, reused/newly-prefilled counts, output-ID parity, pair reduction percentage, and each path's per-turn median/minimum/maximum TTFT.
- [x] Write `session_kv_multiturn_ttft.svg` with standard-library XML generation. Plot all four observations per path, per-turn median lines, and min-to-max range markers; preserve every observed value and label the chart as native Engine TTFT, not HTTP TTFT.
- [x] Call report generation only after raw status becomes `completed`; failed or partial runs retain raw JSON but produce no formal CSV/SVG.
- [x] Test a synthetic 64-pair fixture: CSV has exactly 64 data rows, pair reductions and per-turn medians/ranges match the fixture, SVG parses with `ElementTree`, and all 16 turns, both median series, and range markers are present.
- [x] Test incomplete, mismatched, invalid-metadata, and non-finite fixtures; each must raise a specific validation error and leave no formal report files.
- [x] Run focused report tests and require all to pass.

## Task 5: Add CLI and project-local environment entry point

**Files:**
- Modify: `scripts/run_session_kv_ttft.py`
- Add: `scripts/run_session_kv_ttft.sh`
- Modify: `test/course_project/test_session_kv_ttft.py`
- Verify integration with: `benchmarks/course_project/session_kv_ttft_report.py`

- [x] **Step 1: Add regression tests for the three review findings.** In `test/course_project/test_session_kv_ttft.py`, update `test_run_paths_skip_existing_raw_and_analysis_names` so an occupied analysis path is `results/analysis/session_kv_ttft_occupied` and the fresh result is `results/analysis/session_kv_ttft_fresh`. Add a real-generator integration test using `_synthetic_completed_run()`; write its JSON to `tmp_path`, call `_generate_report_if_completed` with status `completed`, and assert the CSV has 64 rows and the SVG parses with `ElementTree`. Keep the existing failed-status assertion that no analysis directory is created. Add a failure-path lifecycle test that monkeypatches `REPO_ROOT`, `_capture_hardware_snapshot`, `_set_offline_environment`, and `load_model_config` so `main()` fails before importing Transformers or Engine; read the raw JSON and assert `stream_interval == 1`, `concurrency == 1`, UTC-offset-zero `started_at_utc` / `finished_at_utc`, and finish not earlier than start.

```python
analysis_root = project_root / "results" / "analysis"
(analysis_root / "session_kv_ttft_occupied").mkdir()
assert analysis_dir == analysis_root / "session_kv_ttft_fresh"
```

```python
def test_completed_run_generates_report_with_real_generator(
    ttft_runner, tmp_path
):
    raw_path = tmp_path / "run.json"
    raw_path.write_text(
        json.dumps(_synthetic_completed_run()), encoding="utf-8"
    )
    analysis_dir = tmp_path / "session_kv_ttft_synthetic-run"

    csv_path, svg_path = ttft_runner._generate_report_if_completed(
        {"status": "completed"}, raw_path, analysis_dir
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 64
    ElementTree.parse(svg_path)
```

Add `from datetime import datetime, timedelta` to the test imports, then use this failure-path test to inspect the final checkpoint without loading a model or touching an NPU:

```python
def test_main_writes_provenance_and_finish_timestamp_on_failure(
    ttft_runner, tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    output_dir = project_root / "results" / "raw" / "session_kv_ttft"
    monkeypatch.setattr(ttft_runner, "REPO_ROOT", project_root)
    monkeypatch.setattr(
        ttft_runner,
        "_capture_hardware_snapshot",
        lambda *_: {"available": False, "path": None},
    )
    monkeypatch.setattr(ttft_runner, "_set_offline_environment", lambda: None)

    def fail_before_model_loading(_):
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(ttft_runner, "load_model_config", fail_before_model_loading)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_session_kv_ttft.py", "--output-dir", "results/raw/session_kv_ttft"],
    )

    with pytest.raises(RuntimeError, match="stop before model loading"):
        ttft_runner.main()

    raw_path = next(output_dir.glob("qwen35-4b-session-kv-ttft-*.json"))
    run_record = json.loads(raw_path.read_text(encoding="utf-8"))
    assert run_record["stream_interval"] == 1
    assert run_record["concurrency"] == 1
    started = datetime.fromisoformat(run_record["started_at_utc"])
    finished = datetime.fromisoformat(run_record["finished_at_utc"])
    assert started.utcoffset() == timedelta(0)
    assert finished.utcoffset() == timedelta(0)
    assert finished >= started
```

- [x] **Step 2: Run the new tests and verify they expose the defects.**

```powershell
& ..\.conda-env\python.exe -m pytest test/course_project/test_session_kv_ttft.py::test_run_paths_skip_existing_raw_and_analysis_names test/course_project/test_session_kv_ttft.py::test_completed_run_generates_report_with_real_generator test/course_project/test_session_kv_ttft.py::test_main_writes_provenance_and_finish_timestamp_on_failure -q
```

Expected before implementation: the path assertion reports the nested directory mismatch, the integration test raises `FileExistsError` because the runner pre-creates the analysis directory, and the lifecycle test reports missing provenance fields. These tests must use only temporary files and monkeypatched hardware/runtime hooks.

- [x] **Step 3: Correct unique project-local output paths and report handoff.** In `scripts/run_session_kv_ttft.py`, make these minimal changes:

```python
DEFAULT_ANALYSIS_DIR = Path("results/analysis")
analysis_dir = analysis_root / f"session_kv_ttft_{run_id}"

if run_record.get("status") != "completed":
    return None
from benchmarks.course_project.session_kv_ttft_report import generate_report
return generate_report(raw_path, analysis_dir)
```

Leave the analysis directory absent until the report generator creates and atomically publishes it. Failed or partial runs must continue to return `None` without creating an analysis directory.

- [x] **Step 4: Record the complete run provenance.** Add these fields when constructing the initial raw record and in `main()`'s existing `finally` block:

```python
"started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
"stream_interval": 1,
"concurrency": 1,
```

```python
finally:
    run_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )
    run_record["hardware_after"] = _capture_hardware_snapshot(
        output_dir, run_id, "after"
    )
    _write_run_record(output_path, run_record)
```

Preserve the already-recorded pinned model/revision, Engine configuration and CLI args, offline environment, frontier policy, and NPU0 before/after snapshot fields.

- [x] **Step 5: Verify the focused CLI and experiment tests.**

```powershell
& ..\.conda-env\python.exe scripts/run_session_kv_ttft.py --help
& ..\.conda-env\python.exe -m pytest test/course_project/test_session_kv_ttft.py -q
```

Expected: help documents the project-local `--output-dir`; all focused tests pass, including real CSV/SVG generation through the runner hook, exact analysis path naming, and failure-path provenance. No packages are installed and no NPU/runtime is initialized.

## Task 6: Verify locally, on the remote project environment, then measure only NPU0

**Files:**
- Verify: `scripts/run_session_kv_ttft.py`
- Verify: `benchmarks/course_project/session_kv_ttft_report.py`
- Verify: `test/course_project/test_session_kv_ttft.py`
- Generated only after a successful authorized run: `results/raw/session_kv_ttft/` and `results/analysis/session_kv_ttft_<run_id>/`

- [x] Run the focused TTFT tests and complete `test/course_project` with `& ..\.conda-env\python.exe`; do not install or upgrade dependencies. Resolve any new failures before proceeding.
- [x] Sync only the four implementation/test files to the already configured remote project checkout, preserving unrelated remote files; run focused TTFT tests there with the existing project Conda environment. Do not copy model weights or recreate environments. All four remote hashes matched the local files; remote focused tests: 46 passed.
- [x] Perform the approved read-only preflight on NPU0 only. NPU0 process-memory query reported no device processes; utilization was 0% on both chips. Ran one complete four-repeat experiment only on NPU0; did not query or use NPU1–3 or start another workload alongside it.
- [x] After the run, verify raw status is completed, there are exactly 64 pairs/16 turns per repetition, exact output parity holds for every pair, all Session metadata and context limits pass, CSV has 64 data rows, and SVG parses and contains both path medians and all observations. The strict report validator passed; maximum full context was 946 tokens and the SVG contains 128 observation points.
- [x] Stop/close Engine through the runner lifecycle, then check only NPU0 for remaining project processes and capture the after snapshot. Post-run NPU0 process-memory query reported no device processes.
- [x] Copy the raw JSON, NPU0 snapshots, CSV, and SVG into the matching local project paths; all five source/destination SHA-256 pairs matched. Updated the spec, `task_plan.md`, and `notes.md` with measured status and evidence paths without claiming 910C, HTTP TTFT, concurrency results, or statistical significance.
- [x] Final acceptance: all checks above passed. No failed or partial run was represented as complete.

## Final self-review checklist

- The runner keeps each conversation at 16 turns and uses exactly four matched repetitions (64 pairs).
- Stateless input is the full serialized history; Session input is only a verified token suffix.
- TTFT ends at the first non-empty token-ID chunk, while the iterator is still drained fully.
- Output parity, `current-frontier` metadata, context bounds, unique project-local outputs, and NPU0-only scope are enforced before reporting success.
- No step installs packages, queries other NPUs, stages/commits files, or deletes/overwrites existing project artifacts.
