# Session KV Paired Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing report CLI to emit a provenance-preserving CSV and SVG for every valid serial Session KV on/off pair found under `--raw-dir`.

**Architecture:** Keep `load_summary_records` and `generate_standard_reports` behavior unchanged. Add a focused Session KV raw loader and sidecar writer in `benchmarks/report.py`, then call the helper from `main()` with the CLI raw/output directories. Tests drive the public CLI entry point with temporary raw data; no benchmark or NPU execution is involved.

**Tech Stack:** Python standard library (`json`, `csv`, `math`, `html`, `statistics`, `pathlib`), existing unittest/pytest suite, project Conda environment.

---

## Files and responsibilities

- Modify `test/course_project/test_report.py`: temporary raw fixtures, CLI-level output assertions, validation failures, and no-match behavior.
- Modify `benchmarks/report.py`: strict input validation, CSV serialization, deterministic SVG generation, and `main()` integration.
- Generate only in a fresh output folder under `results/plots/`; never rewrite raw JSON or existing report folders.

## Task 1: Add failing CLI-level tests

**Files:**
- Modify: `test/course_project/test_report.py`

- [x] Add `csv`, `sys`, `unittest.mock.patch`, and `xml.etree.ElementTree` imports; load the report CLI in the test helper.
- [x] Add fixture builders that write ordinary `*.summary.jsonl` records plus Session KV JSON using the inspected raw schema. Each synthetic raw file has `status="completed"`, `model_id="Qwen/Qwen3.5-4B"`, `model_revision`, `frontier_policy="current-frontier"`, `generic_prefix_caching=false`, `kvstore_disabled=true`, `measurement_scope="serial Engine.generate request latency; not TTFT"`, and `launch_configuration.model_config.tp_size=1`.
- [x] Add deterministic multi-file fixtures and expected medians without exposing input token IDs.

```python
def _session_pair(
    target: int,
    repetition: int,
    session_ms: float,
    stateless_ms: float,
    *,
    output_ids_match: bool = True,
) -> dict:
    return {
        "context_length_target": target,
        "initial_context_tokens": target + 29,
        "repetition": repetition,
        "execution_order": ["session", "stateless"],
        "measurement": {
            "session": {"request_latency_ms": session_ms},
            "stateless": {"request_latency_ms": stateless_ms},
            "output_ids_match": output_ids_match,
        },
    }


def _write_session_kv_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    (raw_dir / "baseline.summary.jsonl").write_text(
        json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
    )
    common = {
        "status": "completed",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "fixture-revision",
        "frontier_policy": "current-frontier",
        "generic_prefix_caching": False,
        "kvstore_disabled": True,
        "measurement_scope": "serial Engine.generate request latency; not TTFT",
        "launch_configuration": {"model_config": {"tp_size": 1}},
    }
    runs = (
        (
            "run-a",
            [
                _session_pair(512, 1, 80.0, 100.0),
                _session_pair(512, 2, 70.0, 100.0),
                _session_pair(1024, 1, 90.0, 100.0),
            ],
        ),
        (
            "run-b",
            [
                _session_pair(1024, 2, 60.0, 100.0),
                _session_pair(1024, 3, 50.0, 100.0),
            ],
        ),
    )
    for run_id, measurements in runs:
        payload = {**common, "run_id": run_id, "measurements": measurements}
        (raw_dir / f"{run_id}-session-kv-perf-fixture.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
```
- [x] Add a CLI helper and success test. Before implementation, it failed because the CLI did not create the Session KV CSV.

```python
def _run_report(raw_dir: Path, output_dir: Path) -> None:
    with patch.object(
        sys,
        "argv",
        ["report.py", "--raw-dir", str(raw_dir), "--output-dir", str(output_dir)],
    ):
        report.main()


def test_session_kv_cli_writes_all_pairs_and_median_markers(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        raw_dir = Path(temporary) / "raw"
        output_dir = Path(temporary) / "plots"
        _write_session_kv_fixture(raw_dir)

        _run_report(raw_dir, output_dir)

        csv_path = output_dir / "session_kv_paired_reduction.csv"
        svg_path = output_dir / "session_kv_paired_reduction.svg"
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        root = ElementTree.parse(svg_path).getroot()

    self.assertEqual(len(rows), 5)
    self.assertEqual(
        [float(row["paired_reduction_percent"]) for row in rows],
        [20.0, 30.0, 10.0, 40.0, 50.0],
    )
    self.assertEqual(rows[0]["model_revision"], "fixture-revision")
    self.assertEqual(rows[0]["initial_context_tokens"], "541")
    self.assertEqual(rows[0]["execution_order"], '["session","stateless"]')
    self.assertNotIn("initial_input_ids", rows[0])
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    self.assertEqual(len(root.findall(".//svg:circle[@class='pair']", namespace)), 5)
    medians = root.findall(".//svg:polygon[@class='median']", namespace)
    self.assertEqual(len(medians), 2)
    self.assertEqual(
        [float(node.attrib["data-median-percent"]) for node in medians],
        [25.0, 40.0],
    )
    visible_text = " ".join(root.itertext())
    self.assertIn("Target context length", visible_text)
    self.assertIn("not TTFT", visible_text)
```

- [x] Add validation and no-match tests. Before implementation, the success and invalid-input tests failed for the missing behavior; the no-match test passed.

```python
def test_session_kv_cli_rejects_output_mismatch_with_path(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        raw_dir = Path(temporary) / "raw"
        output_dir = Path(temporary) / "plots"
        _write_session_kv_fixture(raw_dir)
        candidate = raw_dir / "run-a-session-kv-perf-fixture.json"
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload["measurements"][0]["measurement"]["output_ids_match"] = False
        candidate.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "run-a-session-kv-perf-fixture.json.*output_ids_match"):
            _run_report(raw_dir, output_dir)


def test_session_kv_cli_rejects_non_finite_latency_with_path(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        raw_dir = Path(temporary) / "raw"
        output_dir = Path(temporary) / "plots"
        _write_session_kv_fixture(raw_dir)
        candidate = raw_dir / "run-a-session-kv-perf-fixture.json"
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload["measurements"][0]["measurement"]["session"]["request_latency_ms"] = float("nan")
        candidate.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "run-a-session-kv-perf-fixture.json.*request_latency_ms"):
            _run_report(raw_dir, output_dir)


def test_session_kv_cli_without_matching_raw_files_writes_no_sidecars(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        raw_dir = Path(temporary) / "raw"
        output_dir = Path(temporary) / "plots"
        raw_dir.mkdir()
        (raw_dir / "baseline.summary.jsonl").write_text(
            json.dumps({"request_count": 1}) + "\n", encoding="utf-8"
        )

        _run_report(raw_dir, output_dir)

        self.assertTrue((output_dir / "summary.csv").exists())
        self.assertFalse((output_dir / "session_kv_paired_reduction.csv").exists())
        self.assertFalse((output_dir / "session_kv_paired_reduction.svg").exists())


def test_session_kv_cli_rejects_incompatible_run_metadata(self) -> None:
    mutations = (
        (("status",), "failed", "status"),
        (("model_id",), "other-model", "model_id"),
        (("frontier_policy",), "legacy", "frontier_policy"),
        (("generic_prefix_caching",), True, "generic_prefix_caching"),
        (("kvstore_disabled",), False, "kvstore_disabled"),
        (("measurement_scope",), "TTFT", "measurement_scope"),
        (("launch_configuration", "model_config", "tp_size"), 2, "tp_size"),
    )
    for key_path, value, field in mutations:
        with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            _write_session_kv_fixture(raw_dir)
            candidate = raw_dir / "run-a-session-kv-perf-fixture.json"
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            target = payload
            for key in key_path[:-1]:
                target = target[key]
            target[key_path[-1]] = value
            candidate.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, rf"run-a-session-kv-perf-fixture.json.*{field}"):
                _run_report(raw_dir, root / "plots")
```

Run: `& ..\.conda-env\python.exe -m pytest test/course_project/test_report.py -q`

Expected before implementation: the new CLI success test fails at the missing CSV assertion; the invalid-input tests fail because no `ValueError` is raised. Existing tests remain green.

## Task 2: Implement the paired raw loader and sidecars

**Files:**
- Modify: `benchmarks/report.py`

- [x] Add a fixed CSV field tuple in the required order.
- [x] Add `load_session_kv_pairs(raw_dir: Path) -> list[dict]` with strict metadata/pair validation, path-and-field errors, complete pair preservation, and `(context_length_target, run_id, repetition)` ordering.
- [x] Add `generate_session_kv_paired_report(raw_dir: Path, output_dir: Path) -> list[Path]`; no matching files returns without creating sidecars, and all candidate files are validated before Session KV outputs are opened. CSV uses compact JSON `execution_order`, the paired formula, and no token-ID arrays.
- [x] Add the dedicated SVG writer with equally spaced context groups, deterministic ±0.12 group-spacing jitter, all pair circles, per-context median diamonds, required labels/warning, escaped dynamic content, and no external plotting dependency.
- [x] Integrate the helper in `main()` after standard reports without changing the Session KV path through ordinary summary loading/writing.
- [x] Re-run the focused report tests; all 14 tests pass.

The metric and plotting core must remain equivalent to:

```python
paired_reduction_percent = 100.0 * (
    1.0 - session_request_latency_ms / stateless_request_latency_ms
)
```

Run: `& ..\.conda-env\python.exe -m pytest test/course_project/test_report.py -q`

Expected: all report tests pass; fixture has five CSV rows, five SVG pair circles, and two median diamonds.

## Task 3: Verify the checked-in raw data end to end

**Files:**
- No source files; outputs go to a new directory under `results/plots/`.

- [x] Confirm `results/plots/session_kv_paired_20261010/` did not exist before generation.
- [x] Run the report CLI against `results/raw` into the fresh output directory.
- [x] Parse the CSV/SVG and verify 12 rows (four each at 512/1024/1536), raw-matching finite reductions, the exact 1536 stateless `6463.689420023002 ms` outlier, 12 SVG pair circles, three median diamonds, valid SVG XML, and the scope warning.
- [x] Confirm `summary.csv` and ordinary summary-derived SVGs are present; Session KV fields were not appended to `summary.csv`.
- [x] Run the complete course suite: 177 passed, 1 skipped (fresh post-confirmation run).
- [x] Run `git diff --check` and inspect the relevant source/tests/spec/plan. No files were staged or committed.

## Scope and safety

- This is a descriptive serial request-latency report, not TTFT, concurrent throughput, significance testing, or proof of general performance gains.
- Preserve every valid pair and retain the measured outlier; never edit raw JSON.
- Do not run an NPU workload or access NPU IDs beyond the existing NPU0 authorization.
- Do not install packages, create files outside the workspace, or modify unrelated report outputs.

## Self-review coverage

- Input selection, run/model/TP/frontier/cache validation, output parity, positive finite latency, and path-specific errors are covered by Task 1 tests and Task 2 loader.
- One CSV row per pair, complete provenance, paired formula, every SVG point, per-context medians, deterministic jitter, exact scope labels, and outlier preservation are covered by Task 1 assertions, Task 2 serializers, and Task 3 inspection of all 12 source pairs.
- No matching raw input and no contamination of `summary.csv` are covered by Task 1 and Task 3 respectively.
- Existing ordinary summary loading and plots remain untouched; the full course suite and `git diff --check` provide regression verification in Task 3.
