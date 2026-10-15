# KV Cache Usage Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible KV cache usage chart from comparable baseline summaries without presenting scheduler page activity as HBM occupancy.

**Architecture:** Preserve the existing `kv_cache.svg` block-size/throughput sweep. Add `kv_cache_usage.svg` from baseline rows only, restricted to TP=1 and 128/32 target input/output; use peak scheduler active-page ratio and the existing median-by-x plot path. Label the ratio explicitly as not HBM occupancy.

**Tech Stack:** Python standard library, existing report helpers, project Conda pytest environment.

---

## Files and responsibilities

- Modify `test/course_project/test_report.py` with one synthetic report test that checks selection, model separation, medians, and labels.
- Modify `benchmarks/report.py` to filter the chosen comparable cohort and emit the additional SVG while retaining the block-size chart.
- Modify `docs/project/experiments.md` to document the new chart's data scope and caveat.

## Task 1: Add and verify the KV usage report

**Files:**
- Modify: `test/course_project/test_report.py`
- Modify: `benchmarks/report.py`
- Modify: `docs/project/experiments.md`

- [x] **Step 1: Add a failing test.** Add this test to `TestCourseReport`; its ordinary baseline inputs are two models with two repetitions at each concurrency, and decoys for wrong input/output length, TP size, unavailable metrics, prefix-cache hits, and Agent family must not affect the plotted medians.

```python
def test_kv_cache_usage_plot_uses_comparable_baselines_and_medians(self) -> None:
    records = []
    values = {
        "qwen35-0.8b": ((0.2, 0.4), (0.6, 0.8)),
        "qwen35-4b": ((0.1, 0.1), (0.9, 0.9)),
    }
    for model, concurrency_values in values.items():
        for concurrency, repeats in zip((1, 4), concurrency_values):
            for ratio in repeats:
                records.append({
                    "result_family": "baseline",
                    "served_model_name": model,
                    "server_metrics_available": True,
                    "prefix_cache_hit_ratio": 0.0,
                    "concurrency": concurrency,
                    "target_input_tokens": 128,
                    "target_output_tokens": 32,
                    "tp_size": 1,
                    "kv_cache_usage_ratio_peak": ratio,
                })
    decoy = dict(records[0])
    records.extend([
        {**decoy, "result_family": "agent", "kv_cache_usage_ratio_peak": 0.99},
        {**decoy, "target_input_tokens": 1024, "kv_cache_usage_ratio_peak": 0.99},
        {**decoy, "target_output_tokens": 64, "kv_cache_usage_ratio_peak": 0.99},
        {**decoy, "tp_size": 2, "kv_cache_usage_ratio_peak": 0.99},
        {**decoy, "server_metrics_available": False, "kv_cache_usage_ratio_peak": 0.99},
        {**decoy, "prefix_cache_hit_ratio": 0.5, "kv_cache_usage_ratio_peak": 0.99},
    ])

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        generate_standard_reports(records, output)
        svg = (output / "kv_cache_usage.svg").read_text(encoding="utf-8")

    polylines = re.findall(r'<polyline points="([^"]+)"', svg)
    self.assertEqual(len(polylines), 2)
    self.assertEqual(polylines, ["90.00,365.00 860.00,165.00", "90.00,465.00 860.00,65.00"])
    self.assertIn("Scheduler KV Active-Page Ratio vs Concurrency", svg)
    self.assertIn("Concurrency", svg)
    self.assertIn("Peak active pages / total pages ratio (not HBM occupancy)", svg)
```

- [x] **Step 2: Confirm the test fails for the missing report.** Run `& ..\.conda-env\python.exe -m pytest test/course_project/test_report.py::TestCourseReport::test_kv_cache_usage_plot_uses_comparable_baselines_and_medians -q`. Expected: failure because `kv_cache_usage.svg` is not generated yet.
- [x] **Step 3: Implement only the missing report path.** In `generate_standard_reports`, derive the subset from `baseline` with:

```python
kv_cache_usage = [
    record for record in baseline
    if record.get("target_input_tokens") == 128
    and record.get("target_output_tokens") == 32
    and record.get("tp_size") == 1
]
```

Add a `kv_cache_usage.svg` spec using `concurrency`, `kv_cache_usage_ratio_peak`, no extra grouping keys, title `Scheduler KV Active-Page Ratio vs Concurrency`, and y-axis label `Peak active pages / total pages ratio (not HBM occupancy)`. Keep the existing `kv_cache.svg` block-size spec unchanged.
- [x] **Step 4: Run focused and full tests.** Repeat the focused command, then run `& ..\.conda-env\python.exe -m pytest test/course_project -q`. Expected: focused test passes and the suite has no new failures.
- [x] **Step 5: Verify the real report.** Generate into a fresh timestamped folder without overwriting existing output:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = Join-Path (Get-Location) ("results/plots/kv_cache_usage_" + $stamp)
if (Test-Path -LiteralPath $outDir) { throw "Output directory already exists: $outDir" }
& ..\.conda-env\python.exe benchmarks/report.py --raw-dir results/raw --output-dir $outDir
```

Parse the new SVG; confirm three model series and four concurrency points per series. Do not overwrite existing plot folders or modify raw data.
- [x] **Step 6: Document the evidence boundary.** State in `docs/project/experiments.md` that this chart is restricted to baseline TP=1, 128/32, reports peak scheduler active-page ratio, and is not HBM occupancy or a KV block-size sweep.

## Scope limits

- Do not change runtime telemetry, add dependencies, or run any NPU workload.
- Keep the existing `kv_cache.svg` block-size chart semantics unchanged.
- Treat chart values as descriptive summaries only; do not infer memory bytes or capacity exhaustion.
