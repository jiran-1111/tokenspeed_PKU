# Session KV Paired Latency Report Design

## Goal

Make the existing serial Session KV on/off sweep reproducibly available from the course report generator, without mixing request-level measurements into the ordinary run-summary CSV or overstating the evidence.

## Input and validation

- Read raw files matching `*session-kv-perf-*.json` beneath the report generator's `--raw-dir`.
- Accept only files with `status="completed"`, `model_id="Qwen/Qwen3.5-4B"`, `launch_configuration.model_config.tp_size=1`, `frontier_policy="current-frontier"`, `generic_prefix_caching=false`, `kvstore_disabled=true`, and `measurement_scope="serial Engine.generate request latency; not TTFT"`. A matching file that fails any required check is an error naming its path and reason; do not silently skip it.
- Require every measurement pair to include positive finite `session` and `stateless` `request_latency_ms` values and `output_ids_match=true`. Reject an invalid candidate file with its path and reason rather than silently plotting incomplete or incorrect measurements.
- Preserve every pair across files. The current data contains 12 pairs: target contexts 512, 1024 and 1536, with four repetitions per target. Actual initial context length, run ID, repetition and execution order remain available as provenance.

## Outputs and metric

- Generate a separate `session_kv_paired_reduction.csv` with one row per validated pair. Include `run_id`, `model_id`, `model_revision`, `frontier_policy`, `context_length_target`, `initial_context_tokens`, `repetition`, `execution_order`, `session_request_latency_ms`, `stateless_request_latency_ms`, `paired_reduction_percent`, and `measurement_scope`. Do not add these raw pairs to `summary.csv`.
- Generate `session_kv_paired_reduction.svg`: x-axis is target context length shown as equally spaced context groups; y-axis is paired whole-request latency reduction in percent, computed per pair as `100 * (1 - session_latency_ms / stateless_latency_ms)`. Show every repetition as a dot, ordered by `(run_id, repetition)` and spread evenly within ±0.12 group-spacing around its context group's x-position; show the per-context median as a distinct diamond marker. Do not add confidence intervals or significance claims.
- Label the scope as serial `Engine.generate` wall time, not TTFT. Include all valid pairs, including the recorded 1536-context stateless outlier; do not trim observations.
- If no matching Session KV raw files exist, generate no Session KV sidecars. Existing raw files are read-only and are never rewritten.

## Integration

Keep ordinary summary loading and plots unchanged. Add a focused loader/report helper in `benchmarks/report.py`; have `main()` call it with the selected raw and output directories, writing sidecars into that output directory. The normal `scripts/generate_report.sh` path should therefore reproduce the new chart when the Session KV raw file is present. If the raw directory has no matching files, the helper returns without creating either sidecar.

## Verification

- Add a temporary raw-JSON fixture with multiple contexts and paired repetitions; assert one CSV row per pair, the exact paired-reduction formula, provenance fields, chart labels, all points and median markers.
- Verify a mismatched-output or invalid-latency pair is rejected and that a raw directory without matching files produces no Session KV sidecars.
- Run focused report tests, the full `test/course_project` suite, and the report generator against the existing project raw data into a new output directory. Confirm the ordinary report files still generate and the new CSV/SVG contain all 12 source pairs.
- Validate the SVG parses and inspect the CSV/SVG values against the existing raw JSON. Do not run another NPU workload.

## Scope limits

This adds a descriptive visualization only. It does not change Session KV runtime behavior, the benchmark protocol, performance conclusions, or the pending 16-turn TTFT experiment. It does not make a claim about concurrency, other models, or Ascend 910C.
