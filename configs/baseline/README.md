# E0 baseline configurations

These files are static, unmeasured E0 templates derived from their matching
`configs/bringup/` configurations with `build_ablation_configs`. They preserve
the bring-up model and runtime settings, omit only the bring-up `status` field,
and set `chunked_prefill_size` to `max_model_len`, `prefix_caching` to `false`,
`disable_kvstore` to `true`, `mixed_batch` to `false`, and `overlap_schedule` to
`false`.

Measured default-config baseline results for Qwen3.5-0.8B, 4B, and 9B are
documented in [`docs/project/baseline.md`](../../docs/project/baseline.md), with
their raw summaries in the [0.8B](../../results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/),
[4B](../../results/raw/baseline/qwen35_4b_20261009_full/), and
[9B](../../results/raw/baseline/qwen35_9b_20261009_full/) result directories.
Those default-config measurements are separate from these E0 templates, which
remain unmeasured and do not imply E0 results.

The `status: bringup_unmeasured` field in bring-up configs is non-runtime
metadata. `load_model_config` parses the field but does not interpret it as a
runtime setting, and it does not describe benchmark run history.

No measured 27B baseline exists. The `qwen35_27b.yaml` configuration uses TP=2
and is unvalidated because it has not reached service readiness.
