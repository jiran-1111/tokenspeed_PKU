# Level1 Ascend Profiler Plan

## Goal

Enable an opt-in Torch-NPU Level1 capture with PipeUtilization metrics without changing the default Level0 behavior, then repeat the 0.8B eight-turn 50%-shared-prefix anomaly workload under profiling and verify the exported profiler configuration and hardware-counter artifacts.

## Constraints

- Keep all project code, test outputs, downloaded files, and profiler artifacts inside the repository workspace.
- The public `/start_profile` JSON is proxied to an external SMG/gRPC API whose installed `StartProfileRequest` has no profiler-level field; do not rely on unknown JSON fields.
- Select the level when launching the service via `TOKENSPEED_NPU_PROFILER_LEVEL`; retain Level0 when unset.
- Preserve the historical anomaly parameters: Qwen3.5-0.8B, prefix length 512, 8 turns, concurrency 1 and 4, shared-prefix ratio 0.5, 32 output tokens, 100 ms client-side wait, 3 repetitions.
- Treat Level1 profiler runs as diagnostic captures, not unprofiled performance comparisons.

## Steps and verification

1. Add adapter tests for Level0 compatibility, Level1 + PipeUtilization configuration, and unsupported configuration failure; then implement the NPU experimental-config mapping. Verify focused unit tests.
2. Add a service-launch option that sets the profiler-level environment variable and test that it reaches the child process without changing unrelated environment values.
3. Add an opt-in profiler wrapper around each Agent benchmark repetition; verify request/profile IDs and guaranteed stop behavior with mocks, and preserve existing benchmark defaults.
4. Sync only the changed project files to the remote workspace, run focused tests, restart the currently stopped 0.8B service with Level1, and capture the matched anomaly workload for all three repetitions at both concurrency levels.
5. Verify each raw profiler metadata file reports Level1 and PipeUtilization, confirm AI Core counter outputs exist and contain rows, then update the profiler analysis, task plan, and reproducibility docs. If the installed CANN parser emits no usable counter artifact, report that limitation without inferring an AICore bottleneck.
