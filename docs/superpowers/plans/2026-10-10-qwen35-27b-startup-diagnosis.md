# Qwen3.5-27B Startup Diagnosis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the narrowest evidence-backed explanation for the Qwen3.5-27B TP=2 startup stall, or document what remains unknown, without repeating an NPU run until a distinct diagnostic is justified.

**Architecture:** Correlate the preserved 27B engine/CANN trace with the previously successful 4B TP=2 initialization and inspect the existing distributed/HCCL initialization path. Keep the work read-only except for project result/planning documents; do not change runtime code, install packages, modify shared CANN, query NPU IDs 2/3, or launch a retry as part of this plan.

**Tech Stack:** Existing project files and saved logs, PowerShell, SSH read-only inspection, `rg`, and `git diff --check`. No model download or NPU workload is needed for the diagnosis tasks.

---

## Files and responsibilities

- Read `results/raw/bringup/qwen35_27b_20261010T1316Z/engine-startup.log`, `hardware-before.txt`, `hardware-after.txt`, and `cann/` as the immutable record of the failed attempt.
- Read `.tmp/qwen35_4b_tp2_server_20261010T0756.log` and `.tmp/qwen35_4b_tp2_repeat_20261010T0821_server.log` as the same-host TP=2 startup comparison; these runs reached model warmup/API use but did not pass strict token correctness.
- Read `configs/bringup/qwen35_27b.yaml` and `configs/bringup/qwen35_4b_tp2.yaml` to isolate the differences relevant to launch and distributed initialization.
- Inspect, without editing, `python/tokenspeed/runtime/execution/distributed_initializer.py`, `python/tokenspeed/runtime/distributed/process_group_manager.py`, `python/tokenspeed/runtime/distributed/comm_backend/hccl.py`, and `python/tokenspeed/runtime/engine/event_loop.py` to map the logged startup point to the initialization sequence.
- Update only `docs/project/final_results.md`, `docs/report/final_conclusion.md`, `task_plan.md`, and `notes.md` with verified findings and remaining gates.

## Task 1: Preserve and verify the failed-run evidence

**Files:**
- Read: `results/raw/bringup/qwen35_27b_20261010T1316Z/`
- Update: `docs/project/final_results.md`, `task_plan.md`, `notes.md`

- [x] Confirm the server artifact directory contains the engine log, before/after NPU snapshots, and selected CANN plog/device traces. The initial local bundle had 11 files; a missing rank-0 device trace was later retrieved separately without overwriting existing evidence.
- [x] Preserve 12 total evidence files locally, including `cann/device-3339127_20261010131810519.log`.
- [x] Compare remote and local SHA-256 for all 12 files; all 12 match.
- [x] Record only these established facts: both TP ranks loaded weights; image/video encoder warmup completed; the service never became ready; CANN reported a 300-second watchdog timeout and two repeated synchronize timeouts for stream 47/task 1952 with `pendingNum=883`; NPU0 had no process and returned to 4% HBM after teardown.
- [x] State explicitly that the HCCL/collective startup path is implicated by the trace, but its root cause is not yet established; no correctness or performance result was produced.

## Task 2: Compare the failing and known-starting TP=2 initialization paths

**Files:**
- Read: `results/raw/bringup/qwen35_27b_20261010T1316Z/engine-startup.log`
- Read: `.tmp/qwen35_4b_tp2_server_20261010T0756.log`
- Read: `.tmp/qwen35_4b_tp2_repeat_20261010T0821_server.log`
- Read: `configs/bringup/qwen35_27b.yaml`, `configs/bringup/qwen35_4b_tp2.yaml`
- Read: the distributed initialization source files listed above
- Update: `docs/project/final_results.md`, `notes.md`

- [x] Extract a compact ordered timeline from each engine log: distributed init begin, communication-buffer completion, weight load begin/end, model warmup, first HCCL collective, and server readiness. Do not include full command lines or environment dumps in the report.
- [x] Compare the two YAML configs and record only launch-relevant differences (model/revision, TP, visible-device selection, model architecture, KVStore/prefix cache, graph and overlap settings).
- [x] Trace `distributed_initializer.py` group creation order through `process_group_manager.py` to the HCCL backend call. Both ranks created the HCCL group and SIO transport resources. The retained rank-1 device trace repeatedly waits for a rank-0 notification on AllReduce `opLaunchIdx=14`; the retained rank-0 trace has no matching runtime event. This captures asymmetric progress but does not establish why rank 0 did not complete or notify.
- [x] Trace the source sequence after multimodal warmup: `build_device_side` prepares communication runtime, builds attention/KV components and the executor, invokes autotune, then calls graph capture. The recorded config disables autotune/prefill graphs and sets decode graph capture sizes 1/2/4. The engine log ends before expected autotune/capture progress markers. Although graph-capture warmup contains a one-element `dist.barrier()` before dummy forwards and the CANN AllReduce has count 1, the trace has no Python callsite and the engine log does not prove graph capture was reached; this remains a candidate, not an attribution.
- [x] Check whether the 4B TP=2 startup has a corresponding CANN task trace. No matching trace was available for the comparison, so its startup comparison is engine-log-only, not a matched HCCL control.

## Task 3: Write an evidence-bounded diagnosis and retry gate

**Files:**
- Update: `docs/project/final_results.md`, `docs/report/final_conclusion.md`, `task_plan.md`, `notes.md`

- [x] Separate confirmed observations, plausible explanations, and unanswered questions in the final-results update.
- [x] No minimal, reversible diagnostic is established by the current evidence; stop at “root cause undetermined” and do not change TP size, device visibility, collective backend, runtime flags, or shared CANN speculatively.
- [x] Require any later NPU retry to have a distinct diagnostic question, unique log/output paths, a pre-run idle/port check, explicit NPU0-only scope, captured CANN logs, a fixed stop condition, and a post-run no-process check.
- [x] Record the next evidence requirement before any retry: add per-rank markers around post-warmup communication preparation, attention/KV construction, executor creation, autotune, and graph-capture entry/each capture barrier. Do not change collective/backend/config behavior in the same diagnostic, and do not start it automatically.
- [x] Do not run 27B correctness or performance measurements unless startup reaches readiness and the model passes the fixed token-ID correctness gate.
- [x] Run `git diff --check` and confirm the updated report links resolve to the local evidence directory.

## Acceptance criteria

1. The 27B failure evidence remains available locally and all 12 copied files match the remote SHA-256 values.
2. The report says “startup failed before readiness”; it contains no 27B correctness or benchmark claims.
3. Any root-cause statement is directly supported by trace/source evidence; otherwise it is labeled unknown.
4. No runtime source, model/config, dependency, shared CANN installation, or NPU state is changed by this diagnosis plan.
5. No NPU retry occurs automatically. A proposed retry is presented for confirmation if it would require a materially different launch or another long TP=2 run.
