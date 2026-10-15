# Baseline Configurations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four static E0 baseline configuration files that exactly match the existing ablation builder without claiming new benchmark results.

**Architecture:** Copy each model's existing `configs/bringup` runtime settings, then apply the E0 switches already defined by `build_ablation_configs`: full-context prefill chunk size, prefix cache off, KVStore disabled, mixed batch off, and overlap scheduling off. A focused test compares every static config against that derived E0 configuration. A short README distinguishes these templates from measured default bring-up baselines and marks 27B as unvalidated.

**Tech Stack:** JSON-compatible YAML files, existing Python configuration loader, pytest, project Conda environment.

---

## File map

- Create `configs/baseline/qwen35_0.8b.yaml`, `qwen35_4b.yaml`, `qwen35_9b.yaml`, and `qwen35_27b.yaml`: static E0 model configurations.
- Create `configs/baseline/README.md`: explain E0 settings, distinguish templates from measurements, and state 27B's unvalidated status.
- Create `test/course_project/test_baseline_configs.py`: ensure each file loads and exactly matches E0 derived from its corresponding bring-up config, excluding the bring-up-only `status` metadata.

## Task 1: Add and verify static E0 configuration coverage

**Files:**
- Create: `test/course_project/test_baseline_configs.py`
- Create: `configs/baseline/qwen35_0.8b.yaml`
- Create: `configs/baseline/qwen35_4b.yaml`
- Create: `configs/baseline/qwen35_9b.yaml`
- Create: `configs/baseline/qwen35_27b.yaml`
- Create: `configs/baseline/README.md`

- [x] **Step 1: Write a failing configuration consistency test.** Added parameterized coverage for all four models, exact YAML directory contents, and full dictionary equality with derived E0 after removing only `status`.
- [x] **Step 2: Run the focused test and verify it fails because the baseline files do not exist.** Observed 4 expected `FileNotFoundError` failures before adding the baseline files; no dependencies were installed and the temporary directory stayed under `tokenspeed/.tmp`.
- [x] **Step 3: Add the four static E0 files and README.** Preserved model/runtime fields, applied only E0 switches, omitted bring-up `status`, and documented the distinction from measured default-config results plus the unvalidated 27B status.
- [x] **Step 4: Run the focused test and verify all four configurations match E0.** Final focused result: 5 passed.
- [x] **Step 5: Run `test/course_project` and confirm the full local suite passes using the project Conda environment and a project-local basetemp.** Final result: 384 passed, 6 skipped. No server, NPU, Profiler, remote host, package installation, or commit was used.

## Self-review

- Coverage: all four required model entries are checked against the existing ablation implementation; the README separates configuration availability from measured evidence.
- Scope: no benchmark code, runtime behavior, report result, or experiment is changed.
- Gate: successful tests validate static configuration structure only; they do not imply 27B readiness or authorize hardware measurements.
