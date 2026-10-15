# Session KV Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in native Engine session API that keeps each session's scheduler request, KV pages, and recurrent state alive between turns while each caller submits only newly appended token IDs.

**Architecture:** The C++ scheduler owns the session table and retains the same `Request` in a `SessionIdle` state after a generation turn, preserving its block tables and request-pool row. A subsequent turn appends its suffix to that request and resumes at the user-selected computed-token frontier; closing an idle session transitions it through the normal resource-release path. Python exposes open/generate/close on the native Engine/AsyncLLM surface and reports per-turn prefill/reuse counts; the OpenAI/SMG HTTP schema remains unchanged.

**Tech Stack:** Python Engine/AsyncLLM, msgspec/ZMQ control messages, nanobind, C++ `tokenspeed_scheduler`, C++ scheduler and FSM tests, project Conda environment, Ascend910 V1 integration validation.

## Progress (2026-10-10 local)

- C++ scheduler lifecycle and bindings are implemented: persistent `Request` association, idle KV/state page retention, suffix-turn continuation, FIFO-tail resume, close/reuse guards, session-only deferred abort release, capacity validation, session statistics, and retraction-aware reuse counts. Review findings were fixed with regression tests. Fresh verification passed 17 Session tests repeated three times, all 591 scheduler tests, Python extension build, and a bound Scheduler open/duplicate/submit/info/active-close smoke test.
- Python Engine API/IPC wiring is implemented: sync/async open-close acknowledgements, suffix routing, text-only/session constraint checks, scheduler-derived per-turn metadata, pause-buffer admission protection, and raw-token output parity. Independent review findings were fixed. Root verification passed the focused suite (25/25), the full `test/course_project` suite (93 passed, 1 skipped), Python `py_compile`, and `git diff --check`. A runtime test could not collect because the project environment lacks NumPy; no dependency was installed. NPU correctness/reuse acceptance is recorded below; no Session KV performance sweep has been run.
- The user selected `current-frontier`: on the next turn, the prior turn's final sampled token is first processed together with the newly appended suffix, with no extra end-of-turn forward.
- Task 3 NPU correctness/reuse acceptance completed on NPU0 with generic prefix caching and KVStore disabled. The initial run disabled overlap; a follow-up run with `overlap_schedule=true` also completed, with all four session turns matching stateless controls. Both runs are correctness/reuse checks, not performance results.

---

## File map

- `tokenspeed-scheduler/csrc/fsm/forward_states.h`, `states.h`, `forward_events.h`, `forward_events.cpp`: represent an idle session as a page-owning state and transition between active turns, idle, and closed.
- `tokenspeed-scheduler/csrc/scheduler/request_spec.h`, `request.h`, `request.cpp`, `scheduler.h`, `scheduler.cpp`, `outside_event_handler.cpp`: keep session identity/context in the C++ scheduler, reuse the same request resources, guard single-writer lifecycle, and release pages on close.
- `tokenspeed-scheduler/bindings/python_module.cpp`: expose session identity and lifecycle operations to Python.
- `tokenspeed-scheduler/tests/cpp/test_scheduler_plan.cpp`: exercise scheduler-owned allocation, turn append, isolation, and close behavior.
- `python/tokenspeed/runtime/engine/io_struct.py`, `protocol.py`, `async_llm.py`, `llm.py`, `entrypoints/engine.py`: expose native sync/async session operations and their acknowledgements.
- `python/tokenspeed/runtime/engine/event_loop.py`, `request_handler.py`, `input_processor.py`, `output_processor.py`: route session control, accept only suffix IDs, preserve ordinary stateless requests, and attach result accounting.
- `test/course_project/test_session_kv.py`: cover public API wiring, lifecycle errors, and metadata without requiring an NPU.
- `scripts/run_session_kv.py`, `scripts/run_session_kv.sh`: run the matched 4B session-on/stateless comparison on the course NPU with generic prefix caching disabled and save raw per-turn records under `results/raw/session_kv/`.
- `docs/project/feature_matrix.md`, `docs/project/experiments.md`, `task_plan.md`: record implementation status and measured evidence without claiming an unrun performance gain.

## Selected behavior: final sampled token frontier

The current scheduler defines `Request::NumComputedTokens()` as `TokenSize() - 1` while decoding: the last sampled token has no KV until the next model forward. The user selected the first behavior:

1. Retain the current frontier: the client still sends only newly appended user IDs; the scheduler internally prefills the prior turn's last sampled token plus the new suffix. This avoids an extra end-of-turn forward and reports that token in `newly_prefilled_tokens`.
2. Materialize the final token at turn end: run one no-sample forward before returning the turn, so the next turn prefills exactly the newly appended suffix. This adds one token of work and end-of-turn latency.

Both choices retain all earlier KV. Under the selected current frontier, the client sends only newly appended IDs, while the scheduler internally includes the prior last sampled token in the next prefill; no extra forward is added at turn end.

## Task 1: Define and test the retained-session scheduler lifecycle

**Files:**
- Modify: `tokenspeed-scheduler/csrc/fsm/forward_states.h`
- Modify: `tokenspeed-scheduler/csrc/fsm/states.h`
- Modify: `tokenspeed-scheduler/csrc/fsm/forward_events.h`
- Modify: `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- Modify: `tokenspeed-scheduler/csrc/core/token_container.h`
- Modify: `tokenspeed-scheduler/csrc/scheduler/request_spec.h`
- Modify: `tokenspeed-scheduler/csrc/scheduler/request.h`
- Modify: `tokenspeed-scheduler/csrc/scheduler/request.cpp`
- Modify: `tokenspeed-scheduler/csrc/scheduler/scheduler.h`
- Modify: `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp`
- Modify: `tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`
- Test: `tokenspeed-scheduler/tests/cpp/test_scheduler_plan.cpp`

- [x] Define the scheduler-owned API as `bool Scheduler::OpenSession(const std::string&)`, `bool Scheduler::CloseSession(const std::string&)`, and `bool Scheduler::HasSession(const std::string&) const`; an empty `RequestSpec.session_id` keeps the existing stateless path. Opening a session registers its ID without allocating a request row or KV pages; the session-map value is null until its first turn. After admission, map the session ID to its persistent `Request*`. Store an idle turn as `SessionIdle { ForwardResources resources; std::int32_t computed_tokens; }` in the FSM `State` variant, retaining the request in the scheduler's request storage. On resume, update the per-turn request ID and move it to the FIFO tail. Closing an unused session removes only its registration; closing an idle session applies a resource-releasing FSM event. Active sessions cannot be closed.
- [x] Add C++ tests for open/duplicate-open, close-before-first-turn, first session turn, finishing into idle without freeing pages, suffix continuation from the scheduler's saved computed-token frontier, distinct-session isolation, rejection of a same-session concurrent turn, unknown-session errors, close releasing retained pages/request row, and scheduler shutdown releasing retained sessions. The user-selected `current-frontier` policy is accepted in the Task 3 NPU matched run recorded below.
- [x] In the close-error test, assert `EXPECT_FALSE(scheduler_->CloseSession("missing"))`; for duplicate-open, assert the first open succeeds and the second returns false. Build and run the focused C++ scheduler test before implementation.

  ```powershell
  . ..\scripts\use_project_env.ps1
  cmake -S tokenspeed-scheduler -B .tmp/session-kv-scheduler-tests -G Ninja -DCMAKE_CXX_COMPILER=g++ "-DPython3_EXECUTABLE=$env:CONDA_PREFIX\python.exe" -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
  cmake --build .tmp/session-kv-scheduler-tests --target tokenspeed_scheduler_tests
  & .tmp/session-kv-scheduler-tests/tokenspeed_scheduler_tests.exe --gtest_filter="*Session*"
  ```

  Expected before implementation: compile failure at the new `Scheduler` session calls/types; expected after implementation: focused tests pass.
- [x] Add `SessionIdle` to `fsm::State`, holding `ForwardResources` and the computed-token frontier. Add `session_id` to `RequestSpec`/`Request`; keep an idle `Request` in the scheduler-owned session map after a successful turn instead of applying the releasing `FinishEvent`. Update all state visitors and lifecycle counters so idle requests are retained for memory accounting but never scheduled as active work.
- [x] Add a turn-append transition that changes the request ID, appends only new tokens, resets the per-turn generation budget, rebases the prompt extent, and creates a prefill window beginning at the saved computed-token frontier.
- [x] Add scheduler open/close/submit handling. Reject duplicate IDs, unknown IDs, active-session overlap, closing an active session, and turns whose cumulative context plus generation reserve exceeds `MaxSingleRequestTokens()`. Close an idle request via the resource-releasing FSM transition; ensure an aborted/failed turn uses the normal resource-release path and invalidates its session registration.
- [x] Rebuild the scheduler test binary and run `tokenspeed_scheduler_tests.exe`; all 591 C++ scheduler tests pass. This CMake project currently has no CTest registrations, so invoke the GoogleTest executable directly. If FetchContent needs GitHub, use a command-scoped proxy override for the active local Clash port; do not modify global Git configuration.

## Task 2: Wire the native Python Engine session API

**Files:**
- Modify: `tokenspeed-scheduler/bindings/python_module.cpp`
- Modify: `python/tokenspeed/runtime/engine/io_struct.py`
- Modify: `python/tokenspeed/runtime/engine/protocol.py`
- Modify: `python/tokenspeed/runtime/engine/async_llm.py`
- Modify: `python/tokenspeed/runtime/engine/llm.py`
- Modify: `python/tokenspeed/runtime/entrypoints/engine.py`
- Modify: `python/tokenspeed/runtime/engine/event_loop.py`
- Modify: `python/tokenspeed/runtime/engine/request_handler.py`
- Modify: `python/tokenspeed/runtime/engine/input_processor.py`
- Modify: `python/tokenspeed/runtime/engine/output_processor.py`
- Test: `test/course_project/test_session_kv.py`

- [x] Add API tests first: `Engine.open_session()` returns a stable ID; `generate(input_ids=suffix, session_id=...)` sends only the suffix and attaches `SessionParams.id`; `close_session()` waits for scheduler acknowledgement; unknown/closed IDs and overlapping writes produce explicit errors; ordinary `generate()` remains unchanged.
- [x] Run the new Python API tests and confirm the session methods/metadata fail before the wiring exists.
- [x] Connect `OpenSessionReqInput` and a new acknowledged close result through the event loop to the scheduler binding. Ensure `AsyncLLM.open_session()` registers its future before sending the control message, and resolve/remove pending futures on every success/error path.
- [x] Add the optional `session_id` argument to native sync/async generation. Keep the HTTP gateway untouched. Grammar-constrained and multimodal session inputs are explicitly rejected rather than silently falling back to stateless generation.
- [x] Remove the unconditional session-ID rejection only after scheduler session validation is available. Route the session ID into `RequestSpec` and preserve the stateless request path when it is absent.
- [x] Add response `meta_info.session` with `session_id`, `total_context_tokens`, `newly_prefilled_tokens`, and `reused_context_tokens`. Counts come from the scheduler's session frontier rather than generic prefix-cache hit counters.
- [x] Run `test/course_project/test_session_kv.py` and the existing `test/course_project` suite; 25 focused tests and all 93 course-project tests passed (1 platform skip), with stateless response behavior covered.

## Task 3: Verify KV reuse and correctness on the course NPU

**Files:**
- Create: `scripts/run_session_kv.py`
- Create: `scripts/run_session_kv.sh`
- Test: `test/course_project/test_session_kv.py`
- Output: `results/raw/session_kv/`

- [x] Add a matched-turn runner using the pinned Qwen3.5-4B revision: each session sends only its appended IDs after turn one, with a stateless full-history control and identical greedy parameters. The CLI supports the selected `current-frontier` policy; final-token materialization remains unsupported.
- [x] Start the 4B service on NPU0 with generic prefix caching and KVStore disabled; confirm readiness and session reuse metadata before validating turns.
- [x] Record raw input/suffix/output IDs, total-context/new-prefill/reused counts, per-turn latency, hardware snapshots, model revision, and configuration for two interleaved sessions; close both after completion.
- [x] All four session output sequences match their stateless controls; turn-2 metadata reports (total/new/reused) 70/20/50 and 67/21/46 tokens; post-close generation is rejected for both IDs.
- [x] The runner stops its own service group; after exit, NPU0 returned to 4% HBM usage with no project serving process. Raw records and snapshots are under `results/raw/session_kv/`.
- [x] Follow-up overlap-enabled acceptance on NPU0: run with `overlap_schedule=true`; both interleaved sessions completed two turns, all four output sequences matched stateless full-history controls, turn-2 metadata again reported 70/20/50 and 67/21/46 tokens, and both post-close probes were rejected. Raw records and snapshots are under `results/raw/session_kv_overlap_20261010/`.

Acceptance scope note: both the initial overlap-disabled run and the follow-up overlap-enabled run are correctness/reuse checks for the recorded 4B/TP1/NPU0 configuration. The overlap-enabled handoff passed this matched run; no Session KV performance sweep was conducted, so do not claim performance gains.

## Task 4: Record verified scope and limitations

**Files:**
- Modify: `docs/project/feature_matrix.md`
- Modify: `docs/project/experiments.md`
- Modify: `task_plan.md`

- [x] Mark the native Engine Session KV API implemented after Tasks 1–3 passed; document retained device memory until close, single-writer/process-local/text-only scope, and the overlap-enabled correctness acceptance boundary.
- [x] Report raw matched correctness and per-turn reuse counts. Make no performance claim; generic prefix caching and KVStore were disabled for the session-specific acceptance run.
- [x] Keep the OpenAI/SMG HTTP API and client-mock Workload F claims unchanged.
