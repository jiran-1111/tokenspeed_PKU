# Agent Tool-Wait Client Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible client-side mode where session-0 waits for a fixed mock tool result while other Agent sessions continue inference requests, then run and document Workload F on the available Ascend 910 V1 host. Course acceptance of V1 in place of 910C remains unconfirmed.

**Architecture:** Keep the existing `all` wait behavior as the default. Add a small request builder in the benchmark runner to decide per-session delay and mock-result injection; the CLI records the selected policy, and the existing thread pool naturally lets non-waiting sessions proceed during session-0's client sleep. No inference-server changes are needed.

**Tech Stack:** Python 3.11, `unittest`, existing TokenSpeed Agent benchmark client, SSH/SCP to the course server, JSONL/CSV/SVG project reports.

---

### Task 1: Build and test per-session Tool-Wait requests

**Files:**
- Modify: `benchmarks/course_project/runner.py`
- Test: `test/course_project/test_benchmark_core.py`

- [x] **Step 1: Add failing policy and request-record tests**

Add this test method to `TestAgentWorkload`:

```python
def test_builds_all_and_one_session_tool_wait_requests(self) -> None:
    build_request = getattr(runner, "build_agent_turn_request", None)
    self.assertIsNotNone(build_request, "agent turn request builder is missing")
    history = [{"role": "system", "content": "context"}]
    common = {
        "turn_index": 2,
        "history": history,
        "user_message": "Continue the task.",
        "tool_wait_ms": 100,
    }

    all_waiting = [
        build_request(
            request_id=f"all-{session_index}",
            session_id=f"session-{session_index}",
            tool_wait_policy="all",
            **common,
        )
        for session_index in range(2)
    ]
    self.assertEqual([request.tool_wait_ms for request in all_waiting], [100, 100])
    self.assertFalse(any(request.mock_tool_result_injected for request in all_waiting))

    waiting = build_request(
        request_id="one-session-0",
        session_id="session-0",
        tool_wait_policy="one-session",
        **common,
    )
    continuing = build_request(
        request_id="one-session-1",
        session_id="session-1",
        tool_wait_policy="one-session",
        **common,
    )
    self.assertEqual(waiting.tool_wait_ms, 100)
    self.assertTrue(waiting.mock_tool_result_injected)
    self.assertIn(runner.MOCK_TOOL_RESULT, waiting.messages[-1]["content"])
    self.assertEqual(continuing.tool_wait_ms, 0)
    self.assertFalse(continuing.mock_tool_result_injected)
    self.assertNotIn(runner.MOCK_TOOL_RESULT, continuing.messages[-1]["content"])

    first_turn = build_request(
        request_id="one-session-first-turn",
        session_id="session-0",
        turn_index=1,
        history=history,
        user_message="Start the task.",
        tool_wait_policy="one-session",
        tool_wait_ms=100,
    )
    self.assertEqual(first_turn.tool_wait_ms, 0)
    self.assertFalse(first_turn.mock_tool_result_injected)
```

Also add this event-coordinated concurrency test to `TestAgentWorkload`:

```python
def test_other_session_reaches_client_during_tool_wait(self) -> None:
    waiting = threading.Event()
    other_called_during_wait = threading.Event()

    class FakeClient:
        def chat(self, *, model, messages, max_tokens):
            if messages[-1]["content"] == "session-1":
                if waiting.wait(timeout=2):
                    other_called_during_wait.set()
            return SimpleNamespace(
                text="ok",
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                tpot_ms=1.0,
                e2e_ms=2.0,
                inter_chunk_latencies_ms=[],
            )

    def controlled_sleep(_seconds):
        waiting.set()
        if not other_called_during_wait.wait(timeout=2):
            raise AssertionError("session-1 did not enter chat during session-0 wait")

    requests = [
        BenchmarkRequest(
            request_id="session-0",
            messages=[{"role": "user", "content": "session-0"}],
            workload="agent_multi_turn",
            session_id="session-0",
            turn_index=2,
            tool_wait_ms=100,
        ),
        BenchmarkRequest(
            request_id="session-1",
            messages=[{"role": "user", "content": "session-1"}],
            workload="agent_multi_turn",
            session_id="session-1",
            turn_index=2,
        ),
    ]
    with patch("benchmarks.course_project.runner.time.sleep", controlled_sleep):
        run_concurrent_requests(
            client=FakeClient(),
            requests=requests,
            concurrency=2,
            model="fake-model",
            max_tokens=1,
        )
    self.assertTrue(other_called_during_wait.is_set())
```

- [x] **Step 2: Run the focused tests and verify RED**

Run from the project root:

```powershell
python -m unittest discover -s test/course_project -p test_benchmark_core.py -v
```

Expected: the new tests fail because `build_agent_turn_request`, `MOCK_TOOL_RESULT`, and the request marker do not exist yet; existing tests still run.

- [x] **Step 3: Implement the minimal request builder and marker**

In `benchmarks/course_project/runner.py`, add `mock_tool_result_injected: bool = False` to `BenchmarkRequest`, define the fixed result constant, and add the builder with this interface and behavior:

```python
MOCK_TOOL_RESULT = (
    "Mock tool result: Search returned two relevant documents. "
    "Continue with the task."
)


def build_agent_turn_request(
    *,
    request_id: str,
    session_id: str,
    turn_index: int,
    history: list[dict[str, str]],
    user_message: str,
    tool_wait_policy: str,
    tool_wait_ms: int,
) -> BenchmarkRequest:
    inject_mock_result = (
        tool_wait_policy == "one-session"
        and session_id == "session-0"
        and turn_index > 1
    )
    should_wait = turn_index > 1 and (
        tool_wait_policy == "all" or inject_mock_result
    )
    message = user_message
    if inject_mock_result:
        message = f"{message}\n\n{MOCK_TOOL_RESULT}"
    return BenchmarkRequest(
        request_id=request_id,
        messages=[*history, {"role": "user", "content": message}],
        workload="agent_multi_turn",
        session_id=session_id,
        turn_index=turn_index,
        tool_wait_ms=tool_wait_ms if should_wait else 0,
        mock_tool_result_injected=inject_mock_result,
    )
```

Add `"mock_tool_result_injected": request.mock_tool_result_injected` to each record returned by `run_concurrent_requests`.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the same `unittest discover` command from Step 2. Expected: all tests in `test_benchmark_core.py` pass, including both policy cases and the event-coordinated overlap test.

- [x] **Step 5: Commit the runner and tests**

```powershell
git add -- benchmarks/course_project/runner.py test/course_project/test_benchmark_core.py
git -c user.name=Codex -c user.email=codex@local.invalid commit -m "feat: model a single waiting agent session"
```

The one-command author identity is local to this commit; it does not alter Git configuration.

### Task 2: Expose the policy, record metadata, and document Workload F

**Files:**
- Modify: `scripts/run_agent_benchmark.py`
- Test: `test/course_project/test_benchmark_core.py`
- Modify: `docs/project/experiments.md`
- Modify: `docs/project/final_results.md` after the remote run

- [x] **Step 1: Add a failing CLI contract assertion**

In the existing CLI help test, for `scripts/run_agent_benchmark.py`, assert that `result.stdout` contains both `--tool-wait-policy` and `one-session`.

```python
if script == "scripts/run_agent_benchmark.py":
    self.assertIn("--tool-wait-policy", result.stdout)
    self.assertIn("one-session", result.stdout)
```

- [x] **Step 2: Run the focused CLI test and verify RED**

Run:

```powershell
python -m unittest discover -s test/course_project -p test_benchmark_core.py -v
```

Expected: the CLI assertion fails because the option is not yet registered.

- [x] **Step 3: Wire the CLI to the tested request builder**

Add this argument in `parse_args()`:

```python
parser.add_argument(
    "--tool-wait-policy",
    choices=("all", "one-session"),
    default="all",
)
```

Replace the inline `BenchmarkRequest(...)` construction in the turn loop with `build_agent_turn_request(...)`, passing the current request ID, session ID, turn index, history, base user message, `args.tool_wait_policy`, and `args.tool_wait_ms`. Add `tool_wait_policy` to every raw record and summary. Preserve the current default and all existing output fields.

Import `build_agent_turn_request` from `benchmarks.course_project.runner` and remove the now-unused `BenchmarkRequest` import from `scripts/run_agent_benchmark.py`.

- [x] **Step 4: Document the command and measurement semantics**

In `docs/project/experiments.md`, add a Workload F command using `configs/bringup/qwen35_4b.yaml`, concurrency `1 4`, shared-prefix ratios `0 0.5 0.9`, 512 prefix tokens, 8 turns, 100 ms wait, 32 output tokens, 3 warmups, 3 repetitions, and `--tool-wait-policy one-session`. State that the mock result is client-side plain user text, individual request TTFT/TPOT/E2E exclude the pre-request sleep, and per-run wall time/throughput include it. Do not describe this as server-side waiting or KV retention.

- [x] **Step 5: Run the complete course-project tests**

Run:

```powershell
python -m unittest discover -s test/course_project -v
```

Expected: all locally runnable course-project tests pass; the known Windows-only Bash test may skip. On the remote server, run the same suite with the project environment and require all tests to pass before starting inference.

- [x] **Step 6: Commit the CLI, regression test, and method documentation**

```powershell
git add -- scripts/run_agent_benchmark.py test/course_project/test_benchmark_core.py docs/project/experiments.md
git -c user.name=Codex -c user.email=codex@local.invalid commit -m "feat: expose client-side agent tool wait policy"
```

### Task 3: Run and verify Workload F on the course server

**Files:**
- Create: `results/raw/agent/qwen35_4b_20261009_tool_wait/` and its hardware snapshots
- Create: `results/plots/after_4b_agent_tool_wait_20261009/`
- Modify: `docs/project/experiments.md`
- Modify: `docs/project/final_results.md`

- [x] **Step 1: Wait for the already-running 4B Agent collection to finish**

Do not overlap the new workload with the active 4B service benchmark. Confirm the existing 4B run has 360 request rows, 18 summary rows, and 36 hardware snapshots; then stop only its known project-owned service group and verify port 8000 is closed. Leave the separate 27B download running.

- [x] **Step 2: Sync only the changed source/tests/docs and run remote tests**

From the local project root, copy only the four changed files:

```powershell
scp benchmarks/course_project/runner.py ASCEND_SSH_HOST:~/tokenspeed/benchmarks/course_project/runner.py
scp test/course_project/test_benchmark_core.py ASCEND_SSH_HOST:~/tokenspeed/test/course_project/test_benchmark_core.py
scp scripts/run_agent_benchmark.py ASCEND_SSH_HOST:~/tokenspeed/scripts/run_agent_benchmark.py
scp docs/project/experiments.md ASCEND_SSH_HOST:~/tokenspeed/docs/project/experiments.md
```

Then run on the remote project:

```bash
cd ~/tokenspeed
source scripts/use_project_env.sh
python -m unittest discover -s test/course_project -v
```

Expected: all 58 course-project tests pass before inference starts.

- [x] **Step 3: Run the 4B Workload F matrix**

Start the 4B service in the background from the remote project root:

```bash
cd ~/tokenspeed
mkdir -p .tmp
nohup setsid bash -lc 'cd ~/tokenspeed && source scripts/use_project_env.sh && bash scripts/serve.sh configs/bringup/qwen35_4b.yaml' > .tmp/qwen35_4b_tool_wait_serve.log 2>&1 < /dev/null &
printf '%s\n' "$!" > .tmp/qwen35_4b_tool_wait_serve.pid
```

Wait for `curl -fsS http://127.0.0.1:8000/health` to succeed, then run:

```bash
source scripts/use_project_env.sh
bash scripts/run_agent_benchmark.sh configs/bringup/qwen35_4b.yaml \
  --concurrency 1 4 \
  --shared-prefix-ratios 0 0.5 0.9 \
  --prefix-tokens 512 \
  --turns 8 \
  --tool-wait-ms 100 \
  --tool-wait-policy one-session \
  --output-tokens 32 \
  --warmup-requests 3 \
  --runs 3 \
  --output-dir results/raw/agent/qwen35_4b_20261009_tool_wait
```

Expected: 360 request records, 18 summaries, and 36 before/after hardware snapshots. Confirm every request has non-null TTFT, TPOT, and E2E values; for turns after the first, only `session-0` has `tool_wait_ms=100` and `mock_tool_result_injected=true`; other sessions have `tool_wait_ms=0` and the marker is false. Every summary must record `tool_wait_policy="one-session"`.

- [x] **Step 4: Sync, hash-check, and render local reports**

Copy the raw JSONL files and hardware snapshots into the same local project paths. Compare remote and local SHA-256 hashes for every copied file. Then run:

```powershell
python benchmarks/report.py --raw-dir results/raw/agent/qwen35_4b_20261009_tool_wait --output-dir results/plots/after_4b_agent_tool_wait_20261009
```

Expected: the output directory is project-local and contains `summary.csv` plus generated SVG charts; no existing result directory is overwritten.

- [x] **Step 5: Record results and close the 4B service**

Add the measured matrix size, per-configuration median TTFT/TPOT/throughput, and the client-side wait caveat to `docs/project/experiments.md` and `docs/project/final_results.md`. Stop only the exact 4B service launch group after the run; verify the service port is closed and do not disturb NPU2 or the independent 27B download.

- [x] **Step 6: Verify project tests and commit only source, test, and documentation changes**

Re-run `python -m unittest discover -s test/course_project -v`. Stage only the source, test, and documentation files:

```powershell
git add -- benchmarks/course_project/runner.py scripts/run_agent_benchmark.py test/course_project/test_benchmark_core.py docs/project/experiments.md docs/project/final_results.md
git -c user.name=Codex -c user.email=codex@local.invalid commit -m "test: record single-session agent tool wait results"
```

Keep raw measurements and generated plots in the workspace without staging them or unrelated files.

### Plan self-review

- Spec coverage: backward-compatible default, fixed session-0 wait/result, concurrent other sessions, per-request and summary metadata, measurement caveat, Workload F invocation, and no server-side state changes are all assigned above.
- Completeness check: no unresolved plan items remain; commands, expected counts, paths, and test behavior are specified.
- Type consistency: `build_agent_turn_request(...)` returns `BenchmarkRequest`; its `mock_tool_result_injected` field is carried unchanged into the raw request record; `tool_wait_policy` is attached by the CLI to both raw rows and summaries.
