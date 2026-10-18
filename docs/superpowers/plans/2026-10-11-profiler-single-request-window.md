# 单请求 TTFT Profiler 时间窗 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 为一个已批准的 0.8B 八轮 Agent 请求记录精确客户端时序，并在时钟核验通过时生成该请求 TTFT 区间与 Torch/CANN 时间线重叠的可复核 CSV。

**Architecture:** 在现有 HTTP streaming client 中增加首个非空块回调和成对 epoch/monotonic 时间戳；在 Agent runner 中增加独立、窄范围的单请求采集选项，不改变既有整轮 --profile 行为。分析器读取客户端记录、Torch trace、CANN task_time 与 profiler_info 时钟锚点；时钟无法核实时只保留分离时间线，不计算重叠。

**Tech Stack:** Python、pytest、Torch-NPU Level1 profiler、CANN profiler 导出 CSV、JSON/JSONL、CSV。

---

## 范围与成功标准

- 只支持已批准的 Qwen3.5-0.8B、BF16、TP=1、Ascend910 V1 NPU0 场景；不接触 NPU ID 1–3，不启动 27B/TP=2。
- 单会话并发 1、上下文目标 512 token、共享前缀比例 50%、8 轮；前 7 轮使用 32-token 上限且不启动 Profiler，第 8 轮使用 max_tokens=1。
- 第 8 轮的现有 100 ms mock tool wait 必须在 Profiler start 前结束。start 钩子位于 wait 之后、HTTP 请求发送之前；首个非空 content 或 reasoning_content 块触发一次 stop，随后继续 drain SSE。
- 原有 TTFT/E2E 仍从 perf_counter 单调时钟计算；额外记录的 epoch/monotonic 时间戳只用于窗口与时钟核验。
- Raw、summary、NPU0 快照、配置 provenance、Torch/CANN 文件和派生 CSV 全部在 TokenSpeed 项目目录内；目标目录已存在时拒绝覆盖。
- focused fake tests 不请求真实 Profiler，不访问 NPU；真实采集须在实现和测试完成后另获用户批准。
- 结论限定为时间重叠观察，不声称请求与具体 operator/task 的因果归属，也不与无 Profiler 的性能基线直接比较。

## 文件职责

- 修改 benchmarks/course_project/core.py：ChatResult 的时间字段、首个非空输出块 callback、正常 SSE 完成标记；保持现有时延计算口径。
- 修改 benchmarks/course_project/runner.py：在 tool wait 后支持一次 before-request callback，并把 first-chunk callback 传给 client；仅用于 c1 profile 时启用。
- 修改 scripts/run_agent_benchmark.py：增加独立的 --profile-single-request-ttft 模式、参数约束、唯一输出路径、final-turn profile 边界和失败记录；保留既有 --profile 行为。
- 修改 benchmarks/course_project/profiler_analysis.py：增加纯函数式时钟核验、Torch/CANN 窗口交集与 CSV 写入逻辑；保留现有汇总输出。
- 新建 scripts/analyze_profiler_window.py：提供项目内路径校验、拒绝覆盖、定位单次采集输入并调用分析器的 CLI。
- 修改 test/course_project/test_benchmark_core.py、test/course_project/test_agent_profiler_capture.py、test/course_project/test_profiler_analysis.py：覆盖时序、边界、时钟和分析器。
- 仅在真实采集验收后更新 docs/project/experiments.md、docs/project/final_results.md、docs/report/ascend_profile.md、task_plan.md 与 notes.md；不提前写实验结论。
- 本计划不包含提交操作；当前工作树有其他未完成改动，保持未提交状态。

## 实施步骤

### Task 1：为 chat streaming 增加精确时间戳和首块 callback

**Files:**
- Modify: benchmarks/course_project/core.py
- Test: test/course_project/test_benchmark_core.py

- [x] **Step 1：先写首个非空块 callback 的失败测试**

在 TestOpenAIChatClient 中增加以下测试。现有本地 SSE fixture 会发送两个 content 块、usage 和 [DONE]；验证 callback 只在第一个块调用、命中的字段被记录、drain 时间晚于首块时间。

```python
def test_first_chunk_callback_and_timestamps_precede_stream_drain(self) -> None:
    port = self.server.server_address[1]
    events = []
    client = OpenAIChatClient(f"http://127.0.0.1:{port}/v1")

    result = client.chat(
        model="qwen35-0.8b",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        on_first_chunk=lambda field, epoch_ns, monotonic_ns: events.append(
            (field, epoch_ns, monotonic_ns)
        ),
    )

    self.assertEqual(len(events), 1)
    self.assertEqual(events[0][0], "content")
    self.assertEqual(result.first_chunk_field, "content")
    self.assertLessEqual(result.request_send_monotonic_ns, events[0][2])
    self.assertLess(events[0][2], result.stream_drain_monotonic_ns)
    self.assertTrue(result.stream_completed)
```

- [x] **Step 2：运行该测试，确认当前接口不满足要求**

Run from the TokenSpeed repository root:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py::TestOpenAIChatClient::test_first_chunk_callback_and_timestamps_precede_stream_drain -q
```

Expected: FAIL because chat currently rejects on_first_chunk and ChatResult has no timestamp fields.

- [x] **Step 3：实现最小字段与 callback**

给 ChatResult 增加以下结果字段；新增 callback 为可选参数，缺省时不改变现有调用方：

```python
@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    inter_chunk_latencies_ms: tuple[float, ...] = ()
    request_send_epoch_ns: int = 0
    request_send_monotonic_ns: int = 0
    first_chunk_epoch_ns: int | None = None
    first_chunk_monotonic_ns: int | None = None
    stream_drain_epoch_ns: int = 0
    stream_drain_monotonic_ns: int = 0
    first_chunk_field: str | None = None
    stream_completed: bool = False
```

保持 started_at、first_token_at、ended_at 继续由现有 time.perf_counter() 计算。仅在 urlopen 前、首个非空 piece、完整 SSE drain 后各记录 time.time_ns() 与 time.monotonic_ns()。遇到 [DONE] 时设置 stream_completed=True。首块时将 field、epoch_ns、monotonic_ns 同步传给 callback；后续块不得再次调用。判断顺序继续优先采用非空 delta.content，否则采用 delta.reasoning_content。callback 异常按请求异常向上传播，由 runner 的 finally 清理活动 Profiler。

- [x] **Step 4：运行 client tests 并检查时延兼容性**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py::TestOpenAIChatClient test/course_project/test_benchmark_core.py::TestMetrics -q
```

Expected: all selected tests pass; existing TTFT/E2E assertions remain unchanged, callback count is one for a normal stream, and a stream lacking [DONE] is not marked complete.

### Task 2：把生命周期钩子放在 mock wait 与首块之间

**Files:**
- Modify: benchmarks/course_project/runner.py
- Test: test/course_project/test_benchmark_core.py

- [x] **Step 1：写 runner callback 顺序测试**

使用 fake client 和 mock clock，要求记录事件严格按 wait → profiler start acknowledgement → request send → first chunk / profiler stop → chat return / stream drain 排列。测试还要验证启用 callback 时只能传入一条请求且 concurrency=1；多请求输入要明确抛 ValueError，避免误把并发 workload 当成单请求窗口。

```python
def test_profile_callbacks_run_after_tool_wait_and_before_stream_drain(self) -> None:
    events = []
    request = BenchmarkRequest(
        request_id="profiled",
        messages=[{"role": "user", "content": "prompt"}],
        workload="agent",
        tool_wait_ms=100,
    )

    class FakeClient:
        def chat(self, *, on_first_chunk=None, **_kwargs):
            events.append("send")
            on_first_chunk("content", 200, 200)
            events.append("drain")
            return SimpleNamespace(
                text="answer",
                input_tokens=2,
                output_tokens=1,
                ttft_ms=1.0,
                tpot_ms=1.0,
                e2e_ms=2.0,
                inter_chunk_latencies_ms=(),
                request_send_epoch_ns=100,
                request_send_monotonic_ns=100,
                first_chunk_epoch_ns=200,
                first_chunk_monotonic_ns=200,
                stream_drain_epoch_ns=300,
                stream_drain_monotonic_ns=300,
                first_chunk_field="content",
                stream_completed=True,
            )

    with patch.object(
        runner.time, "sleep", side_effect=lambda _seconds: events.append("wait")
    ):
        records = run_concurrent_requests(
            client=FakeClient(),
            requests=[request],
            concurrency=1,
            model="fake-model",
            max_tokens=1,
            before_request=lambda _request: events.append("start-ack"),
            on_first_chunk=lambda _request, *_times: events.append("stop"),
        )

    self.assertEqual(events, ["wait", "start-ack", "send", "stop", "drain"])
    self.assertEqual(records[0]["first_chunk_field"], "content")
```

- [x] **Step 2：运行该测试确认 runner 尚不支持生命周期钩子**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py::TestOpenAIChatClient::test_profile_callbacks_run_after_tool_wait_and_before_stream_drain -q
```

Expected: FAIL because run_concurrent_requests does not yet accept before_request or on_first_chunk.

- [x] **Step 3：在 runner 中添加只在明确传入时启用的 callback**

在 run_concurrent_requests 的签名增加可选的 before_request(request) 与 on_first_chunk(request, field, epoch_ns, monotonic_ns)。执行顺序固定为 sleep(tool_wait_ms) → before_request → client.chat(on_first_chunk=...)。任一 callback 被启用时，入口先验证 concurrency == 1 且 len(requests) == 1；默认 benchmark 路径不传 callback，因此维持原行为。将 ChatResult 的时间戳字段原样写入该请求 record，不改变 summarize_records 的既有统计字段。

- [x] **Step 4：运行 runner/client focused tests**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py -q
```

Expected: all tests in the file pass; ordinary concurrent requests still preserve IDs, while the fake profiled request proves wait/start/send/stop/drain ordering.

### Task 3：增加隔离的单请求 TTFT Profile 模式

**Files:**
- Modify: scripts/run_agent_benchmark.py
- Modify: test/course_project/test_agent_profiler_capture.py

- [x] **Step 1：增加 CLI 参数和参数约束测试**

增加 --profile-single-request-ttft 布尔选项。测试要求该模式只接受 concurrency=[1]、shared_prefix_ratios=[0.5]、prefix_tokens=512、turns=8、tool_wait_ms=100、tool_wait_policy=one-session、output_tokens=32、runs=1，且不得同时启用旧 --profile；任意不匹配都在 tokenizer 加载、目录创建、网络或 Profiler 调用前失败。

```python
def test_single_request_ttft_profile_rejects_non_pinned_workload(self) -> None:
    args = self._args(
        profile=False,
        concurrency=[4],
        ratios=[0.5],
        runs=1,
    )
    args.profile_single_request_ttft = True

    with self.assertRaisesRegex(ValueError, "concurrency"):
        agent_script.validate_single_request_ttft_args(args)
```

- [x] **Step 2：运行参数测试确认失败**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_agent_profiler_capture.py::TestAgentProfilerCapture::test_single_request_ttft_profile_rejects_non_pinned_workload -q
```

Expected: FAIL because the flag, validator, and dedicated capture flow do not exist.

- [x] **Step 3：实现独立运行分支和唯一产物目录**

只在新 flag 下启用以下行为；旧 --profile 及默认路径不变：

1. 校验上述固定参数，要求 config 的 served_model_name 为 qwen35-0.8b、tp_size 为 1，并在运行前确认 raw/profile/analysis 目标均不存在。
2. 生成 UTC run_id，并建立 `results/raw/profiler/single_request_ttft/<run_id>/` 与 `results/profiles/qwen35-0.8b/single_request_ttft/<run_id>/`。Raw 目录保存请求 JSONL、summary JSONL、配置 provenance 和仅 NPU0 的前后快照；配置 provenance 包含配置文件 SHA-256 与本次固定参数。原始记录不保存 Authorization header、api_key、request messages 或 output_text。
3. 第 1–7 轮使用 max_tokens=32，不启用 Profiler；第 8 轮使用 max_tokens=1。历史构造继续复用现有 token-sized context 和 conversation history，不额外新增请求或服务端改动。
4. 第 8 轮 run_concurrent_requests 的 before_request 在 100 ms mock wait 后执行：先写 NPU0 before snapshot，再记录 Profiler start 请求/响应时刻并启动现有 Level1 Profiler。首块 callback 记录命中的字段和 epoch/monotonic 时间，马上记录 stop 请求时刻、调用 ProfilerControlClient.stop()、记录 stop 响应时刻；callback 返回后由 OpenAIChatClient 继续 drain SSE。
5. 正常返回后保存 drain 完成时间与 NPU0 after snapshot。只将第 8 轮写为被采集请求记录；记录 1-token usage、[DONE] 完成状态、TTFT/E2E、五类规定时间戳和 start/stop 成对状态，但剔除 output_text、messages 和 Authorization/API-key 字段。若请求错误、无非空块、completion_tokens != 1 或没有 [DONE]，写 status=invalid/failed 与可复核原因，不写成功 summary、不生成窗口 CSV。
6. 用 profile_active / stop_acknowledged 状态保护 finally：Profiler 已开始但首块未到或请求异常时，finally 尝试一次清理 stop；首块 callback 已成功 stop 后不得重复 stop。异常记录只保存异常类型，不序列化请求对象、认证头或敏感正文。

为保持 profile 模式窄且可复现，路径创建使用 exist_ok=False；profile/profile output 的实际文件由现有 Profiler 服务写入。任何启动失败或目标冲突都保留错误状态，不覆盖已有目录。

- [x] **Step 4：用 fake Profiler 和 fake client 验证完整八轮边界**

在 test_agent_profiler_capture.py 中扩展 main() mock harness：前七轮 fake client 收到 max_tokens=32 且没有活跃 Profiler；第八轮的 wait 事件先于 start，只有一个请求使用 max_tokens=1，stop callback 先于 fake stream-drain，start/stop 各成功一次，最终 raw 只含一条 profile 请求记录。增加无首块、usage 非 1、缺少 [DONE]、request exception 和 start/stop RPC exception 分支；失败路径不得产出成功 summary 或 analysis CSV。原有 --profile tests 继续断言旧的整轮采集语义。

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_agent_profiler_capture.py -q
```

Expected: all tests pass; fake path makes no HTTP request and no real Profiler control call.

### Task 4：实现可拒绝不可靠时钟的 Torch/CANN 窗口 CSV

**Files:**
- Modify: benchmarks/course_project/profiler_analysis.py
- Create: scripts/analyze_profiler_window.py
- Modify: test/course_project/test_profiler_analysis.py

- [x] **Step 1：先写纯函数和合成 fixture tests**

覆盖 interval clipping、事件恰好落在窗口外、Torch complete event 与 CANN task_time interval、epoch/monotonic 偏差超过 1 ms、profiler_info 缺少锚点、trace/CANN 文件缺失、已存在 CSV 拒绝覆盖等情况。交集采用半开区间；各事件 overlap 求和列明确标记为 per-event sum，允许重叠事件之间重复计时。

```python
def overlap_us(start_us: float, stop_us: float, window_start_us: float, window_stop_us: float) -> float:
    return max(0.0, min(stop_us, window_stop_us) - max(start_us, window_start_us))


def test_overlap_uses_half_open_window_intersection():
    assert overlap_us(8.0, 15.0, 10.0, 20.0) == 5.0
    assert overlap_us(1.0, 9.0, 10.0, 20.0) == 0.0
    assert overlap_us(20.0, 22.0, 10.0, 20.0) == 0.0
```

- [x] **Step 2：运行分析器测试并确认现有实现**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_profiler_analysis.py -q
```

实现现已存在，因此无法在事后重放“缺少实现”的历史 RED。2026-10-12 使用项目 Conda 环境重跑分析器、采集 callback、runner 和 Profiler control focused tests：32 passed；没有启动服务、NPU、Profiler 或 HTTP 请求。

- [x] **Step 3：实现时钟核验、窗口交集和拒绝覆盖**

在 profiler_analysis.py 新增单采集分析函数，输入唯一 raw run directory、profile directory 和 analysis directory：

- 校验 raw 中恰有一条 status=success 的记录，request_send/first_chunk/drain 的 epoch 差分与 monotonic 差分误差均不超过 1,000,000 ns。
- 读取该 capture 的 profiler_info_0.json；以 end_info.collectionTimeEnd 与 end_info.MonotonicTimeEnd 计算 profiler epoch-minus-monotonic offset，并与客户端 send/first/drain 的 epoch-minus-monotonic offset 做误差核验。缺字段、非数值、误差超限时写明 clock_status=unverified，停止 overlap 计算。
- 仅接受完整 Torch Chrome trace 的 X phase 且 ts/dur 可解析事件；先核实 trace ts 与 CANN task_start(us)/task_stop(us) 的单位和时钟域，并将其时间范围与客户端记录的 Profiler start/stop epoch 时间交叉检查。CANN/Torch 映射无法核实时写明 clock_status=unverified，停止 overlap 计算，但仍写出双方各自的原生时间范围。此前约 57 μs 的观测仅作为先验背景，不写死为本轮 offset。
- CANN 仅解析 task_time CSV 中 task_start(us)、task_stop(us) 为有限正区间的行。用经核验的 epoch/monotonic 映射把两种时间线转换到客户端 monotonic 窗口；不从 Profiler stop 响应后的事件计算 overlap。
- CSV 固定一行，含 run_id、请求发送/首块/stop 请求/stop 响应/drain epoch ns、首块字段、TTFT/E2E、token 数、stream 完成标志、clock status、实际 epoch-minus-monotonic offset 与最大误差，以及 Torch/CANN 各自原生时间域的事件数与起止范围。时钟通过核验时再附窗口内 interval 数及 per-event overlap 微秒和。
- clock_status 不是 verified 时，仍分别报告 Torch 与 CANN 的原生时间域范围，但所有跨时钟 overlap 列留空；缺少 trace/CANN 产物或成功请求记录则报错，不伪造成功报告。
- 新 CLI 只接受解析后仍位于 TokenSpeed project root 内的输入输出路径；analysis 目录存在时退出，不覆盖。不得修改任何 raw/profile 输入。

- [x] **Step 4：运行分析器测试和 CLI help**

Run:

```powershell
..\.conda-env\python.exe -m pytest test/course_project/test_profiler_analysis.py -q
..\.conda-env\python.exe scripts/analyze_profiler_window.py --help
```

Expected: tests pass; help exits 0 and documents required raw/profile run directories and project-confined output behavior.

### Task 5：全量 focused 验收与真实采集门槛

**Files:**
- Verify: scripts/run_agent_benchmark.py
- Verify: scripts/analyze_profiler_window.py
- Verify: docs/superpowers/specs/2026-10-11-profiler-single-request-window-design.md
- Update only after successful real capture: docs/project/experiments.md, docs/project/final_results.md, docs/report/ascend_profile.md, task_plan.md, notes.md

- [x] **Step 1：运行课程项目 focused suite**

Run from TokenSpeed repository root:

```powershell
..\.conda-env\python.exe -m pytest test/course_project -q
```

Expected: exit code 0; all runnable course-project tests pass, with only already documented platform-specific skips. This command must not start a NPU service or Profiler.

- [x] **Step 2：检查变更与计划约束**

Run:

```powershell
git diff --check -- benchmarks/course_project/core.py benchmarks/course_project/runner.py scripts/run_agent_benchmark.py benchmarks/course_project/profiler_analysis.py scripts/analyze_profiler_window.py test/course_project/test_benchmark_core.py test/course_project/test_agent_profiler_capture.py test/course_project/test_profiler_analysis.py
```

Expected: diff check reports no whitespace errors in the scoped tracked files. Run the secret-pattern scan only after real raw files exist; if it unexpectedly matches, do not print the value and replace it with `<REDACTED>` before proceeding.

- [ ] **Step 3：在独立用户批准后才做只读 preflight 和一次采集**

计划、实现测试和 fake dry-run 不授权启动真实采集。获得后续单独批准后，才检查 0.8B 服务 readiness、模型 revision/config、NPU0 使用状态、profile control 可达性与三个唯一输出目标；若 NPU0 被占用、服务不匹配或目录冲突则停止，不抢占、不重启其他任务。仅用 NPU0；不查询 ID 1–3。经检查通过后执行以下唯一采集命令：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python scripts/run_agent_benchmark.py configs/bringup/qwen35_0.8b.yaml \
  --profile-single-request-ttft --concurrency 1 --shared-prefix-ratios 0.5 \
  --prefix-tokens 512 --turns 8 --tool-wait-ms 100 \
  --tool-wait-policy one-session --output-tokens 32 --warmup-requests 3 --runs 1
```

本轮只写计划，不运行此命令。

- [ ] **Step 4：采集后验证完整性，随后更新报告**

真实采集通过条件：1 条 profile 请求；前 7 轮没有 Profiler；最后一轮 max_tokens=1 且 usage 恰为 1；首个非空块与 stream [DONE] 均记录；Profiler start/stop 成对且 stop callback 仅触发一次；Level1 profiler_info 确认为 ACL_AICORE_PIPE_UTILIZATION；Torch/CANN/raw/summary/NPU0 快照均存在；wall/monotonic 与 profiler_info 时钟误差均在 1 ms 内；CSV 可解析、overlap 只覆盖 request_send 到 first_chunk；所有项目内 raw/profile 产物 SHA-256 清单可复核。结束后只检查本次项目进程退出、NPU0 使用回落和服务端口状态，不清理其他进程或文件。只有上述检查通过才更新列出的实验文档，并明确这是单请求带 Profiler 开销的时间重叠观察，不是因果结论或性能对比。

---

## 自审覆盖表

- 请求形状、设备范围、前 7 轮历史、final max_tokens=1 和 100 ms wait 边界：Task 3。
- 发送/首块/Profiler stop 请求与响应/drain 的 epoch ns，发送/首块/drain 的 monotonic ns，命中字段与旧 TTFT/E2E：Tasks 1–3。
- 1 ms wall/monotonic 核验、profiler_info offset/error、clock 未验证时禁止 overlap：Task 4。
- Torch 与 CANN 事件只统计半开 TTFT 窗口，stop 后尾部排除；时钟无效时分别保留两条原生时间范围：Task 4。
- 唯一项目内目录、NPU0 快照、配置 provenance、拒绝覆盖：Task 3。
- Fake dry-run、异常清理、真实采集用户批准门槛、哈希与采后只读检查：Tasks 3–5。
- 不加服务端 request ID/span、不把 overlap 解释为因果、不启动 27B/TP=2、不查询 NPU1–3：范围与 Tasks 3–5。
- 占位项检查：本计划未留占位项；代码接口在 Tasks 1–4 中逐一定义并由 focused tests 固定。
