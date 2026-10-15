# 长 Prefill 干扰持续 Decode 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 增加一个可复现的 HTTP staged 长 Prefill/Decode 干扰实验，在 4B/TP1、2K/3.5K 输入和 chunk size 4096/512 下记录请求与 SSE 块时间线，并生成可审计 CSV/SVG。

**Architecture:** 在现有 `OpenAIChatClient` 上增加不含输出文本的逐块事件；新实验模块负责精确 token 长度、配置校验和 staged 并发；独立 runner 按配对轮次启动服务并只写入唯一 run 目录；分析 CLI 从 raw JSONL 生成 CSV/SVG。实现与实际 NPU 测量分开，真实测量需在本计划复核后另行获准。

**Tech Stack:** Python 标准库（`concurrent.futures`、`csv`、`json`、`socket`、`subprocess`、`xml.etree.ElementTree`）、项目内 Transformers tokenizer、现有 `build_serve_command`、`capture_npu_smi` 与 pytest；不新增依赖、不下载文件。

---

## 文件结构

- 修改 `benchmarks/course_project/core.py`：定义逐块时间事件并让 HTTP streaming client 暴露每个非空 SSE 块的时间戳；保留现有 TTFT 回调语义。
- 新建 `benchmarks/course_project/long_prefill_decode.py`：放置实验条件、配置/提示词校验、单请求与 staged trial、配对摘要和 raw schema。
- 新建 `scripts/run_long_prefill_decode.py`：检查参数与 NPU0/端口前置条件，按配对顺序启动服务、预热、执行 trial、保存 raw JSONL 和配置快照。
- 新建 `scripts/analyze_long_prefill_decode.py`：读取单个 run 的 raw JSONL，验证有效 trial，输出分析 CSV 和 SVG。
- 修改 `test/course_project/test_benchmark_core.py`，新建 `test/course_project/test_long_prefill_decode.py` 和 `test/course_project/test_long_prefill_decode_report.py`：分别覆盖 SSE 事件、实验逻辑/失败边界、CSV/SVG。
- 所有运行数据写在 `results/raw/benchmark/long_prefill_decode/<run_id>/`，派生产物写在 `results/analysis/long_prefill_decode/<run_id>/`；不改 `task_plan.md`、`notes.md` 或 Profiler 结果文档，不提交代码。

## Task 1：逐块记录 HTTP SSE 时间

**Files:** 修改 `benchmarks/course_project/core.py`；修改 `test/course_project/test_benchmark_core.py`。

- [x] **Step 1：增加失败测试。** 在现有 fake streaming HTTP handler 加一个包含空 delta、`content` 与 `reasoning_content` 的 SSE 路由；验证 `on_chunk` 只收到非空块、序号从 1 递增、两个时钟均为正且按序，`first_chunk` 仍只触发一次。测试核心断言为：

```python
events = []
result = client.chat(model="fake", messages=[{"role": "user", "content": "x"}],
                     max_tokens=4, on_chunk=events.append)
assert [event.chunk_index for event in events] == [1, 2]
assert tuple(events) == result.chunk_events
assert all(event.monotonic_ns > 0 and event.epoch_ns > 0 for event in events)
```
- [x] **Step 2：运行测试确认失败（按现有状态处理）。** 执行本计划前，生产接口与主体测试已存在；本轮未删除工作实现来人为制造 RED 状态。新增的 epoch 时钟有序断言已通过 focused test。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py -q
```

预期：新增测试因 `StreamChunkEvent`/`on_chunk` 尚不存在而失败，既有测试结果保持可见。

- [x] **Step 3：实现最小事件接口。** 在 `core.py` 增加：

```python
@dataclass(frozen=True)
class StreamChunkEvent:
    chunk_index: int
    field: str
    epoch_ns: int
    monotonic_ns: int
    text_length: int
```

给 `ChatResult` 添加 `chunk_events: tuple[StreamChunkEvent, ...] = ()`；给 `OpenAIChatClient.chat` 添加 `on_chunk: Callable[[StreamChunkEvent], None] | None = None`。解析到非空 `content` 或 `reasoning_content` 时，创建并保存事件、再调用 `on_chunk`；不把 `piece` 放入事件。保留现有 `inter_chunk_latencies_ms`、`on_first_chunk`、TTFT/E2E 算法。增加可选 `ignore_eos: bool = False`，仅为 True 时在请求 JSON 增加 `"ignore_eos": true`，以供本实验验证 128-token 持续 Decode；其他调用的 payload 不变。

- [x] **Step 4：运行 focused tests。**

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_benchmark_core.py -q
```

预期：新增及既有 benchmark core tests 全通过；fake HTTP 测试不访问 NPU。

## Task 2：实验配置与精确输入构造

**Files:** 新建 `benchmarks/course_project/long_prefill_decode.py`；新建 `test/course_project/test_long_prefill_decode.py`。

- [x] **Step 1：测试配置变体。** 使用 `configs/bringup/qwen35_4b.yaml` fixture，断言只生成 `chunked_prefill_size=4096` 与 `chunked_prefill_size=512` 两种配置，两者均 `prefix_caching=False`，其他字段不变；拒绝非 4B、非 TP1、非 BF16、`max_model_len != 4096` 或未知 chunk size。
- [x] **Step 2：测试 token 长度校验。** 使用 fake tokenizer 覆盖 256、2048、3584 的完整 chat-template token 数；构造器须在限定搜索步数内找到精确目标，否则抛出包含目标和实际长度的 `ValueError`，不得静默截断或超出 4096。
- [x] **Step 3：运行测试确认失败（按现有状态处理）。** 执行本计划前，配置/prompt 实现已存在；本轮未移除现存代码来人为制造 RED 状态。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -q
```

预期：配置构造、精确 prompt 与校验函数尚不存在时失败。

- [x] **Step 4：实现显式数据类型和纯函数。** 定义 `TrialSpec(trial_id, condition, prefill_tokens, repetition, chunk_size, execution_order)`；提供 `build_config_variants(base_config) -> dict[int, dict]`、`build_exact_messages(tokenizer, target_tokens, seed) -> list[dict[str, str]]` 与 `validate_result(result, expected_input_tokens, expected_output_tokens) -> tuple[bool, str | None]`。沿用 `make_token_sized_text` 和 `chat_template_token_ids`；加载 tokenizer 时指定 `local_files_only=True`，并在调用服务前确认模型目录位于项目内且已存在，禁止回退到远程模型 ID 下载。提示词按 seed 固定构造，持久化配方版本与 SHA-256，不写入 raw 日志。
- [x] **Step 5：运行 focused tests。**

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -q
```

预期：所有配置边界、token 精确性和超长拒绝测试通过。

## Task 3：实现 staged HTTP trial 与配对指标

**Files:** 修改 `benchmarks/course_project/long_prefill_decode.py`；修改 `test/course_project/test_long_prefill_decode.py`。

- [x] **Step 1：先写 fake-client 顺序测试。** 覆盖 Decode 先发、首个非空块触发且只触发一次 Prefill、Prefill 在线程池运行、Decode drain 不等待 Prefill、空块不触发，以及任一请求失败时 trial 记为 invalid。
- [x] **Step 2：运行单测确认失败。** 新增空块有效性与失败生命周期回归测试时观察到预期 RED；修复后 focused tests 通过。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -q
```

预期：新增 staged API 尚不存在时失败。

- [x] **Step 3：实现单请求和 staged 流程。** 添加 `run_single_trial(...) -> dict` 与 `run_staged_trial(...) -> dict`。staged 调度遵循以下顺序，确保回调不等待 Prefill，也不阻塞 Decode SSE drain：

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    prefill_future = None

    def submit_prefill(_field, _epoch_ns, _monotonic_ns):
        nonlocal prefill_future
        if prefill_future is None:
            prefill_future = executor.submit(client.chat, **prefill_kwargs)

    decode_future = executor.submit(
        client.chat,
        **decode_kwargs,
        on_first_chunk=submit_prefill,
        on_chunk=decode_events.append,
    )
    decode_result = decode_future.result()
    if prefill_future is None:
        raise ValueError("decode_stream_has_no_nonempty_chunk")
    prefill_result = prefill_future.result()
```

记录每条请求的发送/首块/drain 时间、所有 `chunk_events`、服务端 token 用量、状态和无效原因；不保留 `ChatResult.text`。调用 `ignore_eos=True` 后若预热仍不是 128 个输出 token，runner 必须中止此配置的正式测量。
- [x] **Step 4：实现配对摘要。** 对每轮同一条件的 chunk 4096/512 样本计算原始值、配对差、中位数和范围；Decode SSE 间隔按 Prefill HTTP send 时刻分为完全在前、完全在后、跨边界三类。字段名固定使用 `client_sse_chunk_gap_ms`，不输出 `ITL` 字段。
- [x] **Step 5：重跑 focused tests。**

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -q
```

预期：fake-client 证明 callback 与 drain 的顺序，不启动服务/NPU。

## Task 4：增加受控 runner 与 raw 记录

**Files:** 新建 `scripts/run_long_prefill_decode.py`；修改 `test/course_project/test_long_prefill_decode.py`。

- [x] **Step 1：测试 run 目录与启动安全。** mock `subprocess.Popen`、HTTP readiness、`capture_npu_smi` 和端口检查；覆盖 run 目录已存在、目标端口已占用、缺少 `--confirm-npu0-idle`、NPU0 usage 不可读或计算利用率非零、server readiness 超时等场景，全部必须在测量请求前停止；确保清理只终止本 runner 自己启动的进程。
- [x] **Step 2：运行安全测试确认失败。** 新增快照自定义密钥脱敏回归时观察到预期 RED。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -k 'output_directory or npu0 or port or readiness' -q
```

预期：runner 安全检查尚不存在时，新测试失败。
- [x] **Step 3：定义 CLI 参数。** `scripts/run_long_prefill_decode.py <config>` 接受 `--host`、`--port`、`--startup-timeout`、`--output-root`、`--dry-run` 和显式 `--confirm-npu0-idle`；默认输出根目录为项目内 `results/raw/benchmark/long_prefill_decode`。拒绝项目目录之外的 config、模型目录或 output root；模型与 tokenizer 只从已存在的项目内本地路径读取。用 UTC 微秒时间加随机 UUID 建立唯一 `<run_id>`，运行前检查目标 run 目录不存在；服务子进程 stdout/stderr 丢弃，`service-*.log` 仅保存固定启动状态和数值退出状态。
- [x] **Step 4：实现确定性配对轮次。** 基于 `build_serve_command` 生成两个 config 变体；每个 repetition 依次启动 `4096→512`、`512→4096`、`4096→512`；启动前保存 NPU0-only snapshot（复用 `capture_npu_smi`，该函数命令限定 `-i 0`），检查预期端口未被占用，并要求操作者确认 NPU0 空闲。`NPU Utilization`、Aicore/Aivector 使用率或 HBM bandwidth 使用率任一非零、usage 输出无法解析、现有服务/进程冲突或状态不明，都停止；不以 HBM 已分配容量单独判断忙闲，也不查询其他 NPU ID。每个服务先执行一次 256/128 Decode 和 3584/1 Prefill 预热；确认 `ignore_eos` 得到完整 128 tokens 后再跑五种测量条件，每种条件 3 个配对轮次。
- [x] **Step 5：保存原始记录。** `requests.jsonl` 每个 HTTP 请求一行，含 run/round/trial/config provenance、目标与实际 token 数、时间戳、逐块元数据和完成状态；`trials.jsonl` 每个 trial 一行，含条件、配对 ID、两条请求的生命周期、有效性原因及块间隔分类。写文件使用独占创建；已有 run 目录时失败，不续写旧数据。不得写入输出正文、授权头、密钥或提示词原文。服务子进程 stdout/stderr 使用 DEVNULL 丢弃；`service-*.log` 仅保存固定启动状态和数值退出状态；新增安全日志回归测试通过（65 passed, 1 skipped）。
- [x] **Step 6：验证 CLI dry-run 和 focused tests。**

```powershell
& ..\..\.conda-env\python.exe scripts/run_long_prefill_decode.py configs/bringup/qwen35_4b.yaml --dry-run
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode.py -q
```

预期：dry-run 打印 3 轮×2 配置的完整启动顺序和每配置五种 trial，不调用 `Popen`、HTTP、`npu-smi`，不创建 raw 目录；focused tests 全通过。

## Task 5：生成独立 CSV 与 SVG

**Files:** 新建 `scripts/analyze_long_prefill_decode.py`；新建 `test/course_project/test_long_prefill_decode_report.py`。

- [x] **Step 1：构造 fixture 与失败测试。** 使用 3 轮、两种 chunk、两种 Prefill 长度的有效 trial，并加入提前 EOS、失败请求和跨边界间隔样例；断言无效样本不进入配对统计，跨边界间隔不计入前/后段，CSV 由 `csv.DictReader` 可读、SVG 由 `xml.etree.ElementTree.parse` 可解析。
- [x] **Step 2：运行测试确认失败。** 对新增的指标篡改、gap 缺失/重复、首块元数据、模型别名和输出路径回归测试均观察到预期 RED。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode_report.py -q
```

预期：分析函数尚不存在时失败。

- [x] **Step 3：定义分析函数。** 在 `benchmarks/course_project/long_prefill_decode.py` 添加 `analyze_run(raw_dir, output_dir) -> list[Path]`；验证 raw schema、模型/revision/config 一致性。每个条件/指标至少有 3 个有效配对样本时才输出摘要统计和比较结论；不足时仍生成三项产物、保留原始观测并标记“数据不足”，不输出中位数/范围或比较结论。写 `paired_trials.csv`（逐轮数据与配对差）、`summary.csv`（中位数、范围、n、结论边界）、`decode_chunk_gaps.svg`（显示逐轮原始点与中位数，不画无样本的 p 值/置信带）。输出目录已存在则拒绝覆盖。
- [x] **Step 4：接入分析 CLI。** CLI 接收一个明确的项目内 raw run 目录和一个项目内不存在的输出目录，调用 `analyze_run` 并打印生成文件的项目内相对路径；传入项目外路径或已存在输出目录均安全失败。
- [x] **Step 5：验证 CSV/SVG。** 测试用 `csv.DictReader` 重新读取 CSV，并用 `xml.etree.ElementTree.parse` 解析 SVG；再运行 focused test：

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project/test_long_prefill_decode_report.py -q
```

预期：所有 CSV 行数/字段断言正确，SVG XML 合法，输出冲突时不覆盖。

## Task 6：全套本地验证与实验授权门

**Files:** 仅测试，不改 `task_plan.md`、`notes.md` 或 Profiler 结果文档。

- [x] **Step 1：运行课程项目测试。** 本地回归结果：399 passed, 6 skipped；实际远端项目现有课程测试：350 passed, 1 skipped。真实 tokenizer 的三种目标长度及 21 组正式 seed 格式输入通过，独立复核通过；使用项目 Conda 环境和项目内 pytest basetemp；没有启动推理服务/Profiler或安装新依赖。详见 `docs/project/long_prefill_preflight_20261012.md`。

```powershell
& ..\..\.conda-env\python.exe -m pytest test/course_project -q
```

预期：全部通过；无 NPU、Profiler、外网或新依赖安装。

- [x] **Step 2：执行 dry-run 并核验路径。** 运行 Task 4 的 dry-run；确认只打印 6 次配置启动顺序（3 轮×2 配置），不会加载模型、创建 raw/analysis 目录、启动服务、访问 NPU 或发送 HTTP 请求。
- [x] **Step 3：真实 NPU 运行前单独请求确认。** 用户已批准正式测量，并于启动失败根因核实后批准两组共同关闭 KVStore 的修正。已重新验证 NPU0 空闲及端口可绑定；本机测试 139 passed/5 skipped，远端 143 passed/1 skipped。新 run `20261013T030821237999Z-e4ba674feff4474abe765af68fcebfad` 已启动，验收未完成。任何 NPU0 忙、usage 不可读、端口/服务冲突、`ignore_eos` 未产生 128 tokens 时立即停止。不得启动 Profiler 或访问 NPU ID 1–3。
- [x] **Step 4：实测后验收。** 最终 run `20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4` 完成六次服务启动，30 个正式 trial 和 12 个预热全部 valid；五种条件各三轮配对。54 个请求实际 token 数/时间顺序、24 文件两端 SHA-256、三份报告本机重生成字节一致及 CSV/SVG 解析均通过。完成后 NPU0 无进程、四个端口可绑定。不把客户端块间隔称为 token ITL，不据此声称算子因果。验收与数据不足边界见 `docs/project/long_prefill_acceptance_20261013.md`。

## 自检清单

- Spec 中每项固定条件、三类请求和两种 chunk size、配对顺序、逐块时间戳、无效 trial、唯一输出目录、NPU0-only、Profiler 排除及 CSV/SVG 验收均由以上任务覆盖。
- `ignore_eos` 是正式测量的预检门；若目标服务版本不接受该 HTTP 扩展或预热长度不足，按计划停止，不改成短 Decode 样本。
- 所有测试沿用项目 Conda Python，不安装依赖；不提交改动，保留现有 dirty worktree。
