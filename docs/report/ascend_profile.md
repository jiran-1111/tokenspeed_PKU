# Ascend Profiler 分析

`scripts/run_profiler.sh` 对服务端 `/start_profile` 发起 `CPU`、`NPU` 和内存采集，并在压测结束后调用 `/stop_profile`。NPU trace 使用 TorchNPU 官方 `torch_npu.profiler.profile`；项目环境脚本将 `ASCEND_WORK_PATH` 设为项目内 `.tmp/ascend_work`，以保留 CANN profiler 侧产物。每个场景先预热 3 次，随后采集服务 Prometheus KV/prefix 指标，并保存 `npu-smi info` 原始快照。

采集场景为：

- `prefill-heavy`：本轮实际 summary 为输入/输出 3904/128 tokens、并发 2。
- `decode-heavy`：输入 256、输出 512、并发 8。
- `agent-concurrency`：共享 system/tool 内容、输入约 2K、并发 16。

每次运行前检查模型上下文上限。原始 trace 由 MindStudio Insight 或兼容的 Ascend 工具分析。前三种场景的 Level0 trace 尚未形成可信的 CPU overhead、NPU idle interval、通信或 HBM 结论，也没有可证实的 Top 3 瓶颈排序；后文 8-turn Level1 复采补充了可用的 AICore 计数，但仍不能建立算子与端到端长尾的因果关系或完成上述系统级归因。

## 设备采集结果（阶段性）

已在课程服务器、CANN 8.5.0、torch_npu 2.9.0 上，以 Qwen3.5-0.8B、TP=1 采集三种场景。原始字段为Chip Name Ascend910、Chip Version V1；2026-10-14依据实际SoC Ascend910_9362、CANN系列Ascend910_93及Atlas800I A3产品信息，撤回仅凭V1字段认定“非910C”的早期判断，软件一手来源支持A3/910C系列，详见[硬件身份核对](../project/hardware_identity_20261014.md)。原始字段和历史测量均不修改，不把型号更正当作性能或数值重新验收。

| 场景 | 目标负载 | 请求数 | trace 事件数 | `cpu_op` 数 | CANN `op_summary` 行数 | Task Duration p50 / p95 / max (μs) |
|---|---|---:|---:|---:|---:|
| prefill-heavy | 输入/输出 3904/128、并发 2 | 4 | 133,209 | 5,148 | 229,637 | 2.16 / 41.02 / 72,993.10 |
| decode-heavy | 输入/输出 256/512、并发 8 | 16 | 475,847 | 19,926 | 876,881 | 3.20 / 46.42 / 12,475.36 |
| agent-concurrency | 输入/输出约 2048/128、并发 16 | 32 | 287,229 | 9,220 | 417,188 | 4.38 / 81.68 / 38,931.74 |

trace 可见 CPU operator、enqueue/dequeue task-queue 事件和 async flow。名称中类似 `Dequeue@_zero_byte_ranges_kernel` 的事件仍属于 `dequeue` 类别：Torch-NPU 2.9.0 的解析器从 framework `OP_MARK` 的 enqueue/dequeue 起止标记构造这些事件，并用两端时间差计算队列事件 duration；`_kernel` 名称是关联的 operation 名称，不等于设备 kernel 执行时长。[对应解析器源码](https://github.com/Ascend/pytorch/blob/v7.3.1-pytorch2.9.0/torch_npu/profiler/analysis/prof_parse/_fwk_file_parser.py)

启用项目内 `ASCEND_WORK_PATH` 后，三场景均生成独立的 CANN `op_summary`、`task_time` 和 MindStudio 输出；本轮 CANN parser 日志未见 warning/error。此前未设置该工作路径的旧采集只有 Torch-NPU trace，因此旧结论应限定为旧采集，而不能描述本轮。CANN `op_summary` 中累计 Task Duration 最高的算子如下；这是跨行累计值，不是互不重叠的墙钟时间：

| 场景 | 算子 | 行数 | 累计 Task Duration (μs) |
|---|---|---:|---:|
| prefill-heavy | `_causal_conv1d_fwd_kernel` | 54 | 2,628,384.72 |
| prefill-heavy | `_causal_conv1d_update_kernel_4` | 4,608 | 1,433,042.36 |
| prefill-heavy | `chunk_fwd_kernel_o` | 90 | 895,067.98 |
| decode-heavy | `_causal_conv1d_update_kernel_4` | 17,838 | 11,802,689.82 |
| decode-heavy | `ViewCopy` | 23,808 | 1,063,338.56 |
| decode-heavy | `_fused_gdn_decode_update_kernel_0` | 17,856 | 1,014,764.18 |
| agent-concurrency | `_causal_conv1d_update_kernel_3` | 5,868 | 6,588,598.10 |
| agent-concurrency | `_causal_conv1d_fwd_kernel` | 252 | 1,427,357.72 |
| agent-concurrency | `_causal_conv1d_update_kernel_4` | 1,278 | 932,308.76 |

华为 CANN 8.5 将 `Task Duration(us)` 定义为任务调度至加速器、加速器执行及响应结束所覆盖的耗时；它不是纯 AICore 执行时间，且该轮 `Task Type` 均为 N/A。因此这些统计只用于提出后续验证候选，不构成 AICore 排名或已确认瓶颈。[官方 `op_summary` 字段说明](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0067.html) Torch-NPU trace 中类似 `Dequeue@_zero_byte_ranges_kernel` 的事件仍属于 framework task-queue 标记，不是 kernel 执行时长。[对应解析器源码](https://github.com/Ascend/pytorch/blob/v7.3.1-pytorch2.9.0/torch_npu/profiler/analysis/prof_parse/_fwk_file_parser.py)

虽然本轮已有 CANN `op_summary`/`task_time` sidecar，但当前 trace 中仍没有独立 NPU kernel timeline，Task Duration 也不能代替纯设备执行时长。采集未配置 `experimental_config`；远端 `torch_npu.profiler.profile` 默认 `profiler_level=Level0`、`aic_metrics=ACL_AICORE_NONE`，所以没有 AI Core hardware counters。flow 分位数同样不是 kernel duration，不能据此认定队列或 NPU 瓶颈。

以上是单轮、带 profiler 开销的观察结果，不与无 profiler 的性能测试直接比较。summary 中 TTFT p50 分别为 prefill 1427.42 ms、decode 390.42 ms、Agent 3510.23 ms；这些值不代表无 profiler 服务延迟。Agent trace 是并发 16 的 32 请求单轮场景，不是出现 TTFT 长尾的 8-turn 场景；三种 trace 仍只有 Level0 指标，不能判断 AICore 利用率、Host gap、通信或 HBM 瓶颈，也不能支撑优化收益结论。原始 trace、规范化副本、CANN 解析产物和逐事件审计见 [`results/profiles/qwen35-0.8b/ascend_work_path_fix_full/`](../../results/profiles/qwen35-0.8b/ascend_work_path_fix_full/) 与 [`notes.md`](../../notes.md)。

## 8-turn Agent 异常负载 Level1 复采（2026-10-10）

- 环境：课程服务器NPU0（历史原始Chip Version V1字段；型号解释按上文2026-10-14核对更正）、openEuler 24.03/aarch64、CANN 8.5.0、torch_npu 2.9.0；Qwen3.5-0.8B/TP=1，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。服务通过 `--npu-profiler-level Level1` 启动，并将可见设备限制为 NPU0。
- 匹配负载：512-token 上下文、共享前缀 50%、8 turns、并发 1/4、每档 3 次重复、每轮最大输出 32 tokens、客户端 mock wait 100 ms（`tool_wait_policy=all`）。共 120 条请求、6 份 summary、12 个 NPU 前后快照和 6 份 profile trace。等待在客户端模拟，不表示服务端进入 WAITING_TOOL 或保留 KV。
- 六份 `profiler_info_0.json` 均记录 `_profiler_level=Level1`、`_aic_metrics=ACL_AICORE_PIPE_UTILIZATION`。对应 `op_summary` 含 `aicore_time`、`aic_total_cycles`、MAC/MTE/Scalar 等字段；`Task Type` 中有 `AI_CORE`、`AI_VECTOR_CORE` 和混合类型，不再是 Level0 复采中的全 N/A。

| profile | CANN 采集时间戳 | `op_summary` 行数 | 非零 AICore 时间行 | 累计 AICore 时间最高的算子（μs） |
|---|---|---:|---:|---|
| c1 / r1 | `20261009191541020` | 214,790 | 27,448 | `chunk_fwd_kernel_o` — 201,748.69 |
| c1 / r2 | `20261009191623899` | 221,134 | 28,272 | `chunk_fwd_kernel_o` — 210,028.60 |
| c1 / r3 | `20261009191714887` | 221,134 | 28,272 | `chunk_fwd_kernel_o` — 209,700.12 |
| c4 / r1 | `20261009191758391` | 259,532 | 30,009 | `chunk_fwd_kernel_o` — 591,646.04 |
| c4 / r2 | `20261009191917471` | 272,000 | 31,102 | `chunk_fwd_kernel_o` — 657,059.42 |
| c4 / r3 | `20261009192038548` | 270,264 | 31,103 | `chunk_fwd_kernel_o` — 668,574.22 |

六个原始 CANN 目录（含 `profiler_info_0.json`、原始 profiling 数据库和 MindStudio `op_summary`/`task_time`）保存在 [`results/profiles/qwen35-0.8b/agent_level1_cann/`](../../results/profiles/qwen35-0.8b/agent_level1_cann/)；对应 Torch-NPU trace 位于 `results/profiles/qwen35-0.8b/<profile_id>/`。请求记录、summary 与硬件快照分别见 [`results/raw/agent/`](../../results/raw/agent/) 下 `qwen35-0.8b-20261009T191532Z*` 文件。38 个关键产物（6 trace、6 profiler info、12 张 CANN CSV、2 个 raw/summary、12 个硬件快照）的本地/远端 SHA-256 全部一致。

可在项目根目录运行 `python scripts/analyze_profiler.py`，从上述 Level1 CANN 原始产物重建描述性汇总；默认写入 [`results/analysis/agent_level1_cann_summary.json`](../../results/analysis/agent_level1_cann_summary.json)，不会修改输入 trace。脚本汇总正 `aicore_time` 任务行、同名算子的累计值、任务区间并集/间隙，以及跨 HBM channel 的原始带宽值。HBM 单位未在此独立核验；“任务未覆盖区间”不等于 NPU 空闲；`npu_mem.csv` 的 DDR/HBM 用量字段依 CANN README 在该产品上不受支持，故不报告。JSON 可用于复核下表 AICore 数字，但这些聚合仍不能建立单请求延迟归属或瓶颈因果关系。若要重新生成已存在的默认 JSON，需显式加 `--overwrite`；该选项仅替换汇总文件，不触碰原始采集。

这些表格中的 AICore 时间是 `op_summary` 对同名算子各任务行的累计值，不是互不重叠的墙钟时间，也不是整轮请求延迟。并发 4 的工作量是并发 1 的四倍，不能直接用累计时间比较效率；带 profiler 的延迟也不与无 profiler 基线混比。结果确认异常负载期间存在可测的 AICore 工作，并给出后续候选算子，但尚不能区分算子执行、调度/排队、Host gap 或其他因素对 TTFT 长尾的因果贡献；因此当前不据此选定 kernel/host 优化，也不宣称性能收益。

## Trace 到请求归因审计（离线）

对现有 c1/r1 trace 的抽查可见 5,036 个 `cpu_op`、39,812 个 enqueue 与 39,812 个 dequeue 事件，以及 79,624 个 `async_task_queue` flow；对应 CANN 文件可按 task ID、stream 和时间查看任务级 `Task Duration`、`Task Wait Time` 与 AICore 时间。这些字段尚未形成从单个请求 ID，经服务端阶段 span/队列交接，再到 CANN task ID 的可靠关联，因而不能量化它们对该请求 TTFT 的贡献。

同一 c1/r1 汇总的 task 时间包络为 5,415,831 μs、区间并集为 3,199,867 μs、未覆盖约 2,215,965 μs；“未覆盖”仅描述记录区间，不表示 NPU idle，也不能单独归因于 Host 或排队。代表 trace 位于 [`c1/r1 Torch-NPU trace`](../../results/profiles/qwen35-0.8b/20261009T191532Z-c1-p0.5-t512-n256-r1/20261009-191541-TP0.trace.json)，汇总来源为 [`Level1 CANN summary`](../../results/analysis/agent_level1_cann_summary.json)。

若要完成因果归因，最小后续采集应固定输入、输出、并发与 profiler 配置，为单请求分配唯一 ID，并把 ID 贯穿服务阶段 span、enqueue/dequeue 与 CANN task/stream；重复采集后再一次只改一个调度因素或候选实现。缺少这条关联链之前，只能报告任务级观察，不能写成已确认瓶颈。

## 单请求 TTFT 时间窗（2026-10-13，限定观察）

0.8B/BF16/TP1、单会话八轮，仅第八轮 max_tokens=1 开启 Level1/PipeUtilization。实际请求输出 1 token、正常 SSE 结束，Profiler start/stop 成对。带 Profiler 的 TTFT 为 260.864 ms；时钟核验通过，最大误差 800 ns。TTFT 窗内 Torch 事件 41871、CANN interval 3296；per-event overlap sum 为 798636.98/138195.29 us，重叠区间会重复计时，不能把前者大于 TTFT 解释为性能错误或互斥时长分解。

Raw/profile/analysis 共145文件跨端 SHA-256 一致，本机重新生成 CSV 字节一致，实际 profiler_info 确认 Level1/PipeUtilization。[采集与验收记录](../project/profiler_single_request_capture_20261013.md)提供路径及导出定位。这只补齐可靠时间窗，不建立请求到算子因果关联；不与无 Profiler 性能基线比较。
