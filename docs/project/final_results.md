# Final Results

## 已验证的 bring-up 结果（非最终性能结果）

- 设备：Ascend910 V1（课程 910C 的用户批准替代设备）；Qwen3.5-0.8B，BF16，TP=1，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。
- 修复：NPU Qwen3.5 attention fallback 按 query head 拆分交错的 Q/gate 投影；修复前同一提示词重复输出句点，修复后返回预期的 `OK`。
- 验证：默认 graph-mode 服务启动与预热成功，API 返回 HTTP 200；相关 Q/gate、MHA、GDN 测试 11 项通过。
- 原始响应：[`results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json`](../../results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json)。

这只是单提示词 smoke test，不等同于完整 token-ID 正确性验收或性能实验。

## 指标与正确性采集 smoke（2026-10-08）

- TokenSpeed 指标端点 `:39142/metrics` 已从引擎进程实际读到 `tokenspeed:*` 指标；网关 `:39141/metrics` 不包含这些引擎指标。
- 8 条中文、英文、数学、代码、长上下文和多轮请求已保存至 [`results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl`](../../results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl)。该脚本只采集对话结果，`output_token_ids` 仍不可得，`token_correctness_verified=false`，不记为完整正确性通过。
- 4 请求遥测 smoke 的 summary：TTFT p50 92.89 ms、TPOT p50 10.52 ms、请求吞吐 2.40 req/s；prompt/generation/prefix-hit token delta 分别为 516/128/512。KV usage 采样发生在请求结束后，为 0.0；不能当作峰值或正式结果。可追溯原始数据：[`results/raw/metrics_smoke/qwen35_0.8b_20261008_run17/`](../../results/raw/metrics_smoke/qwen35_0.8b_20261008_run17/)。
- 新增负载期采样后，8 请求/concurrency 4 的设备 smoke 记录 KV usage peak 0.53125、running peak 5；summary/raw 见 [`results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18/`](../../results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18/)。其余 TTFT p50 173.35 ms、TPOT p50 20.62 ms 和吞吐 1.08 req/s 仅用于检查遥测链路，不是正式基线。
- 2026-10-09 首轮 0.8B 普通并发 baseline：目标输入/输出 128/32，concurrency 1/4/8/16，各 3 轮、每轮 32 请求。提示词现按 request id 生成等长变体，实测 prefix hit ratio 为 0。384 条 raw 请求、12 份 summary 和 NPU 前后快照见 [`results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/`](../../results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/)。每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.41/228.89 | 10.48 | 1.82 | 58.17 | 0.12 |
| 4 | 556.17/6297.53 | 17.80 | 2.14 | 68.60 | 0.61 |
| 8 | 884.12/1296.26 | 42.84 | 3.81 | 121.95 | 0.91 |
| 16 | 2760.82/3530.22 | 56.42 | 3.61 | 115.49 | 0.97 |

  并发 16 相对 8 没有增加吞吐，且 KV 接近满载；并发 4 p95 有显著长尾。该结果是 910 V1 上的 0.8B 首轮基线，不是 910C 结果或优化收益。ITL 仍为 null。
- 普通 benchmark 曾有一轮所有请求重复相同 prompt 的试跑，prefix hit ratio 约 0.99；保留在 [`results/raw/baseline/qwen35_0.8b_20261009/`](../../results/raw/baseline/qwen35_0.8b_20261009/)，仅作热共享前缀样本，不混入上述 baseline。
- 0.8B Agent 首轮：并发 1/4/8、共享比例 0/25/50/75/90%、两轮会话、每轮 3 次重复，共 390 条请求。TTFT 波动很大且共享比例曲线不单调，因此只作为工作负载与采样链路 pilot，不据此得出优化结论。数据见 [`results/raw/agent/qwen35_0.8b_20261009_unique_prefix/`](../../results/raw/agent/qwen35_0.8b_20261009_unique_prefix/)。
- 0.8B 8-turn Agent 复测已完成：prefix 512 tokens、并发 1/4、共享比例 0/50/90%、每组 3 次重复，共 360 条请求。50% 组合 TTFT p50 中位数在并发 1/4 下分别为 4776/4238.8 ms，而 0%/90% 约 182–430 ms；曲线非单调且异常重复出现，需 Profiler/更细粒度 scheduler trace 解释，不作为 prefix-cache 优化结论。raw/summary 与硬件快照见 [`results/raw/agent/qwen35_0.8b_20261009_8turn/`](../../results/raw/agent/qwen35_0.8b_20261009_8turn/)。tool-wait 在该 runner 中是客户端延迟，不代表服务端保留待工具 session。
- 项目原有课程/CLI/Q-gate/metrics 回归此前 68 项通过；最新 `test/course_project` 本机运行 61 项（60 通过、1 项因 Windows 无 Bash 跳过），远端项目环境 61 项全部通过。远端复测包含项目内 `ASCEND_WORK_PATH` 环境测试。本机项目前缀 Conda 未装 pytest，因此使用已有 Python 测试运行时，未安装依赖。
- baseline/Agent/ablation/profiler runner 均支持负载中采集 KV usage 和 running/waiting 峰值；已在 0.8B baseline 与 Agent 负载中使用。
- 0.8B/TP=1 E0–E3 重复消融已在 chunk=512 和 1024 两组完成：输入/输出目标 2048/32、并发 4、每变体 12 请求×3 轮。每组 144 条请求、12 条 summary；两组 raw、日志与硬件快照各 30 个文件均跨端 SHA-256 一致。E3 prefix-hit ratio 两组均约 0.965，但 chunk=512 首轮 TTFT p50 为 5.92 s、后两轮约 0.28–0.29 s，显示显著轮间差异；完整表格和限制见 [`docs/report/ablation_analysis.md`](../report/ablation_analysis.md)。这是重复提示词负载下的描述性测量，不是普遍性能结论。
- Qwen3.5-4B bring-up：两分片 SHA-256、离线配置读取与 safetensors header 检查通过；TP=1 服务完成 warmup，`/v1/models` 与引擎 metrics HTTP 200。6 个 correctness case 共 8 条响应已保存至 [`results/raw/correctness/qwen35-4b_20261009_run1.jsonl`](../../results/raw/correctness/qwen35-4b_20261009_run1.jsonl)。这只是 chat/API smoke：引擎输出 token IDs 不可得，`token_correctness_verified=false`，不纳入性能比较。
- Qwen3.5-4B 原始 token-ID 对照：[`TokenSpeed 记录`](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1.jsonl)中的 8 条生成均与同 revision 的 Transformers greedy 输出完全匹配（8/8；每条 64 tokens）。输入 token IDs 相同，比较结果及首次差异索引见[对照 JSON](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1_reference_compare.json)。结论限于 4B/TP1/910 V1 和这组固定样例，不代表其他模型或参数已验收。
- 报告脚本从 raw summary 自动生成 CSV/SVG。原有 `results/plots/` 中保留 summary.csv 与 5 张 baseline/Agent 图；[`results/plots/ablation_20261009/`](../../results/plots/ablation_20261009/) 含 summary CSV 和 7 张消融图；新增 [`results/plots/after_9b_baseline_20261009/`](../../results/plots/after_9b_baseline_20261009/) 含最新 summary CSV 和 7 张图，baseline 图例按模型分开。未运行的持久 Session reuse、KV block sweep 和 TP 对比图不会用单点伪造趋势。

0.8B 的 prefill-heavy、decode-heavy、agent-concurrency 三种 Profiler 场景已在 Ascend910 V1 上重采。项目内 `ASCEND_WORK_PATH` 修复后，每种场景均有 CANN `op_summary`/`task_time` 及 MindStudio 输出，CANN parser 日志未见 warning/error。`op_summary` 可用于描述 Task Duration 和候选算子，但它包含任务调度至设备、设备执行及响应结束耗时；不是纯 AICore 时间。本轮 `Task Type` 均为 N/A，未启用 AICore hardware counters，故仍不能认定设备瓶颈、解释 8-turn TTFT 异常或声称优化收益。详细数据、逐算子值与原始产物见 [`Profiler 分析`](../report/ascend_profile.md) 和 [`results/profiles/qwen35-0.8b/ascend_work_path_fix_full/`](../../results/profiles/qwen35-0.8b/ascend_work_path_fix_full/)。

4B 的固定样例 token-ID 对照已通过。0.8B 两轮固定输入的 engine IDs 逐条一致；按一致 EOS/stop 口径对齐后 6/8 token 序列匹配，长上下文和多轮第 3 轮各有一条可复现正文差异。9B 已完成 bring-up 与首轮 baseline，但 token-ID 对照为 stop-aligned 6/8，也有两条可复现正文差异。0.8B/9B 均不作为完整正确性通过。采集与对照见 [`0.8B 原始 IDs`](../../results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1.jsonl)、[0.8B 第二轮](../../results/raw/correctness/qwen35-0.8b_20261009_native_ids_run2.jsonl)、[0.8B stop-aligned 对照](../../results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1_stop_aligned_reference_compare.json) 与本页的 9B 记录。27B token-ID 对照与 bring-up、27B baseline、四模型完整正确性、逐 token ITL、可用的 kernel-level Profiler、TP=2 和跨模型消融仍未完成。当前性能数据覆盖 0.8B/4B/9B 首轮并发 baseline、4B Agent 客户端 Tool-Wait 描述性测量、0.8B Agent pilot 及固定重复提示词下的 E0–E3 描述性消融；这些结果不等同于优化收益。Profiler/4B smoke 负载指标不纳入性能结论。

## Qwen3.5-4B baseline（2026-10-09）

- 设备/环境：课程服务器 Ascend910 V1（用户批准的 910C 替代设备），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。权重已完成校验，服务 warmup 与 API/engine metrics smoke 通过。
- baseline 配置：输入/输出目标 128/32 tokens，32 条独立 prompt/轮，prefix-hit ratio 为 0，并发 1/4/8/16、各 3 轮，共 384 条请求、12 份 summary、24 份 NPU 前后快照。`max_num_seqs=8`，因此 concurrency=16 会排队。
- 每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 261.96/262.90 | 19.54 | 1.15 | 36.70 | 0.31 |
| 4 | 895.02/914.42 | 33.56 | 2.05 | 65.74 | 0.81 |
| 8 | 772.55/1700.51 | 84.82 | 2.33 | 74.54 | 1.00 |
| 16 | 3972.56/5011.11 | 82.99 | 2.37 | 75.88 | 1.00 |

- c8 到 c16 输出吞吐只从 74.54 增至 75.88 tok/s；c16 TTFT p50 升至约 3.97 s，且配置最多运行 8 个 sequence，故该负载下 c16 有排队，不能据此声称扩展或优化收益。ITL 未采集，仍为 null。
- 原始请求、summary 与硬件快照见 [`results/raw/baseline/qwen35_4b_20261009_full/`](../../results/raw/baseline/qwen35_4b_20261009_full/)；跨端同步后的 JSONL 大小与 SHA-256 一致。

## Qwen3.5-9B baseline（2026-10-09）

- 设备/环境：课程服务器 Ascend910 V1（用户批准的 910C 替代设备），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。四个 safetensors 分片均通过官方 manifest 文件大小与 SHA-256 校验；离线 `AutoConfig` 和权重索引/header 检查通过。服务 warmup 成功，readiness、`/v1/models` 和 TokenSpeed engine metrics 均 HTTP 200。
- baseline 配置：输入/输出目标 128/32 tokens，32 条独立 prompt/轮，prefix-hit ratio 为 0，并发 1/4/8/16、各 3 轮，共 384 条请求、12 份 summary、24 份 NPU 前后快照。`max_num_seqs=8`，因此 concurrency=16 的请求会排队，不等同于 16 条同时运行。
- 每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 276.91/279.19 | 24.79 | 0.95 | 30.49 | 0.25 |
| 4 | 924.27/4634.49 | 38.76 | 1.54 | 49.32 | 0.81 |
| 8 | 803.42/1770.21 | 91.84 | 2.17 | 69.41 | 1.00 |
| 16 | 4261.56/5351.97 | 91.13 | 2.20 | 70.48 | 1.00 |

- c8 到 c16 输出吞吐仅由约 69.41 增至 70.48 tok/s，同时 KV 峰值已满且 c16 TTFT p50 显著升高；这只是该 9B/TP1/910 V1 配置与负载的容量观察，不构成优化收益或其他硬件上的结论。ITL 未采集，仍为 null。
- 原始请求、summary 与硬件快照见 [`results/raw/baseline/qwen35_9b_20261009_full/`](../../results/raw/baseline/qwen35_9b_20261009_full/)。

## Qwen3.5-9B token-ID correctness（2026-10-09）

- 六个固定用例产生 8 条请求；每条最多生成 64 tokens。两轮采集中 8/8 输入 token-ID 与 8/8 引擎输出 token-ID 完全一致，说明该设置下结果可复现。
- 与同 revision 的 Transformers greedy reference 比较，stop-aligned 仅 6/8 匹配。`long_context_recall` 与 `multi_turn_memory` 分别在输出位置 59、58（0-based）出现正文差异；两条均达到 64-token 上限。故不能将 9B 判定为完整 token-level correctness 通过。
- 原始记录及比较结果见 [`run 1`](../../results/raw/correctness/qwen35-9b_20261009_native_ids_run1.jsonl)、[`run 2`](../../results/raw/correctness/qwen35-9b_20261009_native_ids_run2.jsonl) 和[reference 对照](../../results/raw/correctness/qwen35-9b_20261009_native_ids_run1_reference_compare.json)。

## Qwen3.5-4B Agent Workload F（2026-10-09）

- 环境：课程服务器 Ascend910 V1（910C 替代设备），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。只使用默认 NPU 0，NPU 2 未参与。
- 工作负载：共享前缀目标 512 tokens，8 轮对话，最大输出 32 tokens；并发 1/4、共享比例 0/0.5/0.9，各配置 3 次重复。共 360 条请求、18 份 summary、36 份硬件快照。每条配置的下表数字为三次 summary 的中位数；吞吐量为 output tokens/s，包含客户端等待时间。

| 并发 | 共享比例 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 2460.21 / 4101.12 | 19.59 | 10.74 | 0.281 |
| 1 | 0.5 | 3967.35 / 6208.60 | 19.60 | 7.70 | 0.312 |
| 1 | 0.9 | 385.10 / 3068.80 | 19.65 | 20.15 | 0.438 |
| 4 | 0.0 | 2842.40 / 6628.36 | 53.75 | 25.57 | 1.000 |
| 4 | 0.5 | 2963.37 / 9371.43 | 74.74 | 18.34 | 1.000 |
| 4 | 0.9 | 3639.11 / 8593.39 | 71.97 | 18.49 | 1.000 |

- `one-session` 共影响 126 条请求：只有 `session-0` 在首轮后的 7 轮请求前等待 100 ms，并在 user 文本中收到固定 mock 结果。该结果是客户端普通文本，不代表服务端 WAITING_TOOL 状态或 KV 保留。请求级 TTFT/TPOT/E2E 不含此前 sleep；wall time 与输出吞吐包含等待。
- TTFT 随共享比例明显非单调，且并发 4 的 KV 峰值为 1.0；不据此声称 prefix-cache 收益或性能优化。原始数据见 [`raw requests, summaries, and hardware snapshots`](../../results/raw/agent/qwen35_4b_20261009_tool_wait/)，图表与汇总表见 [`Workload F report`](../../results/plots/after_4b_agent_tool_wait_20261009/)。38 个跨端同步文件的 SHA-256 全部一致。

本文件只接受可追溯到 `results/raw/` 的测量值，并同时记录 TokenSpeed commit、模型 revision、启动参数、硬件/软件环境、并发与输入输出长度分布。
