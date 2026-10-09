# Experiments

## 已完成

- GitHub 代理连通性与官方仓库提交固定。
- 源码级 feature inventory 和请求路径追踪。
- 项目本地 Conda 环境与缓存路径验证。
- 四模型启动配置和 serving dry-run。
- 确定性 chat correctness 客户端、普通并发/Agent workload runner、p50/p95/mean 报表代码。
- Ascend TorchNPU profiler 接口适配；在 910 V1/0.8B/TP1 采集三场景 trace。首次采集只有 framework trace；补设项目内 `ASCEND_WORK_PATH` 后重采，三场景均获得 CANN `op_summary`/`task_time` 和 MindStudio 产物。任务时长统计与限制见 [`Profiler 分析`](../report/ascend_profile.md)。
- E0–E3 upstream 开关消融运行器、预热/重复和结果/服务日志保存；只切换当前 Ascend 可用的 overlap、chunked prefill 与 prefix cache，mixed batch 保持关闭，KVStore 在所有变体中统一关闭。
- 0.8B/TP1 E0–E3 功能 pilot 已完成：2048 输入/16 输出、并发 2、每变体 4 条请求、单轮；E2/E3 chunk=512。记录见 [`results/raw/ablation/qwen35_0.8b_pilot_20261009/`](../../results/raw/ablation/qwen35_0.8b_pilot_20261009/)。启动、请求和遥测链路跑通，但单轮小样本且 prompt 重复，仅用于流程检查，不据此得出性能结论。
- 0.8B/TP1 E0–E3 重复消融已完成两组：2048/32 tokens、并发 4、每变体 12 请求×3 轮，E2/E3 分别 chunk=512/1024。每组 144 条 raw、12 份 summary；30 个文件跨端 SHA-256 全匹配。数据及轮次波动见 [`消融分析`](../report/ablation_analysis.md)；固定重复 prompt 使其只代表共享前缀工作负载，不作普遍性能结论。
- Prometheus KV/prefix/scheduler telemetry 解析，并采集可用的 `npu-smi info` 原始快照。
- 模型与客户端 tokenizer 固定为同一个官方完整 commit SHA，并写入各类原始结果和 summary。
- 普通 Benchmark 默认输入为 128/1024/2048 tokens；发送前检查 chat template 后的输入加输出预算是否超出配置上下文。显式指定的超长 workload 会报错，不截短输入；8K/16K 实验需同步提高服务与配置的上下文上限。
- 远端项目环境课程测试、CLI 参数路由、Q/gate 布局和 metrics sampler 回归此前 68 项通过（4 个 CANN 安装目录 owner 警告）；随后为 benchmark 去重、Agent 重复轮次隔离、报告、profiler controller timeout、消融约束和项目内 `ASCEND_WORK_PATH` 补测试。最新 `test/course_project` 本机运行 61 项（60 通过、1 项因 Windows 无 Bash 跳过）；远端运行 61 项全部通过。本机使用已有 Python 测试运行时，未安装依赖。
- Ascend910 V1 上的 Qwen3.5-0.8B 确定性 API smoke test：默认 graph-mode 服务启动、预热成功；提示词 `Reply with exactly the word OK.` 返回 `OK`（HTTP 200）。此前同一请求输出句点；原因是 NPU 分支对 Q/gate 使用了全局二分，而模型布局为每个 query head 内交错。新增 per-head split 后，Q/gate、MHA、GDN 聚焦测试共 11 项通过。响应原始记录见 `results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json`。
- 0.8B correctness runner 已执行 8 条中文/英文/数学/代码/长上下文/多轮请求并保存响应，见 `results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl`。它是记录采集，不是自动 pass/fail，也未验证引擎原始 token ID。
- 2026-10-09 Qwen3.5-4B bring-up：固定 revision 的两个 safetensors 分片均以 SHA-256 校验通过，离线 `AutoConfig` 与分片 header 可读。隔离到空闲 NPU 0 后，TP=1 服务完成 load/warmup，`/v1/models` 和引擎 metrics 均 HTTP 200。correctness collector 记录 6 个用例、8 条请求，见 `results/raw/correctness/qwen35-4b_20261009_run1.jsonl`；引擎原始 output token IDs 不可得，`token_correctness_verified=false`，因此仅算 chat/API smoke，不算完整正确性或性能结果。服务日志在 `.tmp/qwen35_4b_serve_20261009_run1.log` 与 `.tmp/qwen35_4b_correctness_20261009_run1.log`。
- 2026-10-09 Qwen3.5-0.8B/TP=1 token-ID 对照：8 条固定输入均采集到原始 engine IDs。原始数组逐个完全相等 1/8；统一 model/tokenizer EOS/stop 规则后 6/8 完全匹配。第二轮复测的 8/8 输入 IDs 和 8/8 engine 输出 IDs 与第一轮完全一致，因此长上下文第 32 个输出 token、多轮第 3 轮第 2 个输出 token 的正文差异可复现，不判定为通过。两轮采集及比较文件见 `results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1.jsonl`、`qwen35-0.8b_20261009_native_ids_run2.jsonl`、`qwen35-0.8b_20261009_native_ids_run1_stop_aligned_reference_compare.json`；首次未对齐 stop 的诊断文件也保留。
- 后续原生 `/generate` 采集见 [`qwen35-4b_20261009_native_ids_run1.jsonl`](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1.jsonl)：6 个用例共 8 条记录，包含真实引擎输出 token IDs。与同一 pinned revision 的 Transformers greedy 参考逐条比较，8/8 序列完全一致；每条生成 64 tokens（达到 `max_new_tokens=64` 上限），输入使用同一 chat-template token IDs，参考侧显式传入全 1 attention mask。比较详情见 [`qwen35-4b_20261009_native_ids_run1_reference_compare.json`](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1_reference_compare.json)。这只验收 4B/TP1/910 V1 上的这 8 条样例，不等于四模型或全场景正确性完成。
- 2026-10-09 Qwen3.5-9B bring-up：固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` 的四个权重分片均通过官方 manifest 大小/SHA-256 核验；离线 `AutoConfig` 与全部 775 个 safetensors tensor header 可读。Ascend910 V1/NPU 0 的 TP=1 服务完成 warmup，readiness、`/v1/models` 和引擎 metrics 均 HTTP 200。原始 token-ID 对照两轮的 8/8 输入与引擎输出数组完全复现；相对 pinned Transformers greedy 参考为 stop-aligned 6/8，`long_context_recall` 与 `multi_turn_memory` 分别在输出位置 59、58（0-based）出现可复现正文差异，因此不记为正确性通过。记录见 `results/raw/correctness/qwen35-9b_20261009_native_ids_run1.jsonl`、`qwen35-9b_20261009_native_ids_run2.jsonl` 与 `qwen35-9b_20261009_native_ids_run1_reference_compare.json`。
- 2026-10-09 Qwen3.5-9B/TP=1 baseline：输入/输出目标 128/32、独立 prompt、prefix-hit ratio 0，并发 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求、12 summaries 和 24 份硬件快照。设备为 Ascend910 V1，配置 `max_num_seqs=8`，所以 concurrency=16 会排队；c8/c16 的 KV 峰值均为 1.00，吞吐约 69.41/70.48 output tok/s，不能解释为 16 并发带来有效扩展。详细中位数见 [`9B baseline`](final_results.md#qwen35-9b-baseline-2026-10-09)；raw 数据见 `results/raw/baseline/qwen35_9b_20261009_full/`。这是单设备/单配置基线，不是优化收益，逐 token ITL 未采到。
- 2026-10-09 Qwen3.5-4B/TP=1 baseline：输入/输出目标 128/32、独立 prompt、prefix-hit ratio 为 0，并发 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求、12 summaries 和 24 份硬件快照。设备为 Ascend910 V1，配置 `max_num_seqs=8`；c16 排队，c8/c16 KV 峰值均为 1.00，输出吞吐中位数为 74.54/75.88 tok/s，c16 TTFT p50 升至约 3.97 s，不作为扩展或优化结论。详细指标见 [`4B baseline`](final_results.md#qwen35-4b-baseline-2026-10-09)；raw 数据见 `results/raw/baseline/qwen35_4b_20261009_full/`。同步后的 26 个文件已核对大小和 SHA-256；ITL 未采集。
- 修正指标采集路径：`--metrics-port 39142` 现在暴露 TokenSpeed 引擎 Prometheus 指标；`--prometheus-port 39141` 是网关自身指标。0.8B 真实端点返回 KV/scheduler/prefix-cache 样本。对应实现经过 66 项远端测试及服务 smoke 验证。
- 指标 smoke：4 个请求、输入目标 128、输出 32，TTFT p50 92.89 ms、TPOT p50 10.52 ms、请求吞吐 2.40 req/s；采到 prompt 516、generation 128、prefix-cache hit 512 tokens。KV usage 是工作负载结束后的瞬时值 0.0，不代表运行峰值。结果见 `results/raw/metrics_smoke/qwen35_0.8b_20261008_run17/`，此小样本不作正式性能结论。
- 峰值采样设备 smoke：8 请求、concurrency 4、输入目标 128、输出 32，TTFT p50 173.35 ms、TPOT p50 20.62 ms、吞吐 1.08 req/s；采到 KV usage peak 0.53125、running peak 5、prompt/generation/prefix-hit delta 1033/257/1024。结果见 `results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18/`。仅验证采集链路；小样本不作正式性能结论。
- 普通 baseline pilot `run14` 包含 144 条请求（输入目标 128、输出 32，concurrency 1/8/16，各 3 轮），但当时未接通 TokenSpeed 指标且 c8/c16 重复轮次方差很大，仅保留原始记录，不纳入正式 baseline。
- 2026-10-09 完成 Qwen3.5-0.8B 首轮并发 baseline：128/32 tokens，concurrency 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求；引擎指标及 NPU 前后快照均已记录。修正了普通 benchmark 各请求复用同一 prompt 的问题，现为等长独立 prompt；prefix hit ratio 为 0。summary/raw 见 `results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/`，结果表和 910 V1 设备说明见 `docs/project/baseline.md`。修正前的重复全文样本保留在 `results/raw/baseline/qwen35_0.8b_20261009/`，prefix hit ratio 约 0.99，不纳入普通 baseline。
- Agent workload runner 已改为每个重复轮次使用不同的共享/私有上下文，同时保持同一轮内的共享前缀；回归测试覆盖“轮内共享、轮间隔离”。0.8B 两轮 pilot 已完成 390 请求（concurrency 1/4/8、prefix 0/25/50/75/90%）；另完成 8-turn 复测 360 请求（concurrency 1/4、prefix 0/50/90%、512-token context、100 ms 客户端 tool-wait）。两组数据都有显著 TTFT 长尾，尤其 50% prefix 在 8-turn c1/c4 的 TTFT p50 分别约 4.78/4.24 s，且曲线非单调；数据保留待 Profiler 分析，不能宣称 prefix cache 收益。原始数据见 `results/raw/agent/qwen35_0.8b_20261009_unique_prefix/` 与 `results/raw/agent/qwen35_0.8b_20261009_8turn/`。
- Workload F（Qwen3.5-4B/TP=1，Ascend910 V1）指定命令：

  ```bash
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

  `one-session` 仅让 `session-0` 在首轮后的每轮请求前等待，并在其 user 文本中附加固定 mock tool result；它是客户端注入的普通文本，不代表服务端工具协议、请求挂起或 KV 保留。单条请求的 TTFT、TPOT、E2E 从请求发出后计时，不包含此前客户端 sleep；每轮 wall time 与据此计算的吞吐量包含该等待。
- Workload F 已完成：360 条请求、18 份 summary、36 份 NPU 前后快照；126 条请求仅对 `session-0` 注入固定结果并等待 100 ms。原始数据见 `results/raw/agent/qwen35_4b_20261009_tool_wait/`，报告见 `results/plots/after_4b_agent_tool_wait_20261009/`。跨端同步的 38 个文件 SHA-256 全部一致。三轮中位数和结论见 [`Qwen3.5-4B Agent Workload F`](final_results.md#qwen35-4b-agent-workload-f-2026-10-09)。结果曲线非单调，且并发 4 的 KV 峰值达到 1.0；该描述性测量不作为 prefix-cache 优化收益。
- 自动报告读取 `results/raw/**/*.summary.jsonl` 生成 CSV/SVG；按实验类别隔离 baseline 与 Agent 数据，对重复轮次取中位数，并仅在同一配置组存在至少两个横轴取值时生成趋势图。新增矩阵归属后，`results/plots/ablation_20261009/` 生成 summary CSV 与 7 张 SVG，包括 E0–E3 消融图和按 E2/E3 变体隔离的 chunk 512/1024 对比。`results/plots/after_9b_baseline_20261009/` 与最新 `results/plots/after_4b_baseline_20261009/` 均保留独立输出；最新 baseline 图将 0.8B、4B、9B 分成不同模型序列。持久 Session reuse、KV block 与 TP 对比仍无真实 sweep 数据，不生成趋势图。
- 2026-10-09 Profiler 复采：设置项目内 `ASCEND_WORK_PATH` 后，以三 warmup 重采 prefill/decode/Agent 场景，CANN `op_summary` 分别有 229,637/876,881/417,188 行，parser 日志未见 warning/error。CANN `Task Duration` 覆盖调度、加速器执行和响应结束，不是纯 AICore 时间；`Task Type` 为 N/A，且单轮数据不能确认瓶颈。旧 trace-only 采集的 flow 数字不与本轮混用。原始/规范化 trace 与 CANN 产物见 `results/profiles/qwen35-0.8b/ascend_work_path_fix_full/`，分析见 `docs/report/ascend_profile.md`。
- Conda 激活脚本已兼容非登录 SSH PATH；在远端剥离 PATH 的 shell 中成功定位 `/opt/miniconda3`、激活项目 `.conda-env` 并通过课程测试。
- `ASCEND_WORK_PATH` 已在 `scripts/use_project_env.sh` 固定到项目 `.tmp/ascend_work`；远端专用环境测试先失败后通过，并完成服务级三场景 CANN profiler 复采。最新 `test/course_project` 本机运行 61 项（60 通过、1 项按平台跳过），远端 61 项全部通过。

这条 smoke test 仅验证一个提示词的服务端到端生成，不替代完整 token-ID 对照，也不是性能数据。测试结束后已停止服务，NPU 进程清空。

## 尚未验收

### 正确性记录口径

此前 0.8B/4B 的 chat JSONL 仍保留历史口径：`output_token_ids` 为 null，不能追溯补成 token 级通过。当前 `run_correctness.py` 将客户端 chat template 的 `input_ids` 原样提交到 TokenSpeed 控制端口的原生 `/generate`（服务主端口 +1），分别记录引擎原始 `output_token_ids` 与响应文本重新编码得到的 `retokenized_output_ids`。4B 已完成 8 条同输入 greedy 参考对照，结果见上方文件；该采集 JSONL 中的 `token_correctness_verified` 表示采集时尚未比较，最终判断以单独比较文件为准。
4B 比较限定为 910 V1、TP=1、pinned revision、6 个固定用例/8 次生成、每次 64 tokens；不能外推到其他模型、输入长度或参数。0.8B 与 9B 的 stop-aligned 对照均为 6/8，且各有两条可复现正文差异；27B token-ID 对照、incremental decode 与 KV 复用测试仍未验收。原生非流式生成不提供 TTFT/TPOT/chunk latency，这些字段留空；延迟测量使用独立流式 benchmark。

### 流式延迟

OpenAI chat 客户端的 `inter_chunk_latencies_ms` 是非空文本块在客户端的到达间隔，
`chunk_latency_p50_ms/p95_ms` 是其分位数。上游 `generation_output_processor.py`
支持可变 `stream_interval`；文本解码和传输也可能合并多个 token，因此不能将这些间隔
当作逐 token ITL。目前 `itl_p50_ms/p95_ms` 输出 null，真正的 ITL 验收仍未完成，
需要逐 token 事件或设备端时间戳。TTFT 同样是首个非空文本块的到达时间；TPOT
目前按首个文本块到流结束的耗时除以 usage 中的输出 token 数减一计算，含流结束开销。
测试覆盖了 chunk latency 有观测值时 ITL 仍保持 null 的行为。

本机 Windows 没有 NPU。课程服务器实际为 openEuler 24.03/aarch64、Ascend910 V1（用户批准的 910C 替代设备），项目专属 Conda 环境已建立，Triton-Ascend kernel 验证与课程/NPU adapter 测试通过，四个服务配置 dry-run 通过。0.8B 已完成 API/指标 smoke、首轮普通并发 baseline、两组 Agent workload、三场景 Profiler trace、两轮可复现 token-ID 对照（stop-aligned 6/8）及 E0–E3 的 512/1024 chunk 重复消融。4B 权重已校验、8 条固定样例 token-ID 全匹配，且首轮 384 请求 baseline 已完成；9B 已完成权重校验、TP=1 服务 smoke、两轮 token-ID capture（参考对照 6/8）和首轮 384 请求 baseline。Profiler 已获得 CANN Task Duration，但不是纯设备执行时间且没有 AICore counters；阶段 8 尚未验收。4B 服务组已停止，端口已释放；外部 NPU 2 仍被占用。27B 权重正在远端项目目录续传，尚未校验或 bring-up；四模型完整正确性、27B baseline、逐 token ITL、kernel-level Profiler、TP=2、跨模型消融和持久 session KV reuse 仍未完成。KV 峰值采样器已用于 baseline/Agent 实验。所有测量数据应保留在项目 `results/raw/` 与 `results/profiles/`，以原始记录为报告依据。
