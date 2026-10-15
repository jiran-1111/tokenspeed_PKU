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
- 4B/TP1/NPU0 的跨模型 E0–E3 矩阵也已完成：chunk=512/1024 两组各 144 条 raw、12 份 summary；每个 raw 目录 30 个文件，另有 `.tmp/` 下的 detached runner log，31 个相关文件分别跨端 SHA-256 全匹配。生成器目标输入/输出 2048/32，逐条 raw usage 实际为 1855/32；与 0.8B 的 1857/32 有轻微输入长度差异。3/24 个 summary 的 Prometheus token counter delta 比 raw aggregate 多 1，报告已注明；吞吐按请求记录计算。分析未选出通用 chunk 默认，也未把相同提示词带来的 E3 缓存命中外推为一般 Agent 收益。分析、统计附录和图目录见 [`4B cross-model ablation`](../report/ablation_analysis.md)。
- Prometheus KV/prefix/scheduler telemetry 解析，并采集可用的 `npu-smi info` 原始快照。
- 模型与客户端 tokenizer 固定为同一个官方完整 commit SHA，并写入各类原始结果和 summary。
- 2026-10-11 Workload A：Qwen3.5-0.8B、Ascend910 V1/NPU0、TP=1，目标输入/输出 512/128、客户端并发 1/4/8/16/32，每格 3 轮×32 请求；15 份 summary、480 条 raw、30 份 NPU 前后快照，32 个数据文件跨端 SHA-256 全匹配。逐条 usage 实际输入为 481、输出为 128。总 token 吞吐从 c16 的 752.41 增至 c32 的 754.19 tok/s，而 TTFT P50 均值从 3.65 s 增至 11.16 s；仅是该配置的描述性基线，不作因果/优化结论。raw 与 CSV/SVG 见 [`Workload A 结果`](baseline.md) 和 `results/analysis/agent_workload_a_512_128_20261011/`。
- 2026-10-11 Workload A：Qwen3.5-4B、Ascend910 V1/NPU0、TP=1，目标输入/输出 512/128、客户端并发 1/4/8/16/32，每格 3×32 请求；15 份 summary、480 条 raw、30 份 NPU 前后快照，32 个数据文件跨端 SHA-256 全匹配。实际输入/输出为 479/128 tokens。`max_num_seqs=8` 下 c16→c32 总吞吐仅 348.06→349.66 tok/s，TTFT P50 均值为 17.92→25.18 s；仅作描述性基线。详见 [`Workload A 结果`](baseline.md) 与 `results/analysis/agent_workload_a_512_128_20261011/qwen35_4b/`。
- 2026-10-11 Workload A：Qwen3.5-9B、Ascend910 V1/NPU0、TP=1、revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，目标输入/输出 512/128、客户端并发 1/4/8/16/32，每格 3×32 请求；15 份 summary、480 条 raw、30 份 NPU 前后快照，逐条 usage 为 479/128 tokens，prefix-cache hit ratio 为 0。32 个数据文件与两份 runner/server 日志共 34 个文件均跨端 SHA-256 匹配，runner 退出码 0、错误记录为 0。`max_num_seqs=8` 下 c8/c16/c32 总 token 吞吐为 324.07/322.68/324.02 tok/s，c16→c32 TTFT P50 均值为 18.66→26.90 s、waiting 峰值为 13→29；只作固定配置描述性基线。结果见 [`Workload A 结果`](baseline.md) 与 `results/analysis/agent_workload_a_512_128_20261011/qwen35_9b/`。27B Workload A/baseline 仍待处理，TP=2 继续暂缓。
- 普通 Benchmark 默认输入为 128/1024/2048 tokens；发送前检查 chat template 后的输入加输出预算是否超出配置上下文。显式指定的超长 workload 会报错，不截短输入；8K/16K 实验需同步提高服务与配置的上下文上限。
- 2026-10-10 0.8B/4B/9B decode-heavy 补测：910 V1/NPU0、TP1，目标输入 256、最大输出 512/1024，并发 1/4，每格 3 轮×8 请求；均采集 TTFT、服务端 ITL/TPOT、吞吐及 NPU 前后快照。三模型分别有 32/30/29 个原始、硬件及服务文件，合计 91 个文件跨端 SHA-256 全匹配。并发 4 的 TTFT p95 分别为：0.8B/512 输出 4721/4742/1066 ms，4B/512 输出 5584/2141/1976 ms；9B/512 输出 2197/2026/1982 ms、9B/1024 输出 2292/2317/1943 ms。部分轮间长尾波动明显，均只作为描述性 workload 曲线，不代表优化结论，也未测长 prefill 对 decode 的并发干扰。细节见 [`final_results.md`](final_results.md)。
- 远端项目环境课程测试、CLI 参数路由、Q/gate 布局和 metrics sampler 回归此前 68 项通过（4 个 CANN 安装目录 owner 警告）；随后继续补充 benchmark、Agent、报告、profiler、消融、项目内 `ASCEND_WORK_PATH`、decode 诊断及 ITL 回归。2026-10-10 最新本机 `test/course_project` 为 172 项通过、1 项因 Windows 无 Bash 跳过；远端 Session KV/runtime 回归为 90 项通过，Profiler focused tests 为 26 项通过。本机使用已有 Python 测试运行时，未安装依赖。
- Ascend910 V1 上的 Qwen3.5-0.8B 确定性 API smoke test：默认 graph-mode 服务启动、预热成功；提示词 `Reply with exactly the word OK.` 返回 `OK`（HTTP 200）。此前同一请求输出句点；原因是 NPU 分支对 Q/gate 使用了全局二分，而模型布局为每个 query head 内交错。新增 per-head split 后，Q/gate、MHA、GDN 聚焦测试共 11 项通过。响应原始记录见 `results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json`。
- 0.8B correctness runner 已执行 8 条中文/英文/数学/代码/长上下文/多轮请求并保存响应，见 `results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl`。它是记录采集，不是自动 pass/fail，也未验证引擎原始 token ID。
- 2026-10-09 Qwen3.5-4B bring-up：固定 revision 的两个 safetensors 分片均以 SHA-256 校验通过，离线 `AutoConfig` 与分片 header 可读。隔离到空闲 NPU 0 后，TP=1 服务完成 load/warmup，`/v1/models` 和引擎 metrics 均 HTTP 200。correctness collector 记录 6 个用例、8 条请求，见 `results/raw/correctness/qwen35-4b_20261009_run1.jsonl`；引擎原始 output token IDs 不可得，`token_correctness_verified=false`，因此仅算 chat/API smoke，不算完整正确性或性能结果。服务日志在 `.tmp/qwen35_4b_serve_20261009_run1.log` 与 `.tmp/qwen35_4b_correctness_20261009_run1.log`。
- 2026-10-09 Qwen3.5-0.8B/TP=1 token-ID 对照：8 条固定输入均采集到原始 engine IDs。原始数组逐个完全相等 1/8；统一 model/tokenizer EOS/stop 规则后 6/8 完全匹配。第二轮复测的 8/8 输入 IDs 和 8/8 engine 输出 IDs 与第一轮完全一致，因此长上下文第 32 个输出 token、多轮第 3 轮第 2 个输出 token 的正文差异可复现，不判定为通过。两轮采集及比较文件见 `results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1.jsonl`、`qwen35-0.8b_20261009_native_ids_run2.jsonl`、`qwen35-0.8b_20261009_native_ids_run1_stop_aligned_reference_compare.json`；首次未对齐 stop 的诊断文件也保留。
- 2026-10-09 0.8B 长上下文 decode 诊断：同一 979-token 输入、64-token greedy 输出，prefix cache 与 KVStore 均关闭。Eager 连续 decode 和已确认捕获 decode cudagraph 的连续 decode 序列完全相同，均在输出 index 32 首次偏离 HF；每步提交“原 prompt + 已生成 token”的全量重算与 HF 64/64 相同。该证据排除了 prefix cache 和 cudagraph 作为本例偏差的触发因素，并将范围收窄到单请求 incremental-decode/state 路径；尚未定位 MHA KV、GDN state 或其他具体字段。后续 40-token top-2/position capture 显示首错处两路径输入 ID 均为 271、position 均为 1010；continuous margin 为 0.125，full-recompute 在保存精度下并列并选中参考 token。该证据支持数值路径差异在近似并列决策处显现，不指认具体根因。raw 记录位于 [`decode_diagnostic`](../../results/raw/correctness/decode_diagnostic/)，服务配置与日志位于 `configs/diagnostic/` 和 `.tmp/qwen35_0.8b_decode_diag_*.log`；未修改模型/kernel。
- 2026-10-09 Qwen3.5-9B decode-path 诊断：Ascend 910 V1/NPU 0、TP=1、Eager，prefix cache/KVStore 关闭。long-context 与 multi-turn 两例的 continuous 与 full-recompute 输出各自相同，但相对 HF 分别在 index 59/58 首错。long-context index 59 两路径输入 ID/position 均为 11/1035，候选 ID 264/6278；continuous logits 30.375/30.25，full-recompute 30.25/30.25 并列，两者均选 264，参考选 6278。multi-turn index 58 两路径输入 ID/position 均为 10033/76，候选 ID 1149/1；continuous logits 27.125/27.0，full-recompute 27.25/27.125，两者均选 1149，参考选 1。该结果削弱“仅 incremental state 导致这两例偏差”的解释，但未定位具体数值算子/状态。multi-turn full-recompute 的 logits 采集缺失输出位置 20–26（SSH 重置期间短暂 disarm）；完整输出与 index 58 关键采集行保留，限制已记录。raw 文件见 [`decode_diagnostic`](../../results/raw/correctness/decode_diagnostic/)。
- 后续原生 `/generate` 采集见 [`qwen35-4b_20261009_native_ids_run1.jsonl`](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1.jsonl)：6 个用例共 8 条记录，包含真实引擎输出 token IDs。与同一 pinned revision 的 Transformers greedy 参考逐条比较，8/8 序列完全一致；每条生成 64 tokens（达到 `max_new_tokens=64` 上限），输入使用同一 chat-template token IDs，参考侧显式传入全 1 attention mask。比较详情见 [`qwen35-4b_20261009_native_ids_run1_reference_compare.json`](../../results/raw/correctness/qwen35-4b_20261009_native_ids_run1_reference_compare.json)。这只验收 4B/TP1/910 V1 上的这 8 条样例，不等于四模型或全场景正确性完成。
- Session KV matched correctness/reuse acceptance：910 V1/NPU0、Qwen3.5-4B 固定 revision、current-frontier、两条交错 session 各两轮，greedy `max_new_tokens=8`。generic prefix cache 与 KVStore 均关闭。初次验收关闭 overlap；随后在 `overlap_schedule=true` 的复验中，4/4 session 输出序列仍与 stateless full-history 对照完全相同，首轮新预填为 43/39 tokens，第二轮分别复用 50/46 tokens、新预填 20/21 tokens，关闭后的两次探测均按预期拒绝。两次都是正确性/reuse 验收，不是性能实验；首次结果见 [`Session KV raw run`](../../results/raw/session_kv/qwen35-4b-20261009T161435357104Z.json)，overlap 复验及 NPU 快照见 [`overlap-enabled raw run`](../../results/raw/session_kv_overlap_20261010/)。
- 2026-10-10 Session KV 重复 mixed-load 记录：在同一 4B/TP1/NPU0、current-frontier、overlap-enabled、两条交错 session + stateless full-history controls、greedy 8-token 配置下重复 3 次。12/12 session 输出序列均匹配；每轮 turn-2 metadata 均为 session-0 新预填/复用 20/50、session-1 21/46 tokens；六个 turn-2 session latency 中位数为 676.04 ms，stateless controls 为 676.03 ms。由于 session 与 control 请求在同一批次并发，这一组只描述该混合负载，没有显示端到端加速，也不构成参数 sweep。原始数据与 NPU0 快照见 [`三次重复 raw records`](../../results/raw/session_kv_perf_20261010/)；全部 9 个文件已跨端 SHA-256 核验。
- 2026-10-09 Qwen3.5-9B bring-up：固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` 的四个权重分片均通过官方 manifest 大小/SHA-256 核验；离线 `AutoConfig` 与全部 775 个 safetensors tensor header 可读。Ascend910 V1/NPU 0 的 TP=1 服务完成 warmup，readiness、`/v1/models` 和引擎 metrics 均 HTTP 200。原始 token-ID 对照两轮的 8/8 输入与引擎输出数组完全复现；相对 pinned Transformers greedy 参考为 stop-aligned 6/8，`long_context_recall` 与 `multi_turn_memory` 分别在输出位置 59、58（0-based）出现可复现正文差异，因此不记为正确性通过。记录见 `results/raw/correctness/qwen35-9b_20261009_native_ids_run1.jsonl`、`qwen35-9b_20261009_native_ids_run2.jsonl` 与 `qwen35-9b_20261009_native_ids_run1_reference_compare.json`。
- 2026-10-11 Qwen3.5-9B GDN FP32 candidate matched-cap follow-up：在相同 64-token 输出上限下，新的 8-case run 对 pinned Transformers greedy reference 为 7/8 raw/stop-aligned exact；仅 `long_context_recall` 仍在输出 index 59 分歧（engine 264、reference 6278）。与旧 run 的六条同输入记录配对后，五条引擎输出未变，`multi_turn_memory` turn 1 则改为匹配 reference token 1；turn 2–3 的 client histories 已变化，不纳入配对输出比较。证据及跨端 SHA-256 见 `results/raw/correctness/gdn_fp32_candidate_9b_20261011_max64.jsonl` 和 `gdn_fp32_candidate_9b_20261011_max64_reference_compare.json`。严格正确性仍未通过。
- 2026-10-11 GDN FP32 candidate 官方 bring-up 复测：4B/TP1/910 V1/NPU0、`max_tokens=64` 的八条固定样例对 pinned Transformers raw 与 stop-aligned 均为 8/8；raw 与 compare 文件分别为 `gdn_fp32_candidate_4b_20261011_bringup_max64.jsonl` 和 `gdn_fp32_candidate_4b_20261011_bringup_max64_reference_compare.json`。9B 官方 bring-up 配置的八条输出也均为 64 tokens，直接 pinned-reference 对照为 7/8 raw/stop-aligned exact；唯一差异是 `long_context_recall` index 59（engine 264、reference 6278）。9B raw 与 compare 文件分别为 `gdn_fp32_candidate_9b_20261011_bringup_max64.jsonl` 和 `gdn_fp32_candidate_9b_20261011_bringup_max64_reference_compare.json`，本地/远端 SHA-256 一致。两个 reference run 均因可选快速注意力库缺失回退到 PyTorch，未安装依赖。未把结果归因于 FP32 candidate，也未作性能结论。
- 2026-10-11 本机课程测试复验：在 workspace Conda 环境运行 `test/course_project`，224 passed、1 platform skip（6.54 秒）；未安装依赖，未使用 NPU。
- 2026-10-11 README/启动入口复验：README 中的本地 Conda 测试命令复跑通过；远端 Bash serving wrapper 对 0.8B、4B、9B、27B 和 4B TP=2 五个配置的 `--dry-run` 均成功，27B 命令解析为 world-size 2。此项只验证启动命令构造，不代表 27B/NPU 启动通过，也没有加载模型或运行 NPU workload。
- 2026-10-09 Qwen3.5-9B/TP=1 baseline：输入/输出目标 128/32、独立 prompt、prefix-hit ratio 0，并发 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求、12 summaries 和 24 份硬件快照。设备为 Ascend910 V1，配置 `max_num_seqs=8`，所以 concurrency=16 会排队；c8/c16 的 KV 峰值均为 1.00，吞吐约 69.41/70.48 output tok/s，不能解释为 16 并发带来有效扩展。详细中位数见 [`9B baseline`](final_results.md#qwen35-9b-baseline-2026-10-09)；raw 数据见 `results/raw/baseline/qwen35_9b_20261009_full/`。这是单设备/单配置基线，不是优化收益，逐 token ITL 未采到。
- 2026-10-09 Qwen3.5-4B/TP=1 baseline：输入/输出目标 128/32、独立 prompt、prefix-hit ratio 为 0，并发 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求、12 summaries 和 24 份硬件快照。设备为 Ascend910 V1，配置 `max_num_seqs=8`；c16 排队，c8/c16 KV 峰值均为 1.00，输出吞吐中位数为 74.54/75.88 tok/s，c16 TTFT p50 升至约 3.97 s，不作为扩展或优化结论。详细指标见 [`4B baseline`](final_results.md#qwen35-4b-baseline-2026-10-09)；raw 数据见 `results/raw/baseline/qwen35_4b_20261009_full/`。同步后的 26 个文件已核对大小和 SHA-256；ITL 未采集。
- 修正指标采集路径：`--metrics-port 39142` 现在暴露 TokenSpeed 引擎 Prometheus 指标；`--prometheus-port 39141` 是网关自身指标。0.8B 真实端点返回 KV/scheduler/prefix-cache 样本。对应实现经过 66 项远端测试及服务 smoke 验证。
- 指标 smoke：4 个请求、输入目标 128、输出 32，TTFT p50 92.89 ms、TPOT p50 10.52 ms、请求吞吐 2.40 req/s；采到 prompt 516、generation 128、prefix-cache hit 512 tokens。KV usage 是工作负载结束后的瞬时值 0.0，不代表运行峰值。结果见 `results/raw/metrics_smoke/qwen35_0.8b_20261008_run17/`，此小样本不作正式性能结论。
- 峰值采样设备 smoke：8 请求、concurrency 4、输入目标 128、输出 32，TTFT p50 173.35 ms、TPOT p50 20.62 ms、吞吐 1.08 req/s；采到 KV usage peak 0.53125、running peak 5、prompt/generation/prefix-hit delta 1033/257/1024。结果见 `results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18/`。仅验证采集链路；小样本不作正式性能结论。
- 普通 baseline pilot `run14` 包含 144 条请求（输入目标 128、输出 32，concurrency 1/8/16，各 3 轮），但当时未接通 TokenSpeed 指标且 c8/c16 重复轮次方差很大，仅保留原始记录，不纳入正式 baseline。
- 2026-10-09 完成 Qwen3.5-0.8B 首轮并发 baseline：128/32 tokens，concurrency 1/4/8/16，各 3 轮、每轮 32 请求，共 384 请求；引擎指标及 NPU 前后快照均已记录。修正了普通 benchmark 各请求复用同一 prompt 的问题，现为等长独立 prompt；prefix hit ratio 为 0。summary/raw 见 `results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/`，结果表和 910 V1 设备说明见 `docs/project/baseline.md`。修正前的重复全文样本保留在 `results/raw/baseline/qwen35_0.8b_20261009/`，prefix hit ratio 约 0.99，不纳入普通 baseline。
- Agent workload runner 已改为每个重复轮次使用不同的共享/私有上下文，同时保持同一轮内的共享前缀；回归测试覆盖“轮内共享、轮间隔离”。0.8B 两轮 pilot 已完成 390 请求（concurrency 1/4/8、prefix 0/25/50/75/90%）；另完成 8-turn 复测 360 请求（concurrency 1/4、prefix 0/50/90%、512-token context、100 ms 客户端 tool-wait）。两组数据都有显著 TTFT 长尾，尤其 50% prefix 在 8-turn c1/c4 的 TTFT p50 分别约 4.78/4.24 s，且曲线非单调；数据保留待 Profiler 分析，不能宣称 prefix cache 收益。原始数据见 `results/raw/agent/qwen35_0.8b_20261009_unique_prefix/` 与 `results/raw/agent/qwen35_0.8b_20261009_8turn/`。
- 2026-10-09 Qwen3.5-4B 早期 8-turn Agent 客户端等待样本（与后述 Workload F 是不同实验）：共 360 条请求、18 份 summary、36 份硬件快照；每个 session 首轮后请求前等待 100 ms。原始记录中 45 条首轮请求的 `tool_wait_ms` 为 0，其余 315 条为 100；`tool_wait_policy` 与 `mock_tool_result_injected` 字段均未记录，因此只能将其描述为客户端等待，不能据此声称注入了 mock tool result、服务端进入 WAITING_TOOL 状态或保留 KV。该负载不用于优化结论。原始数据见 [`4B 8-turn Agent raw`](../../results/raw/agent/qwen35_4b_20261009_8turn/)。
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
- 2026-10-10 Qwen3.5-9B/TP=1 按同一 Workload F 矩阵完成协议复测：Ascend910 V1/NPU0，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，512-token 目标共享前缀、8 turns、32 输出 tokens、并发 1/4、目标比例 0/0.5/0.9、每格 3 次重复；共 360 条 raw、18 summaries、36 份硬件快照，38 个原始/统计/硬件文件跨端 SHA-256 全匹配。126 条 `session-0` 后续请求带 100 ms client mock wait 和固定 mock result；不是服务端 WAITING_TOOL 或 Session KV 保留测试。并发 4 下 TTFT 和采样 active-page ratio 高、prefix 比例效应非单调，均仅作描述性观察。统计附录、两张 PDF 矢量图和结果报告见 [`9B Workload F analysis`](../../results/analysis/qwen35_9b_agent_tool_wait_20261010/analysis-report.md) 与 [`cross-model replication report`](../../results/analysis/qwen35_9b_agent_tool_wait_20261010/2026-10-10--agent-tool-wait--r00--cross-model-replication.md)。
- 自动报告读取 `results/raw/**/*.summary.jsonl` 生成 CSV/SVG；按实验类别隔离 baseline 与 Agent 数据，对重复轮次取中位数，并仅在同一配置组存在至少两个横轴取值时生成趋势图。新增矩阵归属后，`results/plots/ablation_20261009/` 生成 summary CSV 与 7 张 SVG，包括 E0–E3 消融图和按 E2/E3 变体隔离的 chunk 512/1024 对比。`results/plots/after_9b_baseline_20261009/` 与最新 `results/plots/after_4b_baseline_20261009/` 均保留独立输出；最新 baseline 图将 0.8B、4B、9B 分成不同模型序列。新增 `kv_cache_usage.svg` 仅使用 baseline、TP=1、目标输入/输出 128/32 的记录，展示 peak scheduler active-page ratio；该比例不代表 HBM occupancy，也不是 KV block-size sweep。Session KV 的三次 matched mixed-load 记录没有测出可归因的 latency 提升，也不是参数 sweep；其后另完成 4B/NPU0 serial on/off context sweep（结果见下节）。该 sweep 的专用 raw JSON 由报告生成器单独读取并输出配对 CSV/SVG sidecar，不并入 summary CSV；本次复现产物见下节。KV block 与 TP 性能对比仍未完成。
- 2026-10-11 基于当前 `results/raw/**/*.summary.jsonl` 重建版本化总报告：`summary.csv` 有 356 行、11 张 SVG 均通过 XML 解析，另含独立的 Session KV 配对 CSV/SVG sidecar。目录为 [`current raw report`](../../results/plots/rebuilt_current_raw_5e6b5fe6645a4bcd816bd6a114217e47/)。原有 `results/plots/summary.csv` 仍为 233 行且未覆盖；新版数据只有 `tp_size=1`，因此不生成或声称 TP 性能对比。随后独立重建到 [`repro verification`](../../results/plots/repro_verify_current_raw_20261011/)；输出仍为 356 行和 11 张可解析 SVG，`summary.csv` SHA-256 与前一版本完全一致。
- 2026-10-09 Profiler 复采：设置项目内 `ASCEND_WORK_PATH` 后，以三 warmup 重采 prefill/decode/Agent 场景，CANN `op_summary` 分别有 229,637/876,881/417,188 行，parser 日志未见 warning/error。CANN `Task Duration` 覆盖调度、加速器执行和响应结束，不是纯 AICore 时间；`Task Type` 为 N/A，且单轮数据不能确认瓶颈。旧 trace-only 采集的 flow 数字不与本轮混用。原始/规范化 trace 与 CANN 产物见 `results/profiles/qwen35-0.8b/ascend_work_path_fix_full/`，分析见 `docs/report/ascend_profile.md`。
- Conda 激活脚本已兼容非登录 SSH PATH；在远端剥离 PATH 的 shell 中成功定位 `/opt/miniconda3`、激活项目 `.conda-env` 并通过课程测试。
- `ASCEND_WORK_PATH` 已在 `scripts/use_project_env.sh` 固定到项目 `.tmp/ascend_work`；远端专用环境测试先失败后通过，并完成服务级三场景 CANN profiler 复采。2026-10-10 最新本机 `test/course_project` 为 149 项通过、1 项因 Windows 缺少 Bash 跳过；远端 Session KV/runtime 测试为 90 项、Profiler focused tests 为 26 项通过，较早远端课程测试 63 项通过。

这条 smoke test 仅验证一个提示词的服务端到端生成，不替代完整 token-ID 对照，也不是性能数据。测试结束后已停止服务，NPU 进程清空。

## 尚未验收

状态更新（2026-10-10）：本文件前文“尚无 Session reuse 性能 sweep”已过时；4B/NPU0 的 serial paired sweep 已完成，边界、结果与未覆盖项见 [`Session KV serial paired latency sweep`](#session-kv-serial-paired-latency-sweep2026-10-10)。KV block、TP 性能和跨模型 Session KV sweep 仍未完成。

### 长 Prefill 对持续 Decode 干扰（2026-10-11）

- 已批准并实现 4B/TP=1 的 staged HTTP 实验工具：目标 Prefill 2K/3.5K tokens、Decode 输入/输出 256/128，比较 chunk size 4096/512 下的单请求与并发条件；保存请求/SSE chunk 时间线并输出配对 CSV/SVG。实现与边界见[设计规格](../superpowers/specs/2026-10-11-long-prefill-decode-interference-design.md)、[实施计划](../superpowers/plans/2026-10-11-long-prefill-decode-interference.md)和[运行说明](../../README.md#长-prefill-对持续-decode-的干扰)。
- 本地 Conda 环境的 `test/course_project` 为 385 passed、6 skipped；runner dry-run 展示 3 轮×2 配置且未加载模型、创建结果目录、访问 NPU 或启动服务。这些只验证软件路径，不是硬件测量。
- 真实 NPU 实验尚未运行，没有该 workload 的 raw/CSV/SVG 结果，也不对 chunk size 的 Decode 干扰作结论。计划要求真实测量前另行确认；未获确认前保持待验收。

### KV block 粒度与统计边界（2026-10-10）

- Qwen3.5 `QwenGDNRecipe.prefix_granularity` 固定为 128；课程配置的 `block_size` 实际传给 `--prefix-granularity`。因此 16/32/64 不是该模型当前 recipe 暴露的可比较 block-size 选项，本项目没有强行启动这些配置，也没有声称 128 是由性能 sweep 选出的默认值。
- Scheduler 已在每轮快照中得到 active、cached 与 empty LCM block 数；现有 `tokenspeed:kv_cache_usage_perc` 仅以 active/total 计算，benchmark 尚未分别记录 active/used/free/total。后续补充统计时应以 LCM block 为计数单位，并区分请求持有与 prefix-cache 保留，避免把“活跃占用”误写为总已用容量。

### 正确性记录口径

此前 0.8B/4B 的 chat JSONL 仍保留历史口径：`output_token_ids` 为 null，不能追溯补成 token 级通过。当前 `run_correctness.py` 将客户端 chat template 的 `input_ids` 原样提交到 TokenSpeed 控制端口的原生 `/generate`（服务主端口 +1），分别记录引擎原始 `output_token_ids` 与响应文本重新编码得到的 `retokenized_output_ids`。4B 已完成 8 条同输入 greedy 参考对照，结果见上方文件；该采集 JSONL 中的 `token_correctness_verified` 表示采集时尚未比较，最终判断以单独比较文件为准。
4B 比较限定为 910 V1、TP=1、pinned revision、6 个固定用例/8 次生成、每次 64 tokens；不能外推到其他模型、输入长度或参数。0.8B 与 9B 的 stop-aligned 对照均为 6/8，且各有两条可复现正文差异。0.8B long-context 首错的 Eager、关闭 prefix cache/KVStore 诊断显示 continuous 与 full-recompute 在 index 32 首次分叉；两者该步输入 ID 均为 271、position 均为 1010。fresh 33-token revalidation 再次得到同一首差：continuous top-2 为 100532/96200、margin 0.125，full-recompute 两候选分数 20.25/20.25 并选中参考 token。对齐后的 layer-0 原始数组显示 mixed-QKV 6142/6144 元素相同、z 完全相同，末 token backend output relative-L2 为 7.84e-4；尚缺 backend 前配对状态，因此不能归因到 GDN backend 或某 kernel。另对 0.8B multi-turn 第三轮在 index 2/position 77 复采：TokenSpeed continuous/full-recompute 的 top-2 均为 `[38251, 198]`、margin 0.125；HF `use_cache=True` 在并列 logits 下选择 198，`False` 与 TokenSpeed 选择 38251。本记录绑定 Transformers 5.12.0，支持该用例的 cache-path 近边界敏感性，但不能证明其他 mismatch 或 kernel 根因。9B multi-turn mismatch 仍未解释；逐例证据见 [HF exact-prefix 诊断表](final_results.md)。27B token-ID 对照仍未验收。Session KV overlap 模式已完成正确性复验，另有串行 paired sweep；后者测量 `Engine.generate` 整轮请求延迟，不是 TTFT/并发吞吐。专用 raw JSON 现由报告生成器单独读取并生成配对 sidecar（不并入 `summary.csv`）：[CSV](../../results/plots/session_kv_paired_approval_20261010_182336/session_kv_paired_reduction.csv)、[SVG](../../results/plots/session_kv_paired_approval_20261010_182336/session_kv_paired_reduction.svg)。原生非流式生成不提供 TTFT/TPOT/chunk latency，这些字段留空；延迟测量使用独立流式 benchmark。

2026-10-10 TP=2 追加验证：将 HCCL gather/reduce-scatter 调用切换为远端 PyTorch 2.9.0+cpu 提供的 tensor collectives 后，两进程 HCCL 回归通过，4B TP=2 服务也能启动并完成 warmup。两次固定 8 样例 reference 对照均为 7/8；两轮输入 IDs 与 TP2 输出 IDs 逐条完全相同，唯一的 `chinese_beijing` 差异在生成 index 19（0-based）稳定复现。TP2 correctness 与性能均未验收，根因未定位。原始记录见 [`首次 TP2 token IDs`](../../results/raw/correctness/qwen35-4b-tp2-20261010T000417Z.jsonl)、[首次对照](../../results/raw/correctness/qwen35-4b-tp2-20261010T000417Z-compare.json)、[复跑 raw](../../results/raw/correctness/qwen35-4b-tp2-repro-20261010T0024Z.jsonl) 与[复跑对照](../../results/raw/correctness/qwen35-4b-tp2-repro-20261010T0024Z-compare.json)。

状态更正（2026-10-10 08:16 local）：上文的 06:44 27B 传输快照已过时。远端 staging 的 shard 1–6 已达到 manifest 精确大小且 SHA-256 匹配；shard 7–11 的续传在 08:14 启动，但连接于约 08:17 重置，文件仍保留为 partial。随后仅重启 shard 7、8 两个单写者，待确认其是否仍在运行后再续其余分片。27B 权重尚未晋升到正式目录。HCCL 修复与 TP2 correctness 状态见上段。

TP=2 logit 诊断更新（2026-10-10 08:47 local）：针对可复现的 index-19 分叉，TP2 eager capture 在两 rank 上都得到 token 25701/1286 的 scores 23.375/23.25；Transformers reference 在相同 prefix 对两者均给出 23.375 并列并选择 1286。该证据支持近似并列处一个 BF16 步长改变排序的解释，但不定位产生 0.125 差异的具体算子。TP2 strict token-ID correctness 仍为 7/8、无性能验收。TP2 logits 和 reference scores 的原始文件位于 `results/raw/correctness/decode_diagnostic/`。

TP1/TP2 同前缀补充对照（2026-10-10）：在 eager 模式下对相同 36-token 前缀复采，TP1 在 position 35/input ID 25 的 top-2 为 `[1286,25701]` / `[23.375,23.25]`，目标输出 index 19 为 1286；TP2 两 rank 则为 `[25701,1286]` / `[23.375,23.25]`。HF 对该 prefix 的两候选分数均为 23.375 并列。两路径的 margin 都是 0.125；证据进一步显示排序与 TP 配置相关，但仍不能归因具体 kernel/collective，TP2 correctness 仍是 7/8。TP1 logits/输出及服务、客户端日志的 SHA-256 已跨端核验。

27B 同步更新（2026-10-10 09:08 local）：远端标准 staging partial 为 42,961,887,576 / 55,563,022,432 bytes（77.32%，排除旧 retry partial）。shard1–6 和 shard11 已匹配 manifest 大小/SHA-256；shard7–10 四个 SFTP 单写者处于活动状态，shard10 的 partial 已从 302,153,728 增长至 384,380,928 bytes。仍需等待其余分片完成后逐片校验；当前没有任何权重晋升到正式目录。

### Session KV serial paired latency sweep（2026-10-10）

- Ascend910 V1/NPU0、4B/TP=1、current-frontier、overlap enabled；每次只运行一个请求，prefix cache/KVStore 关闭。512/1024/1536 目标上下文各 4 个新 Session，prime 不计时，session 续轮与 stateless full-history 按轮交替顺序配对。
- 12/12 组 prime 和续轮原始输出 token IDs 一致，metadata 校验通过；每档实际输入长度 541/1053/1565，续轮均预填 20 个 token，复用 548/1060/1572 个 token。`Engine.generate` 请求延迟中位数：Session 264.07/260.96/253.15 ms，stateless 1199.21/2020.91/2840.09 ms；配对降幅中位数 77.95%/87.11%/91.28%。1536 档有一个 6463.69 ms stateless 样本，未剔除；四次重复的局限需保留。
- 原始结果和 NPU0 快照见 `results/raw/session_kv_perf/`。此前 2048 档试跑因设备 kernel `coreDim=66464 > 65535` 硬退出、没有 JSON，因此不纳入比较。为防止后续硬退出丢失已完成配对，runner 已增加原子检查点，并新增本地/远端测试覆盖。该 sweep 仅支持当前 4B/TP1/910 V1 配置下的串行续轮请求延迟观测，不代表 TTFT、并发吞吐或跨设备收益。

### 流式延迟

OpenAI chat 客户端的 `inter_chunk_latencies_ms` 是非空文本块在客户端的到达间隔，
`chunk_latency_p50_ms/p95_ms` 是其分位数。上游 `generation_output_processor.py`
支持可变 `stream_interval`；文本解码和传输也可能合并多个 token，因此不能将这些间隔
当作逐 token ITL。2026-10-09 起，runner 从服务端
`tokenspeed:inter_token_latency_seconds` Prometheus histogram 的 benchmark 前后差分
计算 run-level ITL 均值和 P50/P95；该服务端指标按每个 scheduler update 的时间间隔除以
新增 token 数，并将该均值计入对应数量的样本，因此一次 update 新增多个 token 时不是
逐 token 独立时间戳。分位数由直方图桶插值估计；若目标落入 `+Inf` 桶，报告最高有限桶边界
作为下界，并在 provenance 中标明。ITL 仅在 run summary 级别提供，不伪造每请求值。
首轮 0.8B 复测发现运行时曾把前端墙钟间隔、engine timestamp 间隔和请求结束平均值
混入同一 histogram；RED 回归复现 3 个新增 token 产生 9 个样本。现已只记录 engine
output timestamp 间隔，并按该帧新增 token 数计权。修正后 0.8B/TP1 同负载复测完成
384 请求；每份 summary 的 ITL 观测数均为 992，与原始输出 token 数减一的期望值一致，
26 个原始/summary/硬件文件的跨端 SHA-256 全部匹配。指标定义和分档结果见
[`corrected ITL results`](baseline.md)。
首轮历史 summary 未回填，4B/9B 首轮 summary 仍为 null；两者也已完成独立的 corrected ITL 复测，结果见 [`0.8B/4B/9B corrected ITL`](baseline.md)。TTFT 同样是首个非空文本块的到达时间；
TPOT 目前按首个文本块到流结束的耗时除以 usage 中的输出 token 数减一计算，含流结束开销。
回归覆盖服务端直方图聚合/差分、engine timestamp 计数、缺失数据保持 null，以及 chunk latency 不被误标为 ITL。

更新（2026-10-09）：追加 0.8B 首错位置的逐层状态摘要捕获。continuous 与 full-recompute 各记录 24 层（6 层 MHA KV、18 层 GDN state），形状一致、无采集错误；所有层哈希不同，最早差异落在 layer 0 GDN。L2 norm 汇总最大相对差异为 GDN conv 1.742e-3、SSM 1.269e-3、MHA K 1.054e-4、V 6.374e-5；该统计不是逐元素误差，尚不能证明责任算子或修复根因。两种模式的 40-token 输出均复现此前结果，index 32 首错。摘要与输出见 [`decode_diagnostic`](../../results/raw/correctness/decode_diagnostic/)，本次 4 个采集文件跨端 SHA-256 一致；服务已停止、NPU0 空闲。

早期状态快照（以下述 2026-10-10 12:34 更正为准）：本机 Windows 没有 NPU。课程服务器提供 openEuler 24.03/aarch64、Ascend910 V1；课程原要求为 910C，课程方是否接受替代尚未确认。项目专属 Conda 环境已建立，Triton-Ascend kernel 验证与课程/NPU adapter 测试通过，四个服务配置 dry-run 通过。0.8B 已完成 API/指标 smoke、首轮普通并发 baseline、两组 Agent workload、三场景 Level0 Profiler trace、两轮可复现 token-ID 对照（stop-aligned 6/8）、E0–E3 的 512/1024 chunk 重复消融及修正后的 ITL 复测。4B 权重已校验、8 条固定样例 token-ID 全匹配，且首轮 384 请求 baseline、corrected ITL 复测及 current-frontier Session KV matched correctness/reuse run 已完成；`overlap_schedule=true` 的续轮交接也已在 NPU0 完成一次 matched correctness 复验。9B 已完成权重校验、TP=1 服务 smoke、两轮 token-ID capture（参考对照 6/8）、decode-path 诊断、首轮 384 请求 baseline 和 corrected ITL 复测。2026-10-10 又完成 0.8B 8-turn Level1/PipeUtilization 六次采集（c1/c4 各 3 次、120 请求），`op_summary` 有非零 AICore 时间行；新增只读后处理脚本可复算六轮 AICore、任务区间与 HBM 原始采样，累计值与报告表格匹配，但仍未将算子时间与端到端长尾建立因果归属。新增 0.8B multi-turn 第三轮 exact-prefix 对照显示 HF `use_cache=True` 在近似并列处选择参考 token，而 HF no-cache 与 TokenSpeed 两种 decode 路径选择另一 token；这是单例 cache-path 敏感性证据，不是根因结论或修复。最新本机 `test/course_project` 为 150 项通过、1 项因 Windows 缺少 Bash 跳过；远端 Session KV/runtime 测试 90 项、Profiler focused tests 26 项通过。27B 本机全部 11 个权重分片均已通过 manifest 大小与 SHA-256 校验；离线配置与 1,199 个权重索引/头部张量名完全匹配。已校验分片正在远端项目临时目录续传；06:44 状态累计 21,470,978,048/55,563,022,432 bytes（38.64%），六个活跃 writer 分别负责 shard1–6，远端哈希核验、正式晋升、模型启动与 baseline 尚未完成。0.8B/9B reference 对照仍各为 stop-aligned 6/8；四模型完整正确性、27B bring-up/baseline、Profiler 根因归属、TP=2 和跨模型消融仍未完成。所有测量数据应保留在项目 `results/raw/` 与 `results/profiles/`，以原始记录为报告依据。

状态更正（2026-10-10 12:34 local）：0.8B/4B/9B decode-heavy 补测均已完成，36 条 summary 记录覆盖每模型的输入 256、输出上限 512/1024、并发 1/4、3 次重复；91 个跨端原始/硬件/服务文件 SHA-256 全匹配。汇总 CSV 与 9 张 SVG 单独生成于 [`report refresh`](../../results/plots/refresh_20261010_1234/)，旧报告未覆盖。本机 `test/course_project` 最近一次为 172 passed、1 platform skip；远端 Session KV performance focused tests 为 22 passed。27B 11 个远端权重分片已通过 manifest 大小与跨端 SHA-256 校验并晋升到 `models/Qwen3.5-27B/`；离线 header 检查覆盖 1,199 tensor keys。27B NPU bring-up、固定输入 token-ID 对照和 baseline 仍待完成；0.8B/9B reference 对照、Profiler 因果归属、TP=2 和跨模型消融仍未全部验收。Long Prefill 干扰和 Session reuse 多轮 TTFT 实验仍待设计/批准。
