# Final Results

> 归档说明：本文保留按日期记录的实验过程。标为“归档路径”的本地文件未随公开源码发布，因此以路径文字保留，不提供无效下载链接。当前可公开复核的材料见[最终交付](final_delivery_20261018.md)。

## 当前证据索引（2026-10-14，非最终验收）

完整项目仍未完成，当前逐项状态以[提交验收状态](submission_status.md)为准，下文保留按日期记录的旧快照，不把旧未完成状态当作今天的事实。

- 当前Gamma/QK修正后4B/TP1/BF16生产路径固定max64八样例raw及stop-aligned均8/8，与同输入独立NPU参考一致；八所选源码运行期间未变、20文件cross-SHA，[当前回归记录](current_4b_correctness_20261014.md)。不替代任意长度、TP2、Session或四模型终验。
- 当前源码9B同流程raw/stop-aligned均7/8，long_context_recall index59仍为engine264/HF6278；8/8输入与引擎输出IDs与旧固定运行一致，20文件cross-SHA，[当前回归记录](current_9b_correctness_20261014.md)。这是仍未闭环的strict结果，不自动接受。
- 当前源码27B/TP2在验证HCCL的logical2/3上复验八样例，raw/stop-aligned均8/8、23文件cross-SHA；旧代码样例index54分叉此次未复现，[当前回归记录](current_27b_correctness_20261014.md)。新旧非单变量控制，不只归因某补丁，不外推Session/任意长度或性能。
- 27B恢复后旧E0 Workload A已有480请求/15 summaries、actual479/128、五档并发各三轮，[记录](27b_e0_workload_a_20261013.md)。该旧性能版本当时strict 7/8，不用当前8/8回归去改写其历史状态或冒称新源码性能。
- 4B长Prefill/Decode干扰三轮配对及CSV/SVG已验收，[记录](long_prefill_acceptance_20261013.md)；512在此协议未降低staged Decode E2E，不选定通用最优值。
- 单请求Level1 TTFT窗口实采、800ns时钟误差核验及145文件cross-SHA已完成，[记录](profiler_single_request_capture_20261013.md)，不等于完整请求到算子因果归因。
- 作业1当前0.8B生产基线96请求/12 summaries、121 HBM采样峰值8465MB，[记录](homework1_baseline_hbm_20261014.md)；真实GDN输入独立FP32/64递推通过原模块包络，[记录](homework1_gdn_trained_oracle_20261014.md)。局部证据不替代完整模型。
- 全模型额外1%logits工程合同未过，[验收方向](homework1_numerical_acceptance_decision_20261014.md)待用户确认，未放宽门槛。最终Decode归因、四模型完整正确性/优化消融/配置和复验仍欠；[作业1报告](../report/homework1_compute_subsystem.md)仍草稿。

## 历史状态快照（2026-10-11 Asia/Shanghai）

- A matched 64-token-cap rerun of the 9B GDN FP32 candidate is 7/8 exact against pinned Transformers. On six rows with identical prompt token IDs versus the prior run, five engine outputs are unchanged and `multi_turn_memory` turn 1 now matches the reference; later turns have changed histories and are not paired. The long-context mismatch at output index 59 remains, so 9B is not yet a strict correctness pass.
- The 4B/TP=1 long-Prefill/Decode staged runner and CSV/SVG analyzer are implemented and locally tested for 2K/3.5K Prefill inputs with chunk sizes 4096/512. No real NPU run or result data exists yet; the plan requires separate authorization before measurement, so no interference or chunk-size conclusion is claimed.
- The current raw-summary report was rebuilt into a versioned directory: 356 CSV rows and 11 XML-parseable SVGs, plus the separate Session KV paired CSV/SVG sidecar. An independent regeneration produced the same `summary.csv` SHA-256. The older default `results/plots/summary.csv` remains at 233 rows and was not overwritten. All report rows are TP=1, so no TP comparison is available. See current raw report（归档路径：`results/plots/rebuilt_current_raw_5e6b5fe6645a4bcd816bd6a114217e47`） and reproducibility run（归档路径：`results/plots/repro_verify_current_raw_20261011`）.
- In the current GDN FP32 candidate, the 4B official bring-up config produced 8/8 raw and stop-aligned exact matches across the eight fixed cases (`max_tokens=64`). The 9B official-config run also has a direct pinned-reference comparison: 7/8 raw and stop-aligned exact, with the remaining `long_context_recall` mismatch at output index 59 (engine 264, reference 6278). Both reference runs used the Transformers PyTorch fallback because optional fast-attention libraries are absent; no dependencies were installed. These are fixed-sample correctness checks, not causal or performance claims. See the 4B raw run（归档路径：`results/raw/correctness/gdn_fp32_candidate_4b_20261011_bringup_max64.jsonl`）, 4B comparison（归档路径：`results/raw/correctness/gdn_fp32_candidate_4b_20261011_bringup_max64_reference_compare.json`）, 9B official-config raw run（归档路径：`results/raw/correctness/gdn_fp32_candidate_9b_20261011_bringup_max64.jsonl`）, and 9B comparison（归档路径：`results/raw/correctness/gdn_fp32_candidate_9b_20261011_bringup_max64_reference_compare.json`）.
- A fresh local `test/course_project` run in the workspace Conda environment passed 385 tests with 6 skips; this is software regression evidence, not NPU/model acceptance. No dependencies were installed.
- 2026-10-11 Qwen3.5-0.8B Workload A completed on Ascend910 V1/NPU0/TP=1: target input/output 512/128, client concurrency 1/4/8/16/32, three runs of 32 requests per level. All 480 requests returned 128 output tokens; actual input usage was 481 tokens. Cross-end SHA-256 matched for all 32 raw/summary/hardware files. Total token throughput plateaued from 752.41 tok/s at c16 to 754.19 tok/s at c32, while mean run-level TTFT P50 rose from 3.65 s to 11.16 s. This is a descriptive single-model baseline, not an optimization claim. See [baseline table](baseline.md) and raw/analysis artifacts（归档路径：`results/analysis/agent_workload_a_512_128_20261011`）.
- 2026-10-11 Qwen3.5-4B Workload A completed on Ascend910 V1/NPU0/TP=1 with target input/output 512/128 and client concurrency 1/4/8/16/32, three runs of 32 requests per level. All 480 requests returned 128 output tokens; actual input usage was 479. All 32 raw/summary/hardware files match remote SHA-256. With `max_num_seqs=8`, total token throughput changed only 348.06→349.66 tok/s from c16 to c32, while mean TTFT P50 rose 17.92→25.18 s and scheduler waiting peak rose 14→29. This is a fixed-configuration descriptive baseline, not an optimization claim. See [baseline table](baseline.md) and 4B analysis（归档路径：`results/analysis/agent_workload_a_512_128_20261011/qwen35_4b`）.
- 2026-10-11 Qwen3.5-9B Workload A completed on Ascend910 V1/NPU0/TP=1 with target input/output 512/128 and client concurrency 1/4/8/16/32, three runs of 32 requests per level. All 480 requests returned 128 output tokens; actual input usage was 479. All 32 raw/summary/hardware files and both runner/server logs match remote SHA-256. With `max_num_seqs=8`, c8/c16/c32 token throughput was 324.07/322.68/324.02 tok/s; from c16 to c32 TTFT P50 rose 18.66→26.90 s and scheduler waiting peak 13→29. This is a fixed-configuration descriptive baseline, not an optimization claim. See [baseline table](baseline.md) and 9B analysis（归档路径：`results/analysis/agent_workload_a_512_128_20261011/qwen35_9b`）. 27B Workload A/baseline remain open; TP=2 remains deferred.
- Profiler: the single-request TTFT time-window runner and analyzer, including fake clock/trace tests, are implemented. No real NPU capture or Torch/CANN clock-overlap verification has been performed; capture remains separately gated.
- The documented local Conda test workflow was replayed successfully, and the remote Bash launch wrapper dry-ran all four model configs plus 4B/TP=2. This validates command construction only; 27B remains unverified at runtime.
- The approved 4B/TP=1/NPU0 Session KV multi-turn TTFT experiment completed 64 serial pairs across four 16-turn conversations. Session and stateless output IDs matched 64/64; TTFT medians were 122.71 ms and 1,394.54 ms, with a median paired reduction of 91.52%. This is native Engine TTFT for this fixed workload, not HTTP/concurrent TTFT or a cross-model/hardware claim. Raw evidence, CSV, SVG, and hardware snapshots are recorded under `results/raw/session_kv_ttft/` and `results/analysis/`.
- The 27B TP=2 attempt (`20261010T1316Z`) loaded both ranks' weights and completed image/video encoder warmup, but the service never reached readiness. No 27B token-ID correctness test or performance baseline was run.
- The preserved CANN trace shows `AllReduce_group_name_1` at `opLaunchIdx=14`, a 300-second watchdog timeout, and repeated `SynchronizeExecutedTask` timeouts on stream 47/task 1952 with `pendingNum=883`. This implicates the TP/HCCL startup path but does not establish a root cause. The run was interrupted; the after-snapshot shows no NPU0 process and 4% HBM use.
- The recovered rank-level device trace narrows the observed stall: both ranks created the HCCL group and SIO transport resources, while rank 1 repeatedly waits for a rank-0 notification on the AllReduce. The retained rank-0 trace has no matching runtime event; this is an observed asymmetry, not proof of why rank 0 did not complete or notify.
- Source inspection shows the post-encoder path proceeds through communication-runtime preparation, attention/KV construction, executor creation, autotune, then graph capture. This config disables autotune and prefill graphs and sets decode-graph capture sizes to 1/2/4. The engine log ends before the expected autotune/capture progress markers, so it does not prove graph capture was reached. The one-element AllReduce could match a graph-warmup barrier only if that phase was entered; the CANN trace has no Python callsite, so `opLaunchIdx=14` cannot be mapped to it. The exact stalled startup phase remains unknown; a future retry would first need per-rank phase markers and a distinct diagnostic question.
- Same-host 4B/TP=2 engine logs show distributed-init buffers, weights, image/video warmup, and HTTP server startup all completed. Both configs use TP=2, BF16, 4096-token context, `triton_attn`, prefix caching, and overlap scheduling; they differ in model/revision, `max_num_seqs`/`max_total_tokens` (4/16384 for 27B vs 8/32768 for 4B), and graph capture sizes (1/2/4 vs 1/2/4/8). This is not a matched control: no aligned 4B CANN HCCL task trace was available in the preserved comparison, so the exact cause remains open.
- The original 11 startup evidence files plus the subsequently recovered rank-0 device trace are now present locally (12 total); all 12 match their remote SHA-256 values. See 27B startup evidence（归档路径：`results/raw/bringup/qwen35_27b_20261010T1316Z`）. No retry is planned until a distinct diagnostic is justified.
- Hardware remains Ascend910 V1, not the course-required 910C; course-staff approval for substitution is still undocumented.

## 当前状态摘要（2026-10-10 14:32 local）

- 27B 的 11 个远端权重分片已通过固定 revision manifest 大小与 SHA-256 校验并晋升到正式模型目录；离线 safetensors header 与 index 的 1,199 个 tensor keys 一致。尚未在 NPU 上启动 27B。
- 0.8B/4B/9B 的 decode-heavy 补测已完成：Ascend910 V1/NPU0、TP=1，输入 256 tokens、输出上限 512/1024、并发 1/4、每格 3 轮×8 请求；91 个跨端原始/硬件/服务文件的 SHA-256 全部匹配。结果仅描述固定负载曲线，不代表优化收益；长 prefill 干扰测试仍待设计确认。自动汇总 CSV 与 9 张 SVG 见本轮报告目录（归档路径：`results/plots/refresh_20261010_1234`）。
- 4B/TP=1/NPU0 Session KV 串行 paired sweep 完成 12/12 输出 ID 匹配；整轮 `Engine.generate` 延迟的配对中位降幅为 77.95%/87.11%/91.28%（上下文目标 512/1024/1536），不是 TTFT 或并发吞吐结果。原始数据和边界见下文。
- 4B/TP=1/NPU0 的 E0–E3 跨模型消融已完成 chunk=512/1024 两组矩阵：各 144 请求、12 条 summary，每个 raw 目录 30 个数据/服务日志文件另加 `.tmp/` 下 runner log，共 31 个相关文件跨端 SHA-256 匹配。生成目标 2048/32，逐条 raw usage 为 1855/32。严格分析仅支持固定重复 prompt 下的描述性观察，未选定通用 chunk 默认；详见[`4B 消融分析`](../report/ablation_analysis.md)及[结果表](#qwen35-4b-cross-model-ablation)。
- 0.8B 首错位置的 fresh revalidation 重复了 position 1010/input ID 271 分歧：33-token continuous 首差仍为 index 32，full-recompute 与 reference 的前 33 个 token 全匹配。有效捕获共 48 层记录、无 capture error；两种路径的 layer-0 输入/末 token backend output 均已逐行对齐，15 个本地/远端证据文件 SHA-256 一致。该结果不定位根因，分析见最终结论。
- 最新本机 `test/course_project` 复验为 172 passed、1 platform skip。
- 项目仍未完成：27B 实机验证、0.8B/9B 与 4B TP=2 的 reference 分歧、Session reuse OFF/ON 多轮 TTFT 曲线、完整 Agent/KV/TP 实验矩阵及 Profiler 因果归因仍待处理。实际硬件是 910 V1，课程是否接受其替代 910C 尚未确认。

## 已验证的 bring-up 结果（非最终性能结果）

- 设备：课程服务器当前提供 Ascend910 V1；课程原要求为 910C，课程方是否接受 V1 替代尚未确认。Qwen3.5-0.8B，BF16，TP=1，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。
- 修复：NPU Qwen3.5 attention fallback 按 query head 拆分交错的 Q/gate 投影；修复前同一提示词重复输出句点，修复后返回预期的 `OK`。
- 验证：默认 graph-mode 服务启动与预热成功，API 返回 HTTP 200；相关 Q/gate、MHA、GDN 测试 11 项通过。
- 原始响应：`results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json`（归档路径：`results/raw/qwen35_0.8b_q_gate_fix_20261008_run10.json`）。

这只是单提示词 smoke test，不等同于完整 token-ID 正确性验收或性能实验。

## Qwen3.5-27B 权重状态（2026-10-10；下列带时间条目为进度历史）

- 本机正式模型目录中的 11 个权重分片均已通过固定 revision manifest 文件大小与 SHA-256 校验。
- Offline metadata validation reports `model_type=qwen3_5`; all 1,199 weight-index names match the 11 safetensors headers. The local project environment has no Transformers package, so this used standard-library JSON/header parsing and added no dependency.
- 远端同步最终于 10:22 完成：11 个分片均与固定 manifest 大小及本机 SHA-256 匹配，并已无覆盖地晋升到正式 `models/Qwen3.5-27B/`；离线 header 检查覆盖 1,199 个 tensor keys。以下 06:00–09:32 条目记录的是当时的续传进度，不代表当前状态。
- 06:00 本地时间只读远端 listing 显示 11 个标准 `.partial` 合计 13,825,998,848 / 55,563,022,432 bytes（24.88%），shard 1–6 的现有续传仍在增长。正式远端模型目录仍无权重文件；此临时字节数不等同于已校验或可加载权重。
- 06:04:59 复查时标准 `.partial` 总量为 14,641,922,048 / 55,563,022,432 bytes（26.35%）；shard 1–6 各增长约 136 MB，尚无分片完整或可校验。历史 `.retry.partial` 不计入该总量。
- 06:12:31 复查时总量为 15,966,875,648 / 55,563,022,432 bytes（28.74%）；shard 1–6 均继续增长，shard 7–11 未启动，尚无分片达到 manifest 完整大小。
- 06:16:38 复查时总量为 16,663,502,848 / 55,563,022,432 bytes（29.99%）；shard 1–6 重连后均继续增长，shard 7–11 未变化，尚无分片达到 manifest 完整大小。
- 06:19:26 只读复查时总量为 17,153,486,848 / 55,563,022,432 bytes（30.87%）；6 个 SFTP worker 分属 6 个独立父进程、每个活动分片单写者。shard 1–6 均继续增长，shard 7–11 未变化，尚无分片达到 manifest 完整大小，正式目录仍无权重。
- 06:22:27 只读复查时总量为 17,685,454,848 / 55,563,022,432 bytes（31.83%）；6 个 SFTP worker 仍各自独立，shard 1–6 较前次均增长约 88.6 MB，shard 7–11 未变化。尚无分片达到 manifest 完整大小，正式目录仍无权重。
- 06:29:30 只读复查时总量为 18,922,344,448 / 55,563,022,432 bytes（34.06%）；6 个 SFTP worker 仍各自独立，shard 1–6 较前次均增长约 206 MB，shard 7–11 未变化。尚无分片达到 manifest 完整大小，正式目录仍无权重。
- 08:16 状态更正：远端 staging 中 shard 1–6 已达到 pinned manifest 精确大小，且远端 SHA-256 与本机已验证文件匹配；shard 7–11 的五个续传 writer 于 08:14 启动，仍须等其退出后再校验。所有权重目前仍在临时 staging，尚未晋升到正式模型目录。
- 08:47 只读状态：shard 1–6 仍已完成远端大小/SHA-256 核验；shard 7/8/11 的活动 partial 分别为 2,559,471,616 / 2,358,448,128 / 1,728,073,728 bytes，三路续传持续增长。shard9/10 仍为 301,654,016 / 302,153,728-byte partial 且未启动。全部未晋升到正式模型目录。
- 09:06 状态：远端 11 个标准 partial 共 42,624,069,976 / 55,563,022,432 bytes（76.71%，不计旧 retry partial）。shard1–6 继续保持已校验状态；shard11 的 2,148,512,760-byte staging 文件已在无活动 writer 后与本机 SHA-256 匹配。shard7/8/9 有活动续传；shard10 writer 于 09:06 启动，首次状态轮询仍为 302,153,728 bytes，需后续确认增长。所有文件仍在 staging，未晋升到正式模型目录。
- 09:08 状态更新：标准 staging partial 合计 42,961,887,576 / 55,563,022,432 bytes（77.32%）。shard10 在约两分钟内从 302,153,728 增长到 384,380,928 bytes；shard7/8/9/10 四个单写者均仍在运行。shard1–6 与 shard11 的大小/SHA-256 已核验，所有文件仍在 staging、未晋升。
- 09:15 只读复查：标准 partial 合计 44,214,546,776 / 55,563,022,432 bytes（79.58%，不计旧 retry partial）；四个 SFTP 单写者仍活动。shard1–6 与 shard11 保持大小/SHA-256 已核验，shard7–10 尚未完成；没有对不完整分片计算哈希或进行晋升。
- 09:18 只读复查：标准 staging partial 合计 44,718,661,976 / 55,563,022,432 bytes（80.48%）；shard7–10 均较 09:15 增长。shard1–6 与 shard11 保持大小/SHA-256 已核验；未哈希或晋升任何不完整分片。
- 09:29 只读复查：标准 staging partial 合计 46,723,244,376 / 55,563,022,432 bytes（84.09%）；shard7–10 四个单写者仍活动且持续增长。shard1–6 与 shard11 保持大小/SHA-256 已核验，没有哈希或晋升不完整分片。
- 09:32 只读复查：标准 staging partial 合计 47,291,359,576 / 55,563,022,432 bytes（85.11%）；shard7–10 续传仍在增长，已核验分片未变化，未哈希或晋升不完整分片。
- 10:22 状态更新：11 个远端权重分片均达到固定 manifest 大小，并与本机源文件 SHA-256 匹配；已无覆盖地从临时 staging 移入 `models/Qwen3.5-27B/`。远端离线解析了全部 11 个 safetensors header（共 1,199 个 tensor key）。旧的 69,566,464-byte shard1 retry partial 保留在 staging，未删除。权重同步已完成，但尚未启动 27B NPU 服务、固定输入正确性对照或 baseline。

## 4B TP=2 bring-up 与 correctness（2026-10-10）

- 修复 HCCL backend 对 PyTorch 2.9.0+cpu 的 API 兼容性：使用 `all_gather_into_tensor`、`reduce_scatter_tensor`。远端两进程真实 HCCL 回归覆盖普通 all-gather、普通 reduce-scatter 与 token-aware reduce-scatter，均通过。
- Qwen3.5-4B TP=2 服务启动、warmup 与 health check 通过。两次同输入固定样例对照均为 7/8，8 条输入 ID 与 TP2 输出 ID 在两轮间逐条完全相同；唯一差异是 `chinese_beijing` 在生成 index 19（0-based）分叉（engine 25701，reference 1286）。strict token-ID correctness 仍不通过，也没有性能结论。
- 原始 engine IDs：`TP2 raw JSONL`（归档路径：`results/raw/correctness/qwen35-4b-tp2-20261010T000417Z.jsonl`）；对照摘要：`TP2 reference comparison`（归档路径：`results/raw/correctness/qwen35-4b-tp2-20261010T000417Z-compare.json`）。HCCL red/green test logs 与服务日志保存在 `.tmp/`。
- 同一固定 workload 的第二次运行仍为 7/8；8 条输入 ID 与第一次一致，8 条 TP2 输出也逐条完全相同。因此 `chinese_beijing` index 19 的差异可复现。复跑记录：`TP2 repeat raw JSONL`（归档路径：`results/raw/correctness/qwen35-4b-tp2-repro-20261010T0024Z.jsonl`）；reference comparison（归档路径：`results/raw/correctness/qwen35-4b-tp2-repro-20261010T0024Z-compare.json`）。根因仍未定位；未进行性能测量。
- TP2 eager top-2 capture：两 rank 均在目标 position 35/input ID 25 得到 `[25701,1286]` / `[23.375,23.25]`，margin 0.125。Transformers reference 在同一 36-token prefix 对这两个 token 的分数均为 23.375、精确并列，并选择 1286。证据支持近似并列处约一个 BF16 步长的数值差异改变排序；未指认具体 kernel/collective。见 `TP2 logits`（归档路径：`results/raw/correctness/decode_diagnostic/tp2-eager-chinese19-20261010T0034Z-logits.jsonl`）、reference logits（归档路径：`results/raw/correctness/decode_diagnostic/tp2-chinese19-reference-logits-20261010T0045Z.json`） 和目标请求输出（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-4b-tp2-chinese_beijing-turn1-continuous-20261010T003958Z.jsonl`）。原始证据已跨端 SHA-256 核验。
- TP1/TP2 同前缀对照：在相同 36-token prefix、目标 position 35/input ID 25，TP1 eager 的 top IDs 为 `[1286,25701]`、scores 为 `[23.375,23.25]`，且该 TP1 目标输出的 index 19 为参考 token 1286；TP2 两 rank 的排序相反，为 `[25701,1286]`，scores 同为 `[23.375,23.25]`。Transformers reference 对两候选均给出 23.375 并列并选择 1286。两配置都只有 0.125 的 top-2 间隔；该对照支持近似并列决策对执行/并行路径数值敏感，但没有定位到具体 kernel、collective 或其他算子，也不改变 TP2 7/8 的 strict correctness 结论。见 TP1 logits（归档路径：`results/raw/correctness/decode_diagnostic/tp1-eager-chinese19-20261010T0055Z-logits.jsonl`）、TP1 目标请求输出（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-4b-tp1-diagnostic-chinese_beijing-turn1-continuous-20261010T005518Z.jsonl`）；两份文件及 TP1 服务/客户端日志的本地/远端 SHA-256 均一致。

## 指标与正确性采集 smoke（2026-10-08）

- TokenSpeed 指标端点 `:39142/metrics` 已从引擎进程实际读到 `tokenspeed:*` 指标；网关 `:39141/metrics` 不包含这些引擎指标。
- 8 条中文、英文、数学、代码、长上下文和多轮请求已保存至 `results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl`（归档路径：`results/raw/correctness/qwen35_0.8b_20261008_run17.jsonl`）。该脚本只采集对话结果，`output_token_ids` 仍不可得，`token_correctness_verified=false`，不记为完整正确性通过。
- 4 请求遥测 smoke 的 summary：TTFT p50 92.89 ms、TPOT p50 10.52 ms、请求吞吐 2.40 req/s；prompt/generation/prefix-hit token delta 分别为 516/128/512。KV usage 采样发生在请求结束后，为 0.0；不能当作峰值或正式结果。可追溯原始数据：`results/raw/metrics_smoke/qwen35_0.8b_20261008_run17/`（归档路径：`results/raw/metrics_smoke/qwen35_0.8b_20261008_run17`）。
- 新增负载期采样后，8 请求/concurrency 4 的设备 smoke 记录 KV usage peak 0.53125、running peak 5；summary/raw 见 `results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18/`（归档路径：`results/raw/metrics_peak_smoke/qwen35_0.8b_20261009_run18`）。其余 TTFT p50 173.35 ms、TPOT p50 20.62 ms 和吞吐 1.08 req/s 仅用于检查遥测链路，不是正式基线。
- 2026-10-09 首轮 0.8B 普通并发 baseline：目标输入/输出 128/32，concurrency 1/4/8/16，各 3 轮、每轮 32 请求。提示词现按 request id 生成等长变体，实测 prefix hit ratio 为 0。384 条 raw 请求、12 份 summary 和 NPU 前后快照见 `results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/`（归档路径：`results/raw/baseline/qwen35_0.8b_20261009_unique_prompt`）。每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.41/228.89 | 10.48 | 1.82 | 58.17 | 0.12 |
| 4 | 556.17/6297.53 | 17.80 | 2.14 | 68.60 | 0.61 |
| 8 | 884.12/1296.26 | 42.84 | 3.81 | 121.95 | 0.91 |
| 16 | 2760.82/3530.22 | 56.42 | 3.61 | 115.49 | 0.97 |

  并发 16 相对 8 没有增加吞吐，且 KV 接近满载；并发 4 p95 有显著长尾。该结果是 910 V1 上的 0.8B 首轮基线，不是 910C 结果或优化收益。该首轮原始 summary 的 ITL 仍为 null；运行时 ITL 口径修正后另做了同配置复测，结果见 [`0.8B corrected ITL`](baseline.md)，没有回填历史数据。
- 普通 benchmark 曾有一轮所有请求重复相同 prompt 的试跑，prefix hit ratio 约 0.99；保留在 `results/raw/baseline/qwen35_0.8b_20261009/`（归档路径：`results/raw/baseline/qwen35_0.8b_20261009`），仅作热共享前缀样本，不混入上述 baseline。
- 0.8B Agent 首轮：并发 1/4/8、共享比例 0/25/50/75/90%、两轮会话、每轮 3 次重复，共 390 条请求。TTFT 波动很大且共享比例曲线不单调，因此只作为工作负载与采样链路 pilot，不据此得出优化结论。数据见 `results/raw/agent/qwen35_0.8b_20261009_unique_prefix/`（归档路径：`results/raw/agent/qwen35_0.8b_20261009_unique_prefix`）。
- 0.8B 8-turn Agent 复测已完成：prefix 512 tokens、并发 1/4、共享比例 0/50/90%、每组 3 次重复，共 360 条请求。50% 组合 TTFT p50 中位数在并发 1/4 下分别为 4776/4238.8 ms，而 0%/90% 约 182–430 ms；曲线非单调且异常重复出现，需 Profiler/更细粒度 scheduler trace 解释，不作为 prefix-cache 优化结论。raw/summary 与硬件快照见 `results/raw/agent/qwen35_0.8b_20261009_8turn/`（归档路径：`results/raw/agent/qwen35_0.8b_20261009_8turn`）。tool-wait 在该 runner 中是客户端延迟，不代表服务端保留待工具 session。
- 2026-10-09 Qwen3.5-4B 早期 8-turn Agent 客户端等待样本（不同于后文 Workload F）：360 条请求、18 份 summary、36 份硬件快照。每个 session 首轮后请求前等待 100 ms；raw 中 45 条首轮请求的 `tool_wait_ms=0`、其余 315 条为 100。raw 未包含 `tool_wait_policy` 或 `mock_tool_result_injected` 字段，因此不将其称为服务端 tool wait，也不假设注入了 mock 结果或发生 KV 保留；该数据只描述客户端等待负载，不作为优化结论。数据见 `4B 8-turn Agent raw`（归档路径：`results/raw/agent/qwen35_4b_20261009_8turn`）。
- 项目原有课程/CLI/Q-gate/metrics 回归此前 68 项通过；2026-10-10 最新本机 `test/course_project` 为 150 项通过、1 项因 Windows 缺少 Bash 跳过；远端 Session KV/runtime 测试为 90 项、Profiler focused tests 为 26 项通过。此次新增 Profiler fixture 回归及六份 Level1 CANN 描述性数据的可复算汇总。此前远端课程测试通过 63 项。远端复测包含项目内 `ASCEND_WORK_PATH` 环境和 decode 诊断测试。本机使用已有 Python 测试运行时，未安装依赖。
- baseline/Agent/ablation/profiler runner 均支持负载中采集 KV usage 和 running/waiting 峰值；已在 0.8B baseline 与 Agent 负载中使用。
- 0.8B/TP=1 E0–E3 重复消融已在 chunk=512 和 1024 两组完成：输入/输出目标 2048/32、并发 4、每变体 12 请求×3 轮。每组 144 条请求、12 条 summary；两组 raw、日志与硬件快照各 30 个文件均跨端 SHA-256 一致。E3 prefix-hit ratio 两组均约 0.965，但 chunk=512 首轮 TTFT p50 为 5.92 s、后两轮约 0.28–0.29 s，显示显著轮间差异；完整表格和限制见 [`docs/report/ablation_analysis.md`](../report/ablation_analysis.md)。这是重复提示词负载下的描述性测量，不是普遍性能结论。
- Qwen3.5-4B bring-up：两分片 SHA-256、离线配置读取与 safetensors header 检查通过；TP=1 服务完成 warmup，`/v1/models` 与引擎 metrics HTTP 200。6 个 correctness case 共 8 条响应已保存至 `results/raw/correctness/qwen35-4b_20261009_run1.jsonl`（归档路径：`results/raw/correctness/qwen35-4b_20261009_run1.jsonl`）。这只是 chat/API smoke：引擎输出 token IDs 不可得，`token_correctness_verified=false`，不纳入性能比较。
- Qwen3.5-4B 原始 token-ID 对照：`TokenSpeed 记录`（归档路径：`results/raw/correctness/qwen35-4b_20261009_native_ids_run1.jsonl`）中的 8 条生成均与同 revision 的 Transformers greedy 输出完全匹配（8/8；每条 64 tokens）。输入 token IDs 相同，比较结果及首次差异索引见对照 JSON（归档路径：`results/raw/correctness/qwen35-4b_20261009_native_ids_run1_reference_compare.json`）。结论限于 4B/TP1/910 V1 和这组固定样例，不代表其他模型或参数已验收。
- 报告脚本从 raw summary 自动生成 CSV/SVG。既有 `results/plots/` 和按实验归档的图表均保留；最新汇总单独生成在 `results/plots/refresh_20261010_1234/`（归档路径：`results/plots/refresh_20261010_1234`），含 269 条 summary 记录与 9 张 SVG，其中包括 36 条 decode-heavy summary 记录（0.8B/4B/9B 各 12 条）。Session KV serial sweep raw 尚未纳入自动 summary 报告；KV block 与 TP 参数对比仍未完成，因此不生成这几类趋势图。

2026-10-09 的 0.8B prefill-heavy、decode-heavy、agent-concurrency 三种 Level0 Profiler 场景已在 Ascend910 V1 上重采。项目内 `ASCEND_WORK_PATH` 修复后，每种场景均有 CANN `op_summary`/`task_time` 及 MindStudio 输出，CANN parser 日志未见 warning/error。该轮 `Task Type` 均为 N/A，未启用 AICore hardware counters；`Task Duration` 不是纯 AICore 时间。后续 2026-10-10 已另完成 0.8B 8-turn Level1/PipeUtilization 六次采集并记录到非零 AICore 时间行，但仍不能单凭累计算子时间认定设备瓶颈、解释 TTFT 异常或声称优化收益。详细数据与边界见 [`Profiler 分析`](../report/ascend_profile.md)。

4B 的固定样例 token-ID 对照已通过。0.8B 两轮固定输入的 engine IDs 逐条一致；按一致 EOS/stop 口径对齐后 6/8 token 序列匹配，长上下文和多轮第 3 轮各有一条可复现正文差异。关闭 prefix cache/KVStore 后，Eager 与 cudagraph continuous decode 的 64-token 输出完全一致，均在 index 32 偏离参考；逐 token 全量重算与同输入 HF 参考 64/64 一致。追加的 40-token top-2/position 采集及 fresh revalidation 均在 position 1010/input ID 271 重现差异：continuous 选 100532（margin 0.125），full-recompute 两候选 96200/100532 在记录精度下并列并选择 96200；33-token fresh run 的首差仍为 index 32，full-recompute 对 reference 前 33 个 token 全匹配。有效 raw arrays 显示 layer-0 mixed_qkv 有 6142/6144 元素相同，z 完全相同，末 token backend output relative-L2 为 7.84e-4；这些证据仍不能确定 MHA KV、GDN state 或具体 kernel 根因，也未修复。9B 的 token-ID 对照为 stop-aligned 6/8，也有两条可复现正文差异；0.8B/9B 均不作为完整正确性通过。采集与对照见 `0.8B 原始 IDs`（归档路径：`results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1.jsonl`）、0.8B 第二轮（归档路径：`results/raw/correctness/qwen35-0.8b_20261009_native_ids_run2.jsonl`）、0.8B stop-aligned 对照（归档路径：`results/raw/correctness/qwen35-0.8b_20261009_native_ids_run1_stop_aligned_reference_compare.json`） 与 `decode 诊断记录`（归档路径：`results/raw/correctness/decode_diagnostic`）。0.8B/4B/9B 均已完成修正后的服务端 run-level ITL 复测；逐 token 独立时间戳仍未采集，三模型首轮 summary 中原有的 null 值未被回填。各并发分档结果见 [`baseline.md`](baseline.md)。27B token-ID 对照与 bring-up、27B baseline、四模型完整正确性、可用的 kernel-level Profiler、TP=2 和跨模型消融仍未完成。当前性能数据覆盖三模型首轮并发 baseline 及单独 ITL 复测、4B Agent 客户端 Tool-Wait 描述性测量、0.8B Agent pilot 及固定重复提示词下的 E0–E3 描述性消融；Session KV 已另完成串行 whole-request on/off sweep，但多轮 TTFT、并发吞吐和跨模型性能仍未验收。Profiler/4B smoke 负载指标不纳入性能结论。

### Profiler Level1 更新（2026-10-10）

- 随后针对 0.8B 8-turn、50% 共享前缀长尾异常完成 NPU0 Level1/PipeUtilization 六次重复采集：c1/c4 各 3 次、120 条请求。六份 `profiler_info` 均确认实际 Level1 配置，CANN `op_summary` 中出现非零 AICore 指标行。
- `chunk_fwd_kernel_o` 是六份采集中累计 AICore 时间最高的同名算子；这是各任务行的累计值，不代表端到端 TTFT 或因果瓶颈。当前仍未区分算子执行、Host gap 与队列/调度贡献，也没有据此选择优化或声称收益。
- 38 个关键 trace、Profiler metadata、CANN CSV、raw/summary 与硬件快照跨端 SHA-256 全匹配。详细 per-capture 表和原始路径见 [`Profiler 分析`](../report/ascend_profile.md)；所有原始文件保留在项目 `results/`。

## HF exact-prefix cache-path 诊断（2026-10-10）

比较在相同精确前缀上显式运行 HF `use_cache=True/False`，并对齐 TokenSpeed 的 continuous/full-recompute top-2：

| 用例 | 首错 index / position | TokenSpeed engine / reference | HF `use_cache=True` / `False` | TokenSpeed continuous / full-recompute top-2 |
|---|---:|---:|---:|---|
| 0.8B long-context（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-0.8b-long_context_recall-turn1-hf-cache-paths-20261009T174749076713Z.json`） | 32 / 1010 | 100532 / 96200 | 96200 / 100532 | [100532, 96200] / [96200, 100532] |
| 0.8B multi-turn turn 3（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-0.8b-multi_turn_memory-turn3-hf-cache-paths-20261009T201229183868Z.json`） | 2 / 77 | 38251 / 198 | 198 / 38251 | [38251, 198] / [38251, 198] |
| 9B long-context（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-9b-long_context_recall-turn1-hf-cache-paths-20261009T174837641775Z.json`） | 59 / 1035 | 264 / 6278 | 6278 / 264 | [264, 6278] / [264, 6278] |
| 9B multi-turn（归档路径：`results/raw/correctness/decode_diagnostic/qwen35-9b-multi_turn_memory-turn1-hf-cache-paths-20261009T174942038305Z.json`） | 58 / 76 | 1149 / 1 | 1 / 1 | [1149, 1] / [1149, 1] |

这组精确前缀、首错位置的观测表明 HF cache path 会改变两例近似并列 long-context 选择；新增的 0.8B multi-turn turn 3 中，HF `use_cache=True` 的候选 logits 为 17.125/17.125 并选参考 token 198，`False` 与 TokenSpeed 两路径均选 38251（17.125/17.0）。该新增记录使用 Transformers 5.12.0；此前三份 raw 未记录包版本。9B multi-turn 也处于近似并列：HF 两种模式分别记录 27.0/27.0 与 27.125/27.125 并均选 1，TokenSpeed continuous 与 full-recompute 均选择 1149，top-2 margin 均为 0.125。以上不能定位 TokenSpeed 的引擎状态或具体 kernel 根因，也不代表问题已修复。

## GDN prefill/decode 孤立精度检查（2026-10-10）

在 NPU0 上用同一组 64-token 合成输入比较 GDN chunk prefill 与逐 token decode。Prefill 的 BF16 beta 相对 FP32 sigmoid 最大差 0.001866；decode 对 prefill 的 SSM-state relative-L2 从 BF16 beta 情况的 0.003500 降到 FP32 beta 的 0.003024，输出 relative-L2 仍约 0.00447。说明 gate rounding 有可测贡献，但改用 FP32 beta 后路径差异仍存在，不能据此认定 0.8B 首差的根因。未改实现。原始结果见 `GDN cross-path JSON`（归档路径：`results/raw/correctness/decode_diagnostic/gdn_crosspath_revalidate_20261010.json`）。

## Qwen3.5-0.8B Decode-heavy 补测（2026-10-10）

Ascend910 V1/NPU0、TP=1、输入目标 256 tokens，分别设最大输出 512/1024 tokens；并发 1/4，每格 3 轮、每轮 8 请求（每格 24 请求）。服务配置 `max_model_len=4096`、chunk=4096、block=128，prefix cache 开启但观测 hit ratio 为 0。表内各统计量为三轮 summary 中位数；ITL p95 来自服务端 histogram delta。

| 最大输出 tokens | 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | ITL p95 (ms) | 输出吞吐 (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 299.65/310.18 | 10.47 | 14.76 | 90.60 |
| 512 | 4 | 521.84/4721.35 | 18.76 | 19.98 | 167.24 |
| 1024 | 1 | 302.42/306.79 | 10.45 | 14.76 | 91.69 |
| 1024 | 4 | 325.95/968.09 | 18.73 | 19.94 | 191.43 |

512 输出、并发 4 的三轮 TTFT p95 分别为 4721、4742、1066 ms，长尾波动明显；此专项样本用于补 workload 曲线，不据此声称普遍性能收益，也未覆盖 decode 与长 prefill 同时运行的干扰测试。原始记录与硬件快照见 `512 输出组`（归档路径：`results/raw/benchmark/decode_heavy_20261010_1121b_out512`） 和 `1024 输出组`（归档路径：`results/raw/benchmark/decode_heavy_20261010_1121b_out1024`）；32 个跨端文件的 SHA-256 均匹配。服务结束后 NPU0 回到 4% HBM、0% AICore。

## Qwen3.5-4B Decode-heavy 补测（2026-10-10）

Ascend910 V1/NPU0、TP=1，目标输入 256 tokens，最大输出 512/1024 tokens；并发 1/4，每格 3 轮、每轮 8 请求（每格 24 请求）。服务配置 `max_model_len=4096`、chunk=4096、block=128，prefix cache 开启但观测 hit ratio 为 0。表内各统计量为三轮 summary 中位数；ITL p95 来自服务端 histogram delta。

| 最大输出 tokens | 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | ITL p95 (ms) | 输出吞吐 (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 546.98/562.26 | 19.67 | 19.99 | 48.25 |
| 512 | 4 | 1347.38/2140.53 | 35.60 | 37.77 | 104.83 |
| 1024 | 1 | 552.27/560.38 | 19.75 | 20.19 | 49.35 |
| 1024 | 4 | 1074.77/2195.52 | 35.18 | 37.76 | 110.20 |

512 输出、并发 4 的三轮 TTFT p95 为 5584、2141、1976 ms，轮间差异较大；1024 输出、并发 4 的三轮 p95 为 2232、2192、2196 ms。并发 4 的 KV peak 在重复轮次中最高分别达到 1.00 和 0.97。该专项只补充 4B 的 decode-heavy 工作负载曲线，不代表优化收益或跨模型优劣，也未测长 prefill 与 decode 同时运行的干扰。原始记录与硬件快照见 `512 输出组`（归档路径：`results/raw/benchmark/decode_heavy_4b_20261010_1140_out512`） 和 `1024 输出组`（归档路径：`results/raw/benchmark/decode_heavy_4b_20261010_1140_out1024`）；30 个跨端文件（含服务日志和 PID 记录）的 SHA-256 均匹配。服务停止后 NPU0 为 4% HBM、0% AICore，端口 8000 无服务响应。

## Qwen3.5-9B Decode-heavy 补测（2026-10-10）

Ascend910 V1/NPU0、TP=1，目标输入 256 tokens，最大输出 512/1024 tokens；并发 1/4，每格 3 轮、每轮 8 请求（每格 24 请求）。服务配置 `max_model_len=4096`、chunk=4096、block=128，prefix cache 开启但观测 hit ratio 为 0；模型 revision 为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。表内各统计量为三轮 summary 中位数；ITL p95 来自服务端 histogram delta。

| 最大输出 tokens | 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | ITL p95 (ms) | 输出吞吐 (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 566.72/569.32 | 24.88 | 24.99 | 38.45 |
| 512 | 4 | 1295.86/2025.55 | 40.35 | 39.80 | 90.73 |
| 1024 | 1 | 564.48/573.95 | 24.87 | 25.76 | 38.72 |
| 1024 | 4 | 614.64/2292.46 | 40.98 | 39.76 | 87.14 |

并发 4 的 TTFT p95 三轮分别为 2197/2026/1982 ms（输出 512）和 2292/2317/1943 ms（输出 1024）；两档 KV peak 在重复轮次中最高均达到 1.00。该专项补充 9B decode-heavy 负载曲线，不代表优化收益或跨模型优劣，也未测长 prefill 与 decode 同时运行的干扰。原始记录与硬件快照见 `512 输出组`（归档路径：`results/raw/benchmark/decode_heavy_9b_20261010_1205_out512`） 和 `1024 输出组`（归档路径：`results/raw/benchmark/decode_heavy_9b_20261010_1205_out1024`）；29 个跨端文件（含服务日志）的 SHA-256 均匹配。服务停止后 NPU0 为 4% HBM、0% AICore，端口 8000 无服务响应。

## Qwen3.5-4B baseline（2026-10-09）

- 设备/环境：课程服务器提供 Ascend910 V1（课程原要求 910C，替代是否获课程方确认尚未知），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。权重已完成校验，服务 warmup 与 API/engine metrics smoke 通过。
- baseline 配置：输入/输出目标 128/32 tokens，32 条独立 prompt/轮，prefix-hit ratio 为 0，并发 1/4/8/16、各 3 轮，共 384 条请求、12 份 summary、24 份 NPU 前后快照。`max_num_seqs=8`，因此 concurrency=16 会排队。
- 每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 261.96/262.90 | 19.54 | 1.15 | 36.70 | 0.31 |
| 4 | 895.02/914.42 | 33.56 | 2.05 | 65.74 | 0.81 |
| 8 | 772.55/1700.51 | 84.82 | 2.33 | 74.54 | 1.00 |
| 16 | 3972.56/5011.11 | 82.99 | 2.37 | 75.88 | 1.00 |

- c8 到 c16 输出吞吐只从 74.54 增至 75.88 tok/s；c16 TTFT p50 升至约 3.97 s，且配置最多运行 8 个 sequence，故该负载下 c16 有排队，不能据此声称扩展或优化收益。该首轮 summary 的 ITL 为 null；修正后的独立复测见 [`baseline.md`](baseline.md)。
- 原始请求、summary 与硬件快照见 `results/raw/baseline/qwen35_4b_20261009_full/`（归档路径：`results/raw/baseline/qwen35_4b_20261009_full`）；跨端同步后的 JSONL 大小与 SHA-256 一致。

## Qwen3.5-9B baseline（2026-10-09）

- 设备/环境：课程服务器提供 Ascend910 V1（课程原要求 910C，替代是否获课程方确认尚未知），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。四个 safetensors 分片均通过官方 manifest 文件大小与 SHA-256 校验；离线 `AutoConfig` 和权重索引/header 检查通过。服务 warmup 成功，readiness、`/v1/models` 和 TokenSpeed engine metrics 均 HTTP 200。
- baseline 配置：输入/输出目标 128/32 tokens，32 条独立 prompt/轮，prefix-hit ratio 为 0，并发 1/4/8/16、各 3 轮，共 384 条请求、12 份 summary、24 份 NPU 前后快照。`max_num_seqs=8`，因此 concurrency=16 的请求会排队，不等同于 16 条同时运行。
- 每档三轮 summary 中位数：

| 并发 | TTFT p50/p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 276.91/279.19 | 24.79 | 0.95 | 30.49 | 0.25 |
| 4 | 924.27/4634.49 | 38.76 | 1.54 | 49.32 | 0.81 |
| 8 | 803.42/1770.21 | 91.84 | 2.17 | 69.41 | 1.00 |
| 16 | 4261.56/5351.97 | 91.13 | 2.20 | 70.48 | 1.00 |

- c8 到 c16 输出吞吐仅由约 69.41 增至 70.48 tok/s，同时 KV 峰值已满且 c16 TTFT p50 显著升高；这只是该 9B/TP1/910 V1 配置与负载的容量观察，不构成优化收益或其他硬件上的结论。该首轮 summary 的 ITL 为 null；修正后的独立复测见 [`baseline.md`](baseline.md)。
- 原始请求、summary 与硬件快照见 `results/raw/baseline/qwen35_9b_20261009_full/`（归档路径：`results/raw/baseline/qwen35_9b_20261009_full`）。

## Qwen3.5-9B token-ID correctness（2026-10-09）

- 六个固定用例产生 8 条请求；每条最多生成 64 tokens。两轮采集中 8/8 输入 token-ID 与 8/8 引擎输出 token-ID 完全一致，说明该设置下结果可复现。
- 与同 revision 的 Transformers greedy reference 比较，stop-aligned 仅 6/8 匹配。`long_context_recall` 与 `multi_turn_memory` 分别在输出位置 59、58（0-based）出现正文差异；两条均达到 64-token 上限。故不能将 9B 判定为完整 token-level correctness 通过。
- 原始记录及比较结果见 `run 1`（归档路径：`results/raw/correctness/qwen35-9b_20261009_native_ids_run1.jsonl`）、`run 2`（归档路径：`results/raw/correctness/qwen35-9b_20261009_native_ids_run2.jsonl`） 和reference 对照（归档路径：`results/raw/correctness/qwen35-9b_20261009_native_ids_run1_reference_compare.json`）。
- 追加 decode-path 诊断中，两例的 continuous/full-recompute 输出各自相同，且在首错步都选择相同 token；long-context index 59 两路径输入/position 为 11/1035、候选 264/6278，continuous logits 为 30.375/30.25、full-recompute 为 30.25/30.25 并列，而参考选择 6278。multi-turn index 58 输入/position 为 10033/76、候选 1149/1，continuous logits 为 27.125/27.0、full-recompute 为 27.25/27.125，而参考选择 1。near-tie 现象不等于已定位根因；multi-turn full-recompute 采集还缺少输出 index 20–26 的 logits 行，但完整输出与 index 58 关键行均保留。raw 证据见 `decode_diagnostic`（归档路径：`results/raw/correctness/decode_diagnostic`）。

## Qwen3.5-9B GDN FP32 candidate matched-cap follow-up（2026-10-11）

- 在 diagnostic TP1/BF16 配置、固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`、NPU0 上将输出上限设为 64，和 2026-10-09 run 对齐。八条请求全部生成 64 tokens；与 pinned Transformers greedy reference 比较为 7/8 raw exact、7/8 stop-aligned exact。唯一 mismatch 是 `long_context_recall` 的 index 59（0-based）：engine ID 264，reference ID 6278。它仍是未解决的严格正确性失败。
- 与旧 run 的 8 条记录比较，6 条 `client_input_token_ids` 完全相同；其中 5 条引擎输出序列也相同。相同输入的 `multi_turn_memory` turn 1 从旧 run 的 index 58 / ID 1149 改为候选 ID 1，匹配 reference。由于此处生成历史改变，turn 2–3 输入不同，不把它们计作同输入配对结果。旧 run 与本轮均各自基于实际输入单独计算 Transformers reference。
- 新 raw 64-token candidate run（归档路径：`results/raw/correctness/gdn_fp32_candidate_9b_20261011_max64.jsonl`） 的 SHA-256 为 `91d07e2f2df475b9941e4137f7cbb36cf979113f80594cc2ac4708963f27a0cd`；reference 对照（归档路径：`results/raw/correctness/gdn_fp32_candidate_9b_20261011_max64_reference_compare.json`） 为 `76f4f1a2f89aa6bca4242d30cc57e65a6f50388540e8efbb8e36d5871c79a241`。两文件本地与远端 SHA-256 一致。reference 因可选 fast-attention 库未安装而回退到 PyTorch；未安装依赖。项目服务已停、端口关闭。之后的只读检查发现逻辑 NPU0 的 chip 1 上有另一任务的 Python 进程（约 427 MB、NPU 利用率 0%）；未终止，该进程随后自行退出。最终检查显示 NPU0 两芯片进程表为空、HBM 4%、NPU 利用率 0%。

## Qwen3.5-4B Agent Workload F（2026-10-09）

- 环境：课程服务器提供 Ascend910 V1（课程原要求 910C，替代是否获课程方确认尚未知），openEuler 24.03/aarch64，BF16、TP=1；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。只使用默认 NPU 0，NPU 2 未参与。
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
- TTFT 随共享比例明显非单调，且并发 4 的 KV 峰值为 1.0；不据此声称 prefix-cache 收益或性能优化。原始数据见 `raw requests, summaries, and hardware snapshots`（归档路径：`results/raw/agent/qwen35_4b_20261009_tool_wait`），图表与汇总表见 `Workload F report`（归档路径：`results/plots/after_4b_agent_tool_wait_20261009`）。38 个跨端同步文件的 SHA-256 全部一致。

<a id="qwen35-9b-agent-workload-f-2026-10-10"></a>
## Qwen3.5-9B Agent Workload F protocol replication（2026-10-10）

- 环境：Ascend910 V1/NPU0、openEuler 24.03 (LTS-SP3)/aarch64、BF16/TP=1；Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。课程要求的 910C 尚未获确认；本文不把 910 V1 写成 910C。
- 配置：512-token 目标共享前缀、8 turns、每请求最多 32 输出 tokens、并发 1/4、目标 cross-session prefix ratio 0/0.5/0.9、3 warmups、每格 3 次连续重复；100 ms `one-session` client mock wait。总计 360 raw requests、18 run summaries、36 个 NPU0 前后快照。38 个 raw/summary/hardware 文件本地与远端 SHA-256 全部匹配。
- 表格中的数值为三次 run-summary 的中位数，方括号为这三次重复的最小值–最大值；不是置信区间。输出吞吐按总输出 token 数除以每轮 wall time，包含客户端 sleep；单请求 TTFT/TPOT/E2E 不包含发起请求前的 sleep。

| 并发 | 目标共享比例 | TTFT p50 (ms) | TTFT p95 (ms) | 输出吞吐 (tok/s) | 峰值采样 active-page ratio | 服务端观测 prefix-hit ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0% | 394.00 [393.88–403.42] | 819.55 [817.22–844.80] | 24.82 [24.44–24.83] | 0.250 [0.250–0.281] | 0.760 [0.760–0.760] |
| 1 | 50% | 396.53 [394.94–424.48] | 826.78 [823.42–3,017.51] | 24.75 [18.30–24.86] | 0.281 [0.281–0.313] | 0.768 [0.767–0.769] |
| 1 | 90% | 449.98 [382.47–748.75] | 3,897.74 [832.44–3,956.52] | 14.80 [12.22–25.05] | 0.438 [0.438–0.438] | 0.768 [0.768–0.770] |
| 4 | 0% | 1,210.66 [1,113.05–2,709.26] | 6,272.52 [3,971.41–8,659.52] | 29.44 [18.75–34.42] | 1.000 [0.969–1.000] | 0.657 [0.369–0.755] |
| 4 | 50% | 4,753.41 [3,852.31–7,422.15] | 8,941.72 [6,231.45–13,348.57] | 16.68 [12.77–19.86] | 1.000 [1.000–1.000] | 0.320 [0.235–0.410] |
| 4 | 90% | 3,729.79 [2,190.96–4,428.47] | 8,975.76 [5,657.54–10,105.08] | 18.70 [16.70–22.96] | 1.000 [1.000–1.000] | 0.439 [0.286–0.516] |

- 固定顺序的小样本呈非单调变化：目标共享比例增加没有稳定提高吞吐或降低 TTFT。并发 4 的采样 active-page ratio 接近 1，但它是 scheduler active-pages/total-pages 指标，不是 HBM 利用率或显式 free/used KV block 计数，也不足以证明 TTFT 变高的原因。即使目标 cross-session 比例为 0%，服务端 prefix-hit ratio 仍约 0.76，因其包含同一 session 多轮历史前缀复用，不能将二者混为一谈。
- 126 条请求记录在 `session-0` 首轮后的 7 轮附加固定 mock 文本并在客户端 sleep 100 ms。该实验未让服务进入 WAITING_TOOL 状态，也未验证 Session KV 留存。单条请求的 TTFT/TPOT/E2E 不计前序 sleep；wall-time 吞吐计入。
- 每格只有 3 次按固定顺序执行的 run，分析单位是 run summary，未把 360 条请求伪作独立样本。报告 `mean ± sample SD` 和 run range；未作显著性检验，也不声称因果或通用收益。完整逐轮数据、限制、图表与 claim candidates 见统计附录（归档路径：`results/analysis/qwen35_9b_agent_tool_wait_20261010/stats-appendix.md`）、分析报告（归档路径：`results/analysis/qwen35_9b_agent_tool_wait_20261010/analysis-report.md`）、图表目录（归档路径：`results/analysis/qwen35_9b_agent_tool_wait_20261010/figure-catalog.md`）与实验复盘（归档路径：`results/analysis/qwen35_9b_agent_tool_wait_20261010/2026-10-10--agent-tool-wait--r00--cross-model-replication.md`）。原始记录及 36 份硬件快照位于 `raw bundle`（归档路径：`results/raw/agent/qwen35_9b_20261010_tool_wait`）；runner/server 日志保存在项目 `.tmp/`。
- 这是性能负载采集，不验证生成质量；9B 既有固定用例 stop-aligned reference 对照仍为 6/8。该结果与 4B Workload F 属于同协议在不同模型上的描述性复测，不是隔离模型大小的受控比较。

<a id="qwen35-4b-cross-model-ablation"></a>
## Qwen3.5-4B E0–E3 cross-model ablation（2026-10-10）

- 环境：Ascend910 V1/NPU0、openEuler 24.03/aarch64、Qwen3.5-4B 固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、BF16/TP=1。两组矩阵均为目标输入/输出 2048/32、并发 4、每变体每轮 12 条测量请求、3 轮、2 条预热请求。服务端实际 usage 为 1855/32；相同 tokenizer 的 0.8B 对照为 1857/32。所有请求复用完全相同的提示词。
- 每个 chunk 矩阵含 144 条 raw 请求、12 条 run summary。各 raw 目录内 30 个数据/服务日志文件加 `.tmp/` 下的 detached runner log，共 31 个相关文件，本地与远端 SHA-256 匹配。两个 canonical raw 目录为 `chunk=512`（归档路径：`results/raw/ablation/qwen35_4b_formal_chunk512_detached_20261010`） 与 `chunk=1024`（归档路径：`results/raw/ablation/qwen35_4b_formal_chunk1024_20261010`）；图表 CSV/SVG 另存于 `自动报告目录`（归档路径：`results/plots/refresh_20261010_4b_ablation`）。

下表每个数值是三轮 run-level summary 的中位数，方括号为观测范围。每轮是 12 请求的批次，三轮为连续、固定顺序运行，因此只作描述性比较。

| 矩阵 | 变体 | TTFT p50 (ms) | TPOT p50 (ms) | 输出吞吐 (tok/s) | Prefix-hit ratio | KV sampled peak |
|---:|:---:|---:|---:|---:|---:|---:|
| 512 | E0 | 6257.30 [6239.63–6261.91] | 232.80 [232.74–234.04] | 9.49 [9.47–9.49] | 0.000 | 0.969 |
| 512 | E1 | 6298.94 [6297.71–7720.98] | 233.80 [184.31–233.87] | 9.48 [9.48–9.52] | 0.000 | 0.969 |
| 512 | E2 | 7740.34 [7396.88–9162.29] | 160.80 [160.67–213.09] | 8.35 [7.99–9.08] | 0.000 | 1.000 |
| 512 | E3 | 512.76 [489.06–2319.26] | 33.62 [33.60–33.67] | 55.88 [31.80–82.34] | 0.966 | 0.938 |
| 1024 | E0 | 6253.66 [6248.34–6260.11] | 232.61 [232.21–234.81] | 9.49 [9.47–9.52] | 0.000 | 0.969 |
| 1024 | E1 | 6297.11 [6294.10–7731.77] | 233.68 [184.18–233.82] | 9.48 [9.48–9.52] | 0.000 | 0.969 |
| 1024 | E2 | 7652.35 [7573.23–10638.88] | 193.91 [193.31–194.22] | 9.16 [7.26–9.17] | 0.000 | 1.000 |
| 1024 | E3 | 491.03 [488.28–492.81] | 33.90 [33.90–33.94] | 82.07 [81.51–82.19] | 0.966 | 0.938 |

- 对本次精确重复提示词，E3 的 prefix-hit ratio 为 0.966；相对 E2，TTFT p50 中位数降低 93.4%/93.6%，输出吞吐为 6.69×/8.96×（chunk=512/1024）。该结果不外推到不同用户后缀、一般 Agent 会话或其他模型/并发。
- 与 0.8B 同目标矩阵对照，E2/E1 的 TTFT p50 在 0.8B 上低 5.3%/6.6%，而 4B 上高 22.9%/21.5%；两模型的 E2 输出吞吐均低于 E1。相同 prompt 下 E3 的命中与改善方向相似，但这不能抵消跨模型 E2 方向差异，也不能作为通用策略结论。
- E1 在两个矩阵的第 1 轮 TTFT 均是高值，而变体顺序固定，说明轮次/进程状态可能影响小样本观测。E2 相对 E1 在此 1855-token 固定长度静态输入下未显示 TTFT/吞吐优势。E2/chunk=1024 的 histogram-derived ITL p95 三轮均约 1266.67 ms；两矩阵 E0 第 3 轮落入 `+Inf` 桶，只能报告 ≥2000 ms 下界。不得据当前证据选择全局 chunk 默认。
- `n=3` 连续 run summaries，不将 36 个并发请求视为独立样本；mean±sample SD、逐轮值及 ITL 桶边界见统计附录（归档路径：`results/analysis/qwen35_4b_ablation_20261010/stats-appendix.md`），主要图表见图表目录（归档路径：`results/analysis/qwen35_4b_ablation_20261010/figure-catalog.md`），完整逐图复盘见4B 消融报告（归档路径：`results/analysis/qwen35_4b_ablation_20261010/2026-10-10--qwen35-4b-ablation--r00--ablation-report.md`）。三条 summary 的 Prometheus prompt/generation token delta 各比 raw request 总数多 1；输出吞吐仍由 raw 请求计数和墙钟时间计算，详细口径与异常记录在附录。NPU0 chip 0 的结束快照为 E0/E1 23% HBM、E2/E3 21%，不是峰值/单独 KV 用量。设备是 910 V1，不是 910C；长 prefill + decode 干扰实验仍受设计门控，本结果未启动该实验。

## Qwen3.5-4B Session KV reuse acceptance

- 环境：课程服务器 Ascend910 V1/NPU0、openEuler 24.03/aarch64、BF16、TP=1；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。generic prefix cache 与 KVStore 关闭，current-frontier 策略，greedy `max_new_tokens=8`。
- 缓存边界：轮末不额外执行无采样 forward；下一轮把上一轮最后采样的 token 与新 suffix 一起预填，因此从第 2 轮起 `newly_prefilled_tokens` 比客户端 suffix 多 1。
- 两条独立 session 各执行两轮，所有 session 输出 token IDs 均与同输入的 stateless full-history 对照一致；metadata 与 frontier 计数验证通过：

| Session | 轮次 | 总上下文 tokens | 新预填 tokens | 复用 tokens | 输出 ID 匹配 |
|---|---:|---:|---:|---:|---|
| session-0 | 1 | 43 | 43 | 0 | 是 |
| session-0 | 2 | 70 | 20 | 50 | 是 |
| session-1 | 1 | 39 | 39 | 0 | 是 |
| session-1 | 2 | 67 | 21 | 46 | 是 |

- 初次验收使用 `overlap_schedule=false`；随后在 `overlap_schedule=true` 下完成一次复验。复验中两条交错 session 各执行两轮，4/4 输出 ID 序列继续与 stateless full-history 对照一致，复用计数仍为 50/46 tokens，关闭后的两次探测均按预期拒绝。该 overlap 复验仅证明本配置下的正确性/续轮交接，不是性能实验。
- 初次 raw token IDs、逐轮延迟、配置及 NPU 快照见 `Session KV run`（归档路径：`results/raw/session_kv/qwen35-4b-20261009T161435357104Z.json`） 与同目录 `hardware/`；overlap-enabled 复验及快照见 `overlap-enabled raw run`（归档路径：`results/raw/session_kv_overlap_20261010`）。
- 用户确认 current-frontier 后，另做了 3 次相同的 overlap-enabled matched mixed-load 重复（每次两条 session、两轮，并同时提交 stateless controls）。每次 4/4 session 输出匹配、2/2 close probe 拒绝；turn-2 metadata 稳定为 session-0 新预填/复用 20/50、session-1 21/46 tokens。六个 turn-2 session latency 中位数 676.04 ms，stateless controls 676.03 ms；由于二者在同一批次并发，这组数据没有显示可归因的端到端加速，不构成 Session on/off 参数 sweep。runner 的 `hardware_after` 快照在 Python 进程退出前采集，NPU0 chip 0 当时为 21% HBM；进程退出后的独立检查及后续 `hardware_before` 均回到 4%，且无项目进程。3 个 raw JSON 与 6 个 NPU0 快照见 `session_kv_perf_20261010`（归档路径：`results/raw/session_kv_perf_20261010`），同步文件 SHA-256 全部匹配。

### Session KV serial paired latency sweep（2026-10-10）

- 状态更正：前文概览中“尚无 Session reuse 性能 sweep”是本节记录前的状态；以下是之后完成的独立串行配对结果。
- 配置：Ascend910 V1/NPU0，Qwen3.5-4B 固定 revision，BF16/TP=1，`current-frontier`，`overlap_schedule=true`；请求串行执行，通用 prefix cache 与 KVStore 均关闭。每档 4 次重复，每次新建 Session，先做不计时 prime，再交替顺序测量 session 续轮与 stateless full-history 对照；greedy `max_new_tokens=8`。
- 12/12 次 prime 与测量续轮的原始输出 token IDs 均匹配；metadata frontier 校验全部通过。三个上下文档的实际首轮输入长度为 541/1053/1565 tokens；续轮每次新预填 20 tokens，分别复用 548/1060/1572 tokens。
- 下表为 `Engine.generate` 整个请求的墙钟延迟（不是 TTFT）。paired reduction 是每一对先计算相对降幅，再取 4 次重复的中位数。

| 目标上下文 | 实际上下文 | Session 中位数 (ms) | Stateless 中位数 (ms) | 配对降幅中位数 |
|---:|---:|---:|---:|---:|
| 512 | 541 | 264.07 | 1199.21 | 77.95% |
| 1024 | 1053 | 260.96 | 2020.91 | 87.11% |
| 1536 | 1565 | 253.15 | 2840.09 | 91.28% |

- 1536 档有一个 stateless 配对样本为 6463.69 ms，其余三个为 2828.27–2851.49 ms；中位数和配对降幅使用原始四次重复，未剔除该点。结果支持本模型/设备/参数下 Session KV 显著减少续轮请求延迟，但样本数仅 4，且不是并发吞吐、TTFT 或跨模型/跨硬件结论。
- 原始逐请求 token IDs、延迟、启动配置与 NPU0 前后快照见 raw JSON（归档路径：`results/raw/session_kv_perf/qwen35-4b-session-kv-perf-20261010T015028239838Z.json`）、before（归档路径：`results/raw/session_kv_perf/hardware/20261010T015028239838Z-before.txt`）、after（归档路径：`results/raw/session_kv_perf/hardware/20261010T015028239838Z-after.txt`）。较早的 2048-token 试跑触发 CANN kernel `coreDim=66464` 超过 65535 的设备错误且未产生 JSON，已排除；runner 随后增加了逐组原子检查点。

本文件只接受可追溯到 `results/raw/` 的测量值，并同时记录 TokenSpeed commit、模型 revision、启动参数、硬件/软件环境、并发与输入输出长度分布。
