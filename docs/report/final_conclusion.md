# 最终结论

## 当前验收边界（2026-10-13）

用户已接受实际 Ascend910 V1，并授权使用空闲的全部 NPU。27B TP2 已启动，混合缓存容量与上下文边界计算修复后真实 4095-input/1-output 请求通过；TP1 在95%内存预算下也已启动。27B 固定八例参考对照为7/8，代码样例首差index54。引擎TP1/TP2、eager/graph输出一致；指定位置引擎Decode与完整重算top2 logits并列，与HF完整重算一致，HF cached路径则改变排序。严格一致性未通过，不能据近似并列豁免；见[27B诊断](../project/27b_recovery_20261013.md)。27B E0五档并发三轮基准正在运行，尚无完整性能结论。

4B Long Prefill/Decode 三轮配对及CSV/SVG已验收，512配置没有降低本次staged Decode E2E，不构成优化收益；详见[验收](../project/long_prefill_acceptance_20261013.md)。0.8B单请求Level1 TTFT窗口时钟verified，145文件跨端校验及CSV重生成通过，但只支持时间重叠观察，不满足算子因果归属；详见[采集](../project/profiler_single_request_capture_20261013.md)。本机最新课程回归407 passed/13 skipped。整体项目仍有严格正确性、完整消融/调优、最终配置及整套提交复现等缺项；以下内容为按日期保留的历史记录，不应覆盖本节最新状态。

## 软件验证补充（2026-10-12）

本机完整课程测试为 399 passed、6 skipped；远端现有课程测试为 350 passed、1 skipped，两端测试集合不同。Long Prefill/Decode 实现已同步远端并通过回归；真实 4B tokenizer 的 256/2048/3584-token 输入与 21 组运行 seed 格式检查通过。修复了重复文本拼接后 token 供给不足的构造问题，配方版本为 v2。正式测量尚待已批准计划中的单独确认，因而没有新增该实验的硬件结果。证据与备份位置见[预检记录](../project/long_prefill_preflight_20261012.md)。

当前整体提交状态见[课程验收清单](../project/submission_status.md)；后文按日期保留历史快照。软件回归通过不能替代四模型、目标硬件或性能实验验收。

## 最新状态补充（2026-10-11）

Qwen3.5-9B 的 GDN FP32 candidate 在与旧 run 相同的 `max_tokens=64` 上限下重新验证：8 条固定请求中 7 条 raw token IDs 与 pinned Transformers greedy reference 完全一致，stop-aligned 结果也是 7/8；唯一剩余差异是 `long_context_recall` 输出 index 59（0-based，engine ID 264、reference ID 6278）。这仍不满足严格正确性验收，也不是性能收益证据。与旧 run 对齐后，只有 6 条 `client_input_token_ids` 完全相同；其中 5 条 engine 输出序列不变，`multi_turn_memory` 第一轮则从旧 ID 1149 改为参考 ID 1。后两轮上下文因第一轮生成改变，不能作为同输入配对。原始记录与逐 token 对照见 [64-token candidate run](../../results/raw/correctness/gdn_fp32_candidate_9b_20261011_max64.jsonl) 和[参考对照](../../results/raw/correctness/gdn_fp32_candidate_9b_20261011_max64_reference_compare.json)。

同一 candidate 的官方 bring-up 复测中，4B 的八条固定样例（每条最多 64 tokens）对 pinned reference 为 8/8 raw/stop-aligned exact，见[原始记录](../../results/raw/correctness/gdn_fp32_candidate_4b_20261011_bringup_max64.jsonl)与[对照文件](../../results/raw/correctness/gdn_fp32_candidate_4b_20261011_bringup_max64_reference_compare.json)。9B 官方配置的直接对照为 7/8 raw/stop-aligned exact，唯一差异是 `long_context_recall` 输出 index 59（engine 264、reference 6278），见[官方配置原始记录](../../results/raw/correctness/gdn_fp32_candidate_9b_20261011_bringup_max64.jsonl)及[对照文件](../../results/raw/correctness/gdn_fp32_candidate_9b_20261011_bringup_max64_reference_compare.json)。两项结果均为固定样例正确性证据，不证明 FP32 改动的因果效果或性能收益。

本机 workspace Conda 环境中的 `test/course_project` 最新复验为 385 passed、6 skipped。该套件验证项目软件回归，不代表 NPU 模型验收；本轮没有安装依赖或使用 NPU。单请求 TTFT Profiler 时间窗 runner/分析器及 fake tests 已实现；真实采集和 Torch/CANN 时钟重叠核验尚未进行，Profiler 因果归属仍未验收。

README 启动入口也做了本轮复验：远端 Bash wrapper 对四个模型配置和单独的 4B TP=2 配置均成功生成启动命令；27B 参数为 world-size 2。`--dry-run` 不启动服务，因此不能替代 27B runtime、正确性或性能验收。

## 最新状态更正（2026-10-10 21:54 Asia/Shanghai）

经批准的 Qwen3.5-4B/TP=1/NPU0 Session KV 多轮 TTFT 测量已完成：4 次、每次 16 轮，共 64 个串行配对；Session 与 stateless 输出 token IDs 为 64/64 一致。两侧 TTFT 中位数为 122.71 ms 与 1,394.54 ms，配对降幅中位数为 91.52%。结论仅适用于该固定配置、合成会话与原生 Engine TTFT 口径，不代表 HTTP、并发 Agent、其他模型/硬件或统计显著收益。

Qwen3.5-27B/TP=2 的首次 NPU 尝试已在加载两 rank 权重并完成图像/视频 encoder warmup 后停滞，HTTP 服务未 readiness；没有运行 token-ID correctness 或 baseline。CANN 日志记录 HCCL AllReduce `opLaunchIdx=14`、300 秒 watchdog timeout，以及 stream 47/task 1952、`pendingNum=883` 的重复同步超时。补回的 rank-1 device trace 多次显示等待 rank-0 的 AllReduce 通知；留存的 rank-0 trace 没有对应的运行时事件。这说明观测到 rank 间进度不对称，但不能解释 rank-0 未完成/未通知的原因。源代码中的下一阶段依次包括通信准备、attention/KV 构造、executor 创建、autotune 和 graph capture；服务日志在预期 autotune/capture 进度标记前结束。配置设有 1/2/4 decode-graph capture sizes，但没有 callsite 证据，不能断言已进入 graph capture。单元素 AllReduce 与 capture warmup 中的 barrier 相容，但不能将 `opLaunchIdx=14` 归属到该 barrier；准确停滞阶段和根因仍未知。12 个原始证据文件均已与远端 SHA-256 核对；服务已停止、NPU0 无残留设备进程。详情见[启动证据](../../results/raw/bringup/qwen35_27b_20261010T1316Z/)和[诊断计划](../superpowers/plans/2026-10-10-qwen35-27b-startup-diagnosis.md)。在得到不同于原启动的明确诊断前，不重复长时间占用 NPU。

项目仍未完成：27B 启动/正确性/性能、0.8B 与当前 9B candidate 及 4B TP=2 的 reference 差异、Profiler 因果归因、长 Prefill/Decode 的真实 NPU 测量及完整提交复现/答辩材料仍待处理。4B/TP=1 长 Prefill staged runner 与分析器已在本地实现并通过软件测试，但没有硬件结果；真实测量仍须单独确认。实际设备为 Ascend910 V1，不能写作课程要求的 910C；是否允许替代仍需课程方确认。

## 当前状态更正（2026-10-10 12:34 local）

本节优先于下方较早时间点的状态快照。27B 的 11 个权重分片已在远端正式目录完成 manifest 大小与 SHA-256 校验并晋升；远端离线 header 检查覆盖 1,199 个 tensor keys，但 27B NPU bring-up、TP=2、固定输入正确性和 baseline 仍未完成。0.8B/4B/9B 的 decode-heavy 补测已完成（输入 256 tokens，输出上限 512/1024，并发 1/4，每格 3 轮×8 请求）；91 个跨端原始/硬件/服务文件 SHA-256 匹配，数据仅描述本设备固定负载。4B/TP=1/NPU0 的 Session KV 串行 paired sweep 已完成：12/12 输出序列匹配，512/1024/1536 目标上下文下 `Engine.generate` 整轮延迟的配对中位降幅为 77.95%/87.11%/91.28%；这是 4 次重复的串行请求延迟，不是 TTFT、并发吞吐或跨硬件结论。最新本机 `test/course_project` 为 172 passed、1 platform skip，当前 Session KV focused tests 为 49 passed。0.8B 新增的 33-token GDN revalidation 再次在 continuous path 的输出 index 32 复现首差，而 full-recompute 前 33 个 token 与 reference 全匹配；对齐的 layer-0 capture 与 earlier valid presnapshot 重现相同数组数值，仍无 backend 前配对状态，故不定位具体根因。0.8B/9B reference 对照仍各有两例可复现差异；4B/TP=2 仍为 7/8；Profiler 尚未完成 kernel、Host 与排队/调度的因果归属。课程要求 910C，而当前实测设备为 910 V1，课程方是否接受替代仍未确认。Long Prefill 干扰配置与 Session reuse 多轮 TTFT 方案仍各自待确认；本结论不把 decode-heavy 或 Session 串行 sweep 表述为完整优化验收。

HF exact-prefix `use_cache=True/False` 对照（2026-10-10）显示：0.8B 与 9B long-context 两例中，HF 缓存路径选择参考 token（96200/6278），不缓存路径选择原 TokenSpeed token（100532/264）；新增的 0.8B multi-turn 第三轮在 index 2/position 77 首错，HF `use_cache=True` 因 17.125/17.125 并列选择参考 token 198，`False` 与 TokenSpeed continuous/full-recompute 均选择 38251（17.125/17.0）。该记录使用 Transformers 5.12.0。9B multi-turn 的 HF 两种模式分别记录 top-2 为 27.0/27.0 和 27.125/27.125 并均选 1；TokenSpeed 两种 decode 路径均选 1149，top-2 margin 均为 0.125。总体证据支持特定近似并列选择对 HF cache path 敏感，但不能解释所有差异、定位 TokenSpeed 引擎/状态或具体 kernel 根因，也没有修复。逐例 raw 见 [0.8B long-context](../../results/raw/correctness/decode_diagnostic/qwen35-0.8b-long_context_recall-turn1-hf-cache-paths-20261009T174749076713Z.json)、[0.8B multi-turn](../../results/raw/correctness/decode_diagnostic/qwen35-0.8b-multi_turn_memory-turn3-hf-cache-paths-20261009T201229183868Z.json)、[9B long-context](../../results/raw/correctness/decode_diagnostic/qwen35-9b-long_context_recall-turn1-hf-cache-paths-20261009T174837641775Z.json) 和 [9B multi-turn](../../results/raw/correctness/decode_diagnostic/qwen35-9b-multi_turn_memory-turn1-hf-cache-paths-20261009T174942038305Z.json)。

状态更新（2026-10-10，补充 Session KV overlap 复验；覆盖前文“默认 overlap 交接待复验”的状态）：用户选择 `current-frontier` 作为每轮缓存边界。Qwen3.5-4B/TP=1、Ascend910 V1/NPU0 的 overlap-enabled matched correctness/reuse 复验中，两条交错 session 各执行两轮，4/4 轮输出 token IDs 均与 stateless full-history 对照一致；第二轮分别复用 50 和 46 个上下文 tokens。generic prefix cache 与 KVStore 关闭，关闭后的两次探测按预期拒绝。该结果验证了本配置下的续轮交接正确性，不是性能收益结论；原始记录和 NPU 快照见 [`overlap-enabled Session KV run`](../../results/raw/session_kv_overlap_20261010/)。

TP=1/TP=2 首差数值对照更新（2026-10-10）：对 `chinese_beijing` 的同一 36-token 前缀在 eager 模式下复采，position 35/input ID 25 的 TP1 top-2 为 token 1286/25701、score 23.375/23.25，输出 index 19 选择 1286；TP2 两 rank 都以相同的 0.125 间隔反向排序为 25701/1286。Transformers 对两候选均给 23.375 并列并选 1286。该结果支持近似并列处对数值执行路径敏感，但未定位具体算子或 collective；TP2 strict 对照仍为 7/8，不是通过。

27B 权重状态更新（2026-10-10 04:59 local）：本机正式目录中的 11 个权重分片均已通过固定 revision manifest 大小与 SHA-256 校验；离线 config/index/header 检查确认 `qwen3_5`，且 1,199 个索引张量名与分片头部完全匹配。远端项目 `.tmp/qwen35_27b_sync_fc05daec/` 正通过可续传、SSH 压缩方式接收已校验分片；远端临时副本尚未完成大小/SHA-256 核验，不计为正式同步。27B 服务、TP=2、token-ID correctness 与 baseline 均未验收。

27B 同步快照更新（2026-10-10 09:08 local）：远端 staging 标准 partial 共 42,961,887,576 / 55,563,022,432 bytes（77.32%，不含历史 retry partial）。shard1–6 与 shard11 已匹配 manifest 大小及 SHA-256；shard7–10 正由四个独立 SFTP writer 续传，其中 shard10 的远端长度在两分钟内增长约 82 MB。所有分片仍位于项目 `.tmp/` staging，未晋升；27B 服务启动、正确性和 baseline 仍待完成。

截至 2026-10-10，已在课程服务器提供的 Ascend910 V1 上完成 Qwen3.5-0.8B/4B/9B 的 TP=1 服务 bring-up，并完成三者各自的首轮普通并发 baseline。课程原要求为 910C，课程方是否接受 V1 替代尚未确认，因此本报告不将其称为 910C 实测。Bring-up 中定位并修复了 Ascend attention fallback 对 Qwen3.5 交错 Q/gate 投影的拆分错误；固定提示词修复后返回预期 `OK`。本机 `test/course_project` 172 项通过、1 项因 Windows 缺少 Bash 跳过；当前 Session KV focused tests 49 项通过。4B/TP1/NPU0 current-frontier Session KV correctness/reuse 及 overlap 续轮复验均通过，另完成 serial on/off context latency sweep；其结果是 `Engine.generate` 整轮请求延迟，不是 TTFT/并发吞吐。0.8B 与 9B 的 token-ID greedy 对照均 stop-aligned 6/8，各有两条正文差异；4B 固定样例 8/8 匹配。0.8B long-context 的 Eager/cudagraph continuous decode 与 full-recompute 仍在 index 32 分叉；fresh GDN revalidation 进一步复现相同首差，full-recompute 前 33 tokens 与 reference 匹配，valid layer-0 capture 仍不足以定位具体状态/算子根因。9B 的两例诊断也未定位根因；multi-turn full-recompute 的 index 20–26 logits 行未采到，完整输出与 index 58 关键行保留。Profiler 仍未区分 kernel、Host gap 与排队/调度贡献。27B NPU bring-up、TP2 correctness/performance 与完整四模型矩阵尚未完成。

Profiler 更新（2026-10-10）：在 0.8B/TP=1、512-token 上下文、50% shared-prefix、8-turn、100 ms 客户端 mock wait 负载下，NPU0 完成 Level1/PipeUtilization 六次采集（c1/c4 各 3 次，共 120 请求）。六份 `profiler_info` 均记录 Level1 配置，六份 CANN `op_summary` 均有非零 AICore 时间行。`chunk_fwd_kernel_o` 是各采集中累计 AICore 时间最高的算子，但该值是跨任务行聚合，不是端到端时延，也不建立因果关系；尚未区分 kernel、Host gap 与排队/调度贡献。结果与原始产物见 [`Profiler 分析`](ascend_profile.md)。

首轮普通并发 baseline 共 384 条请求：目标输入/输出 128/32，concurrency 1/4/8/16，每档 3 轮，使用等长独立 prompt，prefix hit ratio 为 0。中位输出吞吐分别为 58.17、68.60、121.95、115.49 tok/s；KV usage 峰值分别为 0.12、0.61、0.91、0.97。并发 16 未超过并发 8 吞吐，且 TTFT 明显升高；并发 4 的 TTFT p95 存在长尾。当前这些是 0.8B 的首轮基线观察，不构成优化收益或普适容量结论。首轮 summary 的 ITL 为 null；修正运行时重复计数后另完成同配置 ITL 复测，结果与口径见 [`docs/project/baseline.md`](../project/baseline.md)。

4B 首轮 baseline 同样为 384 条请求、128/32 tokens、并发 1/4/8/16 各 3 轮。输出吞吐中位数为 36.70、65.74、74.54、75.88 tok/s，KV 峰值为 0.31、0.81、1.00、1.00；服务 `max_num_seqs=8`，c16 排队，c8 到 c16 的吞吐增量很小且 TTFT p50 升至 3.97 s。该结果只描述本次 4B/TP1/910 V1 配置，不构成优化收益。逐档指标见 [`4B baseline`](../project/final_results.md#qwen35-4b-baseline-2026-10-09)。

0.8B Agent 两轮 pilot 完成 390 条请求，覆盖并发 1/4/8、共享前缀比例 0/25/50/75/90%；8 轮会话复测另完成 360 条请求（并发 1/4、比例 0/50/90%、512-token 初始上下文）。两组数据均有较大 TTFT 波动；8-turn 的 50% shared-prefix 组合在并发 1/4 下 TTFT p50 约 4.78/4.24 s，远高于 0%/90% 的部分组合。曲线非单调，属于待 Profiler 定位的异常，不能据此判断 prefix cache 收益。tool-wait 在 runner 中模拟客户端等待，不代表服务端持有待工具 session。普通 benchmark 与 Agent runner 已分别修正为独立请求 prompt、轮内共享且轮间隔离的 Agent 前缀，以减少基准污染。

0.8B/TP1 E0–E3 重复消融已完成两组（E2/E3 chunk=512/1024）：每组目标输入/输出 2048/32、并发 4、每变体 12 请求×3 轮。两组各 144 条测量请求；固定重复提示词下 E3 prefix-hit ratio 约 0.965，但 chunk=512 首轮与后两轮的 TTFT/吞吐差异很大。结果只描述该共享前缀场景；三轮样本、固定执行顺序和缺少 kernel-level Profiler 均不足以证明普遍优化收益。详见 [`消融分析`](ablation_analysis.md)。

追加的 0.8B 逐层状态摘要显示：首错位置的 continuous 与 full-recompute 输入一致，但两者 24 层 MHA KV/GDN state 的哈希均不同，最早层级差异出现在 layer 0 GDN。L2 norm 汇总相近，但这不是逐元素误差证据；现阶段只确认执行路径状态存在数值差异，尚未确定具体算子根因或修复方法。

后续离线审计与 fresh NPU0 revalidation 更正：可用于逐 token 比较的是 `gdn_layer0_presnapshot_20261009_1615` 和新生成的 `gdn_layer0_revalidate_20261010_1053` 原始数组，两次有效捕获逐数组 SHA-256 一致。新捕获在同一 position 1010/input ID 271 重现连续 decode 首错：33-token 输出中 continuous 首差 index 32（100532），full-recompute 与 reference 的 33 个 token 全匹配（index 32 为 96200）；logits 候选分差分别为 0.125 与记录精度下并列。layer-0 `mixed_qkv` 为 6142/6144 元素精确相同（max abs 3.81e-6），`z` 为 2048/2048 精确相同，末 token backend output relative-L2 为 7.84e-4。较早的 `gdn_layer0_full_20261009_1545` 把 full-recompute backend output 保存成 `[1,1011,16,128]`，且其 `mixed_qkv` 与后续重复捕获不一致，不能用于逐 token 归因。两次有效 capture 仍未保存对齐的 backend 前状态，因此证据未定位到 GDN kernel 或其他具体根因。

Qwen3.5-4B 权重已在项目内完成下载和 SHA-256 核验，TP=1 服务 warmup 成功，8 条固定样例与 pinned Transformers greedy token-ID 对照完全匹配；4B baseline 已完成 384 条请求。9B 四个权重分片也已校验，离线结构检查与 TP=1 服务 smoke 成功；两轮固定请求输出可复现，但相对 reference 仅 6/8 stop-aligned；9B baseline 也完成 384 条请求。两种模型的并发配置均为 1/4/8/16、每档 3 轮，`max_num_seqs=8`，c8/c16 KV 峰值达 1.00，需谨慎解释吞吐平台。截至 2026-10-10 02:17 local，27B shard1/2 已通过 manifest 大小与 SHA-256 校验并存在于本机正式模型目录；shard3/4 旧 `.incomplete` 的 SHA-256 不符但原样保留，新的 `.verify` partial（721,420,288 和 620,756,992 bytes）也已保留。两路分块下载均因 curl exit 35 / HTTP status 000 暂停，未写入正式权重；当前证据仅支持 HF 域名/上游路径特定 TLS 失败，Clash 规则/上游仍待确认。远端仍只有 metadata，27B 尚未 bring-up/TP2。4B/9B corrected ITL 已完成；27B TP 对照、长上下文/decode-heavy 矩阵、可用的 kernel-level Profiler 和完整正确性仍未完成，因此目前不能回答全系统吞吐提升、最大并发提升或真实设备瓶颈等最终问题。所有当前结论均绑定原始记录，后续须以更多模型、稳定重复和有效 Profiler 结果继续更新。
27B 状态更正（2026-10-10 05:30 local，覆盖上文 02:17 local 的旧下载快照）：本机 11 个 shard 均已通过固定 revision manifest 大小与 SHA-256 校验。远端项目临时目录中的 11 个标准 `.partial` 累计 8,357,224,448/55,563,022,432 bytes（15.04%）；另有 69,566,464-byte 的旧 shard-1 `.retry.partial` 原样保留，当前批次不会写入它。六路自动续传（shard 1–6）仍在运行，其余标准 partial 均保留待续传。远端哈希核验与正式目录晋升尚未完成。27B 服务、TP=2、token-ID correctness 和 baseline 仍未验收。

27B 状态更正（2026-10-10 05:35 local，覆盖 05:30 checkpoint）：远端 11 个标准 `.partial` 累计 9,585,102,848/55,563,022,432 bytes（17.25%）；六路自动续传（shard 1–6）仍在运行，暂未有分片达到 manifest 精确大小。远端 SHA-256 核验及正式目录晋升仍未完成。

27B 状态更正（2026-10-10 05:41 local，覆盖 05:35 checkpoint）：远端 11 个标准 `.partial` 累计 10,482,843,648/55,563,022,432 bytes（18.87%）；六路自动续传仍在运行，暂未有分片达到 manifest 精确大小。远端 SHA-256 核验及正式目录晋升仍未完成。

27B 状态更正（2026-10-10 05:45 local，覆盖 05:41 checkpoint）：远端 11 个标准 `.partial` 累计 11,186,024,448/55,563,022,432 bytes（20.13%）；六路自动续传仍在运行。逐片大小检查未见完成分片；远端 SHA-256 核验及正式目录晋升仍未完成。

27B 状态更正（2026-10-10 05:53 local，覆盖 05:45 checkpoint）：远端 11 个标准 `.partial` 累计 12,591,464,448/55,563,022,432 bytes（22.66%）；六个 SFTP 进程仍在运行，精确大小/哈希校验与正式目录晋升尚未完成。05:58 本机检查仍看到六个 SFTP 进程，但未确认更新的远端字节数，因此不能据此推断 05:53 后的传输进度。

27B 状态更正（2026-10-10 06:00 local，覆盖 05:53 checkpoint）：只读远端 listing 确认 11 个标准 `.partial` 累计 13,825,998,848/55,563,022,432 bytes（24.88%）；活动的 shard 1–6 仍在增长，六条 SFTP 续传仍运行，shard 7–11 暂无活动进程。正式模型目录仍无权重分片；远端哈希核验和晋升未完成。另有一个历史 shard-1 `.retry.partial`，不计入上述标准分片总量。

27B 状态更正（2026-10-10 06:04:59 local，覆盖 06:00 checkpoint）：远端 11 个标准 `.partial` 共 14,641,922,048/55,563,022,432 bytes（26.35%）；现有 shard 1–6 writer 仍活跃，且每片较 06:00 增长约 136 MB，shard 7–11 未增长。尚无分片达到 manifest 完整大小；旧 shard-1 `.retry.partial` 69,566,464 bytes 不计入总量。正式目录无权重分片，哈希核验及晋升仍待完成。

27B 状态更正（2026-10-10 06:12:31 local，覆盖 06:04:59 checkpoint）：远端 11 个标准 `.partial` 共 15,966,875,648/55,563,022,432 bytes（28.74%）；现有 shard 1–6 writer 均活跃且较前次各增长约 221 MB，shard 7–11 未增长。尚无分片达到 manifest 完整大小；额外 shard-1 `.retry.partial` 69,566,464 bytes 不计入总量。正式目录无权重分片，哈希核验及晋升仍待完成。

27B 状态更正（2026-10-10 06:16:38 local，覆盖 06:12:31 checkpoint）：远端 11 个标准 `.partial` 共 16,663,502,848/55,563,022,432 bytes（29.99%）；shard 1–6 的 writer 已重连/换 PID，六个活动分片均较前次增长约 107–120 MB，shard 7–11 未增长。尚无分片达到 manifest 完整大小；历史 `.retry.partial` 不计入。正式目录无权重分片，哈希核验及晋升仍待完成。

27B 状态更新（2026-10-10 06:19:26 local）：远端 11 个标准 `.partial` 共 17,153,486,848/55,563,022,432 bytes（30.87%）。只读复核到 6 个 SFTP worker/6 个独立父进程，活动 shard 1–6 各单写者；六个分片较 06:16:38 均增长约 81.6 MB，shard 7–11 未变化。尚无分片达到 manifest 大小，正式目录仍无权重；远端哈希核验及晋升未完成。

27B 状态更新（2026-10-10 06:22:27 local）：远端 11 个标准 `.partial` 共 17,685,454,848/55,563,022,432 bytes（31.83%）。只读复核到 6 个 SFTP worker/6 个独立父进程，活动 shard 1–6 各单写者；六个分片较 06:19:26 均增长约 88.6 MB，shard 7–11 未变化。尚无分片达到 manifest 大小，正式目录仍无权重；远端哈希核验及晋升未完成。

27B 状态更新（2026-10-10 06:29:30 local）：远端 11 个标准 `.partial` 共 18,922,344,448/55,563,022,432 bytes（34.06%）。只读复核到 6 个 SFTP worker/6 个独立父进程，活动 shard 1–6 各单写者；六个分片较 06:22:27 均增长约 206 MB，shard 7–11 未变化。尚无分片达到 manifest 大小，正式目录仍无权重；远端哈希核验及晋升未完成。

历史快照（2026-10-09，本地；Session KV 边界选择和 27B 下载状态已由上方/下方 2026-10-10 更新覆盖）：4B 与 9B corrected ITL 复测均已完成，每模型 384 请求、12 份 summary、24 份 NPU0 快照；12/12 组样本数与原始 token 计数一致，两组各 26 个文件跨端 SHA-256 匹配。三模型 corrected ITL 图表见 [`P50`](../../results/plots/after_corrected_itl_20261009/itl_p50_concurrency.svg) 和 [`P95`](../../results/plots/after_corrected_itl_20261009/itl_p95_concurrency.svg)。分档中位数见 [`docs/project/baseline.md`](../project/baseline.md)。当时服务组已停止，NPU0 两个 chip 均无进程，NPU2 未使用。27B 大文件续传当时暂停；0.8B layer-0 数组审计仍不能归因具体 kernel；HF exact-prefix `use_cache=True/False` 对照待做。Session KV 当时已选定引擎级方案，但尚未确定最终采样 token 的 KV 边界。此前 14:22、15:01 的 27B 进度均为历史快照，不代表当前下载状态。

TP=2 正确性状态更新（2026-10-10 08:47 local）：4B/TP2 两轮固定 8-case engine output 逐条一致，均相对 pinned Transformers greedy 为 7/8。`chinese_beijing` 在生成 index 19 的首差可复现；两 rank 的 TP2 top scores 为 token 25701/1286 的 23.375/23.25，而 reference 对相同 prefix 的两 token 都是 23.375 并列并选择 1286。这支持近似并列处一个 BF16 步长改变排序的解释，不证明具体算子根因；因此 TP2 strict correctness 仍未验收，性能尚未测。原始 logits、reference 分数和逐次 raw records 见 [`TP2 correctness evidence`](../project/final_results.md)。
