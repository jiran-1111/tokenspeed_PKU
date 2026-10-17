# 课程提交验收状态（2026-10-15）

依据任务书第 54 节验收清单及第 41 节重点实验核对。项目整体尚未通过验收；文件存在与软件测试通过均不能替代设备实验。这里列出当前证据和继续完成所需工作。

## 当前交付与后续项目

HW1已完成原四项任务、报告、同源服务器复验及GitHub发布：分支 `course/hw1-20261015`，提交 `1d113e54f789a22b14cbbdccbc6230c3f4d0e574`。教学网由用户本人提交；四模型总项目继续执行。后续分支 `course/agent-20261015` 尚未提交/推送。

后续R5源码1398项清单SHA `ef7b3e028233517746147f36da92efea679a29ba50471df4574e3cac9c103a27`，服务器586通过/零skip。新[9B FP32输出头候选](9b_lm_head_precision_20261015.md)在正式图模式原八条raw/stop-aligned均8/8；默认BF16的旧7/8和Session失败保留。四模型同R5原生Session均通过两轮8配对/4次关闭拒绝及重开；4B/27B同R5原八条也各8/8。2026-10-17取回后逐项核对新增41项证据和服务器1398源文件，零差异；这是有限原生回归，尚非完整四模型最终Agent验收。

候选成本已审计：96请求/252 HBM样本，48配对输出ID相同；Decode耗时约+7.5%–9.3%、目标chip观察HBM峰值25874→27814 MB。首轮解析失败、第二轮末臂退出采样失败均保留，末臂只按协议失败补跑，92项成本证据逐项SHA核对通过。跨模型并发Agent、实际工具等待/KV留存、TP匹配性能、完整消融和最终配置/总报告仍欠。下文日期表格中的旧9B“仍7/8”、HW1“未闭环”及旧软件计数是历史状态，以上述分配置/分源码证据为准。

## 第一次作业终态

2026-10-15最终源证据：Stage B R3的272向量冻结门槛全过，旧1%68失败/条件argmax13差异保留；1396相关源码跨端一致、course569passed/0skip；关键模块与Session108passed/0skip、203属性；最终Decode Profiler及96正式请求/12摘要/121个HBM样本已完成。最终原始/分析809文件本机逐项SHA核对零差异。报告已按这些证据定稿，见[HW1报告](../report/homework1_compute_subsystem.md)与[公开证据](../evidence/hw1_20261015/README.md)。本交付采用独立分支`course/hw1-20261015`，目标为`jiran-1111/tokenspeed_PKU`；279项明确交付文件与服务器一致，Git暂存源码与服务器验证的1396文件逐项一致，不覆盖目标main。四模型Agent总项目未完成。

## HW1历史过程（以下“未执行/采集中/Profiler待采”等由上述终态取代）

2026-10-15 Stage B最新终态：原16条验证输入、272个完整logits全部通过冻结双门槛；原1%仍有68向量失败、固定前沿argmax不同13处，分别保留。CPU主参考及引擎退出0，1396源文件对齐，服务器course569passed/0skip，NumPy独立公式全部复算一致。详见[Stage B记录](homework1_stage_b_20261015.md)末节；下方“采集中/尚未执行”为旧状态。最终Decode Profiler、报告定稿及发布仍未完成，无新commit/push。

2026-10-15接续执行：[Stage B执行记录](homework1_stage_b_20261015.md)已建立，冻结16条验证组的独立入口、完整logits捕获与固定续接已实施，CPU FP32/eager主参考正在采集。数值结果尚未完成，不将软件专项测试通过视为验证组通过；原1%失败、greedy和Session证据独立保留。此前下段“尚未执行”现更新为“Stage B采集中”，Profiler、报告定稿和Git发布仍未完成。

2026-10-15最新：四参考采集全部完成16/16，144个原始文件本机重哈希均匹配远端冻结provenance；校准分析与独立只读审计完成。针对最早BF16差异的局部复核定位到CPU BF16 `rsqrt` 的批量尺寸效应，独立复审后用户已批准有限经验判据，冻结记录见[数值判据冻结](homework1_frozen_numerical_gate_20261015.md)；原16条validation输入的CPU主参考和NPU引擎隔离验证尚未执行，旧1%工程失败保持。GDN epsilon正式17项（13NPU+4CPU）通过；MHA D256跨页六项NPU、三项CPU解析自检通过；新正式诊断hook和before-state快照修复已通过服务器同源course全套549 passed/0 skip及NPU Session 1 passed。相关XML和源码跨端SHA已核对、设备释放。最终Decode Profiler、报告定稿、最终版本发布验证和GitHub上传仍未完成；尚无向指定GitHub fork提交的新commit/push/PR。

第一阶段0.8B单chip报告为[计算子系统草稿](../report/homework1_compute_subsystem.md)，三人职责对应的实现、实验和报告由助手继续全部完成，不交回成员执行，也不冒称成员已亲自完成。

后续[源码对齐与复验](homework1_snapshot_alignment_20261015.md)已补：正式诊断脚本及before-state修复双审/服务器专项通过，39个测试/配套源跨端SHA一致，服务器course549passed/0skip；symlink安全检查真实执行。对齐版本0.8B真实Session复验1passed，8组输出/context配对及关闭重开均满足，设备释放。退出阶段pending-task/日志流错误保留，未称无警告退出。参考状态舍入局部CPU诊断准备中，不冻结新判据；Profiler、报告及Git发布仍未完成。

- 算子矩阵、GDN路径、随机典型/边界数值、多步state/隔离及Session证据已有；[真实输入GDN独立递推](homework1_gdn_trained_oracle_20261014.md)对FP32/FP64四项原模块包络通过，不代替完整模型。
- 后续[参考语义审计](homework1_gdn_reference_audit_20261014.md)发现旧Prefill oracle用clamp而生产/固定HF用sum+eps；旧结果保留。两个测试参考已纠正，四ratio近epsilon的CPU解析式RED→GREEN、双审查及17项（13NPU/4CPU）正式设备复验均完成，原NPU容差未改。原训练输入CPU FP64只读核对的单样例及归档限制另列，不与新增epsilon归档混称，不宣布端到端通过。
- [当前生产基线/HBM](homework1_baseline_hbm_20261014.md)完成96正式请求/12 summaries、actual130/940×32/128；121采样的负载观察峰值8465MB，最大采样间隔2.414s。46文件跨端哈希一致，不称瞬时峰值或有效KV总量。
- 现有完整logits额外1%工程合同仍失败，未修改阈值。教师及原任务未指定该数值，用户已批准独立参考校准规格及阶段A。工具服务器80项测试通过；四参考实际320参数canonical值指纹相同、全部完成16条。新[参考分析审计](homework1_reference_calibration_audit_20261015.md)解释当前证据及BF16缓存状态舍入机制，仍不宣称全部误差已解释。候选引擎数值判据未形成/批准，参考自身误差不能自动豁免engine。见[校准规格](../superpowers/specs/2026-10-14-homework1-reference-calibration-design.md)。
- 用户已批准补一次最终0.8B/TP1单chip Decode Profiler，须数值验证结束后源码固定并串行执行，未启动。Git仅local署名和匿名fetch历史核验，当前没有新commit/push/PR；指定fork未被本任务修改，服务器最终对应源码验证后才发布。
- 最终当前版本Decode/Host/HBM归因、提交版报告、最终配置/源码快照与新进程复验未完成；原四模型Agent总目标不变。

## 模型与硬件

| 要求 | 当前证据 | 缺项 |
|---|---|---|
| 0.8B 运行与正确性 | HW1同源16输入/272完整向量冻结双门槛全过，旧1%失败保留；模块/state及Session通过；项目R5 Session通过 | 原历史greedy分叉不取消；项目最终统一版本性能/Agent仍欠 |
| 4B 运行与正确性 | [当前生产源码TP1复验](current_4b_correctness_20261014.md) max64八样例raw/stop-aligned 8/8、20文件cross-SHA；此前Session KV配对通过 | 固定样例不是任意长度/语义普遍正确；TP2、最终同版本Session/性能仍需复验 |
| 9B 运行与正确性 | 原BF16八条7/8保留；[显式FP32 head候选](9b_lm_head_precision_20261015.md)在R5正式graph八条8/8、原生Session通过 | 成本完整验收、更多负载/最终配置与发布仍欠；有限样例不外推质量 |
| 27B 运行与双 chip 策略 | [当前TP2八样例复验](current_27b_correctness_20261014.md)raw/stop-aligned 8/8、23文件cross-SHA；此前4095+1边界及[E0 Workload A](27b_e0_workload_a_20261013.md)480请求/15 summaries已有 | 当前8例不替代任意长度/Session/语义普遍正确；旧E0性能不冒充新源码，匹配TP/通信性能及最终发布复验仍欠 |
| 910C / Kunpeng 规定资源 | [只读身份核实](hardware_identity_20261014.md)SoC为Ascend910_9362、短系列Ascend910_93，映射A3/910C；DMI产品名Atlas800I A3，CPU0x48/0xd02及一手来源支持Kunpeng920系列 | 不凭V1字段认定非910C；完整采购配置、CPU具体子型号及教师single chip/device口径仍待核实 |
| HBM 与 NPU Profiler | 当前0.8B采样与加载/退出快照齐全；[单请求时间窗实采](profiler_single_request_capture_20261013.md)确认Level1/PipeUtilization、clock最大误差800ns、145文件跨端一致 | 瞬时峰值不由周期采样保证；双chip完整证据、当前Decode与系统因果瓶颈归属仍欠 |

## 推理系统与负载

| 要求 | 当前证据 | 仍需完成 |
|---|---|---|
| 统一模型实现、启动与环境 | [架构](architecture.md)、[功能矩阵](feature_matrix.md)、README 和四模型bring-up；27B已实际TP2运行 | 四模型最终同版本完整回归与分发快照 |
| Continuous batching | 复用上游 C++ scheduler；三模型并发曲线已有；[C++ 生命周期与新请求准入测试](kv_block_metrics.md)通过 | 硬件负载下不同输出长度的生命周期时间线、active batch 及 NPU 利用率证据 |
| Prefill/Decode 调度与 chunked prefill | 0.8B/4B E0–E3重复矩阵；[4B staged实测](long_prefill_acceptance_20261013.md)三轮、六次启动、15配对及CSV/SVG验收完成 | 本次512未降低staged Decode E2E，不能据此选通用默认chunk；当前版本完整归因仍欠 |
| KV block/page cache | 上游能力与 active-page ratio 已接入；[LCM 块计数](kv_block_metrics.md)已补齐软件发布/采样/summary并通过验证 | 新计数实际端点验收、碎片与容量证据；Qwen3.5 固定粒度限制需如实说明 |
| Session reuse 与多轮 | 4B原生Engine串行context/16-turn TTFT完成；[0.8B Session验收](homework1_session_acceptance_20261013.md)8组配对、关闭/重开及metadata通过 | HTTP Agent runner仍全历史；并发/完整跨模型Session性能未验证 |
| Prefix cache 与共享前缀 | 共享比例负载与固定重复提示词消融已有 | 一般 Agent 稳定收益、缓存正确性和更完整并发覆盖 |
| Tool wait 与多 Session | 4B/9B 客户端 mock wait 负载已有 | 客户端协议不等同服务端 WAITING_TOOL/KV 留存验收 |
| 长上下文/Decode heavy | 三模型decode-heavy及4B 2K/3.5K staged长Prefill干扰实测已有 | 其他模型/更完整长度曲线、当前版本稳态Decode归因 |

## 重点实验 A–I

| 实验 | 当前完成范围 | 缺项 |
|---|---|---|
| A/B：并发 vs 吞吐、TTFT/TPOT | 三模型历史Workload A及27B E0五档并发各三轮均有实测 | 四模型旧配置不能冒充统一最终配置；客户端c32不等于引擎同时运行32请求 |
| C：Context length vs TTFT | Session context 延迟与多轮 TTFT 已有 | 独立长 Prefill 长度矩阵 |
| D：Chunk size vs 延迟/吞吐 | 0.8B/4B重复prompt的512/1024矩阵，4B staged 4096/512干扰配对已验收 | 更多负载/模型、完整瓶颈归因和最终默认值选择 |
| E：Prefix ratio vs TTFT/吞吐 | 已有 pilot 与多轮共享比例数据 | 异常长尾解释及一般收益验收 |
| F：Conversation turn vs TTFT OFF/ON | [64 对 CSV](../../results/analysis/session_kv_ttft_20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a/session_kv_multiturn_ttft.csv)及 SVG 已有 | 当前范围为 4B 原生 Engine 串行测量 |
| G：KV block size vs HBM/并发/吞吐 | 使用率曲线已有 | block-size sweep 不具备，不能伪造粒度可调性或最优结论 |
| H：Baseline vs Full System | [消融分析](../report/ablation_analysis.md)有 E0–E3 | E4–E7/最终系统、相同协议下完整对照 |
| I：TP1 vs TP2 | 4B TP2 smoke/诊断；27B HCCL基础通信、TP2恢复及E0实测 | 严格正确性闭环及匹配协议TP吞吐/时延/每chip HBM/通信对比 |

## 可复现与提交材料

- 四模型统一启动入口、correctness、普通/Agent benchmark、ablation、Profiler 和报告脚本均存在；实际运行边界见 README。
- `configs/baseline/` 四模型 E0 静态配置已有；**`configs/final/` 尚不存在**，需参数调优证据后选择，不能将 bring-up 默认值直接命名为最终最优配置。
- 自动报告脚本及现有 CSV/SVG 已有；历史 `final_repro_20261011` 中 11 张 SVG 可解析、summary CSV 有 356 行。缺数据的实验图仍未生成。
- 六份报告 Markdown 与答辩提纲已存在，内容仍是带缺项的工作稿；需最终实测结果、图表、结论和整套复现验收后定稿。
- 本机最近课程集合为415 passed/13 skipped；后文不同日期计数是历史证据，不是最新发布验收。KV/Profiler/Long Prefill远端聚焦与整套历史结果见各文档，不能从旧远端集合数宣称与当前本机源码完全一致；最终统一版本仍需新进程复验。
- 清晰分阶段提交、最终版本标识与可分发源码快照仍待完成；当前 dirty worktree 保留，远端没有 Git 元数据。

## 历史过程记录（以下旧状态由上文当前表取代）

2026-10-13 更新：用户已批准 Long Prefill 正式测量，但第一组在服务参数校验阶段退出，未产生任何请求样本。已核实默认 KVStore 与 prefix cache 关闭互斥；拟在两组中共同关闭 KVStore，等待对该控制项修正的确认。失败目录保留且六个文件本地/远端 SHA-256 一致，详见 [启动失败证据](long_prefill_startup_20261013.md)。尚无三个有效配对轮次，不能宣告本实验完成。

同日逐样本复核既有官方配置正确性对照：0.8B 的六个非长上下文 raw 差异均为参考序列末尾含停止 token、引擎输出不含该 token；stop-aligned 后这六个样本一致。仍不一致的是 `long_context_recall`，0-based 首差索引 32，两边均 64 tokens。9B 唯一不一致也是 `long_context_recall`，0-based 首差索引 59，两边均 64 tokens。这两项并非仅终止 token 差异；未新增硬件测量，也不据此认定数值差异原因或放宽验收标准。

用户后续已批准 Long Prefill 两组共同关闭 KVStore 的修正，回归通过并启动新 run `20261013T030821237999Z-e4ba674feff4474abe765af68fcebfad`，正式配对结果尚待验收。Profiler 单请求采集保留其单独授权门。27B 按用户选择暂缓。其余正确性、优化/消融、调优及最终提交缺项继续保留，不能据本地测试通过宣告项目完成。

2026-10-13 最终更新：[Long Prefill 限定实验验收](long_prefill_acceptance_20261013.md)已完成，最终 run 为 `20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4`。六次服务正常启动/退出，五种条件各三轮有效配对；CSV/SVG 通过解析、两端校验和本机重生成。前一个 run 仅有第一组有效数据，因 TIME_WAIT 误判停止并保留，不纳入最终统计。表中 Long Prefill“待实测”条目已由本更新取代；最终默认 chunk、整体系统调优及算子归因仍未完成。

该阶段最后完整回归为本机 404 passed/13 skipped、实际远端 381 passed/1 skipped，取代以上 2026-10-12 的最新测试计数；两端集合仍不完全一致，不声称整体发布验收通过。

27B 恢复更新：用户要求解决 27B、授权全部设备，并接受实际 Ascend910 V1。基础两 rank HCCL 通信通过；27B TP2 已 readiness，八个固定样例执行完成。混合缓存容量规划及上下文边界两项缺陷已修正，真实 4095-input/1-output 请求通过，详见 [27B 恢复诊断](27b_recovery_20261013.md)。参考正确性对照正在空闲逻辑 NPU2 运行，性能矩阵及 TP1/TP2 对比未完成；上述表格中的“暂缓/未 readiness”现已由本更新取代。本机最新回归 407 passed/13 skipped。

Profiler 后续验收更新：[单请求采集](profiler_single_request_capture_20261013.md)已获批准并完成；实际 Level1/PipeUtilization、clock verified（最大误差 800 ns）、145 文件两端 SHA-256 以及本机 CSV 字节重生成均核实。该时间重叠实验不填补完整系统 Top3 算子因果归属缺项。
