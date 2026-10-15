# Session KV 16 轮 TTFT 配对实验设计

## 状态

设计规格已获用户确认；实施计划见 `docs/superpowers/plans/2026-10-10-session-kv-multiturn-ttft.md`。截至 2026-10-10，本地已实现 transcript/stream helper、64 对串行 sweep、raw 检查点、严格 CSV/SVG 报告、CLI、项目环境入口和完整 raw provenance。focused tests 为 46 passed，完整 `test/course_project` 为 224 passed、1 skipped；独立规格复核和代码质量复核均通过。远端 focused tests 为 46 passed；NPU0 的 4B 验收测量已完成，64/64 配对及报告验证通过，五项产物均已同步并通过 SHA-256 对照。

## 已完成测量（2026-10-10）

- 固定 Qwen3.5-4B revision、Ascend910 V1/NPU0、TP=1、原生 Engine、单会话串行；运行状态 `completed`，4 个重复各 16 轮，64 对输出 token IDs 全部一致，Session metadata 与 `current-frontier` 规则通过，最大完整上下文为 946 tokens（上限 4096）。
- 对这 64 个配对观测做描述性汇总：Session TTFT 中位数 122.71 ms，stateless TTFT 中位数 1394.54 ms；配对降幅的中位数为 91.52%。它们是固定提示与固定设备上的 4 次完整对话观测，不是 64 个独立 run；不据此声称统计显著性或普遍因果收益。
- CSV 含 64 行，SVG XML 可解析并含 128 个观测点（两种路径各 64 点）。Engine 退出后，NPU0 进程/显存查询报告两个芯片均无设备进程。
- 本机证据：raw `results/raw/session_kv_ttft/qwen35-4b-session-kv-ttft-20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a.json`；前后硬件快照在同目录的 `hardware/`；CSV 与 SVG 在 `results/analysis/session_kv_ttft_20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a/`。五项文件均与远端 SHA-256 相同。

## 目标

在固定的 16 轮对话中，逐轮比较引擎级 Session KV 与无 Session、每轮完整历史重算的首 token 延迟，观察上下文增长时两者的差异。结果是对固定模型、设备、对话和运行参数的配对描述性测量，不预设 Session KV 必然降低 TTFT。

## 已选方案及不采用的方案

- 采用单会话串行配对：每轮对 Session 路径和 stateless 完整历史路径各发一个请求，轮换请求先后顺序。它能在同一逻辑对话和引擎配置下直接比较两条路径。
- 暂不采用仅绘制 Session TTFT 曲线的方案，因为它没有逐轮 stateless 对照。
- 暂不采用多会话并发方案，因为排队和调度竞争会混入首 token 延迟。
- Session ID 目前通过原生 `Engine` 接口使用，本实验不宣称是 HTTP 客户端 TTFT，也不新增 HTTP Session API。

## 实验参数

- 模型：`Qwen/Qwen3.5-4B`，固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，使用现有 `configs/bringup/qwen35_4b.yaml` 与项目内模型文件。
- 设备：课程服务器 Ascend910 V1、NPU ID 0、TP=1、BF16；设置 `ASCEND_RT_VISIBLE_DEVICES=0`。不查询或使用其他 NPU。
- 引擎配置：保留现有 overlap schedule；关闭通用 prefix caching 和 KVStore；固定 `current-frontier`。模型最大上下文为 4096 tokens。
- 对话：每次重复都使用同一固定合成对话。复用 `scripts/run_session_kv_perf.py` 的 `SYSTEM_PROMPT`（`You are a concise assistant. Keep each reply short.`）、`NEXT_USER_TEXT`（`Please continue with a short reply.`）和 `build_context_prompt` 填充方式；首轮格式化输入目标为至少 512 tokens。之后 15 轮追加同一 follow-up 文本。新增一个最小全历史构造 helper，将 system prompt、全部 user 消息和 assistant 消息重新编码成完整上下文；assistant 内容按现有规则由输出 IDs 去掉末尾 EOS（若存在）后以 `skip_special_tokens=False` 解码，再加 `<think>\n` 前缀。每次续轮都校验新编码以已记录的 session 历史 token IDs 为前缀。现有续轮 helper 只支持一个先前 user/assistant 轮次，不直接用于 16 轮。逐轮记录实际上下文 token 数，并在进入测量前确认完整历史加 8-token 生成预算不超过 4096。
- 生成：greedy，`temperature=0`，`max_new_tokens=8`；不强制忽略 EOS，记录每轮实际输出 token 数。流式间隔固定为 1，以便首个返回 chunk 对应首个生成 token。
- 重复与顺序：做 4 次完整 16 轮对话，每次打开新的 Session，并在结束后关闭。每轮的 Session/stateless 请求串行执行；四次重复的先后顺序依次为 Session→stateless、stateless→Session、Session→stateless、stateless→Session。每个 turn 形成一对观测，总计 64 对。
- 预热：计时前用独立的临时 Session 和相同长度级别的输入，依次完成一次 Session 与 stateless 流式请求；预热结果不进入 64 对测量。

## TTFT 和请求协议

- 使用原生 `Engine.generate(..., stream=True)` 和显式 `input_ids`。每次计时从调用 `Engine.generate` 前开始，到流中第一个非空 `output_ids` 返回为止；不以首个文本 chunk 或整请求完成作为 TTFT。
- 计时包含引擎原生调用路径直到首个 token ID 可读，不包含 HTTP/网络往返、客户端 tokenizer 或调用前的 prompt 构造时间。
- 首轮 Session 与 stateless 两边使用相同的 512-token-target 输入（chat template 可能使实际长度略高于目标；记录精确 token 数）。后续每轮，Session 仅提交新增的 chat-template suffix；stateless 对照提交同一逻辑对话的完整历史 token IDs。两条路径使用相同采样参数。
- 即使已取得首 token，也必须继续读取流至完成，汇总该轮全部 output IDs 并取得最终 Session metadata；不得提前丢弃 iterator。
- 每轮 stateless 对照使用该轮 Session 输出作为下一轮的共同历史。两侧 output IDs 必须逐 token 完全一致。
- 对每轮 Session metadata 按 `current-frontier` 校验 `total_context_tokens`、`newly_prefilled_tokens`、`reused_context_tokens`。最后采样 token 不在本轮额外 forward；由下一轮引擎处理 pending token 与新 suffix。
- 每对的相对 TTFT 降幅按 `100 * (1 - session_ttft_ms / stateless_ttft_ms)` 计算；stateless TTFT 必须是正的有限值。

## 失败处理与有效性

- 仅当 NPU0 的只读预检确认可用时才开始未来的验收运行；若设备正在被使用则停止，不争抢资源。
- 若任一配对缺少首个非空 token、流不能完整结束、output IDs 不匹配、Session metadata 不匹配或上下文超过上限，将整个运行标记为失败并停止后续测量。保留此前完整配对和失败原因，不生成完整成功结论，也不填补缺失测量。
- 使用 `try/finally` 关闭已打开的 Session；所有请求串行，运行期间不叠加其他项目负载。
- `stream_interval=1`、模型 revision、引擎参数、NPU0 前后快照和每轮执行顺序写入 raw provenance。

## 产物与统计口径

- 每次运行的 raw JSON 和 NPU0 前后快照保存于项目内 `results/raw/session_kv_ttft/`。Raw JSON 包含逐轮 input/output token IDs、完整与新增输入长度、两侧 TTFT、Session metadata、正确性校验、配置和运行状态；每完成一对请求即原子检查点保存。
- 只有 `status=completed` 且 64 对均有效的 raw run 才能生成正式派生数据。CSV 一行对应一个 turn-pair，包含 repetition、turn、两侧 TTFT、实际上下文长度、复用计数、output-ID parity 和配对 TTFT 降幅。
- 在唯一的新建 `results/analysis/session_kv_ttft_<run_id>/` 下生成 CSV 与 SVG。SVG 按 turn 显示 Session 与 stateless 两条路径各自的 4 次重复观测和逐轮 TTFT 中位数；CSV 保留每个配对的原始 TTFT 与按上述公式计算的降幅。分析报告逐轮 median 与 observed range，不做显著性检验或置信区间。
- 汇总单位为 4 次完整对话重复；64 个 turn-pair 不是 64 个独立 run。完整报告包含全部逐轮点，不删除异常值。

## 验收与测试

1. 实现前在现有项目 Conda 环境运行新增 runner/stream 测量 focused tests 和完整 `test/course_project`；不安装依赖。测试覆盖：首个非空 output-ID chunk 的 TTFT、流完整 drain、累计 token IDs、全历史 16 轮构造及 suffix 前缀校验、顺序平衡、current-frontier metadata 校验、输出不匹配失败、逐对断点保存及失败状态。
2. 同步到远端项目目录后先运行 focused tests；只有用户复核并确认本设计文件后，才根据实施计划做 NPU0 预检和测量。
3. 成功门槛：单次正式运行产生 16 轮 × 4 重复 = 64 对有效观测；每轮两条输出 IDs 完全匹配；全部 Session metadata 和长度边界通过；CSV 64 行、SVG 可解析且展示所有观测和逐轮中位数；Engine/runtime 进程退出后，NPU0 无项目进程并完成只读清理检查。
4. 本文件当前阶段只产出设计，不生成代码、测试结果、benchmark 数据或图表。完整书面规格获用户确认后，才编写实现计划；实施和测量仍各自遵守后续验证门槛。

## 结论边界

结论仅适用于固定 Qwen3.5-4B revision、Ascend910 V1/NPU0、TP=1、单会话串行、所用 16 轮合成对话与原生 Engine TTFT 口径。它不代表 HTTP TTFT、并发 Agent 负载、其他模型、其他硬件、Ascend910C 验收或统计显著收益。27B 继续暂缓。
