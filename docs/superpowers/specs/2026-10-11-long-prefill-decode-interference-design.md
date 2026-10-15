# 长 Prefill 对持续 Decode 干扰实验设计

## 状态

用户已确认 4B/TP1、2K 与 3.5K 长 Prefill，以及“Decode 首个非空 SSE 内容块到达后提交 Prefill”的 staged HTTP 方案，并确认关闭 prefix cache、使用交替配对顺序和唯一项目内产物目录。本文件仅为待复核规格；当前不实现代码、不启动 NPU 实验或 Profiler 采集。规格通过后再写实施计划。

## 目标与结论边界

通过 OpenAI-compatible HTTP 服务观察长 Prefill 请求进入时，已有 Decode 请求的客户端流式输出间隔和 TTFT 是否变化，并对比 `chunked_prefill_size=4096` 与 `512`。本实验度量的是请求级客户端时间线，不是服务端 token ITL，也不建立请求到算子/硬件任务的因果关联。不能据此声称特定算子被 Prefill 阻塞或获得了精确的服务端排队时长。

## 固定实验条件

- 模型为本地固定 revision 的 Qwen3.5-4B，TP=1、BF16、`max_model_len=4096`；当前课程服务器已知为 Ascend910 V1，并非课程要求的 910C。结果不能替代 910C 验收。
- 以现有 4B bring-up 配置为基线。两个变体仅将 `chunked_prefill_size` 设为 4096 或 512；prefix cache 在两种变体中都关闭，以排除重复前缀命中。其余调度与模型参数一致并记录配置快照/摘要。
- 2026-10-13 经用户批准修正：两种变体共同设置 `disable_kvstore=true`，因为当前服务不允许 KVStore 开启且 prefix cache 关闭的组合。该共同控制项纳入快照及结论边界，不修改运行时校验或原 bring-up 配置。
- Prefill 请求目标为服务端实际输入 2048、3584 tokens，`max_tokens=1`。长度以完整 chat 模板后的服务端 `usage.prompt_tokens` 为准；若请求总长度超过 4096 或实际长度不符合目标，拒绝/标记该 trial，不静默截断。
- Decode 请求目标为服务端实际输入 256 tokens，`max_tokens=128`。生成必须足够长以形成持续 Decode；实施前确认当前服务版本可用的固定长度生成参数，并由非计量预热验证输出长度。若预热不能产生 128 tokens，停止该配置的测量并报告阻塞；测量 trial 提前结束则标记无效、不纳入比较。若任一条件不足 3 个有效样本，只报告数据不足，不作 chunk size 比较结论。
- 两类请求均为独立 HTTP chat 请求，温度为 0。沿用现有首个非空输出判定：`delta.content` 或 `delta.reasoning_content`，并记录命中的字段。

## 实验条件与顺序

每个 chunk 配置包含以下五种 trial：

1. Decode-only：256-token 输入、128-token 输出上限，单请求。
2. Prefill-only-2048：2048-token 输入，单 token 输出上限。
3. Prefill-only-3584：3584-token 输入，单 token 输出上限。
4. Staged-2048：先启动 Decode；客户端收到其首个非空 SSE 内容块后立即异步提交 2048-token Prefill。
5. Staged-3584：同上，提交 3584-token Prefill。

每种条件做 3 个配对轮次。每轮两种 chunk 配置各启动一次服务，轮内顺序交替为 `4096→512`、`512→4096`、`4096→512`。五种条件的运行顺序使用预先固定的轮转表；同一配对轮次的两种 chunk 配置使用相同条件顺序，并记录实际顺序。每次服务启动后先用相同的 Decode 与最长 Prefill 做不计量预热，再执行该配置的测量条件；预热、启动和等待时间不计入请求延迟。测量期间不得并发运行其他基准或修改配置。服务重启前后均需记录模型 revision 与配置摘要。

Staged 请求的首块回调只记录事件并投递 Prefill 任务到独立工作线程，不能在 Decode 的 SSE drain 线程内等待 Prefill。记录两请求的 HTTP 生命周期边界和 Decode 每个非空 SSE 块时间戳；只有两条请求时间区间重叠的 trial 才进入 staged 比较。提交 Prefill 后 Decode 已结束、请求失败、流未正常结束或实际长度/生成长度不符合约束时，保留失败记录但不计入有效样本。

## 时间戳与指标

- 使用单调时钟计算耗时；每个请求记录发送、首个非空块、每个后续非空块及 stream drain 完成时间的 monotonic 纳秒值，并同时记录 epoch 纳秒值供审计。每个 SSE 块另存序号、`content`/`reasoning_content` 字段和文本长度，不保存生成文本。
- 保存服务端 `prompt_tokens`、`completion_tokens`、请求参数、HTTP 状态和结束状态。`TTFT` 为请求发送到首个非空 SSE 块；`E2E` 为请求发送到 stream drain 完成。
- Decode 块间隔按相邻非空 SSE 块的客户端到达时间计算。以 Prefill 请求发送时间划分干扰前后区间：完全早于发送时刻的间隔记为前段，完全晚于发送时刻的记为后段，跨越该时刻的间隔单独标记、不归入任一段。该指标称为“客户端 SSE 块间隔”，不得标成 token ITL。
- 每个 trial 保留单次值及有效性原因。以 3 个配对轮次给出原始点、配对差和中位数/范围；样本量不足以支持显著性或普遍性结论。Prefill-only 与 staged 分别报告两类请求的 TTFT/E2E；重点比较 staged 中 Decode 块间隔及 Decode TTFT 相对同轮 Decode-only 的变化。
- 不启动 Profiler，不采集 Torch/CANN 时间线，也不将请求事件与 NPU 算子、任务 ID 或服务端调度事件作因果归因。

## 产物与安全边界

- 原始请求记录、逐块事件、trial 汇总和配置快照保存到项目内唯一新目录 `results/raw/benchmark/long_prefill_decode/<run_id>/`；分析 CSV 与 SVG 保存到 `results/analysis/long_prefill_decode/<run_id>/`。先检查目标目录，已存在即中止，不覆盖、不追加到旧 run。
- 记录模型 ID/revision、实验配置摘要、chunk size、请求实际 token 数、配对轮次/顺序、预热状态和有效性原因。提示词按固定配方生成并记录配方版本与哈希，不写入日志；不记录授权头或任何密钥。
- 所有新增文件和派生产物留在当前 TokenSpeed 项目目录。后续真实运行前须只读检查 NPU0、服务端口和目标目录；只允许使用 NPU0，若 NPU0 忙或服务状态不明即停止，不查询/启动 NPU ID 1–3。27B/TP=2 与本实验无关。

## 实施验收

1. 单元测试证明：只有 Decode 首个非空块会触发一次 staged 提交；Prefill 在独立工作线程中运行；空块不触发；Decode SSE 可继续 drain；发送/首块/结束和逐块时间戳顺序正确；无效长度、提前结束、请求异常都留下无效状态而非有效样本。
2. fake HTTP/SSE 测试覆盖三类 trial、两种 chunk 配置的参数及 raw/summary 序列化，不访问网络服务、不启动 NPU 或 Profiler。
3. 测试确认目标输出目录冲突时安全失败，CSV 可解析，SVG XML 可解析，所有文件路径在项目目录内。
4. 只有本规格经用户复核、另行编写并确认实施计划、代码测试通过且 NPU0 预检可用后，才进行真实 NPU 运行；该授权不包含 Profiler 采集。

## 解释限制

该实验最多支持如下表述：在所测 4B/TP1、输入长度、调度设置和 HTTP 服务环境下，staged 长 Prefill 与 Decode 客户端流式时间线重叠，并观察到相应的客户端块间隔/TTFT差异。块边界受网络、HTTP 缓冲、服务端 flush 和 detokenizer 影响，不一定一块对应一个 token；小样本也不代表其他模型、并发度、910C 设备或生产 Agent 负载。
