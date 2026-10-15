# 单请求 TTFT Profiler 时间窗设计

## 状态

用户已选择轻量时间窗方案、单请求串行采集和 `max_tokens=1` 边界。本文件供复核；当前没有实现代码变更，也没有启动新的 Profiler 采集。只有规格获用户确认后，才编写实施计划；实施和 NPU 采集仍需遵守后续门槛。

## 目标

在已有 0.8B Level1 CANN/Torch-NPU 采集链路上，为一个 HTTP chat 请求记录客户端发送与首个非空输出块的绝对时间，并观察这段 TTFT 窗口与 Torch/CANN 事件时间线的重叠。结果只支持时间重叠观察，不建立请求 ID 到算子/任务的关联，也不据此声称算子因果归属或优化收益。

## 采用方案与范围

- 设备与服务：课程服务器 Ascend910 V1、NPU0、Qwen3.5-0.8B、BF16/TP=1，沿用已有 Level1 `ACL_AICORE_PIPE_UTILIZATION` 配置。该设备不是课程要求的 910C。
- 请求：复用已测 8-turn Agent 的单会话 c1、512-token 上下文、50% 共享前缀负载；前 7 轮按原始 32-token 上限构造历史，不开启 Profiler。第 8 轮只发一个请求，并设 `max_tokens=1`。现有每轮 100 ms 客户端 mock wait 在启动 Profiler 前完成，不计入请求 TTFT。
- 采集边界：Profiler 在第 8 轮请求发送前启动；客户端检测到首个非空输出块后立即请求停止 Profiler。客户端随后继续 drain SSE 流并记录完成时间。请求仍是单会话、单并发、单个被 profile 的请求；不并发提交其他请求。
- 复用同一 runner 已定义的首个非空输出观测口径（`delta.content` 或 `delta.reasoning_content`），同时记录实际命中的字段。若本轮没有非空输出块、usage 不是恰好 1 个输出 token，或流未正常结束，则该采集标为无效，不作时间线结论。
- 仅增加客户端时间戳与首块回调，不加服务端 request ID、运行时 span、CANN task ID 标记或新的 profiler 插桩。

## 时间戳与时间线核对

- 每个被采集请求的 raw 记录至少保存：请求发送、首个非空输出块、Profiler stop 请求、Profiler stop 响应、SSE drain 完成的 Unix epoch 纳秒时间；发送/首块/drain 同时保留 monotonic 纳秒值。现有 TTFT/E2E 时长仍由 monotonic clock 计算，不被 wall clock 替代。
- 请求发送时间取调用 HTTP `urlopen` 前的时刻；首块时间取现有 TTFT 首块判定处；stop 请求紧接首块记录触发。保存 stop RPC 的请求和响应时间，以显示 profiler 停止控制本身造成的尾部，而不把 stop RPC 延迟计入 TTFT。
- 先检查 epoch 差分与 monotonic 差分是否一致（绝对误差不超过 1 ms）；超限则保留原始证据，但不作窗口重叠分析。
- 将 epoch 纳秒换算为微秒，与 Torch trace 的 `ts` 对齐。对 CANN `task_time` 使用已有只读审计确认的 Torch/CANN 时钟映射；本次记录实际 offset 和可验证误差，不把先前观测到的约 57 μs 当成保证值。若无法验证同一时间基准，则只分别报告两种时间线，不叠图、不计重叠。
- 仅统计 `[request_send, first_nonempty_chunk]` 内的 Torch 事件和 CANN task interval。Profiler stop 响应后仍可能存在采集尾部；窗口外事件不纳入该请求时间窗。

## 产物与报告

- Raw 客户端记录、summary、NPU0 前后快照和配置 provenance 保存到项目内唯一新目录 `results/raw/profiler/single_request_ttft/<run_id>/`；Level1 Torch/CANN 文件保存到唯一新目录 `results/profiles/qwen35-0.8b/single_request_ttft/<run_id>/`。不覆盖既有原始 trace。
- 派生 CSV 保存到 `results/analysis/profiler_single_request_ttft_<run_id>/`，至少含客户端发送/首块/停止时间、校验状态、Torch 与 CANN 的时间基准/offset、窗口内事件/任务数量及重叠时长。保留原始逐事件文件供复核。
- 报告明确这是 1 个被 profile 的请求、`max_tokens=1`、带 profiler 开销的 TTFT 时间窗；不能与无 profiler 的性能基线直接比较，也不能据重叠事件判断该请求独占或触发了这些任务。
- 所有下载和派生文件都同步到当前 TokenSpeed 项目目录。不得把密钥、令牌、私钥或完整授权头写入数据或日志。

## 失败处理与安全边界

- 采集前只读检查 NPU0、服务端口、模型配置和目标输出目录；目标目录已存在、NPU0 被其他任务占用或服务未 ready 时停止，不覆盖、不抢占。
- Profiler stop 由首块回调触发且最多一次；异常路径在 `finally` 停止仍活动的 Profiler。无首块、RPC 失败、请求错误或时间基准不一致均记录失败状态，不伪造完整报告。
- 仅允许使用 NPU0；不查询或启动 NPU ID 1–3。27B/TP=2 不在本实验范围。

## 验收

1. focused tests 验证：时间戳来自预期边界；monotonic TTFT 保持不变；首个非空块只触发一次 Profiler stop；stop 发生在 SSE drain 前；正常 drain 后保存完成时间；错误/空输出不会形成成功记录。
2. 单请求 dry-run 以 fake client/control 覆盖 callback 顺序和时间戳一致性，不启动 NPU Profiler。
3. 只有规格复核确认、实现测试通过且 NPU0 预检可用后，才进行一次实际单请求 Level1 采集。
4. 真实采集成功须有 1 条请求记录、1 个输出 token、正常流结束、Profiler start/stop 成对、raw/CANN/Torch 文件完整、clock check 通过、CSV 可解析、所有本地/远端 SHA-256 一致；结束后项目进程退出、NPU0 利用率回落并完成只读清理检查。

## 结论边界

该实验只回答客户端 TTFT 窗口与采集到的 Torch/CANN 事件在时间上是否重叠。没有服务端请求级关联标记时，重叠不代表任务由该请求引发，也不证明其决定了 TTFT；单次、单输出 token 的结果不能代表长输出、并发 Agent 负载或统计上稳定的性能差异。
