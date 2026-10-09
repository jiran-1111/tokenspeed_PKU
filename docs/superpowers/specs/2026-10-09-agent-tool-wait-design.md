# Workload F：单会话客户端 Tool-Wait 设计

状态：设计稿待用户复核。用户已批准“客户端 mock tool：一个 session 等待并带固定结果续跑，其他 session 同时请求；不改推理服务状态”的方向。

## 目标

让 Agent 基准能够模拟一个会话等待工具返回时，其他会话仍继续请求推理服务；等待和工具结果均在客户端模拟，不改变 TokenSpeed 服务端状态。

## 当前行为

`run_agent_benchmark.py` 在第一轮之后为每个 session 设置相同的 `tool_wait_ms`。请求 worker 在调用推理服务前分别休眠，因此所有 session 都会等待；下一轮也没有注入 mock 工具结果。

## 设计

- 增加 `--tool-wait-policy`，取值 `all` 或 `one-session`，默认 `all`，以保持已有命令及结果语义不变。
- `one-session` 下固定由 `session-0` 模拟工具调用：第一轮之后的每次续跑前等待 `--tool-wait-ms`；其余 session 不等待并照常并发请求。
- `session-0` 的续跑消息附带固定、明确标记的 mock 工具结果：“Mock tool result: Search returned two relevant documents. Continue with the task.”。它作为普通 user 消息文本传入，不伪装成真实工具协议或工具角色消息。
- 继续记录每条请求的 `tool_wait_ms`，并在原始记录及汇总中记录等待策略；原始记录另标记该请求是否注入 mock 结果。所有数据字段须能区分等待会话和未等待会话。
- 不改现有 `all` 策略的等待方式。Workload F 的运行命令显式选择 `one-session`。

## 指标解释

等待发生在客户端发起该轮 HTTP 请求之前，因此单个请求的 TTFT、TPOT、E2E 延迟不包含该轮前置等待；按轮次计算的 wall time 及由其计算的吞吐量包含等待。报告必须说明这一点，并说明这是客户端模拟，不代表服务端保留了等待中的请求或 KV 状态。

## 验收标准

1. 未指定策略时维持现有行为：第一轮后所有 session 均按 `tool_wait_ms` 等待，且没有 mock 工具结果注入。
2. `one-session` 多会话测试中，第一轮之后只有 `session-0` 的请求带等待时长和 mock 结果；其他 session 的请求等待为零且不含该结果。
3. 并发测试能证明 `session-0` 等待期间，其他 session 的请求已进入客户端推理调用；测试不依赖脆弱的固定时序阈值。
4. 请求记录与汇总包含策略信息，既有 Agent 基准测试继续通过。
5. Workload F 的命令明确选择 `one-session`，产物说明上述等待与延迟口径。

## 范围外

- 不在服务端实现 WAITING_TOOL、会话挂起/恢复或 KV 状态保留。
- 不调用真实工具或依赖网络工具结果。
- 不改变模型、服务配置、并发度定义、缓存行为或现有 `all` 策略默认值。
