# 系统设计

## 目标与负载

Agent 推理把长 system/tool 定义、逐轮增长的对话历史、短暂的工具等待和多个并行会话放在同一服务里。主要压力来自重复 prefill、decode 对长 prefill 的敏感性以及 KV cache 占用增长。

## 系统结构

课程系统沿用 TokenSpeed 的 C++ scheduler 与 cache 管理，模型执行仍走统一 Qwen3.5 model executor 和 attention backend。NPU 分支由 `tokenspeed-kernel-npu` 提供算子，并通过 HCCL 处理 TP 通信。课程新增代码负责四模型配置、确定性正确性请求、Agent/普通负载生成、采样指标、自动报告和 Ascend Profiler API 适配。

请求路径、核心文件和 cache/scheduler 边界见 [`../project/architecture.md`](../project/architecture.md)。

## 配置策略

从 0.8B TP=1 开始验证，逐步升到 4B、9B；27B 用 TP=2。配置上限是 bring-up 初值，不等同于已经验证的模型容量。每次调整都应保存具体配置、模型 revision、软件栈与原始 JSONL。
