# 答辩提纲

1. Agent 工作负载：多轮历史增长、共享 system/tool 前缀、工具等待和并发 session。
2. TokenSpeed 基线：请求路径、C++ scheduler、chunked prefill、paged KV/prefix cache 与 NPU backend。
3. 目标与实际环境：课程原目标 910C 2-chip；实际提供 openEuler 24.03/aarch64、Ascend910 V1、CANN 8.5。明确说明替代设备，并列出已实测与未实测的型号/软件栈。
4. 负载与指标：固定 correctness、普通并发、长 prefill、decode-heavy、多轮 Agent、共享前缀与 tool wait。
5. 优化和消融：只展示真实可控的 upstream 开关；展示 0.8B/TP1 E0–E3、chunk=512/1024 重复矩阵，同时说明固定重复提示词、轮间差异和不能外推的边界。
6. Profiler：展示 Level0 三场景历史 trace，以及 0.8B 8-turn/50% shared-prefix 的 Level1/PipeUtilization 六次采集（c1/c4 各 3 次、120 请求）。已有非零 AICore 时间计数，但累计算子时间不是端到端延迟；目前不能据此归因 Host gap、排队/调度或 HBM 瓶颈，也不能声称优化收益。
7. 四模型结果：TTFT、TPOT/ITL、吞吐、HBM 与最大稳定并发，重点说明 27B TP=2。GDN FP32 candidate 的 64-token 固定样例：4B 官方配置为 8/8 greedy-reference exact；9B 官方配置直接对照为 7/8，长上下文首错仍在 index 59。不能把 9B 从旧 run 的 6/8 到 7/8 表述成通用优化收益。
8. 结论与限制：优化收益、代价、可复现命令，以及未验证/未实现功能。

本提纲是材料结构，不代表设备实验已完成；未填写结果前不得按已验证结论答辩。
