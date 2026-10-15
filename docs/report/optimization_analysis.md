# 优化分析

## 当前可控变量

TokenSpeed 提供真实运行参数，可独立控制 chunked prefill、prefix cache、mixed batch 与 overlap scheduler。`run_ablation.py` 以相同模型、输入长度、输出长度和并发启动各配置，预热后重复测量。

| 阶段 | 配置差异 |
|---|---|
| E0 | 上游 scheduler；关闭 mixed batch、overlap、chunked prefill、prefix cache |
| E1 | 在 E0 上只开启 overlap scheduler；mixed batch 仍关闭 |
| E2 | 在 E1 上开启 chunked prefill |
| E3 | 在 E2 上开启 prefix cache |

本次 E0–E3 消融的所有变体均关闭 mixed batch（Ascend 后端不支持）；它不是本组实验的对照变量。E1 与 E0 的受控差异仅为 overlap scheduler。

Session KV correctness/reuse 已在 910 V1/NPU0 的 Qwen3.5-4B 上完成 matched acceptance：current-frontier 下两条 session 的第二轮分别复用 50/46 tokens，4/4 输出 ID 序列与 stateless full-history controls 一致；后续 `overlap_schedule=true` 的 correctness 复验也通过。用户确认边界后，3 次同配置 mixed-load 重复中 12/12 session 输出匹配，turn-2 session/control latency 中位数为 676.04/676.03 ms；两类请求在同批次并发，不能据此分离性能贡献。另完成 4 组重复的串行 Session on/off context sweep，12/12 输出匹配；目标上下文 512/1024/1536 的 `Engine.generate` 配对中位延迟降幅分别为 77.95%/87.11%/91.28%。这是本设备/模型上的整轮请求延迟，不是 TTFT 或并发吞吐，也不能外推为通用收益。另完成 4B/TP=1/NPU0 原生 Engine 单会话的 16-turn TTFT 配对（4×16 共 64 对），输出 IDs 64/64 一致；Session/stateless TTFT 中位数为 122.71/1394.54 ms，配对降幅中位数为 91.52%。该结果仅适用于此固定配置和合成会话，不代表 HTTP、多会话并发或跨模型/硬件收益。两类实验均关闭 generic prefix cache 与 KVStore；现有 OpenAI 客户端仍每轮发送完整历史，Session API 未接入该路径。

## 保留标准

每项优化需比较相同负载至少三轮结果，并同时观察 TTFT/TPOT/ITL、输出吞吐、显存/HBM 和正确性。只有测量改善且无正确性回归时才作为最终配置。
