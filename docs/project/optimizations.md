# Optimizations

当前没有保留任何未经 Benchmark/Profiler 证明的优化。

课程分支已增加 E0–E3 的可复现配置运行器，实际调用上游开关。continuous batching 是上游 scheduler 基础能力，不伪造关闭版本。E0 关闭 overlap、chunked prefill 和 prefix cache；E1–E3 逐阶段启用，所有 Ascend 变体均关闭不支持的 mixed batch，并固定关闭 KVStore。

已在 Ascend910 V1、Qwen3.5-0.8B、TP=1 上完成两组重复消融矩阵：目标输入/输出 2048/32、并发 4、每变体 12 请求×3 轮；E2/E3 分别使用 chunk=512 和 1024。结果为固定重复提示词工作负载下的描述性测量，不足以支持普遍性能结论：E3 prefix-hit ratio 约 0.965，且 chunk=512 的首轮 TTFT 明显高于后两轮。完整条件、轮次结果和限制见 [`消融分析`](../report/ablation_analysis.md) 及 `results/raw/ablation/qwen35_0.8b_formal_chunk512_20261009/`、`qwen35_0.8b_formal_chunk1024_20261009/`。

4B/TP1/NPU0 的匹配 E0–E3 两组矩阵也已完成。固定重复 prompt 下 E3 同时出现约 0.966 prefix-hit、较低 TTFT 和较高吞吐，但这不能代表多样用户前缀或一般 Agent 会话；E2 对 E1 的 TTFT 变化方向还与 0.8B 矩阵相反，因此 chunked prefill 效果依赖模型/负载。chunk=1024 E2 的 ITL p95 histogram estimate 约 1.267 s，因而不据 TTFT/吞吐单独选择默认 chunk。该比较仅在本次 910 V1、单设备、固定顺序和三轮重复范围内成立，详见[消融分析](../report/ablation_analysis.md)。

已确认应优先复用的上游能力：

- C++ continuous batching scheduler
- prefill chunking
- paged/block KV cache
- prefix matching/cache admission
- HCCL tensor parallel backend
- NPU MHA/RMSNorm/RoPE kernels

已实现独立的 Engine 级 Session KV API 和 native Engine Session-vs-stateless runner。910 V1/NPU0 的 4B current-frontier matched correctness/reuse run 已通过；`overlap_schedule=true` 下两条交错 session 的 4/4 轮也匹配 stateless 对照，generic prefix cache 与 KVStore 关闭。另完成串行 paired on/off context sweep：12/12 输出匹配，目标上下文 512/1024/1536 下 `Engine.generate` 整轮请求延迟配对中位降幅为 77.95%/87.11%/91.28%。这是单设备串行整轮延迟观察，不是 TTFT 或并发吞吐；Session 多轮 TTFT 曲线仍待验收，OpenAI chat endpoint 目前仍不使用该 API。除此之外，尚未实现新的 prefill/decode 调度策略或自定义 NPU kernel。TokenSpeed Prometheus 已有 KV 使用率、prefix hit、prompt/generation token 与运行/等待请求指标；课程 runner 会把这些指标写入 summary，并在设备上保存 `npu-smi info` 原始快照。

状态更正（2026-10-12）：上文“Session 多轮 TTFT 曲线仍待验收”已由完成的独立测量取代。4B/TP1/NPU0 四次 16 轮串行测量共 64 对，输出 IDs 全部匹配；Session/stateless TTFT 中位数为 122.71/1394.54 ms，配对降幅中位数为 91.52%。[原始记录](../../results/raw/session_kv_ttft/qwen35-4b-session-kv-ttft-20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a.json)与[图表](../../results/analysis/session_kv_ttft_20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a/session_kv_multiturn_ttft.svg)对应原生 Engine 串行 TTFT；HTTP、并发和跨模型性能仍需独立验证。

后续每项候选修改必须记录 baseline 问题、修改机制、相同 workload 的前后数据和 profiler 解释。
