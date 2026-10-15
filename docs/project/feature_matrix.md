# Feature Matrix（阶段 0）

“存在”表示已追踪到实际实现；“Ascend 状态”在没有 910C 实测时不会写成已通过。

| 功能 | 当前是否存在 | Ascend 状态 | Qwen3.5 状态 | 是否需要修改 |
|---|---|---|---|---|
| Qwen3.5 model | 是：统一 config/model，含 dense/MoE/NextN | 910 V1 上 0.8B/4B/9B TP=1 服务已实测；GDN FP32 candidate 的 64-token 固定样例：0.8B stop-aligned 7/8，4B 官方配置 8/8，9B 官方配置直接对照 7/8；4B TP=2 固定样例 7/8；27B TP=2 曾加载权重并完成多模态 warmup，但未到 readiness | 上游已支持 Qwen3.5 系列架构 | 仍需定位 0.8B/9B 与 TP2 正确性差异，并完成 27B、容量及四模型完整验证 |
| Ascend backend | 是：`device=npu`、HCCL、NPU kernel 包 | 本项目 openEuler/aarch64、CANN 8.5、910 V1 上已验证 0.8B/4B/9B TP=1 部分路径；上游另有 CANN 9.0.0 验证栈 | 未提供课程四模型的 Ascend recipe | 继续补齐采样/GDN/graph、27B、跨模型与 TP 路径检查 |
| Continuous batching | 是：C++ Scheduler + EventLoop | 已随 0.8B/4B NPU 服务和 0.8B benchmark 路径运行；独立 scheduler/continuous-batching 性能收益尚未隔离 | 可进入统一调度路径 | 先补不同并发与排队场景测量，不重写 |
| Chunked prefill | 是：`operations/prefill_chunk.*`，有 C++ 测试 | 910 V1/0.8B/TP1 已测 E2/E3，2048/32 tokens、并发 4、每 chunk 配置 36 请求；比较 chunk=512/1024，限固定重复提示词。另有 4B/TP1、2K/3.5K staged Prefill/Decode 干扰 runner、CSV/SVG 分析器和本地 dry-run/测试；真实 NPU 运行未执行 | 统一 scheduler 路径 | 先完成已设计的长 Prefill 干扰实测，再扩展更多并发/轮次及其他模型；未实测前不作收益结论 |
| Paged/block KV | 是：cache groups、block table、page table | NPU paged MHA 消费 block table；当前 Prometheus 仅发布 active LCM block / total 的比例，没有单独的 used/free 计数 | Qwen3.5 GDN recipe 将 `prefix_granularity` 固定为 128，配置中的 `block_size` 会映射到该 scheduler 参数；16/32/64 不是当前 Qwen3.5 recipe 可直接测试的粒度 | KV block 统计仍待补；不把固定的 128 描述成经 block-size sweep 得出的性能最优值 |
| Prefix cache | 是：CacheCoordinator prefix probe/admission/publish | 910 V1/0.8B 重复提示词消融已采到约 0.965 hit ratio；尚未独立验收缓存正确性或一般 Agent 负载收益 | Qwen3.5 cache groups 可参与匹配 | 补不同共享前缀/增长历史下的正确性和稳定性分析 |
| Session KV reuse | 已实现独立的 Engine 级 Session API，由 C++ Scheduler 持有 session→request/KV 状态；OpenAI chat endpoint 仍不暴露该 API | 910 V1/NPU0 的 4B current-frontier correctness/reuse 通过；3 次 overlap-enabled mixed-load 重复为 12/12 输出匹配、turn-2 稳定复用 50/46 tokens，但同批次 session/control latency 中位数几乎相同（676.04/676.03 ms）。另完成独立串行 on/off context sweep：12/12 输出匹配，目标上下文 512/1024/1536 的 `Engine.generate` 配对中位延迟降幅为 77.95%/87.11%/91.28%；另完成 16-turn native Engine TTFT 测量，64 对输出 IDs 全匹配，配对降幅中位数 91.52% | 当前 Agent/chat benchmark 仍发送完整历史；独立 native Engine runner 提供 Session 与 stateless 对照 | 串行结果只适用于本配置，不是 HTTP TTFT、并发吞吐或跨模型结论；保留 single-writer、process-local、text-only 限制 |
| Tensor parallel | 是：Mapping、并行线性层、distributed collectives | 910 V1 上 4B TP=2 服务启动、warmup 与 health check 通过；真实两进程 HCCL all-gather、reduce-scatter、token-aware reduce-scatter 回归通过；两次固定 token-ID 对照输出逐条相同且均为 7/8。首差处 TP2 rank logits 相差 0.125，HF reference 对相同前缀在两候选上精确并列；TP1 对同一前缀 top-2 排序与 TP2 相反，margin 同为 0.125。27B TP=2 一次启动在权重加载和 multimodal warmup 后卡于 HCCL AllReduce、未 readiness；rank-1 trace 等待 rank-0 通知，留存的 rank-0 trace 未见对应运行时事件，但根因未定位 | Qwen3.5 使用并行线性层 | 4B strict correctness 仍未通过，27B 未完成启动/正确性；不得据此给出 TP 性能结论 |
| Benchmark | 是：random、long-context、agentic、kernel benchmark | 910 V1 上 0.8B/4B/9B TP=1 已有 128/32、并发 1/4/8/16 首轮 baseline；27B TP=2 启动尝试未到 readiness，未有 correctness/baseline | 现有 agentic 脚本主要面向 NVIDIA/AMD 与其他模型；课程矩阵尚未完整 | 诊断 27B startup；其余模型扩展长上下文/Agent/TP2 及跨模型矩阵 |
| Profiler | 有 `/start_profile`、`/stop_profile` 与 PyTorch profiler；另有单请求 TTFT 时间窗 runner/分析器与 fake tests | 910 V1/0.8B 三场景有 Level0 历史 `op_summary`/`task_time`；另有 0.8B 8-turn Level1/PipeUtilization 六次历史采集和非零 AICore 时间行。新单请求时间窗实现尚未做真实采集，也没有核实 Torch/CANN 窗口重叠 | 与模型无关 | 在单独获批后采集单请求窗口并核实时钟；再用 trace 与可控实验分析瓶颈，当前不作请求-算子因果或收益结论 |
| Tool-wait scheduling | 未发现显式 `WAITING_TOOL` 请求状态 | 未验证 | 与模型无关 | 先构造 workload 证明是否会阻塞，再决定最小修改 |

## 主要源码证据

- Qwen3.5：`python/tokenspeed/runtime/models/qwen3_5.py`、`configs/qwen3_5_config.py`、`kv_cache/recipes/qwen35.py`
- Scheduler：`tokenspeed-scheduler/csrc/scheduler/`、`python/tokenspeed/runtime/engine/event_loop.py`
- KV/prefix cache：`tokenspeed-scheduler/csrc/cache/`、`docs/design/cache-concepts.md`
- NPU：`tokenspeed-kernel-npu/`、`tokenspeed-kernel/.../attention/mha/triton.py`、`distributed/comm_backend/hccl.py`
- Benchmark：`test/random_benchmark/`、`test/long_context_benchmark/`、`test/agentic_benchmark/`
- Profiler：`python/tokenspeed/runtime/engine/request_handler.py`
