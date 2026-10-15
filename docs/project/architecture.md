# 当前推理架构（阶段 0 静态检查）

基线提交：`cf4a4f89ae5f9423a9af490d8f3a0613adfba3fd`（`main`，2026-09-26）。

## 请求路径

```text
ts serve
  python/tokenspeed/cli/__main__.py::main
  python/tokenspeed/cli/serve_smg.py::run_smg_from_args
        ↓
SMG gateway / TokenSpeed engine
  python/tokenspeed/runtime/entrypoints/engine.py::Engine
  python/tokenspeed/runtime/engine/async_llm.py::AsyncLLM
        ↓ tokenizer + request IPC
C++ control-plane scheduler
  python/tokenspeed/runtime/engine/event_loop.py::EventLoop
  tokenspeed-scheduler/csrc/scheduler/scheduler.{h,cpp}::Scheduler
  tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp
        ↓ ExecutionPlan
Python execution plane
  python/tokenspeed/runtime/execution/model_executor.py::ModelExecutor
  python/tokenspeed/runtime/execution/forward_step.py::ForwardStepRunner
        ↓
Qwen3.5 model
  python/tokenspeed/runtime/models/qwen3_5.py
  python/tokenspeed/runtime/configs/qwen3_5_config.py
        ↓
Paged attention + KV metadata
  python/tokenspeed/runtime/layers/paged_attention.py::PagedAttention
  python/tokenspeed/runtime/layers/attention/backends/paged/mha.py
  python/tokenspeed/runtime/layers/attention/kv_cache/
        ↓
tokenspeed-kernel dispatch
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/triton.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/__init__.py
        ↓ current_platform().is_npu
Ascend implementation
  tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py
  tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/layernorm.py
  tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/rotary_embedding.py
        ↓
torch_npu / CANN / Ascend NPU
        ↓
sampling + detokenization
  python/tokenspeed/runtime/sampling/backends/
  python/tokenspeed/runtime/engine/generation_output_processor.py
  python/tokenspeed/runtime/engine/async_llm.py
```

## 调度与缓存边界

- `EventLoop.event_loop()` 接收请求/缓存事件，通过绑定的 C++ `Scheduler.next_execution_plan()` 获取本轮执行计划，再交给 `ModelExecutor`。
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` 统一处理 decode、resident prefill chunk 和新 prompt；`PrefillChunkTokens` 受剩余 token budget 与 cache promotion boundary 约束。
- `tokenspeed-scheduler/csrc/cache/` 管理 cache group、物理块、请求 block table、prefix matching、claim/release 和分层缓存。
- Python KV recipe 位于 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/`；Qwen3.5 使用 `qwen35.py` 构造与模型层类型相符的 cache groups。
- NPU paged MHA 将 `page_table`、cache sequence lengths 和 page size 传给 `torch_npu.npu_fused_infer_attention_score`。

## 已确认的边界

- 调度器和 KV/prefix cache 已存在，不应为课程项目重写。
- Ascend 使用 HCCL：`python/tokenspeed/runtime/execution/distributed_initializer.py` 在 `device=npu` 时选择 `hccl`，collective backend 位于 `python/tokenspeed/runtime/distributed/comm_backend/hccl.py`。
- 当前公开 Ascend recipe 只验证 Qwen3-0.6B；Qwen3.5 四个课程型号的完整调用链仍需在 910C 上执行后才能确认。
