# KV LCM 块计数（2026-10-12）

任务书要求 KV allocated/used/free 统计。本轮复用 scheduler 已有分区计数，补齐 Prometheus 发布与 benchmark 记录；没有修改缓存分配算法。

## 数据来源与含义

`Scheduler::ActiveLcmBlocks()` 统计活动请求 block tables 引用的物理 LCM parents；`EmptyLcmBlocks()` 统计空闲 parents。`EventLoop._get_scheduler_stats()` 已按 `total - empty - active` 计算仅缓存驻留数量。本轮把该快照传给 `EngineMetrics.record_scheduler_iteration()`。

| 指标 | 含义 |
|---|---|
| `tokenspeed:kv_lcm_blocks_total` | 设备池可用 LCM parent 总容量，不含预留 null pages |
| `tokenspeed:kv_lcm_blocks_active` | 活动请求引用的 LCM parents，对应本口径 used |
| `tokenspeed:kv_lcm_blocks_cached` | 仅作为缓存驻留的 LCM parents |
| `tokenspeed:kv_lcm_blocks_empty` | 空闲 parents，对应本口径 free |
| `tokenspeed:kv_lcm_blocks_allocated` | active + cached，即非空 parents |

不把 `available_lcm_blocks()` 当成 free：上游 available 包含可回收缓存，而且该方法扫描池和 prefix indexes，源码将其用于诊断而非每步 gauge。上述计数描述 LCM parents，不是每个 cache group 的 child pages、缓存 token 数或 HBM 字节；空闲块增加不意味着预分配的 HBM 被释放。

Prometheus 保留引擎原有 rank 标签。课程 telemetry 对计数求各 series 之和；TP 多 rank 时需按该汇总口径解释，不能当成单 rank 容量。summary 保留结束快照与 `_peak`（采样到的最大值）；empty_peak 是最大空闲块数，不是最低余量。旧 endpoint 没有新指标时保持缺失，不回填 0。

## 验证与部署状态

- 新测试覆盖 active/cached/empty 分区、全缓存驻留、零容量、benchmark 采样峰值及旧 endpoint 缺失行为。
- 远端 RED 为四个预期失败；实现后独立目录真实 prometheus_client 验证为 18 passed。
- 本机完整课程测试为 401 passed、9 skipped；新增三项 skip 因 Windows 项目环境未安装 prometheus_client，已在远端既有环境验证，没有安装依赖。
- 远端独立验证目录：`/path/to/tokenspeed/.tmp/kv_counts_validate_20261012`。为 RED 临时复制到实际项目的本轮测试已移入该目录保存。
- 新 runtime/telemetry 改动尚未部署到实际远端项目，避免改变已准备的 Long Prefill 实验；后续在独立配置下部署并核对真实指标端点。
- 当前仅证明软件接口与汇总行为；没有新的 NPU 负载、碎片、容量或 block-size sweep 结果。

## 接线与生命周期补充验证

从当前 EventLoop 源码抽取并执行 `_get_scheduler_stats()` 与 `_record_scheduler_iteration_metrics()`，以 scheduler 快照覆盖 active/cached/empty 三种分区状态；使用真实 Prometheus registry 检查计数。独立目录相关测试共 21 passed，包含新增三项接线用例。此测试只隔离依赖导入，执行的是实际源码方法，尚不等同 NPU 服务端点验收。

远端已安装 C++ scheduler 的原有 finish 生命周期用例为 2 passed。另在 `tokenspeed-scheduler/python/tests/test_fsm_and_scheduling.py` 增加 `test_new_request_is_admitted_before_long_decode_finishes`：短、长请求先进入 decode；短请求 finish 后提交新请求；下一计划中短请求被移除、新请求进入 prefill，而长请求仍为 decode。远端独立目录执行为 1 passed，运行结束清理测试请求。该证据证明本 scheduler 软件行为；没有生成模型 tokens，不证明硬件吞吐或 NPU 利用率收益。
