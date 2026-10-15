# Long Prefill / Decode 测量验收（2026-10-13）

本次限定实验已完成测量和 CSV/SVG 验收；不代表整个课程项目已完成。

## 实验与证据

- Run：`20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4`。
- 4B/TP1/BF16，实际 Ascend910 V1，仅 NPU0；模型 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- max_model_len=4096；Decode 输入 256、输出 128；Prefill 输入 2048/3584、输出 1。prefix cache 与 KVStore 在两组都关闭，其余条件一致。
- 三轮交替配对，共六次服务启动；服务日志均为 ready、exit_status=0。12 项预热加 30 项正式 trial 均 valid，共 54 个 HTTP 请求。
- 五种条件各有三轮配对，配对 CSV 共 15 行，summary 共 18 行。54 个请求的实际 token 数和发送/首块/drain 顺序通过核验；12 个 staged trial 均存在请求生命周期重叠。
- [原始目录](../../results/raw/benchmark/long_prefill_decode/20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4/)、[配对 CSV](../../results/analysis/long_prefill_decode/20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4/paired_trials.csv)、[摘要 CSV](../../results/analysis/long_prefill_decode/20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4/summary.csv)、[块间隔 SVG](../../results/analysis/long_prefill_decode/20261013T031215346959Z-132fde4d591240c38762e3bd5bc383b4/decode_chunk_gaps.svg)。
- 21 个 raw 文件与 3 个报告文件的本地/远端 SHA-256 全部一致。CSV 可重读、SVG XML 可解析；本机从 raw 重新生成三项报告，字节与远端一致。
- `measurement-source/` 保存 run.json 指定的五个测量脚本/模块，两端逐一核验 SHA-256。分析器在测量后修正了轮内启动序号校验；归档保留测量时的模块版本，不用修改后的分析模块冒充测量版本。此归档不是完整推理运行时源码快照。
- 完成后 NPU0 两个 chip 无设备进程，API 8000 连接探测为拒绝连接；未启动 Profiler。

## 数值观测（ms，三轮描述性统计）

| 条件/指标 | chunk 4096 中位数 | chunk 512 中位数 | 配对差中位数（512−4096） |
|---|---:|---:|---:|
| Decode 单独 E2E | 2923.39 | 2917.03 | −9.91 |
| 2048 Prefill 单独 TTFT | 2990.00 | 3099.84 | +109.84 |
| 3584 Prefill 单独 TTFT | 5116.13 | 5413.34 | +299.53 |
| staged 2048 Decode E2E | 6010.82 | 6064.30 | +75.01 |
| staged 3584 Decode E2E | 8058.04 | 8317.29 | +248.12 |

配对差是逐轮作差后取中位数，不等于两列中位数相减。完整范围及逐轮值以 CSV 为准。这里没有显著性检验、置信区间或“最佳默认配置”结论。

所有 staged 测量的 Prefill 都在 Decode 首个非空块后提交，因此没有完全位于 Prefill send 之前的块间隔，相关 summary 为 n=0 并留空中位数/范围，SVG 标记数据不足。每条 staged 流有一个 crosses 和 126 个 after 间隔；跨界间隔不进入前/后统计。after 间隔中位数约 19.6 ms，但每条流的最大客户端块间隔分别约为 3.05–3.17 秒（2048）或 5.13–5.42 秒（3584）。不能用约 19.6 ms 的中位数声称没有长停顿。

观测到的 staged Decode E2E 高于独立 Decode；本次 512 配置未呈现降低 staged Decode E2E 的结果。这只支持当前配置下的客户端时间线观测，不能把 SSE 块间隔称作 token ITL，也不能据此给某个算子作因果归属或推广到 Ascend910C、其他模型/并发度。

## 失败记录与后续边界

前两次启动的 KVStore 配置冲突和 TIME_WAIT 误判、修正授权及回归证据见 [启动诊断](long_prefill_startup_20261013.md)。旧目录保留，不拼接进本次配对统计。分析器轮内序号问题由真实 raw 暴露，fixture 修正后先确认失败，再修正校验；本机 Long Prefill 集合 141 passed/6 skipped，远端 146 passed/1 skipped。

本次达到已批准的 CSV/SVG 测量验收；整体项目仍有模型正确性、27B/TP、Profiler、完整消融及最终配置等缺项，见 [提交状态](submission_status.md)。

完成分析器修正后的完整课程回归：本机 404 passed/13 skipped，实际远端 381 passed/1 skipped；均使用项目 Conda 与项目内临时目录，未安装依赖。最后端口探测确认 8000/8001/39141/39142 均可绑定，NPU0 两 chip 计算利用率及 HBM bandwidth 为 0。
