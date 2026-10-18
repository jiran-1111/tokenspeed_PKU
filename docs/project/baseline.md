# Baseline

> 归档说明：本文保留按日期记录的实验过程。标为“归档路径”的本地文件未随公开源码发布，因此以路径文字保留，不提供无效下载链接。当前可公开复核的材料见[最终交付](final_delivery_20261018.md)。

## 固定条件

- TokenSpeed 上游基线提交：`cf4a4f89ae5f9423a9af490d8f3a0613adfba3fd`；当前运行包含为课程 Ascend/Qwen3.5 路径所需的兼容修复。
- 实测设备：课程服务器提供 Ascend910 V1（openEuler 24.03、aarch64）；课程原要求为 910C，课程方是否接受替代尚未确认，因此结果不表述为 910C 实测。
- 模型：Qwen3.5-0.8B，BF16，TP=1，revision `2fc06364715b967f1860aea9cf38778875588b17`。
- 服务配置：`configs/bringup/qwen35_0.8b.yaml`，max model length 4096、KV block 128、chunked prefill 4096、overlap scheduler 开启、prefix cache 开启；NPU 不支持的 mixed batch 关闭。
- 负载：目标输入/输出 128/32 tokens；并发 1/4/8/16；每档 3 轮，每轮 32 请求，共 384 请求和 12 份汇总。请求提示词按 request id 生成等长变体，避免意外复用完整 prompt。
- 数据：`results/raw/baseline/qwen35_0.8b_20261009_unique_prompt/`（归档路径：`results/raw/baseline/qwen35_0.8b_20261009_unique_prompt`），含 JSONL、summary 和 NPU 前后快照。

## 首轮并发结果

下表是每档三轮 summary 指标的中位数。TTFT/TPOT 来自客户端流式观测；这些原始历史 summary 中 ITL 为 null。运行时 ITL 采集口径修正后，0.8B 同配置另行重跑，数据见下节；没有回填或覆盖这些历史文件。

| 并发 | TTFT p50 / p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | KV 峰值 | waiting 峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 225.41 / 228.89 | 10.48 | 1.82 | 58.17 | 0.12 | 0 |
| 4 | 556.17 / 6297.53 | 17.80 | 2.14 | 68.60 | 0.61 | 0 |
| 8 | 884.12 / 1296.26 | 42.84 | 3.81 | 121.95 | 0.91 | 2 |
| 16 | 2760.82 / 3530.22 | 56.42 | 3.61 | 115.49 | 0.97 | 10 |

首轮数据表明并发从 8 提高到 16 时吞吐没有继续提升，且 TTFT、TPOT 和 KV 使用率上升；并发 4 的 TTFT p95 有明显长尾。它们是后续优化的待解释基线观察，不单独证明某项优化有效。输入/输出吞吐、E2E、运行期采样和各轮数值均可从 summary/raw JSONL 复算。

## 修正后的 0.8B 服务端 ITL 复测（2026-10-09）

复用上方模型 revision、服务配置与 128/32 独立提示词负载：并发 1/4/8/16，各 3 轮、每轮 32 请求，共 384 条请求、12 份 summary 和 24 份 NPU 前后快照。每档三轮 ITL summary 取中位数：

| 并发 | ITL P50 (ms) | ITL P95 (ms) | ITL mean (ms) |
|---:|---:|---:|---:|
| 1 | 12.500 | 14.750 | 10.517 |
| 4 | 17.616 | 19.970 | 18.763 |
| 8 | 27.566 | 30.400 | 40.540 |
| 16 | 31.983 | 262.609 | 52.121 |

ITL 来自服务端 `tokenspeed:inter_token_latency_seconds` 直方图：用 engine output timestamp 的差分除以该帧新增 token 数，再按新增 token 数计权；它是 run-level 估计，不是每 token 独立时间戳。每组的 `itl_observation_count` 均为 992，与该组 32 个请求的原始 `sum(output_tokens - 1)` 精确一致。分位数受 Prometheus 桶精度限制；c16 的 P95 长尾明显。数据与硬件快照见 `results/raw/benchmark/itl_20261009_0.8b_corrected/`（归档路径：`results/raw/benchmark/itl_20261009_0.8b_corrected`），本次 26 个文件同步后 SHA-256 全部匹配。该复测补齐 0.8B ITL，不改变上方历史首轮 latency 结果，也不是优化收益。

## 4B/9B 修正后的服务端 ITL 复测（2026-10-09）

沿用相同的 128/32、并发 1/4/8/16、每档 3 轮×32 请求设置；4B 与 9B 各完成 384 请求、12 份 summary 和 24 份 NPU0 硬件快照。每份 summary 的 ITL 观测数均为 992，且与对应 32 条原始记录的 `sum(output_tokens - 1)` 相同。下表为每档三轮中位数：

| 模型 | 并发 | ITL P50 (ms) | ITL P95 (ms) | ITL mean (ms) |
|---|---:|---:|---:|---:|
| 4B | 1 | 17.567 | 19.878 | 19.485 |
| 4B | 4 | 32.570 | 34.883 | 34.889 |
| 4B | 8 | 45.443 | 280.339 | 81.346 |
| 4B | 16 | 45.247 | 416.842 | 79.337 |
| 9B | 1 | 22.630 | 24.997 | 24.876 |
| 9B | 4 | 37.544 | 39.838 | 40.211 |
| 9B | 8 | 63.227 | 368.077 | 88.452 |
| 9B | 16 | 62.760 | 525.000 | 85.292 |

ITL 定义和 Prometheus 桶估算方法与 0.8B 相同；c8/c16 P95 显示明显长尾，不能把这些单机数据解释为优化收益。4B 原始记录/硬件快照位于 `results/raw/benchmark/itl_20261009_4b_corrected/`（归档路径：`results/raw/benchmark/itl_20261009_4b_corrected`），9B 位于 `results/raw/benchmark/itl_20261009_9b_corrected/`（归档路径：`results/raw/benchmark/itl_20261009_9b_corrected`）；两组各 26 个文件同步后 SHA-256 全部匹配。历史首轮 summary 的 null 值未被改写。

三模型 corrected ITL 的并发曲线见 `P50`（归档路径：`results/plots/after_corrected_itl_20261009/itl_p50_concurrency.svg`） 和 `P95`（归档路径：`results/plots/after_corrected_itl_20261009/itl_p95_concurrency.svg`）；图表由 36 份 corrected summary 汇总，按模型、输入/输出长度和 TP 配置分组。

## 4B 与 9B 首轮基线

4B 与 9B 也在 Ascend910 V1、TP=1 上按相同的 128/32 tokens、并发 1/4/8/16、每档 3 轮×32 请求运行，各有 384 条请求、12 份 summary 和 24 个硬件快照。模型 revision、每档中位数及 `max_num_seqs=8` 下的排队/KV 峰值限制分别记录在 [`4B 结果`](final_results.md#qwen35-4b-baseline-2026-10-09)和 [`9B 结果`](final_results.md#qwen35-9b-baseline-2026-10-09)。三组数据均为该替代设备上的首轮基线，不是 910C 结果或优化收益；4B/9B 首轮历史 summary 的 ITL 仍为 null，三模型 corrected ITL 复测见上节。

## Workload A：512/128 普通聊天并发扩展（2026-10-11）

Qwen3.5-0.8B、BF16、TP=1，在 Ascend910 V1/NPU0 上使用默认 bring-up 配置。客户端目标输入/输出为 512/128 tokens，并发 1/4/8/16/32；每档 3 轮、每轮 32 请求，共 15 份 summary、480 条请求记录和 30 份 NPU 前后快照。固定 revision 与配置见 raw summary；请求使用唯一 request ID 生成，prefix-cache hit ratio 为 0。目标输入为 512，但逐条 usage 实测输入长度为 481；输出长度均为 128。服务配置 `max_num_seqs=16`，并发列表示客户端负载，不等于引擎中同时执行的请求数。

在服务器项目目录开两个终端。终端 1 启动服务并保持运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/serve.sh configs/bringup/qwen35_0.8b.yaml
```

确认 `/readiness` 返回 HTTP 200 后，在终端 2 运行 benchmark：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/run_benchmark.sh configs/bringup/qwen35_0.8b.yaml \
  --concurrency 1 4 8 16 32 --input-tokens 512 --output-tokens 128 \
  --num-requests 32 --warmup-requests 3 --runs 3 \
  --output-dir results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_0.8b
```

下表为每档三轮的 run-level 指标均值；`±` 是三轮之间的样本标准差，p95 区间是三个 run-level p95 的观测范围，不是合并 96 个请求后的 p95：

| 客户端并发 | TTFT P50 均值±SD (ms) | run-level TTFT P95 均值 [范围] (ms) | TPOT P50 均值±SD (ms) | run-level TPOT P95 均值 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 432.63 ± 2.93 | 442.54 [434.30, 457.16] | 10.48 ± 0.02 | 11.09 |
| 4 | 1,435.35 ± 2.70 | 4,182.38 [1,520.67, 9,500.63] | 17.72 ± 0.10 | 20.80 |
| 8 | 2,205.22 ± 669.52 | 5,476.00 [2,749.07, 10,924.87] | 34.73 ± 0.06 | 62.42 |
| 16 | 3,650.80 ± 77.31 | 8,373.90 [8,085.28, 8,602.03] | 67.70 ± 2.48 | 75.95 |
| 32 | 11,161.94 ± 153.92 | 21,515.53 [21,477.43, 21,574.54] | 67.45 ± 2.30 | 76.17 |

| 客户端并发 | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | 总 token 吞吐 (tok/s) | 峰值 active-page ratio | scheduler waiting 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5615 | 71.87 | 341.97 | 0.1719 | 0 |
| 4 | 1.0007 | 128.09 | 609.41 | 0.6406 | 0 |
| 8 | 1.1134 | 142.52 | 678.08 | 1.0000 | 2 |
| 16 | 1.2355 | 158.14 | 752.41 | 1.0000 | 10 |
| 32 | 1.2384 | 158.52 | 754.19 | 1.0000 | 27 |

从并发 16 增到 32，总 token 吞吐仅从 752.41 增至 754.19 tok/s，而 TTFT P50 均值从 3.65 s 增至 11.16 s；在本配置和固定负载下呈现吞吐平台与排队/TTFT 代价。该现象是描述性基线，不证明某项优化的因果效果。active-page ratio 是 scheduler 页面指标，不是 HBM 占用；30 份 NPU0 前后快照的 Chip 0 `npu-smi` HBM 使用率均显示 10%，是离散快照而非运行期间峰值。

原始 JSONL、summary 和 NPU 快照见 `raw workload`（归档路径：`results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_0.8b`），跨端 32 个文件 SHA-256 全匹配。逐轮 CSV、按并发聚合 CSV 和专用 SVG 位于 `analysis`（归档路径：`results/analysis/agent_workload_a_512_128_20261011`）；聚合保留每轮数据，不把重复轮次伪装为独立样本。

## Workload A：4B 512/128 普通聊天并发扩展（2026-10-11）

Qwen3.5-4B、BF16、TP=1，固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，运行于 Ascend910 V1/NPU0。使用默认配置 `configs/bringup/qwen35_4b.yaml`（`max_num_seqs=8`、prefix cache 开启），客户端目标输入/输出为 512/128 tokens，并发 1/4/8/16/32；每档 3 轮、每轮 32 请求，共 15 份 summary、480 条请求记录和 30 份 NPU0 前后快照。逐条 usage 实际输入为 479 tokens、输出为 128 tokens；prefix-cache hit ratio 为 0。客户端并发高于 `max_num_seqs` 时会出现排队，不表示相同数量的请求同时在 NPU 执行。

复现时，服务端终端运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/serve.sh configs/bringup/qwen35_4b.yaml
```

服务 `/readiness` 返回 HTTP 200 后，在另一个终端运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/run_benchmark.sh configs/bringup/qwen35_4b.yaml \
  --concurrency 1 4 8 16 32 --input-tokens 512 --output-tokens 128 \
  --num-requests 32 --warmup-requests 3 --runs 3 \
  --output-dir results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_4b
```

下表为三轮 run-level 指标均值；`±` 为三轮间样本标准差。p95 区间是三个 run-level p95 的最小值–最大值，不是合并 96 条请求后的 p95。

| 客户端并发 | TTFT P50 均值±SD (ms) | run-level TTFT P95 均值 [范围] (ms) | TPOT P50 均值±SD (ms) | run-level TPOT P95 均值 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 825.53 ± 1.10 | 834.12 [832.48, 835.38] | 19.56 ± 0.00 | 19.81 |
| 4 | 1,998.37 ± 337.91 | 4,838.91 [3,208.23, 8,092.66] | 43.33 ± 2.83 | 47.93 |
| 8 | 5,321.60 ± 1,152.08 | 7,799.74 [7,168.71, 8,718.74] | 66.32 ± 5.62 | 71.30 |
| 16 | 17,916.25 ± 1,325.83 | 21,917.89 [21,476.12, 22,720.44] | 69.40 ± 5.65 | 72.55 |
| 32 | 25,176.48 ± 266.70 | 48,932.84 [48,179.01, 50,414.19] | 69.34 ± 5.52 | 73.82 |

| 客户端并发 | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | 总 token 吞吐 (tok/s) | 峰值 active-page ratio | scheduler waiting 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2964 | 37.94 | 179.91 | 0.5000 | 0 |
| 4 | 0.5186 | 66.38 | 314.78 | 1.0000 | 2 |
| 8 | 0.5653 | 72.36 | 343.16 | 1.0000 | 6 |
| 16 | 0.5734 | 73.40 | 348.06 | 1.0000 | 14 |
| 32 | 0.5760 | 73.73 | 349.66 | 1.0000 | 29 |

c16 到 c32 总 token 吞吐仅从 348.06 增至 349.66 tok/s，TTFT P50 均值则从 17.92 s 增至 25.18 s，scheduler waiting 峰值从 14 增至 29。该结果描述 `max_num_seqs=8` 的固定单机配置下吞吐平台与排队代价，不代表某项优化的因果收益。active-page ratio 是 scheduler 页面指标；30 份 NPU0 前后快照中，Chip 0 HBM 使用率均为 21%、Chip 1 均为 4%，只是离散前后快照而非运行期间峰值。

原始 JSONL、summary 与硬件快照见 `4B raw workload`（归档路径：`results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_4b`），32 个数据文件本地/远端 SHA-256 全匹配。15 行 run summary、按并发聚合 CSV 和五张 run-level SVG 位于 `4B analysis`（归档路径：`results/analysis/agent_workload_a_512_128_20261011/qwen35_4b`）；三轮是统计重复单位，不将 96 条请求当作独立 run。

## Workload A：9B 512/128 普通聊天并发扩展（2026-10-11）

Qwen3.5-9B、BF16、TP=1，固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，运行于 Ascend910 V1/NPU0。使用 `configs/bringup/qwen35_9b.yaml`（`max_num_seqs=8`、prefix cache 开启），客户端目标输入/输出为 512/128 tokens，并发 1/4/8/16/32；每档 3 轮、每轮 32 请求，共 15 份 summary、480 条请求记录和 30 份 NPU0 前后快照。逐条 usage 实际输入为 479 tokens、输出为 128 tokens；所有 summary 的 prefix-cache hit ratio 为 0。客户端并发高于 `max_num_seqs` 时会出现排队，不表示相同数量的请求同时在 NPU 执行。

复现时，服务端终端运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/serve.sh configs/bringup/qwen35_9b.yaml
```

服务 `/readiness` 返回 HTTP 200 后，在另一个终端运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/run_benchmark.sh configs/bringup/qwen35_9b.yaml \
  --concurrency 1 4 8 16 32 --input-tokens 512 --output-tokens 128 \
  --num-requests 32 --warmup-requests 3 --runs 3 \
  --output-dir results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_9b
```

下表为三轮 run-level 指标均值；`±` 为三轮间样本标准差。p95 区间是三个 run-level p95 的最小值–最大值，不是合并 96 条请求后的 p95。

| 客户端并发 | TTFT P50 均值±SD (ms) | run-level TTFT P95 均值 [范围] (ms) | TPOT P50 均值±SD (ms) | run-level TPOT P95 均值 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 849.25 ± 2.66 | 856.68 [851.75, 861.10] | 24.82 ± 0.02 | 25.39 |
| 4 | 1,707.56 ± 1.91 | 2,507.58 [2,506.21, 2,509.35] | 51.54 ± 0.13 | 54.72 |
| 8 | 4,296.17 ± 3.87 | 7,633.59 [7,620.28, 7,653.67] | 79.28 ± 0.06 | 81.96 |
| 16 | 18,661.31 ± 79.22 | 23,059.60 [23,006.15, 23,096.67] | 78.92 ± 0.52 | 81.97 |
| 32 | 26,898.21 ± 10.69 | 52,371.71 [52,172.76, 52,734.98] | 78.81 ± 0.75 | 82.13 |

| 客户端并发 | 请求吞吐 (req/s) | 输出吞吐 (tok/s) | 总 token 吞吐 (tok/s) | 峰值 active-page ratio | scheduler waiting 峰值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2494 | 31.92 | 151.39 | 0.3438 | 0 |
| 4 | 0.4861 | 62.23 | 295.08 | 1.0000 | 1 |
| 8 | 0.5339 | 68.34 | 324.07 | 1.0000 | 5 |
| 16 | 0.5316 | 68.04 | 322.68 | 1.0000 | 13 |
| 32 | 0.5338 | 68.33 | 324.02 | 1.0000 | 29 |

c8 到 c32 总 token 吞吐约为 322.68–324.07 tok/s，显示此固定配置下吞吐平台；c16 到 c32 的 TTFT P50 均值由 18.66 s 升至 26.90 s，scheduler waiting 峰值由 13 升至 29。该结果是描述性基线，不代表某项优化的因果收益。30 份离散 NPU0 快照中，Chip 0 HBM 为 35%（6 份）或 36%（24 份），Chip 1 均为 4%；这不是运行期间峰值。

原始 JSONL、summary 与硬件快照见 `9B raw workload`（归档路径：`results/raw/benchmark/agent_workload_a_512_128_20261011/qwen35_9b`），32 个数据文件本地/远端 SHA-256 全匹配；runner/server 日志也已同步并核对，相关 34 个文件均匹配。15 行 run summary、按并发聚合 CSV 和五张 run-level SVG 位于 `9B analysis`（归档路径：`results/analysis/agent_workload_a_512_128_20261011/qwen35_9b`）；三轮是统计重复单位，不将 96 条请求当作独立 run。

## 区分共享前缀样本

修正提示词前的首轮试跑同样包含 384 请求，但所有请求使用完全相同的提示词，prefix-cache hit ratio 约 0.99。该结果属于“重复全文/热前缀”压力样本，不作为普通并发 baseline；原始记录保留在 `results/raw/baseline/qwen35_0.8b_20261009/`（归档路径：`results/raw/baseline/qwen35_0.8b_20261009`）。

消融入口 E0–E3 用于比较可独立切换的优化阶段；continuous batching 保留，因为它是 TokenSpeed scheduler 的基础能力。Ascend 变体统一保持 `mixed_batch=false`，并统一传入 `--disable-kvstore`，避免关闭 prefix cache 时与默认 Host KVStore 冲突，也避免 E3 同时改变两项设置。E0/E1 使用 `chunked_prefill_size=max_model_len`，保证一个合法请求可在单次 prefill 中处理；当前分支的 `-1` sentinel 会传入 KV 容量模型并触发 `max_scheduled_tokens must be > 0`。E2/E3 的 chunk size 由 `--chunked-prefill-size` 指定，E3 另启用 prefix cache。E0 不替代本节默认配置基线，应分别留存。上游 OpenAI chat 路径仍未提供持久 Session KV API；项目 Engine 级 Session API 已通过 4B/NPU0 matched correctness/reuse 验收，并在 `overlap_schedule=true` 下完成续轮复验。独立串行 Session on/off context sweep 为 12/12 输出匹配，目标上下文 512/1024/1536 的 `Engine.generate` 配对中位延迟降幅为 77.95%/87.11%/91.28%；这不是 TTFT 或并发吞吐。另完成 4B/NPU0/TP=1 单会话原生 Engine 的 16-turn TTFT 配对：4×16 共 64 对，输出 IDs 64/64 匹配；Session/stateless TTFT 中位数为 122.71/1394.54 ms，配对降幅中位数为 91.52%。该结果仅适用于固定配置和合成会话，不代表 HTTP、多会话并发或跨模型/硬件收益。详见 [Session KV 串行 paired sweep](final_results.md#session-kv-serial-paired-latency-sweep-2026-10-10)、TTFT raw JSON（归档路径：`results/raw/session_kv_ttft/qwen35-4b-session-kv-ttft-20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a.json`）、TTFT CSV（归档路径：`results/analysis/session_kv_ttft_20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a/session_kv_multiturn_ttft.csv`） 和 TTFT SVG（归档路径：`results/analysis/session_kv_ttft_20261010T125937824616Z-92dd0788c7c0406e8d81a7aa6eaa806a/session_kv_multiturn_ttft.svg`）。
