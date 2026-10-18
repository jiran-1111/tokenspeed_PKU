# TokenSpeed Ascend Agent 推理课程项目

本仓库从上游 TokenSpeed `cf4a4f89ae5f9423a9af490d8f3a0613adfba3fd` 开始，复用其 Qwen3.5 模型、C++ continuous-batching scheduler、chunked prefill、paged KV/prefix cache 与 NPU backend。课程代码集中在 `benchmarks/course_project/`、`configs/bringup/` 和 `scripts/`。


## 最终课程交付（2026-10-18）

见[最终入口与精确复现命令](docs/project/final_delivery_20261018.md)、[总报告](docs/report/course_project_final_20261018.md)、[最终源与四模型验证证据](docs/evidence/course_final_20261018/README.md)和[答辩材料](docs/presentation/README.md)。最终分支`course/final-20261018`，1415源清单d05a47d9；原1409实现与R9r2逐字节一致，仅新增六配置。四模型最终128原生/重算配对通过，R9主性能768正式请求全部匹配，完整软件637服务器通过/零skip。有限TP候选与失败、旧数值/greedy失败、统计和硬件边界均如实保留；教学网用户本人提交。后文bring-up与旧报告按各自日期解释，以最终入口为准。

## 第一次作业材料（2026-10-15）

[阶段报告](docs/report/homework1_compute_subsystem.md)与[源码清单、标量证据及复现命令](docs/evidence/hw1_20261015/README.md)已整理。0.8B原16条隔离轨迹的272个完整logits通过预先批准的有限经验双门槛；原1%门槛68个向量失败、条件argmax13处差异独立保留。最终源服务器course569项、关键模块/Session108项通过；最终Decode Profiler及96请求的2×2性能/HBM测量已完成。它们不代表自由greedy完全一致或四模型Agent项目完成。

交付使用独立分支`course/hw1-20261015`，不覆盖目标fork的main；教学网报告由用户本人提交。

## 环境

课程目标硬件为Linux Ascend910C节点，并需匹配的驱动、固件和CANN。2026-10-14只读核实实际SoC为`Ascend910_9362`、CANN短系列为`Ascend910_93`；一手软件来源将其映射到Atlas A3/910C系列。DMI产品名为Huawei Atlas800I A3，CPU标识及一手来源支持Kunpeng920系列，具体CPU子型号和完整采购配置未直接读取，见[硬件身份与来源](docs/project/hardware_identity_20261014.md)。此前凭`Chip Version V1`认定“非910C”的结论撤回；这不改变已测设备、原始数据或数值失败。上游NPU安装脚本声明PyTorch2.9.0、torch_npu2.9.0.post2和Triton-Ascend3.2.1，应以目标节点实际兼容矩阵为准。Windows项目Conda环境只用于脚本和无设备单元测试，不代表NPU运行环境。

项目环境、缓存、权重和实验产物均限定在当前工作区/仓库目录内，不修改其他项目环境。Windows 开发机复用工作区根目录的 Conda 环境 `..\.conda-env/`（相对仓库根），由 `..\scripts\use_project_env.ps1` 激活；Linux 目标机的 setup 脚本则在仓库根目录创建 `.conda-env/`。新缓存、模型、数据和结果分别放在 `.cache/`、`models/`、`data/` 和 `results/`；Conda 可复用已有 base package cache，但项目环境前缀仍独立。本次实测复用已经建立的项目Ascend环境；`setup_env.sh`是环境搭建入口，其建议版本不等于本次实际版本，不能据此承诺跨不同CANN栈直接安装。已建立环境从仓库根目录执行：

```bash
source scripts/use_project_env.sh
```

首次下载权重需要 Hugging Face 网络权限。脚本把 Hugging Face cache 指向仓库内 `.cache/huggingface`；模型 ID 固定在下列配置中。

## 模型与启动

统一启动入口接收配置文件。首次 bring-up 从 0.8B、短上下文和 TP=1 开始：

```bash
bash scripts/serve.sh configs/bringup/qwen35_0.8b.yaml
```

其余配置为`qwen35_4b.yaml`、`qwen35_9b.yaml`和`qwen35_27b.yaml`。27B默认TP=2；bring-up配置的`bringup_unmeasured`标签不应代替实际实验记录。2026-10-10曾在HCCL启动阶段失败，旧日志保留；之后已恢复TP2 readiness并完成E0 Workload A。2026-10-14当前生产版本八个固定样例与同输入HF参考raw/stop-aligned均8/8，见[当前27B复验](docs/project/current_27b_correctness_20261014.md)。这些结果不代表全项目或任意输入正确性通过，历史性能也不冒充当前源码性能；详情见[当前与历史结果](docs/project/final_results.md)。

服务兼容 OpenAI Chat Completions：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-0.8b","messages":[{"role":"user","content":"你好，请用一句话介绍北京。"}],"temperature":0,"max_tokens":64}'
```

## 正确性、Benchmark 与 Profiler

```bash
bash scripts/run_correctness.sh configs/bringup/qwen35_0.8b.yaml
bash scripts/run_benchmark.sh configs/bringup/qwen35_0.8b.yaml
bash scripts/run_agent_benchmark.sh configs/bringup/qwen35_0.8b.yaml
bash scripts/run_ablation.sh configs/bringup/qwen35_0.8b.yaml
bash scripts/run_profiler.sh configs/bringup/qwen35_0.8b.yaml
bash scripts/generate_report.sh
```

普通 benchmark 参数可调整并发、输入/输出 token 数和请求数；Agent benchmark 覆盖共享前缀、多轮上下文增长与客户端 mock tool wait。mock wait 只让客户端暂停并注入固定结果，不会让服务端进入 `WAITING_TOOL` 状态，也不单独证明 Session KV 留存。消融脚本每个配置预热 3 个请求，正式测量 3 轮，自动启动并停止服务。Profiler 会采集 prefill-heavy、decode-heavy 和高并发 Agent 场景，原始 trace 写入 `results/profiles/`，请求统计写入 `results/raw/profiler/`。

原生 Engine API 的 Session KV correctness/reuse 复现固定使用 4B 配置，并要求可用的 Ascend NPU：

```bash
bash scripts/run_session_kv.sh --frontier-policy current-frontier --overlap-schedule --max-new-tokens 8
```

`current-frontier` 不在轮末额外执行 forward；下一轮预填上一轮最后采样的 token 和新追加后缀。该 runner 使用固定的 4B 配置，在通用 prefix cache 与 KVStore 关闭时比较两条交错 session 和 stateless 对照；结果保存在 `results/raw/session_kv/`。串行 Session/stateless 延迟复现入口为 `scripts/run_session_kv_perf.sh`，参数说明可通过 `--help` 查看。此 Session API 属于原生 Engine，不是 OpenAI Chat Completions 接口的一部分。

16 轮 Session KV native Engine TTFT 的实现入口为 `scripts/run_session_kv_ttft.sh`。当前已完成 4 次、每次 16 轮的串行配对：64/64 输出 ID 一致，Session/stateless TTFT 中位数为 122.71/1394.54 ms，配对降幅中位数为 91.52%。这是固定 4B/TP=1/NPU0 workload 的原生 Engine TTFT，不代表 HTTP、并发或跨模型收益；raw、CSV、SVG 与边界见 `docs/project/final_results.md`。

所有脚本默认把 JSONL、SVG、CSV 和硬件快照放在 `results/` 下。Benchmark summary 记录 mean/p50/p95 TTFT、TPOT、E2E、ITL，TokenSpeed KV cache 使用率、prefix hit、prefill/generation token 计数，以及压测前后的 `npu-smi info` 路径。报告生成器只绘制原始数据中实际存在的指标，不补造缺失数据。

### 长 Prefill 对持续 Decode 的干扰

该实验使用 4B/TP=1、`max_model_len=4096`，对比 2048/3584-token Prefill 与 256-input/128-output Decode；chunk size 为 4096/512、共同关闭 prefix cache 和 KVStore，按配对顺序执行三轮。SSE 块间隔指标是客户端观测值，不代表 token ITL，也不能单独用于算子因果归因。

先运行不加载模型、不检查 NPU、不启动服务的 dry-run：

```bash
python scripts/run_long_prefill_decode.py configs/bringup/qwen35_4b.yaml --dry-run
```

真实实验只能在已获准使用的节点上、确认 NPU0 空闲且目标端口可用后执行。`--confirm-npu0-idle` 是操作者确认门，不替代课程/管理员授权；NPU 不空闲或状态不明时应停止，不要抢占其他任务：

```bash
python scripts/run_long_prefill_decode.py configs/bringup/qwen35_4b.yaml --confirm-npu0-idle
```

Runner 会将 raw JSONL 写入 `results/raw/benchmark/long_prefill_decode/<run_id>/`。分析时为输出选择一个尚不存在的项目内目录：

该 run 的 `run.json` 保存配置/模型版本和五个 benchmark 源文件的 SHA-256，用于核对测量代码；完整 runtime/内核版本还需结合环境记录与源码快照。

```bash
RUN_ID="REPLACE_WITH_RUN_ID_PRINTED_BY_RUNNER"
python scripts/analyze_long_prefill_decode.py \
  --raw-run-dir "results/raw/benchmark/long_prefill_decode/${RUN_ID}" \
  --output-dir "results/analysis/long_prefill_decode/${RUN_ID}"
```

2026-10-13 已完成 Ascend910 V1/NPU0 三轮正式配对和 raw/CSV/SVG 验收，见 [验收记录与产物索引](docs/project/long_prefill_acceptance_20261013.md)。512 配置未降低本次 staged Decode E2E；没有 Prefill send 之前的块间隔样本，相关统计明确标记数据不足。此结果不代表整体课程项目完成，也不替代 Profiler 算子归因。

## 本地代码测试

Windows/CPU 开发机可运行不需要模型或 NPU 的课程单元测试：

```powershell
. ..\scripts\use_project_env.ps1
$env:PYTHONPATH = "$(Get-Location)\python;$(Get-Location)\tokenspeed-kernel\python"
$pytestTemp = Join-Path '.tmp' ('pytest-course-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $pytestTemp) { throw '测试临时目录已存在，请勿复用' }
python -m pytest -q test/course_project --basetemp $pytestTemp
```

这组测试验证配置构造、流式 API 客户端、指标统计、Profiler 请求封装、报表和命令入口；不能替代目标硬件上的模型正确性、吞吐、HBM 或 NPU Profiler 验收。

## 结果复现与项目记录

逐项完成范围与剩余缺项见[课程提交验收状态](docs/project/submission_status.md)。

第一次作业的当前入口为[计算子系统报告草稿](docs/report/homework1_compute_subsystem.md)、[要求对照](docs/project/homework1_requirements_audit_20261013.md)及[增量计划](docs/superpowers/plans/2026-10-13-homework1-compute-subsystem.md)。当前0.8B单chip已有模块/state/Session证据和[生产基线/HBM实测](docs/project/homework1_baseline_hbm_20261014.md)，但全模型数值验收、当前Decode热点、提交版报告与最终复验尚未完成；四模型Agent总任务继续保留。

环境版本、源码路径、基线定义、实验状态与报告草稿位于 `docs/project/` 和 `docs/report/`。动态数字必须能追溯到 `results/raw/`；Profiler 原始 trace 位于 `results/profiles/`。当前若这些目录没有实验文件，即表示目标 NPU 实验尚未执行，不应将静态代码检查称为运行结果。
