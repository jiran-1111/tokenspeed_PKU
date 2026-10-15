# 作业1：当前生产版本2×2基线与HBM采样

## 配置与范围

固定0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`、Ascend910 V1单logical chip0、TP1/BF16。配置为`configs/baseline/qwen35_0.8b.yaml`，chunk4096、block128、max_model_len4096、max_num_seqs16、max_total_tokens65536；prefix cache、KVStore、mixed batch及overlap关闭。生产版包含Gemma FP32有效gamma和Q/K norm输出恢复BF16修正；没有加载临时精度/逐层诊断钩子，也没有启动Profiler。

本次是当前源码的真实性能基线，不代表未通过的全模型数值合同已被豁免，也不提前宣布最终交付版本冻结。后续若生产计算改变，应重新核验性能。

编排脚本`.tmp/homework1_baseline_hbm_20261014.py`复用原benchmark，target输入128/1024×输出32/128，c1，每条件预热3次、三轮正式测量，每轮8请求，共96正式请求、12 summaries、24轮次前后快照。九份所选源码/配置归档并记录SHA；结束后重新读取远端源码，九个hash全部未变。采样与服务全部在项目Conda和项目目录内，没有安装依赖、覆盖旧raw或修改共享环境。

## 核验后的性能

raw根目录：`results/raw/homework1_baseline_hbm_20261014/`。分析：`results/analysis/homework1_baseline_hbm_r3_20261014/`，`summary.csv`、`hbm_samples.csv`及`audit.json`可由`.tmp/homework1_baseline_hbm_analysis_20261014.py`在新独占目录复算。以下先取每轮P50，再对三轮取均值，不是合并24请求后计算分位数。

| actual输入/输出 | TTFT P50均值(ms) | 客户端TPOT P50均值(ms) | 服务端ITL mean均值(ms) | E2E P50均值(ms) | 输出吞吐(tok/s) |
|---|---:|---:|---:|---:|---:|
| 130/32 | 220.159 | 11.623 | 11.658 | 580.079 | 54.822 |
| 130/128 | 203.311 | 10.795 | 10.779 | 1574.293 | 81.404 |
| 940/32 | 724.547 | 11.139 | 11.231 | 1070.054 | 29.786 |
| 940/128 | 718.725 | 10.854 | 10.871 | 2099.258 | 60.925 |

四条件均24请求、三轮、实际输出恰为32/128，无提前结束。目标输入不是实际长度；网关模板/构造后actual为130/940，不能称精确128/1024 native测量。12轮服务端ITL observation count全部等于`8×(output−1)`，即248/1016，metrics可用；ITL来自Prometheus histogram差分，不是SSE块间隔。TTFT包含服务、主机、HTTP等，不能称纯Prefill算子时间。

较长输入的额外延迟主要出现在首Token前，而稳态每Token指标变化较小；该观察区分了阶段趋势，但不单独区分计算、访存、布局转换或Host因果贡献。没有用新旧两版数据计算优化收益。

## HBM占用证据

以完整`npu-smi info`连续采样，保存查询开始/结束epoch ns、阶段、返回码和原始stdout/stderr。共121次查询，均返回0，覆盖startup、两档输出的负载阶段及teardown；加载与退出另有完整独立快照。

| 状态 | chip0设备HBM(MB) |
|---|---:|
| 启动前 | 3122 |
| health通过后 | 8461 |
| out32负载采样最大 | 8465 |
| out128负载采样最大 | 8465 |
| 退出快照 | 3117 |

每次查询后等待1秒，并非严格1Hz：最长查询耗时1.413秒，最大相邻查询起点间隔2.414秒。峰值是采样观察到的最大值，可漏掉更短的尖峰，也包括条件预热；不是任意瞬时峰值保证。原始设备表有时把AICore/Memory/HBM字段合在同一列，解析通过chip/PCIe行及最后的used/capacity数对锁定HBM，避免把容量或进程内存行当占用。

这些是driver设备总占用，不能把差值称作全部模型有效tensor/KV，也不能把退出后的约3.1GB基础占用判成泄漏。最后设备检查四物理NPU均无进程，服务PID929388已不存在。

## 终态与失败记录

编排95888正常退出0，持有服务正常退出0；同步27041退出0。46个raw/provenance/日志文件跨端SHA全部一致。本机课程测试415通过、13跳过（8.17s），不替代NPU模型验收。

首次空闲guard拒绝发生在结果目录和服务创建前；随后重新查询确认设备空闲后才启动，未捕获首次stdout，原因不做无依据推断。离线分析第一次和第二次仅留下summary.csv：错误假设HBM总是独立竖线列，第二次PCIe筛选仍未解决字段合并；两个旧分析目录保留。完整检查实际行后按尾部usage/capacity解析，在新r3目录成功；一次诊断使用错误cwd失败后纠正，未改raw。

当前已补齐这一版本的完整矩阵、有效负载期采样及加载/退出HBM证据。仍欠全模型数值闭环、当前版本稳态Decode/Host/HBM完整热点归因、最终提交报告与新进程复验；原四模型Agent优化/消融范围不变。

远端独立执行同一分析脚本成功，生成的summary.csv及hbm_samples.csv与本机逐字节一致；不需要重跑服务或覆盖任何raw。
