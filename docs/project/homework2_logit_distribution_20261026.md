# 原4B/9B分叉：分数平移与分布差异

本记录只重算原128输入的共同前沿，未跑新NPU实验，未改核心源、固定续接、样例或判据。原4B/9B自由输出失败仍保留。目的在于区分完整logits的大relative-L2与对候选分布的影响，不能由量度变化宣称正确性通过。

## 数据和方法

沿用诊断R2的4B18步、9B10步，各有CPU参考、TP1/rank0、TP2/rank0及rank1，共112个248320维向量。180个原始case、张量、采样、参考状态、controller和旧分析文件逐SHA固定。导出器验证输入/位置/原始argmax/广播及固定续接、guard与外层退出释放；84个NPU张量SHA还逐项关联原分析，原CPU权重指纹和前沿身份保持。核心1531项清单仍为2a1a2196...。这不是112个独立题目，也不是新一轮模型生成。

以a为实际、r为参考，δ=a−r，分数整体平移分量为mean(δ)。其平方能量份额为`V*mean(δ)^2 / sum(δ^2)`，V=248320；它只是正交代数分解，不是误差来源的因果贡献。中心化relative-L2用各自去均值后的差作分子、`||r−mean(r)||`作分母，不能从raw与中心化L2的降幅反推平移份额。

对保存的向量另计算温度1 softmax，给出TV=`sum(abs(p−q))/2`、双向KL、JS（自然对数、nats）和top10重叠；稳定log-softmax，不clip概率或设通过线。**温度1是离线诊断变换，不是原temperature=0 greedy的实际采样分布。** top10并列按低token ID裁切；未定义的零分母保留null及flag。浮点舍入可能使数学上为零的JS出现约±5e-17，原值保留。

## TP2与TP1的关键步骤

step为0起始，step0是Prefill最后位置127，step17/9分别对应第18/10个输出。

| 模型与步骤 | raw relative-L2 | 平移平方能量份额 | 温度1 TV | 温度1 JS |
| --- | --- | --- | --- | --- |
| 4B step0 | 0.30816710 | 0.7811% | 0.01026838 | 0.00240787 |
| 4B step17 | 0.02293070 | 1.3333% | 0.03122613 | 0.00048787 |
| 9B step8 | 0.38528013 | 21.1815% | 0.06402380 | 0.02117342 |
| 9B step9 | 0.12439306 | 4.5437% | 0.34141633 | 0.11764650 |

4B最大raw误差出现在step0，平移能量仅0.7811%，不能把整个向量差异归成统一偏移。其step17候选198/271在TP1并列最高，TP2以0.125分差选择271；TV为0.031226。该末步支持并列附近翻转的解释，但不解释Prefill差异的全部来源。

9B最大raw误差在step8，平移能量份额21.1815%；step9的TV升到0.341416、JS为0.117646。step9中35378在TP1排第10、TP2排第1，TP1选25389，独立CPU亦选25389；TP1与TP2各自最高/次高分差为0.243282/0.153355。因此此处不只是严格并列选择，也不能解释成全部logits一起平移。以上不构成新质量门槛。

## 与独立CPU参考的关系

| 模型 | TP1/CPU最大TV | TP2/CPU最大TV | TP1/CPU最大JS | TP2/CPU最大JS |
| --- | --- | --- | --- | --- |
| 4B | 0.00955181 | 0.02533696 | 0.00137343 | 0.00051895 |
| 9B | 0.17541535 | 0.32276233 | 0.01908499 | 0.10458380 |

TP2的rank0/1原始向量逐位相同，TV与KL为0；不能由两rank自一致认证对CPU的正确性。各模型、各量度的最大值可以来自不同步骤。没有随机重复或总体推断，不给置信区间、显著性或一般模型质量通过率。

## 复算与限制

[全部140项配对指标](../evidence/homework2_20261026/frontier_distribution.json)保留逐步结果；[归档与工具身份](../evidence/homework2_20261026/frontier_distribution_provenance.json)记录180原文件、112导出向量、模型权重指纹、40项两端测试及本地/服务器分析对应。全部15文件归档已下载逐SHA，原首次30秒CPU分析超时保留；仅限制BLAS线程为1后重算，模型、公式和输入不变。两端2528个浮点字段最大差1.3001e-13，在重算比较1e-12数值容差内，非浮点字段完全相同；这项计算容差不是引擎验收门槛。

先恢复工作区归档的export、原frontier_analysis_r2.json，再核对所有SHA。从仓库根目录、现有工作区NumPy/Pytest环境执行；完整导出还需匹配的CPU Torch环境与原180文件，不能用公开摘要替换原张量。输出路径必须未存在。

```bash
python -m zipfile -e docs/evidence/homework2_20261026/frontier_distribution_tools_r1.zip .tmp/frontier-distribution
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
python -B -m pytest .tmp/frontier-distribution -q
python -B .tmp/frontier-distribution/analyze_distribution.py   --root /path/to/restored/export   --manifest .tmp/frontier-distribution/selected_frontier.json   --baseline-analysis /path/to/restored/frontier_analysis_r2.json   --output .tmp/new-frontier-distribution.json --top-k 10
```

这些数据排除了“只要去掉统一偏移就没有问题”的简单解释，也说明最大向量误差与最大分布变化不一定同一步。它没有定位最早差异的完整起因，不能排除核心bug，也不认证生成质量。下一步数值定位应优先对原9B step9的实际输入、FA缓存与GDN状态做同前沿核对，并沿此前step8诊断追踪传播；任何修复仍须新源固定后重验原失败。作业2主报告继续保持原失败结论，作业1冻结验收不变。
