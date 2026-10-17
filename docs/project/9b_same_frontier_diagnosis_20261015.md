# 9B 长上下文首差前沿诊断（2026-10-15）

## 问题与固定条件

历史固定八条 greedy 为 7/8；唯一分叉是 `long_context_recall` 的 0-based 输出 index59：engine 为264，HF BF16缓存参考为6278。原977个输入ID及前59个共同输出组成1036-token相同前沿，最后输入ID为11、position1035。本次不改样例、采样、生产源码或数值门槛，也不把0.8B冻结经验判据应用给9B。

模型为 Qwen3.5-9B、revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，BF16/TP1。生产源码与已发布HW1分支 `course/hw1-20261015` 对应，1396文件清单SHA为 `f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e`，各次采集前后逐项核对未变。每次NPU运行前核对逻辑2→物理NPU1/chip0及设备无进程，两个NPU实验串行，其他设备任务未干预。

## 已采集观察

| 路径 | token264 logit | token6278 logit | 实际生成/观察 |
|---|---:|---:|---|
| HF BF16缓存，原prompt后逐token处理59个共同ID | 30.25 | 30.375 | 6278分数更高 |
| HF BF16完整1036-token前沿重算 | 30.25 | 30.25 | 两候选精确并列 |
| Engine BF16连续Decode，诊断eager | 30.25 | 30.25 | index59实际输出264 |
| Engine BF16完整1036-token前沿重算，诊断eager | 30.25 | 30.25 | 实际输出264 |

HF探针复用 `scripts/probe_reference_prefix_logits.py`，默认选择attention实现，未人为指定eager/SDPA；其结果不能代替同投影权重CPU FP32/eager参考。Engine探针复用正式 opt-in logits hook，仅采集目标前沿的top2；为保证Python hook实际执行，显式设置eager。该eager请求的全部64输出与历史graph baseline逐ID一致，前59共同ID核对通过；这只验证了这一条轨迹，不能推出任意输入的eager/graph等价。

不能把 `topk` 对并列值的返回顺序当作实际采样结果；上述engine token264来自实际输出ID。两条hook记录分别确认为DECODE/1输入token及EXTEND/1036输入tokens，词表均248320。三轴文本MRoPE的末position一致；此旧诊断hook记录展开后的position_count，3及3108不是输入token数。

该观察表明：在首差前沿，HF自身缓存/重算已出现一个BF16量化步长的候选排序变化，引擎连续与重算这两个候选得分一致。它支持继续核查参考状态精度和近并列敏感性，尚未定位产生差值的具体层、kernel或运算；没有证明引擎全向量正确，更没有取消原strict 7/8失败。

## 证据与后续

- HF原始目录：`results/raw/correctness/9b_same_frontier_20261015/`，退出0，源未变，目标设备释放。
- Engine原始目录：`results/raw/correctness/9b_engine_frontier_20261015/`，两条目标logits记录、连续64输出及重算1输出齐全，进程退出0。退出日志保留，不宣称无警告退出。
- 以上20个数据/源码/配置/日志/快照已跨端SHA逐项验证一致；归档SHA `7f4219900d0f79f1480be5075d11153158aee39d24da29b0b37f3a28757075c7`。
- 原HW1发布提交保持不动，9B新资料待完整验证后另行交付。

## 完整向量与最后投影的独立检查

新增完整引擎向量目录为 `results/raw/correctness/9b_engine_frontier_full_20261015/`。同时保存真实LM head输入hidden及264/6278两行实际权重，不修改生产计算或采样；对保存值以CPU FP64重新点积，结果为：

| 实际引擎输入状态 | token264点积 | token6278点积 | 6278优势 | 两者舍入BF16 |
|---|---:|---:|---:|---|
| 连续Decode | 30.2059670341 | 30.2860613632 | 0.0800943291 | 30.25 / 30.25 |
| 完整前沿重算 | 30.2200519778 | 30.2848570026 | 0.0648050247 | 30.25 / 30.25 |

这直接显示实际hidden/weight所支持的两候选高精度排序在最终BF16输出表示时丢失，而不是通过替换输入或强制参考token制造差异。两路引擎全248320维向量仍有180733项不同，连续对重算relative-L2为0.0071423603、maxabs0.15625；不能由两候选相同推导全部logits相同。

同投影权重CPU FP32/eager已完成，427参数值指纹 `3fb5b1bfbf1fe8817ecafdbe4b0fc0bf58447cd6b6b4a9738f73670578b86834`，1线程；加载后逐参数检查CPU，实际输入及缓存前向为CPU张量。最终依赖懒导入记录 `npu_imported=true`，如实保留，不把模块导入等同NPU计算。运行约1013.69秒，源码前后核对一致；原始目录 `results/raw/correctness/9b_cpu_fp32_frontier_20261015/`。

CPU缓存/重算都选择6278，优势分别0.0118503571/0.0118389130；两CPU全向量间relative-L2为1.68310985e-6、maxabs3.43322754e-5。引擎连续向量对CPU缓存参考的relative-L2/maxabs为0.0067328996/0.1235775948；引擎完整重算对CPU完整重算为0.0077830749/0.1291558743。这些是单个已知首差前沿的描述量，没有应用0.8B门槛或新定义9B通过阈值。

CPU结果、完整引擎捕获及对应源码/日志已同步逐项SHA核对。数据支持对“仅保留最终LM head FP32输出”的单变量试验进行检验，不支持直接宣布整个9B已修复，也不解释所有中间层误差。
