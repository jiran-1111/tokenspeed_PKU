# 作业1：隔离验证数值判据候选（待用户确认）

## 状态与用途

当前状态：**候选已独立复审，尚未批准/冻结，验证组引擎数值未采集**。用户确认后记录本文件SHA，再执行阶段B；不能把一般“继续”解释为对本候选数值的专项确认。

用途限定：固定Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`、当前投影权重、BF16/TP1单chip、冻结的16条验证输入与16个固定续接，检查每条末端Prefill与连续16步Decode的完整248320维向量。这是一项有限配置的经验回归约定，不是普遍误差上界或充分正确性证明。

原始greedy、多步模块state、Session隔离和历史1%工程检查分别报告，不能互相替代。旧1%失败保留，不改成通过。

## 主参考与逐向量条件

主参考为同投影learned-weight值展开后的CPU HF FP32/eager；缓存递推输入前沿与被测引擎完全一致。使用全248320维向量、CPU FP64比较计算，`relative_L2 = norm(actual-reference) / max(norm(reference), 1e-12)`，另计算最大绝对误差。

| 阶段 | relative-L2 ≤ | max-abs ≤ |
|---|---:|---:|
| Prefill（step0） | 0.04160680933973707 | 0.4934459924697876 |
| Decode（step1–16） | 0.027781371644764415 | 0.4458632469177246 |

每个向量必须**同时**满足两项，所有规定向量都须通过；不能平均后通过、剔除样例或只看argmax。输入、位置、权重、源码身份或finite检查失败即停止；超限即该协议失败，不根据验证结果回调门槛。

完整向量在改变采样选择前采集。teacher forcing只固定下一输入ID，不省略模型计算；原始greedy另行运行，不用强制续接的输出匹配代替greedy验证。测试/诊断钩子不得进入性能样本。

验证输入来自既定manifest文件SHA `23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb`、canonical manifest SHA `f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936`，其中validation组16条；不能依据结果更换输入/续接/分组。未见验证引擎数值。

## 独立校准依据

只使用阶段A的CPU BF16/SDPA和NPU BF16/SDPA各自对CPU FP32/eager的误差，汇集cached/recomputed两条路径；不使用BF16设备互比或self误差，也不使用旧引擎失败值或验证组引擎值。不增加任意倍率。

Prefill两项最大值来自CPU BF16、calibration/1/0、step0；Decode relative-L2来自CPU BF16重算、calibration/2048/1、step5；Decode maxabs来自NPU BF16缓存、calibration/512/1、step2。可追溯表及完整CSV见`results/analysis/homework1_reference_calibration_20261015/`和[参考审计](homework1_reference_calibration_audit_20261015.md)。

校准身份、finite、输入前沿及计算分析经独立核对。FP32实现/缓存差异为数个1e-6且top1一致，支持本批主参考实现一致性，不证明精确真值。针对最早BF16分歧的实际局部重放，已分别定位BF16缓存写回和CPU BF16 rsqrt调用尺寸效应；统一归一化输入后recurrent/chunk核心state差异约7.5e-8。该局部机制解释不外推为全部层/logits误差分解，也不自动豁免未来新异常。

## 必须保留的限制

- 校准只有16条固定轨迹、两个相关来源族；17个连续向量不是17个独立样本。没有总体覆盖概率、置信上界或显著性结论。
- 最大值是observed maximum；Prefill最大值来自1-token样例，不能冒称对其他长度同等紧。
- 两指标最大值来自不同点，矩形约定可能允许参考未实际出现的联合误差组合。
- 通过的准确表述仅为“满足本次冻结经验回归判据”；结合模块、状态、greedy和实际NPU端到端证据描述作业结果，不宣布普遍等价。
- 新发现的具体语义异常、身份/finite问题或验证超限必须独立报告，不能套用“浮点误差”解释后直接通过。

## 待用户决定

是否接受上表作为本次固定配置的经验回归约定，确认后冻结本文件SHA并开始原16条验证组的阶段B？拒绝或要求修订时不采验证引擎数值；校准原始数据和旧失败保持不变。
