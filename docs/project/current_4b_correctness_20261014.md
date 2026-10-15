# 当前生产源码：4B固定样例复验

## 范围与配置

复验近期Gemma有效gamma及Q/K norm修正后的生产路径，复用原`run_correctness.py`与`compare_correctness_reference.py`，不修改模型、采样或验收政策。实际Ascend910 V1、logical2（物理NPU1/chip0）、TP1/BF16，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。

配置为`configs/baseline/qwen35_4b.yaml`：prefix cache/KVStore/overlap/mixed batch关闭，chunk4096、block128、context4096、max_num_seqs8、max_total_tokens32768。没有临时诊断钩子、Profiler或性能测量。首轮检查发现远端仅缺该本机已有配置，确认不存在后同步；模型和依赖已存在，没有重新下载或覆盖其他配置。

新独占raw根目录：`results/raw/correctness/current_4b_20261014/`，编排源`.tmp/current_4b_correctness_20261014.py`及八份所选源码/配置归档。结束后逐份源码SHA复验未变；这不是完整仓库快照，也不是最终版本发布声明。

## 固定greedy对照

引擎记录原生input/output IDs及文本，生成上限64；参考为同revision/BF16的Transformers NPU greedy `generate`，`do_sample=false, num_beams=1`，使用每条引擎记录的完全相同input IDs。先关闭自有服务并重新确认目标卡空闲，再在该chip串行运行参考。可选fast-path库未安装，使用既有NPU torch参考，不安装依赖；不是CPU serving替代。

| 样例 | turn | 实际input | 实际output | raw/stop-aligned |
|---|---:|---:|---:|---|
| chinese_beijing | 1 | 17 | 64 | exact/exact |
| english_capital | 1 | 23 | 64 | exact/exact |
| math_arithmetic | 1 | 24 | 64 | exact/exact |
| code_python | 1 | 24 | 64 | exact/exact |
| long_context_recall | 1 | 977 | 64 | exact/exact |
| multi_turn_memory | 1 | 19 | 64 | exact/exact |
| multi_turn_memory | 2 | 109 | 64 | exact/exact |
| multi_turn_memory | 3 | 190 | 64 | exact/exact |

八条原始IDs及停止边界对齐均8/8一致。多轮通过全历史重发构造，不是Session KV复用测量；全部64-token上限不保证最终答案已输出完整，不把ID一致推导为无限长度、所有提示词或语义质量普遍正确。此处不检验0.8B完整logits的1%合同，也不改变其失败状态。

## 归档与终态

编排47132正常退出0、自有服务938577退出0且进程不存在；结束后物理NPU1无运行进程。NPU0属于另一用户的服务始终未干预，不声称全服务器空闲。20个raw/配置/源码/日志文件本机与远端SHA-256全部一致，记录结果见`result.json`与`reference_compare.json`。

服务关闭阶段CancelledError、CANN目录owner和参考base-format警告均保留；关闭取消提示未影响已完成八条数据和最终正常退出。初次因远端缺配置在结果目录/服务创建前退出，未重启任何活进程；一次只读误猜参考脚本名后按实际仓库入口纠正。

这补齐当前生产路径4B/TP1的固定样例回归，尚不能替代4B TP2/Session/优化性能完整验证；0.8B、9B、27B及最终四模型同版本验收、完整Agent优化/消融仍未完成。
