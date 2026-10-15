# 当前生产源码：9B固定样例复验

## 协议

原固定八条greedy流程，当前生产Gamma/QK修正后源码，不修改采样或验收标准。实际Ascend910 V1、logical2（物理NPU1/chip0）、TP1/BF16，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；baseline9B关闭prefix cache、KVStore、overlap/mixed batch，chunk4096、block128、context4096、max_num_seqs8、max_total_tokens32768，max_new_tokens64。

空闲设备/四端口检查通过，远端已有模型，仅在确认目标不存在后同步已有baseline9B配置；无模型/依赖下载。引擎与同输入HF NPU参考串行运行，先关闭本次自有服务并复查目标卡空闲，未启动Profiler/性能测量或临时精度钩子。多轮为全历史重发，不称Session KV实验。

raw独占目录`results/raw/correctness/current_9b_20261014/`；一次性编排`.tmp/current_9b_correctness_20261014.py`及八份所选源码/配置归档，结束后hash未变。不是完整仓库/最终版本发布快照。

## 严格结果

八条完整native IDs已采集，raw及stop-aligned均7/8，不是完全正确性通过。

| 样例 | turn | actual输入 | actual输出 | raw/stop-aligned |
|---|---:|---:|---:|---|
| chinese_beijing | 1 | 17 | 64 | exact/exact |
| english_capital | 1 | 23 | 64 | exact/exact |
| math_arithmetic | 1 | 24 | 64 | exact/exact |
| code_python | 1 | 24 | 64 | exact/exact |
| long_context_recall | 1 | 977 | 64 | mismatch/mismatch |
| multi_turn_memory | 1 | 19 | 64 | exact/exact |
| multi_turn_memory | 2 | 108 | 64 | exact/exact |
| multi_turn_memory | 3 | 190 | 64 | exact/exact |

唯一分叉仍为long_context_recall的0-based output index59：engine token264，HF参考6278；双方64 tokens，首59个ID相同，不是终止边界导致。对同一固定样例的旧2026-10-11官方运行逐条核对，当前8/8输入IDs及8/8引擎输出IDs与旧记录完全一致。不同日期及配置不是单变量因果实验，该对照只说明这批固定输入/输出未变，不能证明所有数值或性能无变化。

参考使用固定revision/BF16的Transformers NPU `generate`、do_sample=false、num_beams1；可选fast-path缺失采用既有torch NPU参考，没有安装依赖。每条参考均使用当前engine原生input IDs，不用较早历史替代当前多轮前缀。

## 终态与边界

编排56644正常退出0、自有940989服务退出0且不存在；20文件（原始ID/对照/结果/源码/配置/日志/快照）跨端SHA-256全部一致。物理NPU1已无运行进程，NPU0另一用户服务未干预，不宣称全服务器空闲。

保留CANN owner、内部格式及退出提示；正常结束采集不等于通过strict检查。全部输出64-token上限不保证答案语义已完整。未调整0.8B额外logits合同或数值方向，不因旧参考cache路径敏感或近并列诊断自动豁免本次9B分叉。

当前源码9B固定回归确认7/8，仍需正确性闭环与其他配置/Session/优化性能证据；4B当前8/8结果不外推给9B、27B或0.8B，原四模型Agent总目标保持。
