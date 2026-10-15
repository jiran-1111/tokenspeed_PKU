# 当前生产源码：27B TP2固定样例复验

## 设备与协议

实际Ascend910 V1，使用空闲logical2/3（物理NPU1的chip0/1），避开NPU0另一用户服务。先复用既有`probe_hccl_pair.py`验证这对设备：FP32/BF16、1/4096元素，两rank八项all_reduce均得到3，62818退出0，目标卡回到无进程；六份基本通信证据跨端SHA一致。该探针不是模型级通信或性能验收。

随后运行当前生产Gamma/QK/归约helper及容量/上下文边界修正后的baseline27B，revision `fc05daec18b0a78c049392ed2e771dde82bdf654`、TP2/BF16、max_new_tokens64。prefix cache、KVStore、overlap/mixed batch关闭，chunk4096、block128、context4096、max_num_seqs4、max_total_tokens16384。不改判据、不开临时精度钩子、不采Profiler或性能。

raw独占目录`results/raw/correctness/current_27b_20261014/`，编排`.tmp/current_27b_correctness_20261014.py`。十一份所选源码/配置归档，含实际Norm、QK、cache recipe、event loop及request handler；运行期间hash未变，不等于完整发布快照。

服务完成双rank初始化、权重/encoder/graph warmup、HTTP health后运行八固定样例。自有服务完全退出且目标卡重新检查空闲后，在logical2单chip串行执行同revision/BF16的独立HF NPU greedy参考；参考使用每条engine的真实input IDs、do_sample=false、num_beams1。未下载模型/依赖，不是CPU serving替代。

## 结果

raw及stop-aligned均8/8一致。

| 样例 | turn | actual输入 | actual输出 | raw/stop-aligned |
|---|---:|---:|---:|---|
| chinese_beijing | 1 | 17 | 64 | exact/exact |
| english_capital | 1 | 23 | 64 | exact/exact |
| math_arithmetic | 1 | 24 | 64 | exact/exact |
| code_python | 1 | 24 | 64 | exact/exact |
| long_context_recall | 1 | 977 | 64 | exact/exact |
| multi_turn_memory | 1 | 19 | 64 | exact/exact |
| multi_turn_memory | 2 | 107 | 64 | exact/exact |
| multi_turn_memory | 3 | 188 | 64 | exact/exact |

与旧`qwen35_27b_capacityfix_20261013_r1.jsonl`逐条核对，8/8输入IDs相同，7/8引擎输出IDs仍相同；唯一改变为code_python的0-based output index54起，当前序列与本次独立参考完整一致，旧321/1472分叉未复现。旧失败数据和HF cache路径诊断全部保留。

新旧还存在源码、配置及设备对差异，不是只切换Gamma一项的因果实验，不能将改善只归因某个补丁。本次没有为了选择参考token改采样。八个64-token样例不证明任意上下文/长度/语义质量普遍正确；三轮对话为全历史重发，不代替Session KV验收，也不重新证明4095+1上边界。

## 终态与剩余范围

11733编排正常退出0，自有942891服务退出0且不存在，结束后物理NPU1无进程；NPU0他人任务未干预，不声称全服务器空闲。23文件（源码/配置/原始IDs/参考/结果/日志/快照）跨端SHA全部一致。Owner/base-format及关闭提示保留，不隐去日志；没有性能或优化收益结论。

当前源码27B/TP2固定greedy回归通过，取代旧7/8作为此配置下的最新样例结果，但不覆盖原旧结果。4B当前8/8、9B当前7/8；0.8B全模型数值方向待用户确认，合同未变。四模型最终一致版本、Session/TP对照、Agent完整矩阵/优化消融、最终配置、报告及复验仍未完成。
