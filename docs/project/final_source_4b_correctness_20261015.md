# 发布运行时源码的 4B 固定样例复验

2026-10-15，沿用既有八条 greedy/max64 协议，对齐 HW1 已发布提交 `1d113e54f789a22b14cbbdccbc6230c3f4d0e574` 的运行时源码。1396文件清单SHA为 `f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e`，开始/结束逐项一致；仅更新一次性实验入口及端口，没有改生产源码。

Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、BF16/TP1，沿用baseline配置；prefix/KVStore/overlap/mixed batch关闭，chunk4096、block128、context4096、max_num_seqs8、max_total_tokens32768，graph启用。隔离HTTP端口18570/18571、metrics39611/39612。逻辑2对应物理NPU1/chip0；引擎启动前及引擎完全退出后启动HF参考前，分别实时检查映射与整卡无进程。全程与其他NPU实验串行，未影响其他设备上的任务。

## 结果

八条raw及stop-aligned均8/8匹配，参考为同revision的HF NPU BF16 `generate`、do_sample=false、num_beams1，使用引擎实际input IDs，未重新选择前缀。

| 样例 | turn | 实际输入tokens | 实际输出tokens |
|---|---:|---:|---:|
| chinese_beijing | 1 | 17 | 64 |
| english_capital | 1 | 23 | 64 |
| math_arithmetic | 1 | 24 | 64 |
| code_python | 1 | 24 | 64 |
| long_context_recall | 1 | 977 | 64 |
| multi_turn_memory | 1 | 19 | 64 |
| multi_turn_memory | 2 | 109 | 64 |
| multi_turn_memory | 3 | 190 | 64 |

与2026-10-14的4B记录逐条比对，8/8输入ID及8/8引擎输出ID均相同；不把跨日期结果相同解释为严格单变量因果实验。三轮对话是全历史重发，不是Session KV验收；64-token输出上限及八样例覆盖不保证任意长输入或语义质量。

## 终态与证据

编排退出0，服务退出0，HF参考退出0，目标设备最终无进程。17个原始ID/对照/配置/源码/日志/设备快照均已跨端逐文件SHA验证一致。原始目录 `results/raw/correctness/final_source_4b_20261015/`，校验清单 `results/raw/correctness/final_source_4b_20261015_sha256.json`；归档SHA为 `2fda81ba23edeae1bf3a3bb7b47c0f047dab81e87e56f314edda764c1745aaac`。Owner/base-format及退出日志完整保留，不宣称无警告退出。

本次填补4B与发布运行时源码的固定样例对齐，不替代性能、并发Session、TP2、工具等待或整项目最终验收。新资料尚未加入HW1已发布提交，后续统一交付单独处理。
