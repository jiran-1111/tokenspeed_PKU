# 发布运行时源码的 27B TP2 固定样例复验

2026-10-15，Qwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654`、BF16/TP2，采用HW1已发布提交 `1d113e54f789a22b14cbbdccbc6230c3f4d0e574` 的同一运行时源码。1396文件清单SHA为 `f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e`，采集前后逐项一致。

引擎使用逻辑2/3，即物理NPU1的chip0/1，启动前确认映射和整卡无进程；原baseline配置不变，仅隔离HTTP端口18580/18581、metrics39621/39622。prefix/KVStore/overlap/mixed batch关闭，graph启用；chunk4096、block128、context4096、max_num_seqs4、max_total_tokens16384。服务完全退出后，重新检查设备空闲，仅在逻辑2运行独立HF NPU BF16 greedy参考，不与引擎或其他本任务NPU实验重叠。

## 结果与终态

固定八条、max_new_tokens64，raw及stop-aligned均8/8匹配。actual输入长度依次17、23、24、24、977、19、107、188，输出均64 tokens。参考使用每条引擎实际input IDs、do_sample=false、num_beams1；三轮对话为全历史重发，不是Session KV。对2026-10-14的记录逐条核对，8/8输入及8/8引擎输出ID完全相同；不从跨日期一致性推导特定补丁的因果作用。

编排、引擎服务和独立参考均退出0，目标整卡释放，其他设备任务未干预。17项原始记录/协议/源码/日志/设备快照已同步并逐项SHA一致；目录 `results/raw/correctness/final_source_27b_20261015/`。与9B完整前沿诊断一起归档为 `course_followup_20261015_r1.tar.gz`，归档SHA `df3cba6ae8cc17a036350328f4be102b3e1307e6124c2cb788d51159a70285be`，其中全部32文件本机复核无差异。

结果填补27B与发布运行时的固定样例对齐，不证明任意长度、语义质量、原生Session、TP1/TP2性能收益或完整Agent验收。新资料尚未加入HW1发布提交，后续总项目交付单独处理。
