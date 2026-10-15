# 最终源码 Decode Profiler（2026-10-15）

Stage B最终源R3的272向量全部通过后，在fresh idle/mapping检查确认逻辑2→physical1/chip0后串行采集，NPU0既有任务未操作。

## 成功采集

- raw：`results/raw/homework1_decode_profile_final_20261015_r2/`，meta同名`_meta/`；分析：`results/analysis/homework1_decode_profile_final_20261015_r2/`。
- 同R3源码清单，native128输入、BF16/TP1；三个8-token预热；单请求256输出，首输出后start、累计128输出时stop，随后完成256输出。
- 退出0，started/stopped=true、cleanup_stop=false；设备释放。实际profiler_info确认Level1与ACL_AICORE_PIPE_UTILIZATION，而非只看声明。
- CANN原目录保留，并完整复制到本次`profile/cann`；167份raw文件建立SHA。最终归档共809份原始/分析文件同步到本机后逐项SHA一致。
- 762次full attention/6层=127步，2286次fused GDN Decode=18×127；没有GDN chunk Prefill，范围与控制窗口相符。

## 统计与解释

全部op_summary108331行，累计Task Duration981676.691µs。卷积状态更新36.944%、MatMulV2 21.107%、ViewCopy13.334%，随后Mul3.636%、fused GDN Decode3.223%、Cast2.840%。分母是记录任务累计时长，不是端到端互斥墙钟占比。

AICore-only计数主要只有MatMul和FIA，无法覆盖卷积等自定义kernel；其中MatMul92.15%不能外推为完整Decode92%。Task timeline114231个正区间，envelope1531812µs、union996448.75µs、uncovered535363.25µs、最大gap1405.75µs。未覆盖不自动等于空闲/Host耗时；流水计数互有重叠，HBM带宽原始单位未校准，不虚构带宽利用率。

本批证据支持后续优先分析卷积状态更新、布局拷贝和矩阵权重读取/小算子发射，不足以给出全系统因果分解。Profiler请求耗时及导出暂停不进入无Profiler的性能表。

## 首次失败保留

首次`homework1_decode_profile_final_20261015/`产出trace但控制脚本把原生增量流当累计流，导致窗口未于128输出停止，最终检查失败、退出1。实际256个块各1token、completion_tokens256；cleanup触发stop，原record/trace/log完整保留，不当作成功窗口。

基于原生stream实现修正测试，真实RED 1failed/2passed；最小改为累计IDs，GREEN3passed。只修改Profiler脚本及测试，不改生产模型/engine/kernel。为保证最终上传同源，又对R3重采全部Stage B：272原始PT与R2逐项SHA一致，冻结门槛及NumPy独立公式复算全过，再进行本次成功采集。

公开标量、原始数据SHA及复现命令见[交付证据](../evidence/hw1_20261015/README.md)。
