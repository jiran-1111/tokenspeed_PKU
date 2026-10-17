# 原生Agent有限验证计划

- [x] 读取Session生命周期、HoldsPages/ActiveLcmBlocks及HTTP metrics服务接线，限定SessionIdle语义。
- [x] 固定16轮双Session/8输出/单次2s工具workload，SHA `26cd9cf606589da095f064d9985ccf30a35a7014c711c73b09398a6aeb15a794`；新入口缺失RED明确记录，helper15项及完整入口/拒绝/清理测试通过。
- [x] 最小append-only构造与原生并发验证入口，复用既有流计时、context验证和指标解析，不新增调度路径。
- [x] 本机完整575 passed/34 skipped，服务器609 passed/零skip；完整pre-commit round6通过。R7/r4清单1403项SHA `dc3142dba99acac6e874d82b55de58343da067dd0befa6c1658bbe620b68a394`，前后源匹配。fake时序及Windows15.625ms边界失败保留；真实指标tuple键落盘错误先RED再修复六计数记录，21专项通过；workload和判据未改。
- [x] 4B真实图模式已采集并独立审计：32对输出/context均匹配，max实际上下文968；19项证据SHA同步。最早26项采集失败证据保留。
- [ ] 原并发控制点整体失败：2s工具窗口无另一Session开始/完成、无窗口内KV采样。保留负结果，分离采样及真实执行/之后重算阶段，不能改称原点通过。
- [ ] 通过后串行扩展四模型并归档/报告；独立OFF/ON性能及TP/消融继续单独实施。
