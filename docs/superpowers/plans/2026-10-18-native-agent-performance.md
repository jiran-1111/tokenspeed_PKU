# Native Agent独立性能实施计划

> 执行采用executing-plans，延续当前隔离工作树；不分派新代理。

**目标：**四模型固定R8轨迹上独立比较Session ON/OFF，质量、延迟、吞吐和逐chip HBM均可追溯。

**结构：**新纯函数模块`benchmarks/course_project/native_performance.py`负责轨迹合法性、反平衡次序和HBM解析；`scripts/run_native_agent_performance.py`负责并发重放、采样/生命周期、固定源码与安全启动；`configs/validation/native_agent_performance_20261018.json`固定轨迹SHA及协议。Python/asyncio、既有Engine/stream/Prometheus，无新依赖。

- [x] 新增`test/course_project/test_native_agent_performance.py`：两臂输入路径/实际输出/context检查，坏轨迹/选择器拒绝，分叉drain/关闭，HBM双chip与进程行排除，采样shutdown顺序；先14项RED，核心GREEN后新增采样/输出guard的4项RED。
- [x] 最小实现上述三个源与协议，原1405源逐项不变；专项18项GREEN。
- [x] 完整格式round2通过、本机600通过/34无Torch跳过；冻结1409项源SHA 8c0914167f44d2142c01df6d83ec961b6d7f8e5706ce7a37e8b0aa31c10b888a，服务器逐项匹配、完整634通过/零skip。
- [x] 独立只读审查发现坏chunk剩余流/部分创建清理两P2，新3项RED复现后修复并21项GREEN，复核无新阻塞（仅修复范围）。已成功正常路径r1数据保留，原协议不变。
- [x] r1任务结束后固定r2源1409项SHA `2c9e47616eb3c460a740e2542e776ab655934a9ad7ed0ac0c1401a008a97d712`，备份部署后服务器637通过/零skip、本机603通过/34 skip，完整pre-commit round4通过；r1独立保留。
- [x] r2四模型硬件复采全部通过192正式/64预热，1409源后验零差异，退出后所需chip空闲。
- [x] 先4B、再0.8B/9B/27B，逐模型串行：双方一次完整预热、三轮6臂，原始失败保留、不按指标补跑。
- [x] 四模型503项原始/分析证据SHA取回，独立本机/服务器复算一致；三轮描述性报告与TTFT图已生成，图已视觉核验，不作显著性推断。
- [ ] 按allowlist完成同源服务器/Git交付发布；继续TP、完整消融、最终配置/总报告答辩。
