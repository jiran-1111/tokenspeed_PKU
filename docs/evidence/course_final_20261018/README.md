# 最终同源验证与TP候选证据

1415项完整源清单SHA `d05a47d932b519b6e50ed0c668725935d93b744fffffa68e95f37a0227f6c6ea`，原R9r2的1409项实现文件逐字节不变，只新增六配置。四个final配置逐字节等同各自已验证来源，两个TP候选各只改tp_size；`config_selection.json`记录来源、对应字节SHA和单项差异。

四模型final配置各32组原生/完整实际历史重算输出配对、32组context检查通过，累计128组输出和128组context；工具期间其他会话进展、活动块留存、恢复复用、idle持页与close后active归零、控制阶段边界分别检查。成功运行同时要求身份/完整配置/revision/9B头精度/固定工作负载SHA正确、source一致、退出0、无超时及目标chip释放，模型集合与总数从通过审计的实际结果派生。

- `verification.json`：四模型和各TP运行的分状态汇总、配置身份、范围、软件检查与原始归档SHA；所有数字来自实际审计，不用标签代替模型身份。
- `source_manifest.json`、`config_selection.json`：完整源码和六配置的选择与差异。
- 四个`final_*`目录：`native_agent_audit.json`保留真实历史/配对/工具/池计数/close，`phase_audit.json`保留原生与串行重算的阶段边界。
- `tp4b_tp2`与`tp4b_tp1_baseline`：各192正式/64预热真实请求全部逐ID匹配；CSV与audit给全部臂、P50/P95及观察HBM。不同TP为独立Engine、固定顺序，非随机独立重复，不作显著性或HCCL子阶段因果归因。
- `tp27b_tp1/failure.json`：固定TP1候选在启动缓存容量检查处拒绝（num_lcm_blocks=31 cannot admit one token with configured qwen_gdn cache scheduler limits），退出-10且没有run记录，未形成有效性能结果。不是权重OOM或完整质量向量失败的证明，也不表示所有TP1配置均不可能。没有改内存/调度参数或输入重试；最终27B仍用已验证TP2。
- `review_receipt.json`：只读独立复核发现并修复的四项审计/表述问题，九个新guard先失败后通过，复核范围仅这些修复，不替代真实NPU结果。
- `raw_manifest.json`：377项完整原始/分析文件逐项SHA，已跨端核对；完整私有归档SHA `76607b0cf61362a162dc507151a53c3daf064138135d693218e3980bf8776fd8`。保留所有失败、启动日志、退出记录、源后验与驱动源码，不公开私有服务器路径/帐号或不透明IDs。

所有成功运行均经本机独立复算，与服务器JSON语义一致；模板和统计不会根据验证结果改阈值、删样例或择优重跑。未有效完成的实验单独分类，不自动解释成资源/质量负结果。主性能768正式请求仍见[R9r2证据](../native_performance_20261018/README.md)，真实清单按来源标记；最终四配置设备验证在本1415源执行。

精确启动、软件、原生验证、TP和性能复现命令见[最终入口](../../project/final_delivery_20261018.md)。`runners/audit_native.py RUN_JSON OUTPUT_DIR`和`audit_phases.py RUN_JSON OUTPUT_JSON`为标准库独立审计，使用原始run而不是audit作为输入，检查报告中的通过布尔值与全部计数；不能只看分析进程退出0。主性能独立分析器为`../native_performance_20261018/runners/analyze.py`，支持本1415源的TP记录。原采集外置预算质量900秒、性能1800秒，只有核实属于本次的子进程才可发SIGINT，失败目录不覆盖。

实际实现、实验、分析与材料由AI工具辅助完成，三人责任见[总报告](../../report/course_project_final_20261018.md)。教学网用户本人提交；有限实验不代表未知课程隐藏负载或任意语义质量已验收。
