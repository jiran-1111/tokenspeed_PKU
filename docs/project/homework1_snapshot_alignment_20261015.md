# 作业1待交付源码对齐与真实启动复验

## 对齐范围与软件测试

先核对本机32个course测试与远端27个：远端缺5个、3个版本不同。进一步核对23个依赖，发现6源版本不同、startup_trace缺失，其余辅助源及8份模型配置相同。只读审查确认增量为KV block观测、启动日志和离线报告，不直接改变模型数学计算或调度决策；相关必填参数及新import成组部署，旧源/旧测试先备份。

成组15个既有文件部署后服务器547passed/1skipped；唯一skip查明是symlink测试漏建models父目录、将ENOENT误称权限/平台不支持。修复测试准备及异常筛选，保留原越界拒绝断言，并新增ENOENT不得误skip回归。SPEC与quality均通过；Linux两项实际通过，不拿Windows权限skip替代安全检查。

最终服务器course：**549passed、0failure/error/skip，17.746秒**。32个测试及7个配套源共39文件，本机与服务器最终SHA全部一致。此为明确范围的软件回归，不等于仓库全部源或所有NPU模型完成验收。

原始目录：`results/raw/homework1_snapshot_alignment_20261015/`。

| 产物 | SHA256 |
|---|---|
| remote_before.tar（部署前9文件备份） | 0871c5c6e665c324b36b6299e4f2bd7e577e81b4d4a5bb9f395231e69301f1c4 |
| course_final.xml | 3ea383496e8fc50c697a3eb2f195d1c452f77f3e2664ee8b39fb79febaeff5c8 |
| symlink.xml | d3716bbff1ab71334bca77b470231ac740d3b07553b089f71c920260c98fb345 |
| source_final.sha256 | c95843146bf44f7d5bfed2a63961a0bf6baecd16dcf24e2acf12b112473ba37d |

表内产物均同步后本机重哈希匹配。一次子代理测试未设置项目basetemp，曾在系统temp生成目录；能确认仍存在的3个本次目录已用原生移动收回项目`.tmp/recovered_pytest_20261015/`，未删除文件，另1个在核查前已不存在，不冒称已收回。后续测试显式使用项目环境和项目内basetemp。

## 对齐版本真实0.8B Session复验

在fresh空闲的physical1/chip0（logical2）单独运行已有`test_homework1_session_npu.py`，0.8B/TP1/BF16、baseline固定配置、prefix cache关闭、current-frontier。两个指标端口先检查可绑定；测试与本机SHA一致，没有执行新校准验证组或Profiler。

实际1passed、0failure/error/skip，29.684秒。每轮4组Session/control共两轮：8组输出及context metadata一致，4次关闭后拒绝，重开两轮的4组输入/输出一致；两个round的session IDs互不复用。包括真实模型启动与连续推理，不从AST测试推断硬件启动通过。

测试进程退出0，结束后fresh npu-smi确认physical1无进程，NPU0他人任务未操作。10个选定配置/测试/模型/Norm/观测源结束后SHA与本机一致。原始目录`results/raw/homework1_aligned_session_20261015/`。

- session.xml SHA256：`d139fde9f6b5d256084ece5fcdfc6865172caec8be2b0531c32a0b1d20526eb8`。
- run.log SHA256：`c18fb67200ac4e4933c6b80ee81d296b583c7bd424a969d85a1102647d5ff8c7`。

两文件同步后哈希相同。保留CANN owner及HF弃用警告；进程结束阶段另有3条pending Task销毁及已关闭日志流错误。行为测试通过和设备释放不消除这些清理日志问题，不能写“无警告干净退出”，也未定位其根因或通过修改测试掩盖。

## 仍未完成

独立参考数值可信性/候选判据及隔离引擎验证、最终Decode Profiler、对应最终版本性能与报告冻结、完整发布allowlist及格式化后服务器复验仍未完成。没有新Git提交、推送或PR。
