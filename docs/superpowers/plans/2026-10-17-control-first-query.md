# 首次控制查询实施计划

- [x] 读取真实挂起证据、控制客户端及event-loop设计，确认get/set入口缺少已有回复循环启动。
- [x] 添加getter/setter带超时异步回复测试，实际2项TimeoutError确认RED；最早Python路径命令错误未冒称RED。
- [x] 在两个入口各补一行现有启动调用，专项2 passed（0.07s）。
- [x] 完整course回归：初次遗漏basetemp父目录的143个fixture setup错误保留；创建工作树内.tmp后554 passed/34 skipped，服务器588 passed/零skip。
- [x] 完整pre-commit round2通过后固定1399项新清单，SHA `8cd3fd2bde06ff94d10e7d4f862d8e8121e0232559a2c1b2d804aaae2c17d729`；服务器两文件补丁已备份/部署，全部源匹配。原dirty主树该文件仅CRLF差异经核实后先备份再同步，未覆盖实质用户改动。
- [x] 真实NPU图模式首次生成前两次查询、16-token生成及之后查询通过，外置240s超时未触发，退出0且目标chip空闲，前后1399源一致；14项证据跨端SHA核对通过。
- [ ] 更新证据与发布状态，并继续原生多轮/并发工具等待项目任务。
