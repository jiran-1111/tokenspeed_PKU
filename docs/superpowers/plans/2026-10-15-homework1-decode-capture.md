# 作业1诊断快照可复现性修复 Implementation Plan

> **For agentic workers:** Use subagent-driven-development。用户已批准具体设计；dirty main原地授权不变，不commit/push。

**Goal:** 全course不再依赖临时文件，并真实记录不受更新污染的GDN执行前状态。

**Architecture:** 正式化已有可选sitecustomize hook，仅启用时复制真实输入页并接入原raw输出；不改变生产计算。一个实现子任务，随后两阶段独立审查、根代理服务器验证。

**Tech Stack:** 标准库、现有pytest；真实Torch验证用已有服务器环境，不新增依赖。

## Task 1：正式脚本与回归

- [ ] 读取设计、旧356行hook、现有测试及实际Mamba backend调用签名。正式文件`scripts/diagnostics/sitecustomize.py`，测试修改`test/course_project/test_correctness_decode_diagnostic.py`；保留目录组织与旧raw。完整MIT头。
- [ ] 将测试导入目标指向正式路径，保留alias测试，并新增wrapper原地更新/输入输出不同页/未启用无副作用检查。运行专项确认预期RED，不把缺依赖异常误称数值失败。
- [ ] 正式化现有hook；最小helper行为为`return pages[page].detach().clone()`，必须由实际before捕获路径使用。before捕获发生在`original(...)`前，明确输入页；实际raw输出包含before状态。保留旧post-state字段/guard/日志与算法；不扩大为通用Profiler框架。
- [ ] `../.conda-env/python.exe -m pytest test/course_project/test_correctness_decode_diagnostic.py -q`专项GREEN，然后全`test/course_project`用新项目内basetemp；报告真实fail/skip。
- [ ] SPEC先、quality后；未解决重要问题不部署。根代理审查完整新增源、SHA并独立重跑。

## Task 2：服务器对应源码验证

- [ ] fresh目标与旧测试SHA检查；已有目标不同则备份到项目新raw而非直接覆盖。部署审核版本，核对SHA。
- [ ] 服务器source项目环境、CPU-only专项/全course软件测试，新XML和basetemp；不与Profiler/性能同时采集。
- [ ] 同步XML与源码身份，核对跨端SHA，更新状态/报告。此次是诊断工具回归，不冒称完整模型通过；Git发布继续等待最终整体验收。
