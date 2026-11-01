# 当前原生Agent同源验证

本轮于2026-10-31开始接续，跨日执行以原始记录的实际UTC时间为准。此目录与20261018历史证据分别保存，不将旧测量改称本轮结果。

最终配置保持0.8B TP1、4B TP1、9B FP32输出头 TP1、27B TP2。原生质量负载继续使用两会话、16轮、每轮8个输出token及脚本工具等待；性能继续使用原固定参考IDs、完整预热和三轮交替Session ON/OFF。它们验证有限轨迹，不代表未知统一评测或任意输入语义质量已通过。

## 已完成的软件和身份核查

- `software_initial.json`：整合提交528ebba7的1532项源码，在新服务器副本上903通过、8项Windows专用测试跳过；源码前后全部匹配。首次副本目录命名为baseline导致报告分类测试4项失败，更正本次副本目录名后完整重跑通过，原日志和哈希保留，未修改被测源码或断言。
- `model_identity.json`、`model_files.json`：四模型55项文件（含18个权重分片）的完整SHA检查与固定revision，未分发权重。
- 当前设备入口与共享守卫已通过规格和代码质量审查。守卫轮询观察实时设备、背景PID及UID/starttime，只能中断自身采集父进程；它不能预留其他用户的设备。

当前完整源码和工具在实验前分别固定清单。CPU检查、NPU运行和独立分析按实际记录分类，不能互相替代。新的硬件结论只在对应运行、退出释放和独立审计均有效时成立。

## 当前入口

两份原生采集入口均要求显式提供`--npu-id`、`--chip-logic-ids`和`--resource-mode idle|shared`；共享模式还要求`--min-free-mb 8192`或更大的明确安全余量。8192MB是设备运行保护量，不是课程正确性门槛。`--chip-logic-ids`使用`物理chip:逻辑device`配对，TP1也允许选择chip1；实际编号必须取自本次`npu-smi info -m`，不能照抄旧编号。

`TOKENSPEED_HW2_WORKSPACE`显式指向所有本组实验共同使用的服务器工作区根目录，用于沿用同一个`.tmp/hw2-npu.lock`。历史队列保持暂停。激活已核验的项目环境后，把当前源码、kernel与隔离scheduler目录放在PYTHONPATH中；运行前核对真实导入来源、扩展哈希、模型指纹、端口及实时占用。

采集程序为`scripts/run_native_agent_validation.py`和`scripts/run_native_agent_performance.py`；其他必填参数和原固定负载保持原接口。每次输出必须使用不存在的新目录，失败目录不能覆盖。

外层串行工具见`runners/controller.py`与`runners/jobs.json`。将工具复制到新的任务目录运行，使用其中的显式参数提供当前源码、已验证项目Python、环境激活脚本、设备选择、当前清单及本机重新准备的模型状态和软件收据。不得复用旧机器的PID、inode、ready或模型stat文件。外层驱动先核对软件与来源收据，再逐项检查资源、运行质量/性能、验证自身进程退出，并立即调用独立分析；质量失败时对应性能项不会运行。超时清理只允许处理经UID、启动时间及会话/进程组核验属于本次的进程，两个纯CPU清理用例见`test_owned_processes.py`。

## 独立重算

当前分析器由历史统计工具选择性适配，MIT许可和来源说明保留；旧文件原样保存。分析器显式接收当前源根和完整源清单，重新解析原始映射、进程和HBM表，核对运行与外层`post_exit_verification.json`的SHA关联。共享场景要求本实验进程释放，不能要求原有服务全部退出。

```bash
python docs/evidence/project_agent_20261031/runners/audit_native.py \
  RUN_JSON NEW_ANALYSIS_DIR --source-root "$PWD" \
  --source-manifest docs/evidence/project_agent_20261031/source_manifest.json

python docs/evidence/project_agent_20261031/runners/analyze.py \
  RUN_JSON ORIGINAL_TRACE_JSONL NEW_ANALYSIS_DIR --source-root "$PWD" \
  --source-manifest docs/evidence/project_agent_20261031/source_manifest.json
```

质量入口包含阶段审计；任一必要判据失败均退出非0。性能保持原P50/P95、逐臂配对、包含工具等待的吞吐及观察HBM口径。原始npu表没有进程祖先链、UID或模型文件内容；这些只能作为外层已保存的观察身份检查，不冒称离线可重新获取的事实。

共享结果只说明记录条件下的观察性能，不推导独占加速保证。课程统一负载、规定Kunpeng资源预算和评测细则尚未取得，继续单独标注尚未验证。
