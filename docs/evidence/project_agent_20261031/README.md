# 当前原生Agent同源验证

本轮于2026-10-31开始接续，跨日执行以原始记录的实际UTC时间为准。此目录与20261018历史证据分别保存，不将旧测量改称本轮结果。

最终配置保持0.8B TP1、4B TP1、9B FP32输出头 TP1、27B TP2。原生质量负载继续使用两会话、16轮、每轮8个输出token及脚本工具等待；性能继续使用原固定参考IDs、完整预热和三轮交替Session ON/OFF。它们验证有限轨迹，不代表未知统一评测或任意输入语义质量已通过。

## 当前已完成的同源实测

被测实现固定`1d6099da5909e38fdbc737fe77bcb4176c978de1`，1535项清单SHA `633f4b745e540048325b997dc8357ab8383380dcf70cfe8a1052f6d3865f0d23`。完整结果见[验收汇总](final_verification.json)、[整轮最终核查](server_final.json)和[当期报告](../../report/course_project_final_20261101.md)。四质量和四性能运行均退出0/无超时，独立设备/阶段/释放审计通过；1710原始运行文件跨端SHA匹配，本地分析与服务器JSON语义相同。

| 模型 | 原生输出/context各通过 | 正式/预热请求 | 三轮TTFT P50下降 | 三轮吞吐比 |
|---|---:|---:|---:|---:|
| 0.8B TP1 | 32/32 | 192/64 | 87.99%–89.15% | 3.51–4.39× |
| 4B TP1 | 32/32 | 192/64 | 94.04%–94.61% | 4.51–5.66× |
| 9B FP32头 TP1 | 32/32 | 192/64 | 93.43%–94.00% | 4.20–4.25× |
| 27B TP2 | 32/32 | 192/64 | 89.24%–89.79% | 3.84–4.76× |

全部为共享观察，非独占收益保证。每模型三轮同Engine，不是独立进程样本，不报告CI或显著性。TTFT/E2E和近似TPOT完整P50/P95见[24正式臂CSV](performance_arms.csv)，配对见[摘要CSV](paired_summary.csv)，逐请求数据在各`performance_*/requests.csv`。共享HBM见[资源CSV](resource_summary.csv)：包括背景用量、非瞬时峰/模型净占用，单臂无完整覆盖即null。

每个`quality_*`和`performance_*`目录公开脱敏`summary.json`与`raw_manifest.json`，后者关联本地保留的原始tar SHA及每文件SHA；原始含私有路径/设备PID/不透明请求ID的日志不发布。离线审计可重算历史、配对、阶段和设备原表；PID祖先链/UID/starttime和模型stat只能核查当时收据及哈希，不能冒称可从npu表重新获取。

`software_current.json`记录当前999课程测试，本地966/33、服务器991/8；[逐项跳过](software_skips.json)列明原因与JUnit SHA。`startup_imports.json`核对parent/spawn/forkserver来自当前源码和同一隔离扩展；`environment_current.json`保留实际主机与运行Python，不能把系统探针Python3.11.6混为运行Python3.11.16，也不能把可见cgroup当课程规定预算。

## 已完成的软件和身份核查

- `software_initial.json`：整合提交528ebba7的1532项源码，在新服务器副本上903通过、8项Windows专用测试跳过；源码前后全部匹配。首次副本目录命名为baseline导致报告分类测试4项失败，更正本次副本目录名后完整重跑通过，原日志和哈希保留，未修改被测源码或断言。
- `model_identity.json`、`model_files.json`：四模型55项文件（含18个权重分片）的完整SHA检查与固定revision，未分发权重。
- 当前设备入口与共享守卫已通过规格和代码质量审查。守卫轮询观察实时设备、背景PID及UID/starttime，只能中断自身采集父进程；它不能预留其他用户的设备。

当前完整源码和工具在实验前分别固定清单。CPU检查、NPU运行和独立分析按实际记录分类，不能互相替代。新的硬件结论只在对应运行、退出释放和独立审计均有效时成立。

测量实际8工具为`tool_manifest.json`；后补`prepare_reproduction.py`、其5项CPU回归和实际bootstrap快照另见`reproduction_manifest.json`，不混称测量前工具。完整新机器准备及运行命令见[当前复现入口](../../project/final_delivery_20261101.md)。历史模型全SHA检查和各次模型stat核查分别记录；后补复现检查重新计算模型SHA不追认成测量前的另一次全权重检查。

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
