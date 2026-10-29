# 作业2软件与首轮实机证据

当前状态见[要求核对](../../project/homework2_requirements_audit_20261028.md)，报告见[四任务报告](../../report/homework2_communication_subsystem.md)。本目录是较早的软件与功能实机记录；旧JSON中的未完成标志为记录产生时的状态，不代表当前缺项。后续性能、通信、HBM和容量见[新证据](../homework2_20261025/README.md)。

`hardware_supplements_20261024.json`保留四模型TP1/TP2完整有限序列比较、双rank执行、设备映射、内存与原始失败。0.8B/27B为16/16序列一致，4B/9B为8/16；未将共享功能实验追认为性能实验。

## 已核验的范围

- `source_manifest.json`：1531个源码、配置与课程测试文件；清单SHA为 `2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。服务器前后逐项核对一致。
- `software_verification.json`：本地878通过/33跳过；服务器903通过/8跳过；新编译scheduler的259项Python测试全部通过。服务器课程测试使用真实Torch，并禁用NPU导入，不能替代NPU验证。
- `scheduler_build.json`、`runtime_dependencies.json`：按同一源码在工作区隔离编译的扩展，记录实际二进制SHA；未安装或替换共享环境的包。
- `source_normalization.json`：只规范两个新增Python文件的行结束符，scheduler源码和构建定义均不变；编译记录保留原快照身份，最终快照重新完成全部软件复验。
- `startup_imports.json`：父进程、spawn、forkserver都解析到当前快照，并加载相同新扩展。工作区启动钩子仅将旧scheduler editable加载器移至普通路径查找之后，不修改全局安装。
- `model_geometry.json`、`model_artifacts.json`：四模型配置字段、revision与既有本地模型文件SHA；不包含模型权重内容。驱动在每次实验前后检查文件身份未变化。
- `timeout_selftest.json`：两个纯CPU进程组清理测试通过，包含父进程退出后子进程暂存的情况，独立对照进程均未受影响。

首次部署比较、OpenSSL发现、测试临时目录与旧editable导入等准备失败均保留，分类见 `software_verification.json`。早期逐chip监控缺身份字段，不能用其历史证明连续占用；新监控改为带设备身份的整卡查询。没有将准备失败解释成模型数值失败，也没有删除失败记录后只留成功。

## 首轮历史队列（已暂停）与复验

当前报告采用的0.8B性能与分块策略实测入口见[20261025证据](../homework2_20261025/README.md)，不应拿历史4B策略候选替代。以下是首轮历史驱动与队列的复现说明，队列保持暂停。

`configs/homework2/queue.json`固定16项任务，包括四模型TP1/TP2、27B条件性95%预算候选、4096上下文边界、4B通信粒度策略和独立Profiler采集。配置、负载和源清单在运行前固定；不根据结果更换输入或修改阈值。候选、跳过和失败必须各自保留。

`runners/queue_driver.py.txt`是实际驱动的逐字节复现副本，`runners/sitecustomize.py.txt`是隔离启动钩子。两者遵循仓库[MIT许可](../../../LICENSE)。运行时需将钩子放到快照的 `.tmp/import_bootstrap/sitecustomize.py`，显式设置 `TOKENSPEED_SOURCE_SNAPSHOT`，并让该目录位于本次进程的PYTHONPATH前端；正常路径还须包含本快照的runtime、kernel、NPU kernel及scheduler Python目录。不能修改系统Python或其他任务的环境来替代隔离启动。

驱动只有在准备凭据与源码、队列、模型指纹、依赖及自身SHA匹配后才启用。它每30秒重新查映射与整卡进程，等待同卡两chip空闲；同一对照保持同一张物理卡。每项实验预算1800秒，仅允许清理自己创建并核验身份的进程组。新的模型实验始终串行，不停止未知任务。

原始 `run.json`不改写。父进程NPU上下文可能在Engine关闭后暂存，最终释放由外层驱动在父进程退出后查询，并以 `device_release.json`关联原始run SHA。`audit_run`和`compare_runs`独立重算记录、HBM与输出对照；这些是未签名的内部一致性检查，不构成防篡改签名。没有完整硬件证据时不能标记课程作业已完成。

完整原始准备日志保存在项目工作区的任务目录，不公开含服务器私有路径的原始日志。重新编译或更换依赖需要新的依赖身份与运行记录，不能将旧结果重标成新二进制结果。
