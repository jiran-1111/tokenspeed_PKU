# R6首次控制查询证据

本轮仅修复控制客户端getter/setter发送前遗漏回复循环启动的问题。1399文件清单SHA `8cd3fd2bde06ff94d10e7d4f862d8e8121e0232559a2c1b2d804aaae2c17d729`，服务器588 passed/零skip，本机554 passed/34skip。两项入口测试先超时失败，再通过；真实0.8B/TP1/BF16/图模式首次generate前连续两次查询、16-token生成及之后查询均通过，退出0、未触发240s超时，进程退出后目标chip空闲。

`source_manifest.json`约束源码版本，`npu_result.json`保存有限存活性结果，`verification.json`给出验证范围，`raw_manifest.json`记录全部14项已下载原始证据SHA。原始日志和XML保留于工作区，不公开私有服务器路径；归档SHA `963c010338bb3ac010def598785662b61837d4f24597c8de156924bbf06d629e`。

从对应源码快照的仓库根目录，在项目环境和固定revision本地模型已配置后运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
python docs/evidence/control_first_query_20261017/runners/run_npu_query.py
```

脚本实时检查逻辑2→物理NPU1/chip0映射、目标无进程及端口空闲；与其他NPU实验串行。拒绝覆盖输出目录，执行前后逐项核对清单，超时只中断经命令行确认的本次自有调用进程。请勿用其他源码的通过结果替代此次检查。

公开脚本从实际驱动整理，仅调整证据/输出路径、补许可证并格式化；原始实际脚本哈希另存，不称二者字节一致。getter当前仍返回空内部状态，setter仍updated=False；本轮不新增KV观测、配置修改或WAITING_TOOL能力，也不是新的正确性或性能基线。HW1及R5证据仍分别属于其发布快照，复现旧证据须切到相应版本。
