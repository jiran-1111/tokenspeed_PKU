# 当前4B E0测量终态审计

本记录是原始证据与退出问题审计，不是完整统计报告、正常清理验收或整个项目完成声明。

## 测量证据

原测量目录：[raw](../../results/raw/benchmark/current_4b_e0_20261014/)。使用已部署4B/TP1/BF16、logical2（physical NPU1/chip0）、FP32 SSM，prefix/KVStore/mixed/overlap关闭，无Profiler或诊断。固定revision和完整参数见原[protocol](../../results/raw/benchmark/current_4b_e0_20261014/protocol.json)。本次不覆盖远端生产源；远端尚缺本机三文件的LCM计数更新，不能冒称本机最新全树验收。

原[result.json](../../results/raw/benchmark/current_4b_e0_20261014/result.json)记录benchmark_exit=0、service_exit=0；480请求、15 summaries、30份physical NPU1快照，各并发1/4/8/16/32三轮、每轮32请求。全部实际输入479、输出128，无短输出。目标输入512不等于服务端实际输入，不推断未保留的finish_reason。

15个选定源/配置/helper结束后hash未变；这不是完整二进制、全部权重或整个仓库的发布证明。限速同步1814正常退出0，56本地/远端文件逐一SHA-256一致，没有额外或缺失文件。独立跨端清单保存在[项目临时目录](../../.tmp/current_4b_e0_crosshash_20261014.json)，未往raw目录追加或修改数据。

## 退出失败及直接证据

原包装流程status=failed，cleanup_errors=[OwnedGroupRequiredSIGKILL]；该失败保持原样，不能把测量程序退出0改述为整个编排正常完成。

本机SSH句柄22235先因Connection reset退出1，重新连接时远端orchestrator966173、worker967135和service966428实际存活且数据持续增长，故未重启实验。观察器85059/99086/52485仅只读核验，不是新的测量进程。

480/15/30收齐后worker及service主进程消失，但自有service PGID/SID966428中剩余966686，ps状态Z。随后/proc/966686/status明确State=Z (zombie)、PPid=1；清理循环只检查进程组存在性，不区分活进程与已退出的僵尸，因此等待60秒后进入SIGKILL分支并记录失败。僵尸不能靠再次signal完成回收；本轮没有额外手动signal、重启或干预PID1/他人进程。

最终/proc/966173/status和/proc/966172/status也为Z、PPid=1（python编排和原sshd）。先前kill(pid,0)的true只证明PID存在，在这里不再代表仍在执行。worker/service主进程已不存在；fresh npu-smi证明physical NPU1无运行进程。其它用户NPU0任务不在清理范围，不能据此宣称全服务器空闲。

直接证据支持“僵尸被当作未退出进程组触发清理失败”；PID1为何未回收以及SSH断开是否导致子进程收养，尚未完整调查，不宣称已确定。后续修复必须有生命周期回归与适当的存活判断，不能重写原result或靠删除失败目录掩盖问题。

## 下一步与准入边界

- 对完整480/15记录再核对唯一键、cell、actual usage、计时/指标和snapshot身份，才写独立统计分析；测量数据完整性与退出清理结果分别列示。
- 以每并发档三轮为单位报告生成吞吐量及客户端TTFT的均值/SD/范围、逐轮值；固定档位顺序和共享主机，不声称显著因果收益，不以旧prefix-on异版本结果为同控对照。
- 排查并修复一次性编排退出判断，保留此次失败。不得为获得complete标签无依据重跑全部480请求；是否需要重测须由数据有效性和修复影响决定。
- 0.8B校准B第三节仍待确认；原1%失败、模块参考覆盖欠项、阶段热点/最终报告/复验与四模型Agent项目继续，不将本次性能采集当作数值验收。
