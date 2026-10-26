# 作业2新增实机与通信证据（2026-10-25）

本批补齐作业2性能、逐chip HBM、双rank设备时间线、大模型容量与通信粒度实验材料。[报告](../../report/homework2_communication_subsystem.md)按四项任务组织。**4B/9B短输入差异仍未解决，不能宣称四模型全部数值通过或完整大项目完成。** 本分支仍标记进行中，供继续正确性诊断。

## 数据入口与适用范围

- `shared_performance.json`：0.8B同卡TP1/TP2各96正式、27B TP2 96正式、两策略各24正式，共336正式请求；保留全部轮次、慢轮及主矩阵采样缺口。0.8B短输入HBM由另一张卡的独立延长补测取得，两TP各128正式，不回填原轮次。吞吐是三轮中位与范围，延迟是客户端观察；共享背景不等于独占加速比。
- `new_27b_sequence_comparison.json`：新TP2对同源前次TP1的96/96逐ID配对；前次TP1不作为新性能基线。
- `native_profile_summary.json`：三份有效双rank采集的六个正式窗口、锚、输入/导出SHA、计算/通信并集、交集、主机同步等待与未知分类。Profiler输出16，普通性能输出32，不混算。
- `additional_numerical_capacity.json`：HF11独立FP32参考、未采纳精度候选和27B容量成功/失败。旧4B/9B自由生成失败不被这些结果替代。
- `next_numerical_diagnosis.json`：现有逐层数据的CPU聚合，仅用于制定下一步诊断，不认定根因。
- `archive_receipts.json`：本地取得完整原始归档后的逐文件SHA核验；大型原始trace和含账户信息的日志保留在工作区，不随公开源码发布。`report_state.json`记录本批范围。

## 固定源、工具和历史失败

核心1531项运行源清单为[原清单](../homework2_20261023/source_manifest.json)，SHA `2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`，全部硬件实验前后未变。采集器和分析器是额外工具，不能冒充属于原1531项；身份见`tool_snapshots.json`。scheduler二进制身份和软件测试见[原软件记录](../homework2_20261023/README.md)。

`runners/measurement_r1`、`hbm_r4`、`profile_r4`及历史`analysis_r6`保存逐字节工具。为避免仓库格式化改写已测试历史源，Python副本采用`.py.txt`，恢复时只移除最后`.txt`；其manifest仍以原`.py`名称校验。最终`analysis_r7.zip`保存14项工具和清单，解压后逐项SHA核对；保留原始字节而不将其格式化为另一个“已测试”快照。ZIP不包含凭据或二进制可执行程序。

R6仅统计显式Hccl外层范围，图重放内核仍为unknown，故其结果保留但不作报告最终通信表。R7按原设备trace核对的三个精确名增加图重放分类；Hccl外层范围与内核重叠取并集，不重复计时。其他AI_CPU、CPU enqueue及无确认Stream元数据的活动不自动计为通信。未知事件原样保留，未覆盖区间不代表空闲。

原主批三份Profiler合并导出失败、显存不均衡导致TP2补测启动失败、早期容量清理失败和R4/R5分析准备失败均保留在归档。有效native文件由未改写CPU准备数据及同worker官方完整CANN时间线组成，双输入SHA/事件数/顺序与原文件不改写标志均记录；原失败文件没有被改写成成功。

## 恢复与CPU复算命令

以下从仓库根目录执行；使用已有工作区环境，不安装全局包。先从已保存归档恢复对应原始采集到新工作目录，再核对原始SHA。没有原始trace时只能检查公开标量，不能声称重跑了设备分析。

```bash
mkdir -p .tmp/hw2-reproduce/analysis
python -m zipfile -e docs/evidence/homework2_20261025/runners/analysis_r7.zip .tmp/hw2-reproduce/analysis
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('.tmp/hw2-reproduce/analysis')
m = json.loads((p / 'analysis_manifest.json').read_text())
assert all(hashlib.sha256((p / name).read_bytes()).hexdigest() == expected for name, expected in m.items())
print('analysis snapshot verified')
PY
export TOKENSPEED_SOURCE_SNAPSHOT="$PWD"
export PYTHONPATH="$PWD/.tmp/hw2-reproduce/analysis:$PWD"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
python -B -m unittest discover -s .tmp/hw2-reproduce/analysis -p 'test_*.py' -q
python -B .tmp/hw2-reproduce/analysis/analyze_shared_profiler_native_cached_20261025.py   --run-dir /path/to/restored_capture --source-root "$PWD"   --tools-dir /path/to/restored_capture_tools --clock-tolerance-ns 2000   --collective-pattern '(?i)(?:"opName"\s*:\s*"Hccl(?:Allreduce|Allgather|Broadcast)"|^(?:allreduceAicpuKernel|allgatherAicpuKernel|broadcastAicpuKernel)$)'   --output .tmp/hw2-reproduce/new-timeline.json
```

路径占位符须替换为实际恢复目录。分析器验证source、run/protocol、工具manifest、授权范围、采集控制确认、原始trace身份与时钟锚，不会用缺失材料补造有效结果。正式复现实验还需设备、同revision权重和匹配CANN/torch_npu环境；每轮重新查设备映射与占用。本组共享性能和Profiler经过明确授权，不把该授权扩展到停止其他服务。新硬件采集始终串行；CPU分析可独立并行。

## 同源FA7补充诊断

[FA7完整Prefill观察](../../project/homework2_fa7_diagnosis_20261025.md)已取得两模型48项实际参数分片核对、逐算子FP32参考及84向量结果保持性复验。该层被测attention与MLP SUM均与实际部分和的BF16舍入和逐位一致；原4B/9B自由输出失败仍未修复，cached step8 KV尚未覆盖。大型原始和分析归档收据、工具ZIP及公开标量见上述记录；作业2主报告保持已发布的四任务版本。

后续[9B cached step8实际KV与GDN状态观察](../../project/homework2_cached_step8_diagnosis_20261026.md)已补采完成，另保留30向量保持性、178项归档及CPU输入/历史交叉替换。它限制了该步KV写入和SSM单步更新错误的解释，仍未修复原输出差异。

## 后续多输入比较

2026-10-26另完成[四模型冻结语义请求](../../project/homework2_semantic_quality_20261026.md)，主报告第2.5节记录CPU/TP1/TP2比较、表达差异及代码截断。本页原性能、Profiler、原4B/9B失败及配置身份不变，新生成协议不回填本页统计。
