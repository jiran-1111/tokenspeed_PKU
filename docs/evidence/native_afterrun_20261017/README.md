# R8原生Agent验证证据

四模型各2 Session×16轮×8输出，累计128组原始输出配对、128组current-frontier上下文检查通过。工具为脚本受控2秒等待与求和，不是模型自主选择工具，也未新增服务端WAITING_TOOL状态。

`source_manifest.json`包含1405项源，字节SHA为`e4a19a33a96322a19617273ce43e84695c68813c6e984953bfd4e119dfa852aa`。`raw_manifest.json`索引工作区保留的原始日志、运行记录及分析数据，不公开私有服务器路径。`verification.json`分别记录原始run.json与公开精简轨迹的哈希；两者字节不同。

各模型子目录提供独立历史/配对审计、阶段审计和32条`native_trace.jsonl`。精简轨迹保留全部实际输入/suffix/output IDs，省去不透明的引擎Session/请求ID。没有用参考token替代真实生成。

额外`4b_concurrent/comparison.json`保留同R8代码并发控制点：32组质量与窗口判据也通过，窗口另一会话1次完成、19个正活动块样本；其32组输入/输出与已公开4B轨迹及旧R7完全相同。这个单点没有反平衡预热/次序，不能把旧失败只归因于并发控制。五次运行共90项原始/分析证据已逐项SHA一致，归档SHA `8687e69c5456459b5c82e1c6cd68f19fd2e405f3b34e15a9b91b517c134f2b2d`。

在本交付源码、项目环境与固定revision本地权重下，从仓库根目录运行4B示例：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
python scripts/run_native_agent_validation.py \
  --config configs/baseline/qwen35_4b.yaml \
  --workload configs/validation/native_agent_16turn_afterrun_20261017.json \
  --source-manifest docs/evidence/native_afterrun_20261017/source_manifest.json \
  --output-dir results/raw/reproduce_native_afterrun_4b \
  --engine-port 18780 --metrics-port 39982
```

0.8B使用`configs/baseline/qwen35_0.8b.yaml`；9B使用显式`configs/candidates/qwen35_9b_fp32_head.yaml`；27B使用`configs/baseline/qwen35_27b.yaml`并设置`ASCEND_RT_VISIBLE_DEVICES=2,3`。每次更换独立且不存在的输出目录，所有硬件实验串行；入口检查逻辑2/3→物理NPU1两个chip的实时映射、目标空闲和端口，拒绝覆盖。原采集外置900秒预算，保存失败与源后验，不按指标选择重跑。

新协议必填`control_schedule`和`tool_metrics_interval_seconds`，没有默认回退。保留的`native_agent_16turn_concurrent_observed_20261017.json`显式并发质量控制，原R7负记录需在R7对应源码快照重放；不能把新协议通过改写成旧实验通过。

idle时active>0、关闭后active=0表示池内活动引用留存/解除，cached或allocated仍可非零；不是释放预分配HBM给系统。TP指标按端点既有series汇总解释，不当成逐chip HBM。工具采样按实际时间戳判断，不能把采样次数当成独立统计样本。

本组验证不是独立OFF/ON性能比较，也不是完整项目验收。将质量重算移出执行窗口改变了竞争负载，不能据阶段时长宣称系统加速。完整条件、负结果、后续工作见[验证记录](../../project/native_agent_four_models_20261017.md)；实际实现与实验由AI工具辅助完成。
