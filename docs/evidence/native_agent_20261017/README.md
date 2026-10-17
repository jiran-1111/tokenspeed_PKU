# R7原生Agent负结果与配对证据

4B/BF16/TP1/图模式，2 Session×16轮×8输出，32组输出与context配对通过，工具窗口综合判据失败。此目录不是完整Agent通过凭证，原失败保留。

- `source_manifest.json`：1403源文件，字节SHA `dc3142dba99acac6e874d82b55de58343da067dd0befa6c1658bbe620b68a394`。
- `audit.json`：独立核对历史追加/输出/metadata/实际长度，及2.00006638s窗口内零完成、零新请求、零KV采样；缺失关闭快照为null。
- `native_trace.jsonl`：32条完整输入、suffix和实际输出的精简选择。省去原始引擎Session/请求ID和无关metadata；输入输出数值逐项保留，字节哈希与原run.json不同，分别记录在`verification.json`。
- `raw_manifest.json`：19项已同步原始/分析数据SHA；原始日志及run.json保留在工作区，不公开私有服务器路径。

在对应源码快照、项目环境及固定revision本地模型下，从仓库根目录运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
python scripts/run_native_agent_validation.py \
  --config configs/baseline/qwen35_4b.yaml \
  --workload configs/validation/native_agent_16turn_20261017.json \
  --source-manifest docs/evidence/native_agent_20261017/source_manifest.json \
  --output-dir results/raw/reproduce_native_agent_4b \
  --engine-port 18780 --metrics-port 39982
```

入口逐项校验源码、映射逻辑2→物理NPU1/chip0、目标空闲及端口，并拒绝覆盖输出。实际原采集另有420s外置驱动，只中断经命令行确认的自有进程；结束后独立核对源及设备，失败不自动重跑。不要并行启动其他本任务NPU实验。

本次记录的程序退出1来自工具进展/采样判据，不能把32组配对通过改称整组成功，也不能把窗口内无请求完成解释成服务端工具状态或NPU空闲。完整条件与后续独立阶段设计见[实验记录](../../project/native_agent_4b_validation_20261017.md)。
