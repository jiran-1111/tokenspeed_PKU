# HW1之后的9B精度与Session交付证据

HW1已发布提交为 `1d113e54f789a22b14cbbdccbc6230c3f4d0e574`。本目录对应后续候选R5：1398文件，`source_manifest.json` 字节SHA-256为 `ef7b3e028233517746147f36da92efea679a29ba50471df4574e3cac9c103a27`；不替换HW1的原快照或冻结判据。

- `formal_correctness.json`：正式图模式、无诊断hook、原八条生成raw/stop-aligned均8/8。
- `session_summary.json`：四模型同R5原生Session的有限配对、上下文、关闭重开结果；9B用显式候选、其他用baseline，即时退出标志与进程完全退出后的设备确认分开记录。
- `uniform_regression.json`、`uniform_raw_manifest.json`：4B/27B同R5原八条各8/8及41项原始证据哈希，2026-10-17取回逐项核对；对照沿用此前采集的HF参考，没有冒称重新采集。
- `frontier_full_vector_cpu_comparison.json`：原9B已知前沿的完整logits描述量，没有给9B新设数值门槛。
- `verification.json`：软件测试计数、原始/公开选择数据的不同字节哈希、归档同步核对信息。原始输入ID、采样配置及参考输出ID均保持；公开选择文件去掉了无关历史输出和计时字段，不能称与原捕获文件字节相同。
- `formal_raw_manifest.json`：已同步121项原始数据/日志/脚本的SHA，包括首轮HBM采样失败；原始日志不公开私有服务器路径。
- `cost_summary.json`、`cost_raw_manifest.json`：6个有效成本臂、96请求/252 HBM采样及全部92项成本原始证据哈希。最后候选臂收尾查询失败后补跑，原失败保留，统计JSON明确列出排除原因。
- `native_inputs.jsonl` 和 `native_reference.json`：重跑正式八条需要的原始输入ID、采样参数及已有参考ID。重跑既有参考是复现，不等于重新独立采集HF参考。

## 复现

从仓库根目录执行，使用工作区项目环境及已校验固定revision本地模型。脚本每次核对全部1398源文件，拒绝覆盖已有输出目录。输出头候选仅增加 `hf_overrides.npu_lm_head_dtype=float32`，不改默认配置。硬件绑定是本次实测映射：逻辑2→物理NPU1/chip0，TP2再使用逻辑3→chip1；每次启动实时核对映射与目标chip空闲，不可盲目换编号绕过检查。

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
python docs/evidence/agent_20261015/runners/correctness.py
python docs/evidence/agent_20261015/runners/session.py 9b \
  --config configs/candidates/qwen35_9b_fp32_head.yaml \
  --run-label reproduce_9b_fp32_head
python docs/evidence/agent_20261015/runners/session.py 0.8b \
  --config configs/baseline/qwen35_0.8b.yaml \
  --run-label reproduce_08b_baseline
```

成本矩阵须与其他NPU实验串行：

```bash
python docs/evidence/agent_20261015/runners/cost_matrix.py
python docs/evidence/agent_20261015/runners/analyze_cost.py \
  results/raw/reproduce_9b_head_cost results/analysis/reproduce_9b_head_cost
```

成本完整审计通过。每臂3次8-token预热/输入长度，实际128/1024输入×32/128输出，每条件4请求，3轮独立进程/精度，共96正式请求。输出token计数、单token流、全部HBM采样及采样线程异常均通过分析器，进程退出0本身不够。整理脚本同时修复了采样线程与引擎退出清理的竞态，先停止查询再退出Engine；实际第二轮末臂因此失败并保留，只有这一臂按协议失败原因补跑，不按性能结果选优。

若使用已归档的原第二轮和补跑目录复算，应给分析器追加 `--retry-root results/raw/9b_head_cost_20261015_r2_retry`；从上述整理脚本新跑完整矩阵则不需要此参数。没有补跑目录时分析器会拒绝原第二轮的采样错误，不能静默通过。

复现脚本由本次实际实验驱动整理，调整证据/输出相对路径并补许可证；原始执行脚本字节哈希保存在原始清单中，不冒称整理后脚本字节与实际驱动相同。额外输出头逻辑权重1.89453125 GiB与周期采样HBM必须分别解释。有限样例不代表普遍质量保证，3轮描述统计不宣称显著性或最优配置。完整论述见[9B候选报告](../../project/9b_lm_head_precision_20261015.md)。
