# HW1 补充实验公开证据

对应[实验记录](../../project/homework1_supplement_20261022.md)及[统一HW1报告](../../report/homework1_compute_subsystem.md)第3.5、3.6节。全部结果来自本组TokenSpeed计算路径的新运行；原模型、16条验证输入及固定续接保持不变。

## 文件

| 文件 | 内容 |
| --- | --- |
| `source_manifest.json` | 最终1407项源码/测试清单，SHA `402ba211fdae786fe6f4f3866e2d7972d9e90a2477ed233b493abc43ec8e2bf6` |
| `capture_source_manifest.json` | 初次补充运行的1401项清单；原1396实现与最终快照相同，后续整理显式parent清单参数和格式 |
| `head_replay_summary.json`、`head_replay_vectors.csv` | 全部272向量的局部评分重放，原13处、新9处、消除8处及新增4处分别记录 |
| `head_rounding_summary.json`、`head_rounding_vectors.csv`、`head_rounding_audit.py.txt` | 纯后处理将保存的CPU FP32分数舍入回BF16，272处选词均回到原NPU选择；完整分数仍有差异 |
| `gdn_protocol.json`、`gdn_results.json` | 固定困难输入、五种长度和每种14组数值对照及未选槽检查 |
| `verification.json`、`regression.xml` | 最终源码一致性、114项回归和设备释放；XML移除私有机器/绝对路径信息 |
| `independent_numpy_audit.json`、`independent_numpy_audit.py.txt` | 本机独立向量误差/选词及GDN递推复算，未使用Torch或NPU |
| `raw_evidence_sha256.json`、`captured_raw_sha256.json` | 未纳入Git的大型张量、日志、退出记录等的文件身份 |
| `run_receipts.json` | 初次、准备失败及最终发布快照的分别计时和退出状态 |

完整张量及原始日志保留在工作区；模型权重、输出头权重和大型张量不加入Git。原[HW1证据](../hw1_20261015/README.md)保持不变。

## 复现

以下在 `course/hw1-supplement-20261022` 检出根目录运行，使用匹配的项目Ascend环境和固定revision模型。先按[原HW1证据](../hw1_20261015/README.md)准备原manifest、单线程CPU FP32/eager cached参考及原BF16采集。例中的原结果目录只是路径占位，须指向实际原始文件，不是仓库自带的张量。

```bash
source scripts/use_project_env.sh
npu-smi info -m
npu-smi info -t proc-mem -i 1 -c 1
# 仅当本机实时映射确认逻辑3对应此芯片且显示No process in device时使用。
export ASCEND_RT_VISIBLE_DEVICES=3
mkdir -p .tmp/hw1_supplement
gzip -dc configs/validation/homework1_inputs.json.gz > .tmp/hw1_supplement/inputs.json
sha256sum .tmp/hw1_supplement/inputs.json
# 应为23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb

python scripts/run_homework1_head_capture.py \
  --logical-device 3 \
  --config configs/baseline/qwen35_0.8b.yaml \
  --manifest .tmp/hw1_supplement/inputs.json \
  --reference results/raw/homework1_stage_b_20261015/cpu-fp32-eager-cached-threads1 \
  --source-manifest docs/evidence/hw1_supplement_20261022/source_manifest.json \
  --parent-source-manifest docs/evidence/hw1_20261015/source_manifest.json \
  --output .tmp/hw1_supplement/head_capture
```

采集退出后，仅在CPU上重放最后投影；生产模型和共享embedding不变。

```bash
ASCEND_RT_VISIBLE_DEVICES='' python scripts/analyze_homework1_head_replay.py \
  --capture .tmp/hw1_supplement/head_capture \
  --old-engine results/raw/homework1_stage_b_20261015/engine_r3 \
  --reference results/raw/homework1_stage_b_20261015/cpu-fp32-eager-cached-threads1 \
  --output .tmp/hw1_supplement/head_replay \
  --threads 1 --vocab-block 4096
```

再次核对设备映射和空闲后，单独运行GDN；不要与任何本任务NPU实验并行。输出目录必须为新目录。

```bash
npu-smi info -m
npu-smi info -t proc-mem -i 1 -c 1
ASCEND_RT_VISIBLE_DEVICES=3 python scripts/run_homework1_gdn_stress.py \
  --output .tmp/hw1_supplement/gdn
```

再次确认设备空闲，运行原HW1测试组合及新增CPU保护测试：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python -m pytest \
  tokenspeed-kernel-npu/test/test_gdn_homework1.py \
  tokenspeed-kernel-npu/test/test_gdn_prefill.py \
  tokenspeed-kernel-npu/test/test_mha_homework1.py \
  test/runtime/test_homework1_norm_rope_npu.py \
  test/runtime/layers/test_gemma_rmsnorm_canonical_npu.py \
  tokenspeed-kernel-npu/test/test_qk_norm_fp32_gamma.py \
  test/runtime/test_causal_conv1d_homework1_npu.py \
  test/runtime/layers/test_layernorm_gated_npu.py \
  test/runtime/test_homework1_session_npu.py \
  test/test_homework1_head_capture.py \
  --basetemp=.tmp/hw1_supplement/pytest \
  --junitxml=.tmp/hw1_supplement/regression.xml -q
```

原13处argmax差异和68个1%门槛失败继续保留。局部FP32重放、模块状态和Session不能相互替代，更不等于自由生成质量评价。这里的运行时间属于诊断，不与原性能基线或Profiler混合。
