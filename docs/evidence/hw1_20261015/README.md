# HW1 最终源码与证据（2026-10-15）

报告：[计算子系统](../../report/homework1_compute_subsystem.md)。本目录提供可公开的标量数据及哈希；完整logits、CANN/Torch trace、退出日志和XML保存在工作区原始目录，未把模型权重或大型张量加入Git。

## 数据与结论

| 文件 | 内容 |
|---|---|
| `source_manifest.json` | 1396个计算源码、配置和课程测试文件的服务器匹配SHA；清单自身SHA为`f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e` |
| `numerical_vectors.csv` / `numerical_summary.json` | 全部272向量的双门槛结果；冻结协议全过，原1%门槛68失败、条件argmax13差异分别保留 |
| `numerical_independent_audit.json` / `numerical_repeat_audit.json` | NumPy独立公式全量复算；最终重采272个PT与上一轮逐文件SHA一致 |
| `decode_op_types.csv` / `decode_scope_audit.json` | 全Task Duration分类、127步Decode范围；不把AICore-only子集当全部算子 |
| `decode_profiler_summary.json` | 原始Profiler汇总及解释限制；HBM带宽原始单位未校准，不据此给出带宽利用率或访存墙钟占比 |
| `baseline_summary.csv` / `baseline_audit.json` / `hbm_samples.csv` | 最终版本96正式请求/12摘要；actual130/940×32/128；121次HBM查询 |
| `verification.json` | 服务器course569项、关键模块/Session108项的终态、203属性、原XML哈希及9测试源身份 |
| `raw_evidence_sha256.json` | 809个最终原始/分析文件的SHA，路径均相对仓库根；包含并保留未公开的大型数据身份 |
| `*.py.txt` | 本次性能编排及只读分析的原始文本，便于精确复现；以`.txt`归档保留采集时源码字节，Python可直接执行 |

这些结论只覆盖固定模型/输入/配置。条件argmax不等于自由greedy轨迹，Session与模块结果也不由logits回归替代。原失败及Profiler流计数修复过程见[Stage B记录](../../project/homework1_stage_b_20261015.md)和[最终Profiler记录](../../project/homework1_final_decode_profile_20261015.md)。

## 复现入口

以下在仓库根目录执行，使用已经建立的项目Ascend Conda环境、匹配CANN及本地固定revision权重；实际软件版本见[环境记录](../../project/environment.md)。本次课程节点逻辑2对应物理NPU1/chip0，采集入口有显式映射保护；不要省略运行前设备检查或占用已有任务。

```bash
source scripts/use_project_env.sh
npu-smi info -m
npu-smi info -t proc-mem -i 1
export ASCEND_RT_VISIBLE_DEVICES=2
mkdir -p .tmp/hw1_repro
# 首次复现使用新目录；解压不改变原manifest字节。
gzip -dc configs/validation/homework1_inputs.json.gz > .tmp/hw1_repro/inputs.json
sha256sum .tmp/hw1_repro/inputs.json
# 必须为23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb
```

先离线CPU参考，再重新确认设备空闲后采集引擎；诊断采用eager且不进入性能样本。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/collect_homework1_validation_reference.py \
  --reference cpu-fp32-eager --model models/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --manifest .tmp/hw1_repro/inputs.json \
  --manifest-sha f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936 \
  --threads 1 --output results/raw/hw1_repro/reference

python scripts/run_homework1_stage_b.py \
  --config configs/baseline/qwen35_0.8b.yaml \
  --manifest .tmp/hw1_repro/inputs.json --reference results/raw/hw1_repro/reference \
  --source-manifest docs/evidence/hw1_20261015/source_manifest.json \
  --output results/raw/hw1_repro/engine

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 python scripts/analyze_homework1_stage_b.py \
  --reference results/raw/hw1_repro/reference --engine results/raw/hw1_repro/engine \
  --manifest .tmp/hw1_repro/inputs.json --output results/analysis/hw1_repro
```

只有272个向量全部通过且源码不变，才在设备再次空闲后单独采Profiler。另开干净进程，不继承`TOKENSPEED_DIAG_*`或`TOKENSPEED_STAGE_B_ARM`；实际采集器会检查。

```bash
python scripts/run_homework1_decode_profile.py \
  --config configs/baseline/qwen35_0.8b.yaml --manifest .tmp/hw1_repro/inputs.json \
  --stage-b-summary results/analysis/hw1_repro/summary.json \
  --source-manifest docs/evidence/hw1_20261015/source_manifest.json \
  --output results/raw/hw1_repro/decode_profile
```

复现软件集合及本次关键模块/状态组合，保持项目内独占basetemp，NPU测试与所有其他NPU实验串行：

```bash
python -m pytest test/course_project -q --basetemp .tmp/hw1_repro/course
python -m pytest \
  tokenspeed-kernel-npu/test/test_gdn_homework1.py \
  tokenspeed-kernel-npu/test/test_gdn_prefill.py \
  tokenspeed-kernel-npu/test/test_mha_homework1.py \
  test/runtime/test_homework1_norm_rope_npu.py \
  test/runtime/layers/test_gemma_rmsnorm_canonical_npu.py \
  tokenspeed-kernel-npu/test/test_qk_norm_fp32_gamma.py \
  test/runtime/test_causal_conv1d_homework1_npu.py \
  test/runtime/layers/test_layernorm_gated_npu.py \
  test/runtime/test_homework1_session_npu.py \
  -q --basetemp .tmp/hw1_repro/npu --junitxml results/raw/hw1_repro/modules.xml
```

无诊断、无Profiler的正式基线在独立进程执行归档的编排文本；它仍会检查当前源码、设备和端口。首次运行要求目标结果及临时配置不存在；重复实验应使用新run路径，不能覆盖原记录。该脚本对应本课程节点的逻辑2及固定测量端口，引用的R3清单可从本目录复制到原记录位置：

```bash
mkdir -p results/raw/homework1_stage_b_20261015
cp docs/evidence/hw1_20261015/source_manifest.json \
  results/raw/homework1_stage_b_20261015/source_manifest_r3.json
python docs/evidence/hw1_20261015/homework1_final_baseline_hbm_20261015.py.txt
python docs/evidence/hw1_20261015/homework1_final_baseline_hbm_analysis_20261015.py.txt
```

原benchmark每轮硬编码physical0的快照仅作原始辅助记录；本次实际HBM结论来自独立全设备采样器解析physical1/chip0，不能混用。所有输出目录保留失败记录，统计不得删失败或根据结果更改冻结门槛。教学网提交由用户本人完成。
