# 9B缓存step8补充证据

用途与结论见[诊断记录](../../project/homework2_cached_step8_diagnosis_20261026.md)。仅针对原9B短输入的实际缓存step8；核心1531源未变，30个原前沿向量逐位保持。**4B/9B自由输出差异仍未修复，不新增通过阈值，不作为性能结论。**

`cached_step8_diagnosis.json`保存各rank局部指标、dtype、槽/位置绑定及原始文件SHA。`gdn_state_input_counterfactual.json`保存三层的CPU状态/当前输入交叉替换、跨TP权重身份和全部非零指标。公开JSON去掉服务器绝对路径、账户和进程信息；逐head明细压缩成最坏head标量，完整分析的SHA和原始归档收据仍保留。

四个ZIP保存逐字节工具及各自manifest。采集R1为18文件/46先行测试，局部分析R1为7文件/1失败17通过（准备阶段未运行真实分析），分析R2为7文件/18通过，交叉分析R1为8文件/26通过。R1与R2分析数学相同，仅合成单测容差和控制脚本目标目录不同；模型门槛及原始结果未动。`tool_snapshots.json`同时记录失败和成功身份，不能只复算R2就宣称重现了历史失败。

以下从仓库根目录执行。使用工作区已有匹配环境，无需全局安装；工具本身只用CPU，显式禁止NPU自动加载。先从工作区保留的原始归档恢复`jobs`，没有原始张量只能核对公开标量，不能声称重跑了数学参考。输出文件名必须未存在。

```bash
python -m zipfile -e docs/evidence/homework2_20261026/cached_analysis_r2.zip .tmp/cached-analysis
python -m zipfile -e docs/evidence/homework2_20261026/cached_propagation_r1.zip .tmp/cached-propagation
python - <<'PY'
import hashlib, json
from pathlib import Path
for name in ('cached-analysis', 'cached-propagation'):
    p = Path('.tmp') / name
    manifest = json.loads((p / 'analysis_manifest.json').read_text())
    assert all(hashlib.sha256((p / n).read_bytes()).hexdigest() == s for n, s in manifest.items())
print('tested snapshots verified')
PY
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
python -B -m pytest .tmp/cached-propagation -q
python -B .tmp/cached-analysis/invoke.py \
  --root /path/to/restored/jobs \
  --jobs-manifest .tmp/cached-analysis/selected_jobs.json \
  --output .tmp/new-cached-analysis.json --threads 8
python -B .tmp/cached-propagation/invoke.py \
  --root /path/to/restored/jobs \
  --jobs-manifest .tmp/cached-propagation/selected_jobs.json \
  --output .tmp/new-cached-propagation.json --threads 8
```

两个CLI显式要求root、原任务选择清单、输出和CPU线程数，并校验case、guard、位置135/有效前缀136、dtype和真实状态映射。交叉分析验证A_log、dt_bias及逻辑Q/K/V卷积权重逐位匹配才计算替换；不把当前输入差异和历史差异当成可加贡献。重新采硬件还需同revision权重、匹配CANN/torch_npu及设备，须重新检查映射与后台并串行执行；恢复ZIP本身不是硬件运行授权。
