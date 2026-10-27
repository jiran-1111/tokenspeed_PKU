# 原4B/9B首层完整参考与分片存储对照

[结论和范围](../../project/homework2_prefill_reference_20261027.md)。本批仅服务器CPU分析与本地复算，未启动NPU、未修改运行数学、未新增模型阈值；原全模型分叉仍保留。

- `conditional_prefill.json`：六份rank操作数×四种完整128步参考、所有逐token误差、权重证书及条件初态标记。相对原JSON只压缩空白，字段和值完全相同。
- `row_storage_reference.json`：相同输入的FP64完整/分片BF16存储对照、各自实际输入参考及原HCCL逐位求和检查。
- `provenance.json`：工具、原JSON/公开JSON、数组、归档、两端测试及本地核验身份；历史9B两种TP不同卡的限制见完整记录。
- 两个`*_cpu_tools.zip`：服务器实际执行工具及内部`tool_manifest.json`，逐字节保存。
- `local_verification_tool.zip`：本地核验脚本另加完整MIT注释头，两侧SHA及变换见provenance；没有内部manifest，使用ZIP外部SHA核验。

原始数据在工作区 `.tmp/hw2-20261023/supplement_export_prefill_conditional_20261027/`，旧输入在 `.tmp/hw2-20261023/supplement_export_first_layer_r4_20261024/`。大型NPZ、PT和完整权重不随公开源码再分发；没有原数组时只能检查公开指标，不能声称重做了数学参考。

先解压工具到工作区新目录，按内部manifest核对每个文件SHA。使用已有CPU环境，保持NPU不可见，不执行旧controller或恢复旧队列：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
export CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -B -m pytest /path/to/restored_tools/test_binding.py /path/to/restored_tools/test_prefill_oracle.py -q -W error -p no:cacheprovider
```

入口`analyze_prefill.py`和`row_reference.py`绑定本项目既有相对目录，恢复时保留布局并提供同revision checkpoint。输入、guard、退出/释放、实际权重证书和源码不匹配则拒绝分析。改变恢复路径应另留工具身份，不能将修改后脚本当成原执行字节。
