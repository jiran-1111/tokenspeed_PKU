# 作业1 MHA：实际D256跨页与独立CPU参考

## 结果

服务器正式测试：3项CPU解析自检通过（1.220秒），6项NPU边界通过（7.479秒）；均0 failure/error/skip。测试完成后physical1无运行进程，NPU0他人任务未操作。两端XML与变异检查JSON的SHA一致。SPEC与quality独立审查后部署，未修改生产MHA或原有容差。

## 固定协议

实际8 Q heads、2 KV heads、D256、page128；Extend/Decode各测试KV长度127/128/129。两个序列的非连续页表`[[2,0],[3,1]]`使用四个物理页；Extend query长度2/3，Decode为1/1，缓存已包含本轮token。输入固定CPU seed311+长度生成BF16。

CPU参考将实际BF16值展开FP32，独立按页表gather、截取有效长度、GQA重复、QK/sqrt256、softmax和V聚合；Extend采用下右对齐causal mask。不是原NPU BF16参考，也不调用生产MHA helper。比较原先冻结的MHA atol=0.02/rtol=0.02，不沿用GDN包络、不按结果调大。检查全部七个输入逐位不变、输出shape/finite，并记录maxabs与relative-L2。

CPU自检使用手算常量检验页序/截断、causal/GQA和缩放softmax。另对源码仅内存变异：反转页序、删除causal下右偏移，两项均实际触发预期AssertionError；磁盘测试源SHA前后未变。

| 路径 | KV长度 | 输出shape | max-abs | relative-L2 |
|---|---:|---|---:|---:|
| Extend | 127 | 5×8×256 | 0.0019299984 | 0.0021082857 |
| Extend | 128 | 5×8×256 | 0.0021448731 | 0.0021851168 |
| Extend | 129 | 5×8×256 | 0.0015723109 | 0.0021507819 |
| Decode | 127 | 2×8×256 | 0.0021048188 | 0.0020458184 |
| Decode | 128 | 2×8×256 | 0.0016995072 | 0.0021858311 |
| Decode | 129 | 2×8×256 | 0.0012763739 | 0.0022215168 |

六项全部finite。有限固定样例不能证明全部输入或任意状态分布正确；该证据不代替完整引擎或全模型误差判据。

## 来源与复现

- 测试：`tokenspeed-kernel-npu/test/test_mha_homework1.py`，SHA256 `0a3876fe204cc0f65994bb466817d530e255b70a5eaa038963632649c794276f`。
- 生产包装：`tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py`，SHA256 `3d6791372605e0ab3d505ee1a85a5f8297d113b9886c4ee6eea371ed99bb5d73`。
- 原始目录：`results/raw/homework1_mha_boundaries_20261015/`，含cpu.xml、npu.xml、cpu_mutations.json及前后设备快照。
- CPU XML SHA256：`4a9e0d8b00a833af845b4ba2d558367b553c34ff38bae87f9e047136062f80e6`。
- NPU XML SHA256：`e2434ebde3ca810eb03692b1d5ed3d1c117809165347ad7c9505ab44c9544edb`。
- 变异JSON SHA256：`5eda75fe89be556aa78d185bf2adb0755b8365fb79a3fac20193ca430d813c15`。

```bash
source scripts/use_project_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tokenspeed-kernel-npu/test/test_mha_homework1.py -q -k cpu_reference
# 先fresh核对空闲芯片及逻辑映射，下面2只适用于本次记录的映射。
ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest tokenspeed-kernel-npu/test/test_mha_homework1.py -q -k paged_boundary
```

本机环境缺Torch时仅模块skip，不计为通过；上述数值结果均来自服务器实际执行。
