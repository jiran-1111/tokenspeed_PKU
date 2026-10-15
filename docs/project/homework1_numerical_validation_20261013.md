# 作业1单chip数值与状态验证记录

后续勘误：下面是原测试的实测通过记录，不覆盖或改写。2026-10-14确认旧Prefill参考使用clamp而生产/固定HF使用平方和加eps，近epsilon边界未由这些随机输入证明；详见[参考语义审计](homework1_gdn_reference_audit_20261014.md)。旧容差、状态隔离断言和XML仍有效记录，但不能据此声称参考与生产数学定义完全相同；参考纠正与边界复验仍欠。

2026-10-13，Ascend910 V1，`ASCEND_RT_VISIBLE_DEVICES=0`，项目Conda，无依赖安装。先确认27B基准正常结束、保存结果并停止其服务，再确认设备无其他进程，串行运行测试。版本实际读取：PyTorch `2.9.0+cpu`、torch_npu `2.9.0`、Transformers `5.12.0`、Triton `3.2.0`；PyTorch wheel版本中的cpu后缀不表示本测试在CPU执行，实际被测张量通过torch_npu创建/迁移到NPU，CPU仅用于参考。

## 新增真实形状与隔离测试

原始结果：`results/raw/homework1_numerical_20261013/results.xml`。运行10项全部通过，无skip/error/failure，耗时43.52秒。阈值采用实测前的[合同](homework1_numerical_contract.md)，没有事后放宽。两个新增测试源文件跨端SHA-256相同：

- GDN测试：`5e3e32a536515d92744726d02981bb7e87950ef4a548077622dd763427e311e9`。
- runtime卷积测试：`259144e42d7d3c65cb8e98b6eee344a9efd1dbc0133a42f8ff651dc2fe62e909`。

XML跨端SHA-256相同：`83eb4518a2bbe853368ef0e609ea39eef2cb1e6b26aaf97dc5f81534e60998b4`。逐条解析确认10个testcase及92条误差properties：5×2个Prefill输出/状态，8×2×2个交错Decode输出/状态，2个读写分离输出/状态，3×8×2个卷积输出。最初离线核验误写预期116导致计数断言失败；按测试真实循环核对为92并重新验证通过，没有改XML或数值测试。

| 测试 | 记录内最大max-abs | 记录内最大relative-L2 |
|---|---:|---:|
| GDN Prefill length 1 | 1.1921e-7 | 9.5119e-8 |
| GDN Prefill length 8 | 3.8147e-6 | 7.0038e-6 |
| GDN Prefill length 127 | 6.1035e-5 | 4.1915e-5 |
| GDN Prefill length 128 | 6.1035e-5 | 2.9775e-5 |
| GDN Prefill length 129 | 6.1035e-5 | 3.5145e-5 |
| GDN双slot交错各8步 | 7.6294e-6 | 2.7233e-5 |
| GDN读slot1/写slot3 | 5.9605e-8 | 6.1010e-8 |
| 卷积channels16/6143/6144，各两slot×8步 | 0 | 0 |

GDN表中聚合了该testcase内输出和state的各自误差指标；不是说最大绝对误差和最大relative-L2来自同一张量。逐张量值保存在XML。卷积输出与从同一BF16输入出发的CPU FP32参考再量化结果相等，不表示任意输入均逐位相等。

每次交错更新都验证未选slot精确不变；卷积历史移位精确等于参考；GDN读写分离验证读slot及其他slot精确不变。覆盖证明这些形状与八步范围内的实现行为，不证明任意长序列、完整引擎请求隔离或端到端正确性。

## 既有关键kernel复验

独立pytest运行MHA、LayerNorm、RoPE、GDN Prefill/Decode及solve-tril，30项通过，无skip/error/failure，20.39秒。原始结果 `results/raw/homework1_numerical_20261013/existing_kernel.xml`。这是对原有测试断言的复验，不声称这些测试全部新增了逐张量误差properties；各模块参考/容差来源仍需与算子矩阵一同说明。

## 真实Norm与MRoPE补充

`test/runtime/test_homework1_norm_rope_npu.py`于首次实测前在数值合同固定判据，2026-10-13logical chip0执行70项全部通过、无skip/error/failure，7.54秒。standard/Gemma Norm宽度256/1024、rows0/1/129/1024、BF16/FP32、有/无残差共64项；实际runtime MRotaryEmbedding text1D/三轴相同/三轴不同，BF16/FP32共6项。head256、rotary64、theta1e7、sections11/11/10与固定配置一致。未旋转192维及残差更新精确相等；FP32均额外通过atol=rtol=1e-5，未改阈值。

`results/raw/homework1_model_20261013/norm_rope.xml`解析验证70 cases、76误差properties，跨端SHA-256 `2401c0b26bdf4e5734790b384f16441224325b4bc939092f4ed4e57cf5cb8638`。Norm全部记录最大max-abs0.03125、relative-L2最大0.00312357；不同记录的两种最大值不视为同一张量。MRoPE最大max-abs0.015625、relative-L2最大0.000965875。FP32 Norm max-abs9.536743e-7、MRoPE max-abs0，所有逐元素判据通过。BF16 Norm最大绝对误差可超过atol0.002，因其判据为atol+rtol×|expected|，不能只按max-abs与atol单独对比判失败或通过。

MRoPE参考使用独立CPU频率和axis选择，以与模块相同dtype量化cos/sin后FP32旋转；测试不是通用npu_mrope入口。空输入记录0误差只是无元素的边界检查，不计作有数值样本。新增测试只验证现有实现，不引入生产源码变更，完整模型分叉仍未闭环。

## runtime复验、警告与边界

runtime层单独复验gated RMSNorm、GemmaRMSNorm、NPU attention backend及Q/gate布局，12项通过，无skip/error/failure，5.63秒；`runtime.xml`已同步。此12项含CPU上的布局helper检查，不统称12项都是NPU kernel执行。此前三个批次共52项通过，另有本次70项真实尺寸补充；既有kernel/runtime XML均逐一跨端SHA-256一致。对外报告仍按测试范围区分新增数值、原有kernel、runtime wiring及CPU布局。

共享CANN路径所有者不同的警告未改权限；NPU内部格式关闭时自动使用base format的警告未隐藏。首次新增测试JUnit采用xunit2，pytest提示record_property与其schema不兼容，但实际XML包含完整92条properties并已解析验证；后续复验采用 `-o junit_family=legacy`。原始XML保留，不为消除警告覆盖重跑。

多长度原生采集、Session交错/关闭重开及统一2×2性能已有独立实测记录；完整模型共同前缀数值分叉、有效负载HBM峰值/热点及最终报告仍未闭环。不能用本记录把第一阶段全项标记通过。
