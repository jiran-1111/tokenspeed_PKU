# 作业2：实际分叉步、局部参考与精度对照

运行核心仍为1531项清单`2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。本目录补充原4B/9B共同前沿的诊断，不新增验收门槛，不删除原失败。R10未采纳，既有性能与Profiler仍只属于原核心数学。[分析与范围](../../project/homework2_cached_precision_20261027.md)

| 材料 | 内容 |
| --- | --- |
| [cached_step9_diagnosis.json](cached_step9_diagnosis.json) | 原9B step9实际FA/GDN缓存、65规范化边界和输出头自身输入参考 |
| [cached_step9_provenance.json](cached_step9_provenance.json) | 原始归档、四份工具快照SHA、服务器测试和失败保留 |
| [gdn_sensitivity.json](gdn_sensitivity.json) | 原9B step8九项softplus、十二个core/z替换及逐head统计 |
| [independent_checks_provenance.json](independent_checks_provenance.json) | 独立NumPy复算脚本身份和范围 |
| [norm_precision_r10.json](norm_precision_r10.json) | 原CPU/R2/R9与R10的逐向量比较，无模型通过线 |
| [norm_precision_r10_provenance.json](norm_precision_r10_provenance.json) | 三次尝试、失败/释放、collector交叉绑定及最终工具身份 |
| [norm_precision_r10_local_verification.json](norm_precision_r10_local_verification.json) | 280向量、588配对指标的本地独立重算 |
| [full_text_parameter_identity.json](full_text_parameter_identity.json) | 原4B/9B六rank观察、2472项参数/派生视图认证、84原向量保持及未认证视觉范围 |
| [full_text_parameter_ledger.zip](full_text_parameter_ledger.zip) | 完整2472项ledger和服务器只读分析源，含成员SHA |
| [full_text_parameter_provenance.json](full_text_parameter_provenance.json) | 全层工具、原始归档、首次scope失败和本地复核身份 |

公开zip是工具或数据快照。`independent_numpy_checks.zip`按[独立检查身份](independent_checks_provenance.json)核对两份脚本SHA；其余zip各含自己的`tool_manifest.json`，使用前核对每个成员SHA：

- `cached_step9_capture_r1.zip`：只读观测；服务器70测试/零跳过，原30向量逐位保持。
- `cached_step9_analysis_r1.zip`：严格自身操作数分析，两端41测试/零跳过。
- `cached_step9_head_reference_r2.zip`：期望checkpoint投影权重与实际hidden做CPU FP64矩阵乘法，两端45测试/零跳过；没有认证旧采集的实际完整head矩阵。
- `gdn_sensitivity_r1.zip`：绑定已认证norm权重与实际操作数的CPU分析，服务器52测试/零跳过。
- `norm_precision_r10_retry3.zip`：外部精度候选及逐forward dtype检查，服务器66测试/零跳过；不是生产修复。
- `norm_precision_r10_analysis_r1.zip`：完整候选/旧结果比较，两端32测试/零跳过，使用说明见zip内README。
- `independent_numpy_checks.zip`：本地独立公式检查。FP32超越函数跨平台存在末位差异；GDN最终BF16候选与实际输出9/9精确一致，不等于全模型质量通过。
- `r10_independent_numpy.zip`：独立重算全部280向量的588项比较；只需NumPy，脚本实际数学主体SHA与[执行身份](r10_independent_numpy_provenance.json)对应，显式传入`--bundle`和新的`--output`。
- `full_text_parameter_tools_r2.zip`：原数学路径的只读全层文本认证，服务器29测试/零跳过；视觉参数明确不认证，原scope失败不删除。
- `full_text_identity_numpy_check.zip`：本地ledger/配置/artifact/SHA与84向量逐字节交叉检查，不在本地重读大checkpoint。

完整张量、NPZ、设备/进程收据及失败日志留在工作区`.tmp/hw2-20261023/`，没有把含账户信息的raw日志公开。已下载归档均逐成员SHA复核：

| 本地归档 | SHA256 | 文件数 |
| --- | --- | --- |
| `supplement_export_step9_capture_20261026.tar.gz` | `f738bea2bf6df10704d2cfdce40b691de0ff079145d31c76cc6cd6ce4288d8cf` | 142 |
| `supplement_export_cpu_step9_and_sensitivity_20261027.tar.gz` | `f4975719d27926848e2f6b0e30d9c85642447b721bf6a7c58d04246d8336127f` | 51 |
| `supplement_export_norm_precision_r10_20261027.tar.gz` | `ce8ca649f37b82ac0f4c086855866a43782c6e484b92abc0858b831f04b7c012` | 614 |
| `supplement_export_full_text_identity_20261027.tar.gz` | `2d69d94bf2598880650d7b3472fc7a9b473719f3ed4a76465173a6e82b24f30e` | 429 |

复现时先还原原始归档并核对`export_manifest.json`，不能用公开指标替代原张量。各入口路径、参考根目录、输出和线程数均须显式传入，输出路径不得覆盖已有结果。张量分析使用匹配的CPU Torch环境；本地NumPy检查不导入设备库。

`gdn_sensitivity.json`与`norm_precision_r10.json`只压缩JSON空白，反序列化后的值与相应归档完全相同，公开字节SHA另见[紧凑序列化记录](compact_public_serialization.json)；不能拿归档原始排版的SHA直接比公开紧凑版。全层ledger按原字节压入zip，避免将数MB的展开JSON作为单个新增文件；归档原SHA保持。

R10的离线温度1 TV不等于原temperature=0采样分布。实验中出现的公式重算容差也不是模型误差验收门槛。两个rank一致、局部数学复现、固定续接logits比较、自由生成与语义质量是不同证据，不互相替代。
