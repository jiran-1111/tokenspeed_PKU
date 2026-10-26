# 作业2：四模型固定语义请求比较

本实验补充[主报告](../report/homework2_communication_subsystem.md)第2.5节的多输入证据。原主矩阵的4B、9B短输入自由输出与完整logits差异仍未修复。此次只报告新冻结协议下的生成结果，不更换原失败，不增加通过阈值，不把输出相同当作答案正确。

## 协议和数据身份

生成前从已有六类正确性问题固定八条请求：中文、英文、算术、代码、上下文召回和三轮记忆。完整messages、固定assistant前史、tokenization选项、输入ID、EOS来源、模型revision、配置SHA及权重文件身份见[原始冻结manifest](../evidence/homework2_20261026/semantic_quality_frozen_manifest.json)。文件SHA为`1e8e3ae38444191c7dd9d90ffa97b30f730e9e7ab98a2a7fc7a9e17b2d7ef774`，canonical SHA为`59603dd069177872d06827af445ff8a68716d1db9056f5d0e4654886f04e092e`。

四模型均显式禁用thinking、greedy、EOS停止、最多128输出，输入长度依次19/25/26/26/979/21/55/86。原生Engine保持1531项核心源SHA `2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`，沿用成对TP配置；27B两TP都是既定95%预算。配置开启Decode图模式，不据此断言每次forward都实际图重放。采集不安装前沿强制或数学覆盖hook，端口之外不改配置。

CPU参考采用已验证的同投影权重合同：checkpoint参数按原BF16值投影后转FP32，A_log/dt_bias保留FP32，使用独立HF eager实现、8个CPU线程，禁止NPU自动加载。核对模型文件、投影指纹、参考源与运行源前后身份。四CPU各8条，原生TP1/TP2各模型各8条，共96条输出；三种配对共96组，但共享两侧，且记忆三轮相关，不是96个独立样例或随机抽样。

## 完整输出配对

| 模型 | TP2 / TP1 | TP1 / CPU | TP2 / CPU |
| --- | --- | --- | --- |
| 0.8B | 6/8 | 6/8 | 6/8 |
| 4B | 8/8 | 7/8 | 7/8 |
| 9B | 8/8 | 8/8 | 8/8 |
| 27B | 8/8 | 6/8 | 6/8 |

计数为完整输出token IDs逐位相同；原始与EOS对齐后计数完全相同。完整输入、输出ID、文本、停止原因、首差位置和原始文件SHA见[逐条证据](../evidence/homework2_20261026/semantic_quality.json)。0.8B英文和上下文召回、4B确认句以及27B北京描述和代码注释有表达差异；不同ID不自动表示答案错误，ID相同也不能认证语义。

必须保留的质量问题：0.8B、9B三路代码均达到128上限，没有完整求和实现；9B8/8也包含这条截断输出。0.8B三路北京介绍共同包含未经核实称谓。4B、27B代码有`return sum(numbers)`主体，未执行代码功能测试，不写为代码能力通过。固定前史不是原生Session状态或缓存验收。CPU margins属于各自生成历史，分叉后不作为共同前沿数值比较。此次没有采完整logits，也不从这些样例估计总体准确率、置信区间或显著性。

## 资源保护、失败和最终释放

每项NPU实验串行，重新读取设备映射、进程身份和显存，保护原有服务。0.8B/4B/9B两TP均在获准共享功能模式完成。首次27B TP1在初始化触发既有共享保护的8192MB余量下限，0条请求；这不是数值结果，失败原样保留。随后确认目标单chip空闲，使用已有exclusive lease完成TP1；TP2继续原共享保护。没有放宽共享余量、变更模型/配置或杀后台服务。

内部guard的子进程释放与外层driver退出分别校验。最终R2配对器显式读取两份原始controller文件及SHA，逐个成功job要求唯一对应、exit0、8条完成、guard通过和own contexts released；第一份controller整体failed不改写，因为其中最后27B任务失败，先前六项仍按其实际终态认证。关联的quality当前SHA由分析器记录，不虚构旧controller曾记录该SHA。[失败与退出证据](../evidence/homework2_20261026/semantic_quality_failures_and_release.json)

准备阶段缺少可选generation_config、默认BatchEncoding及同工作区模型链接边界失败全部保留；最终使用显式tokenizer返回类型和workspace边界，未改冻结输入。R1分析器只有本地14测试及分析，未强制外层释放；R2新增九项反例，两端23测试通过，并对同一真实数据复算一致。比较时只规范化带SHA文件记录的位置字段，其余字段全部相同。

## 归档与复算

原始日志含服务器账户、进程与绝对路径，不公开再分发。工作区保留初批468文件、27B重试91文件及最终分析8文件三个完整归档，均下载逐SHA核对；[公开收据](../evidence/homework2_20261026/semantic_quality_archive_receipts.json)记录身份。公开摘要保留全部输出和负结果。八份[工具快照及测试身份](../evidence/homework2_20261026/semantic_quality_tools.json)覆盖CPU R1–R4、原生R1–R2及分析R1–R2；原失败未删除。

从仓库根目录、已有工作区环境执行CPU复算。先恢复原始`jobs`及两份原始controller到`jobs/_controllers`，并核对归档SHA；没有原始文件时只能复核公开输出比较，不能声称重跑完整身份验证。输出路径必须未存在。

```bash
python -m zipfile -e docs/evidence/homework2_20261026/semantic_analysis_r2.zip .tmp/semantic-analysis
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('.tmp/semantic-analysis')
manifest = json.loads((p / 'tool_manifest.json').read_text())
assert all(hashlib.sha256((p / n).read_bytes()).hexdigest() == s for n, s in manifest.items())
PY
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
python -B -m pytest .tmp/semantic-analysis/test_analyze_semantic_quality.py -q
python -B .tmp/semantic-analysis/analyze_semantic_quality.py \
  --root /path/to/restored/jobs \
  --manifest docs/evidence/homework2_20261026/semantic_quality_frozen_manifest.json \
  --selection .tmp/semantic-analysis/selected_quality_runs.json \
  --source-root "$PWD" --output .tmp/new-semantic-analysis.json
```

本实验扩展了有限输入观察，没有解决原4B/9B差异，也没有认证一般生成质量。原共同前沿、局部诊断、性能及Profiler各自保留独立协议。任何运行数学修改仍需固定新源并重做对应验证，不能沿用本次性能结果替代。
