# 作业2：原4B/9B失败输入的完整路径验证

本批完成原128-token Prefill及原固定续接截至首次分叉输出的全32层核对：4B为step0–17，9B为step0–9。未更换输入、续接、阈值或原始失败；未修改1531项运行核心。四次NPU任务在同卡3串行完成，全部84个原logits保持字节一致。

本目录提供完整数值JSON的无损压缩包、实际执行的采集/CPU参考代码与测试、独立对应审计、1820组边界指标及来源清单。原始PT及大向量在工作区归档，体积约6.84 GB和102 MB，不提交Git；不含模型权重、账户或受保护进程信息。

## 先看结果

- [完整路径结论及限制](../../project/homework2_fullpath_validation_20261028.md)：与主报告配套的有限验证结论。
- [数值摘要](numerical_summary.json)：各自实际输入上的非零参考残差，区分整组投影与单次前向最大值；不是模型通过阈值。
- [对应及离散检查](correspondence.json)、[附加连接与实际竞争分数](connections.json)：22422项离散不变量和24644项实际连接，全部核对无失败；它们不是独立样例或统计重复。
- [本地独立复算](local_metric_verification.json)：1910组指标、128组六向量重建，与导出值一致；不冒称本地重新读取checkpoint或原始PT。
- [全部归一化边界指标](norm_boundary_metrics.csv)：65边界×28前向，图像只展示step0与step17/9。
- [来源与资源保护](provenance.json)、[所有压缩包及成员SHA](tool_and_result_archives.json)。

## 完整数值与工具

| 文件 | 内容 |
| --- | --- |
| `4b_history.zip`、`9b_history.zip` | 原服务器历史分析JSON，字节不变；包括输入清单、初态、卷积/SSM历史、FA有效缓存核对及非零残差 |
| `4b_operations.zip`、`9b_operations.zip` | 原服务器全算子JSON，字节不变；全部层投影、局部数学、行SUM分解、归一化边界及输出头，不抽掉失败或非零项 |
| `collector_r3_tools.zip` | 实际R3采集代码、测试和固定case，以及原工具manifest；公开包仅省略包含运行资源授权元数据的`authorization.json`，保留它的摘要，不能直接当可启动的授权文件 |
| `history_cpu_tools.zip`、`operations_cpu_tools.zip` | 实际CPU分析程序、参考和测试，分别对应39项和80项服务器测试，零跳过；原manifest完整保留 |
| `correspondence_tool.zip`、`additional_connections_tool.zip` | 实际执行的两项对应/连接审计脚本 |
| `local_metric_verifier.zip` | 实际本地NumPy复算脚本；依赖工作区分析归档的路径布局 |
| `figure_and_summary_tools.zip` | 本地数值摘要及图像生成脚本；字体路径可按系统字体调整，不参与模型数学 |

每个ZIP小于500 KiB，避免Git大文件；完整JSON只是压缩，没有截断或重新编码。用Python标准库可解压并校验，例如在本目录运行：

```python
import hashlib
import json
import zipfile
from pathlib import Path

ledger = json.loads(Path("tool_and_result_archives.json").read_text())
for name, item in ledger.items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    with zipfile.ZipFile(name) as archive:
        assert set(archive.namelist()) == set(item["members"])
        for member, digest in item["members"].items():
            assert hashlib.sha256(archive.read(member)).hexdigest() == digest
```

重跑CPU分析需本地同revision checkpoint、原始PT与sidecar、固定核心及前次基线，按工具中的显式目录布局恢复。历史参考由`run_history_analysis.py`执行，全算子参考由`run_operations_analysis.py`执行，审计随后执行。运行源码、模型、case、原logits及工具身份不符时分析会拒绝。软件测试可在解压后的独立工具目录使用`python -B -m pytest -q -p no:cacheprovider`；完整集合需要Torch，不能把无Torch环境的跳过写成通过。

采集并非启动示例：必须另取得当前硬件授权，重新核对同卡映射、占用、余量及受保护进程身份，按原串行保护流程运行。目录中的历史授权摘要不授权停止任何任务，也不恢复旧队列。

## 失败与解释范围

R1只准备和CPU测试；R2的9B采集成功，4B采集在捕获前因观察器通过`named_modules()`找不到共享权重别名`lm_head`而失败，原退出记录保留。回归测试重现后，R3改用实际`model.lm_head`与`model.logits_processor`对象观察，四任务全部重采。修复对象是观察器，没有修改模型数学。旧R2成功raw保留服务器；本地capture归档保留全部R3、R1–R3工具及R2失败4B。

FA历史由每步真实K/V重建，末步实际有效页直接读取；不宣称每个早期前向都独立读取全部物理缓存。GDN历史使用每层真实操作数，真实Prefill初态、g/beta和末步状态；不从CPU重算hidden独立运行整个模型。局部参考不模拟全部NPU中间舍入、FMA或规约顺序。

SUM差分分解中的重建残差为0只证明恒等式记账。分解包含局部参考残差，不能用它证明全部差异均由BF16分片存储引起，不能计算独一的因果百分比。局部非零残差与较大的全模型差异均保留；原greedy分叉未修复，未制定课程没有给出的数值容差。

这些材料完成了作业2原文要求的比较和有限验证工作，不等于四模型所有输入数值均通过，也不等于整个Agent大项目完成。
