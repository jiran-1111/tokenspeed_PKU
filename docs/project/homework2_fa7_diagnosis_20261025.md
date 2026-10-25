# 作业2补充诊断：4B/9B的FA第7层

本记录接续[作业2报告](../report/homework2_communication_subsystem.md)的正确性分析。它只补充同一固定短输入下FA层7及其MLP的Prefill观察，**没有修复原4B/9B自由输出分叉，也没有新增通过阈值**。层号从0开始；本次不覆盖9B cached step8的实际KV/状态。

## 协议与结果保持性

保持1531项核心运行源不变，清单SHA为`2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。仍用原128输入、4B共同17-token续接和9B共同9-token续接，以eager实际缓存路径取得18/10个完整logits。新观察器只在第0步保存FA7全部128行操作数，保留原dtype，复制发生在RoPE、门控和残差原地更新之前；多个FA共享的RoPE对象通过层7入口/出口限定观察范围，异常也移除hook。

两个模型分别串行运行TP1与同卡TP2。TP1在card0/chip1，TP2在card0/chip0、chip1；每项重新检查映射、后台身份和余量，保留其他服务。四组有效采集最终均正常退出、自身设备上下文释放，原独占队列保持暂停。所有操作数和新旧完整logits均保留，正式计时资格为false，不把观察器开销当作性能。

相对原同源首层诊断，`18×3+10×3=84`个完整logits向量逐位一致，实际输入、位置、原始argmax及采样/强制续接决定也一致。这说明新观察器没有改变这些被测结果，不证明任意输入或图模式正确。

## 实际分片与局部参考

独立读取checkpoint，分别切分Q/gate、K、V和MLP逻辑段；Q/gate按head保持交错布局，不把整块packed矩阵二等分。两模型各有TP1 rank0、TP2 rank0/1，每份核对FA7八项实际参数，共48项全部逐位匹配期望分片。另保留实际规范化有效gamma，并核对其等于原始权重转FP32后加1。

每个算子的CPU FP32参考使用该算子自己真实抓取的输入和实际权重，而非把前一级重算输出一路传下去。原始/舍入后参考均存档，以下是全部128行上的relative-L2观察范围，没有将其作为新验收门槛。

| 观察项 | 本次结果 | 可以支持的结论 |
| --- | --- | --- |
| 四类矩阵投影，六份rank捕获 | 相对自身FP32参考0.00162185–0.00178074 | 被测局部矩阵计算的差异量级约0.16%–0.18%，不能替代全模型比较 |
| causal GQA attention，六份rank捕获 | 相对真实RoPE后Q/K/V的FP32参考0.00167285–0.00171788 | 本次完整Prefill注意力结果与该参考接近；不认证后续缓存读取 |
| attention行投影的实际SUM | 两模型两个rank返回值均与实际BF16部分和的FP32相加再BF16舍入逐位一致 | 未发现这次实际部分和合成错误 |
| MLP down的实际SUM | 同上，最大绝对差均为0 | 未发现这次实际部分和合成错误 |
| 残差原地相加 | 与实际输入之和按BF16舍入逐位一致 | 验证被测残差存储舍入，不证明所有误差都合理 |

全128行的跨TP规范化结果如下；它与前次只比较最后token行的数字口径不同，不能相互替换。

| 模型 | FA7输入norm | FA7注意力后norm | 下一层输入norm |
| --- | --- | --- | --- |
| 4B | 0.01537498 | 0.03268946 | 0.03646953 |
| 9B | 0.00638277 | 0.01355372 | 0.01213308 |

差异在进入FA7前已存在，经过注意力、MLP、规约和规范化后继续变化。结合本次参数身份、局部操作数参考及实际SUM，可以限制“FA7分片错装或该次SUM错误”的解释；不能把这些复合边界差异全部归给某个单算子，也不能排除其他层、其他步骤或语义路径的问题。

RoPE保存实际positions与所用cache前缀，分别以原表FP32和表转Q dtype构造参考；没有完整认证内核中所有舍入顺序，因此两者明确标为参考候选。Prefill实际路由和metadata满足冻结源码的`prefix=0 / postwrite`分支条件，记录是attention前置观察，未冒称捕获了分支内部调用事件。由此构造本次无历史前缀的attention算术参考，不等于验证cache写入或Decode。

## 失败、工具与可复现材料

首批9B TP2已经保存10个前沿向量，但退出保护器发现自身worker暂存，记录为`captured_pending_guard_exit`，exit1；源码前后相同且无后台资源违规。外层随后确认自身上下文释放。该组失败完整保留，未改写成成功。

只为9B TP2建立R2，采集、模型、输入和数学路径不变；外部保护器复用已有四项测试覆盖的15秒限时等待，每次仍做严格身份/资源快照，未知进程或资源违规直接传播。R2正常退出，分析清单明确选择前三项R1与这项R2，没有替换原失败文件。

- [公开标量、参数身份及原始文件SHA](../evidence/homework2_20261025/fa7_prefill_diagnosis.json)：48项分片、逐rank/逐算子指标、跨TP重组及84向量结果保持性。
- [完整原始与分析归档收据](../evidence/homework2_20261025/fa7_archive_receipts.json)：420项硬件文件含全部五次成功/失败、16项CPU分析文件均已下载，并逐文件跨端SHA匹配。权重和大型张量只保存在工作区，不公开再分发。
- [冻结工具身份](../evidence/homework2_20261025/fa7_tool_snapshots.json)：R1为14文件、28项服务器CPU测试；R2为16文件、32项；CPU分析为6文件、13项本地及服务器测试。先行失败、准备阶段测试收集不足和后续修正日志全部保留。

分析工具可从公开ZIP恢复，再用已保存的原始采集在匹配环境中复算。所有命令从仓库根目录执行，目录占位符必须替换为实际恢复路径；输出须使用新文件名。

```bash
python -m zipfile -e docs/evidence/homework2_20261025/fa7_analysis_r1.zip .tmp/fa7-analysis
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES= \
python .tmp/fa7-analysis/invoke.py \
  --root /path/to/restored/jobs \
  --jobs-manifest .tmp/fa7-analysis/selected_jobs.json \
  --output .tmp/fa7-new-analysis.json --threads 8
```

下一步应优先补9B cached step8的真实KV、请求位置与GDN状态边界；这批Prefill结果不能证明后续缓存没有问题。任何数学实现修改都须另固定源码并重新验证，不拿本次相同源的性能数据认证新实现。
