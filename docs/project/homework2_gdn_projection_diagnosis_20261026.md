# 作业2补充诊断：9B GDN投影与门控

本记录接续[实际缓存step8诊断](homework2_cached_step8_diagnosis_20261026.md)，补齐其中尚未验证的GDN层8–10投影权重身份、带门控规范化及MLP局部运算。**原4B/9B短输入自由输出差异仍未修复；没有新增阈值，也不把局部参考接近改写成全模型正确性通过。** [作业2主报告](../report/homework2_communication_subsystem.md)继续保留原比较结论。

## 同源、同输入与实际权重

核心1531项运行源及清单SHA仍为`2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。模型为Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，沿用原128输入、9-token固定续接和step8位置135。BF16、SSM FP32、9B输出头FP32及eager实际缓存路径不变。

先用已保存的操作数与checkpoint期望权重做CPU参考（R1），得到局部运算接近参考的结果。但这个结果本身不能证明运行时加载了正确权重，故另加只读身份观察器，串行重跑原TP1/TP2。每项前重新查映射、身份和余量；TP1使用card0/chip1，TP2使用同卡chip0/1，保留后台服务和暂停的旧队列。两项均exit0、自身上下文释放，核心源码前后相同，没有保护违规。观察器不写回权重或状态，本批不作为性能数据。

每层检查八项实际参数：融合Q/K/V/Z/B/A投影、GDN输出投影、门控规范化、MLP gate/up、MLP down、卷积权重、A_log及dt_bias。融合矩阵每个逻辑段分别切分，行投影按输入列切分；规范化权重复制，A_log/dt_bias按value-head切分。三层×三份rank×八项，共**72项实际形状、dtype与参数字节均匹配checkpoint的正确分片**。

GDN内部norm实际为每head128宽的`norm(x) * silu(z)`，epsilon为1e-6，`group_size=None`、`norm_before_gate=True`、`sigmoid_gate=False`；没有Gemma权重加1。checkpoint norm为FP32，实际加载权重为BF16，身份比较包含这个转换，CPU参考使用同样的BF16权重值。A_log/dt_bias实际保持FP32。

新TP1 rank0、TP2 rank0/1各10个完整logits，共**30个向量与原同源FA7参考逐位一致**；实际输入前沿、位置、原始argmax及采样/固定续接决定也相同。随后将相同CPU数学分析器重新运行在本次带身份记录的捕获上（R2），逐项核对checkpoint原张量SHA、配置和索引SHA、捕获文件SHA以及实际参数身份。该对应验证针对这批forward，未倒推历史运行中未抓取的参数身份。

## 各算子自身操作数的参考

每种矩阵乘都使用该投影自身的实际输入，GDN norm使用实际递推输出和实际z，MLP激活使用实际gate/up输出；没有把CPU重算结果一路串起来冒充原执行。FP32参考之后再给出最终BF16存储候选，它不模拟所有设备内核中间舍入。

| 局部运算 | 比较数 | 相对自身FP32参考的relative-L2范围 | 最终BF16候选逐位一致数 |
| --- | --- | --- | --- |
| 四种GEMM：GDN输入/输出、MLP gate/up与down | 36 | 0.00156156–0.00195675 | 10/36 |
| GDN内部带门控norm | 9 | 0.00139119–0.00226555 | 9/9 |
| MLP SiLU乘法 | 9 | 0.00157074–0.00234203 | 9/9 |

共54项，28项最终存储候选逐位一致，其余26项均为GEMM，共59个元素不同；最大绝对差0.00390625，单项最大relative-L2为9.34052298e-5。所有非零项均保留，不能说“全部只是最终舍入、全部逐位通过”。融合投影到卷积、z/b/a及带门控norm到输出投影的45项操作数连续性检查均逐位一致。

这里的GDN内部带门控norm不同于上一批hidden norm边界。上一批TP2两rank在层10 hidden post-attention norm各有1元素的最终候选差异，仍然保留，不被本表9/9替代。

## 传播差异与局部误差的区别

以每个rank真实输入分别进行相同FP32数学，再按逻辑head或通道重组，跨TP差异仍然很大。以下是两侧实际输入不同条件下的relative-L2，不能与上表的“相同输入下局部参考误差”混同。

| GDN层（0起） | 带门控norm实际TP2/TP1 | 带门控norm CPU参考TP2/TP1 | MLP down实际部分和/TP1 | MLP down CPU部分和/TP1 |
| --- | --- | --- | --- | --- |
| 8 | 0.23165841 | 0.23176406 | 0.33146592 | 0.33135470 |
| 9 | 0.42414439 | 0.42422983 | 0.68156496 | 0.68212521 |
| 10 | 0.82117559 | 0.82139074 | 0.99167496 | 0.99193171 |

MLP down“实际部分和”列明确指两个rank实际BF16部分结果用CPU FP32相加，尚未作返回dtype舍入，**不是直接测得的HCCL返回值**；因此它与前批实际SUM返回后的数字略有不同。CPU部分和列同样以各自真实输入计算局部投影，再相加。两列各以对应TP1结果为分母，不将它们的差值解释为某种可加贡献。

这说明在所测步骤，输入已经分叉时，相同权重的CPU数学也产生接近实际的后续差异。结合上一批真实KV、SSM、SUM及输入/历史交叉替换，可优先检查早期小差异如何经连续投影、门控和规范化放大；不能直接认定某次HCCL、状态槽或本地GEMM算错。它也没有证明最早差异的完整来源、整段prefix的状态正确性、自由生成质量或所有模型均可接受。输入分叉前的独立全模型参考与更完整质量评价仍需继续。

## 归档与复算

- [54项指标、非零结果和同批身份对应](../evidence/homework2_20261026/gdn_projection_diagnosis.json)。原独立分析器的`runtime_projection_and_norm_weights_certified=false`如实保留，因为它只读取checkpoint；另列的`same_capture_actual_parameter_certificate`将本次72项实际参数认证与完全相同的捕获SHA关联。这个对应结果不通过改写原分析标志产生。
- [三份rank的72项参数及norm模式](../evidence/homework2_20261026/gdn_runtime_parameter_identity.json)。
- [182项完整文件的下载校验收据](../evidence/homework2_20261026/gdn_projection_archive_receipts.json)：初版期望权重CPU分析18项、实际身份硬件采集145项、同批CPU复算及对应记录19项，均已下载并逐SHA核对。参数值、原始张量及含账户信息的日志不公开再分发。
- [工具身份](../evidence/homework2_20261026/gdn_projection_tool_snapshots.json)：新增采集20文件/51项服务器CPU先行测试；两版CPU工具各9文件/27项服务器测试。R1/R2数学分析器逐字节相同，变化是任务选择、对应控制脚本及输出目录。历史R1没有被删除或改写成已取得实际权重。

CPU复算示例从仓库根目录执行，使用已有工作区环境。先恢复相应原始`jobs`并验证SHA，输出名须未存在。没有原始数据时只能核对公开标量，不能宣称重跑了实验。

```bash
python -m zipfile -e docs/evidence/homework2_20261026/gdn_projection_r2.zip .tmp/gdn-projection
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
python -B -m pytest .tmp/gdn-projection -q
python -B .tmp/gdn-projection/invoke.py \
  --root /path/to/restored/jobs \
  --jobs-manifest .tmp/gdn-projection/selected_jobs.json \
  --source-root "$PWD" \
  --output .tmp/new-gdn-projection.json --threads 8
```

该CLI复算期望权重数学；实际参数身份还须另核对三份`gdn_actual_checkpoint_identity.json`、72项匹配及完整对应SHA。工具ZIP的manifest和SHA核对方式同[缓存证据说明](../evidence/homework2_20261026/README.md)。没有同revision权重时不得用另一模型补算。需要新采硬件时仍先检查设备和后台，串行执行，不停止其他服务。
