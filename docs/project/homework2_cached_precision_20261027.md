# 原9B分叉步的局部复核与精度对照

本轮继续检查原4B/9B短输入跨TP差异，不替换原样例、固定续接或失败，不新增通过门槛。核心1531项仍为`2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。局部操作在自己的输入上可复现，与两种TP最终结果一致，是不同层次的结论。

## 9B step9实际执行与局部参考

对原128-token输入和固定9-token续接，TP1、TP2在同卡串行观测。step9实际位置136，FA有效缓存由136增长至137；记录FA7、GDN8–10实际缓存和状态、65个规范化边界及真实输出头操作数。全部30个原logits向量逐位保持，两任务退出0、自身NPU上下文释放。观测工具服务器70测试通过、零跳过；严格分析两端各41测试通过，原137项输入和1531项核心源前后核对。

三个rank的195个规范化输出BF16候选中，189个逐位一致，其余6个各有一个元素不同；192个残差BF16存储候选全部逐位一致。FA7以自己的实际操作数进行FP32参考，relative-L2约0.00156–0.00166；GDN8–10的自身SSM更新参考relative-L2为3.56e-8–1.79e-7。这些是局部误差测量，没有自动通过线，未认证全部层或全部输入。

同一步两种TP的第一层输入一致，第一层注意力后规范化差异约0.002523；后续部分边界逐渐增长，final norm差异为0.266842。局部运算大体遵循各自操作数，仍不能排除其他位置的实现问题，或据此认定所有差异均属正常浮点误差。[完整边界及局部指标](../evidence/homework2_20261027/cached_step9_diagnosis.json)

## 输出头：差异已经出现在输入中

本次真实记录为matmul HF32=false、conv HF32=true、float32 matmul precision=high；这只描述本次采集，不倒推旧的未记录开关。输出头实际输入均能精确往返BF16，随后以FP32计算。

固定各自真实hidden，与同一checkpoint期望的BF16投影权重值做CPU FP64矩阵乘法：TP1自身参考relative-L2为4.6752e-7，TP2完整输出为4.7280e-7，max-abs分别约2.05e-5、1.95e-5。CPU参考分别重现TP1的25389、TP2的35378；两侧自身参考的跨TP relative-L2仍为0.12439306。该结果支持本步输出层的分叉跟随已有输入差异，而非仅由最终argmax或输出头计算造成。

没有导出实际输出头完整矩阵，`runtime_weight_identity_certified=false`；不能把期望checkpoint复算写成运行时参数身份认证。R1因Torch字典整数键与JSON字符串键比较而失败，日志保留；R2只规范化键比较，数学、输入与选择不变，两端45测试通过。完整向量NPZ已本地用独立NumPy重算误差、分片拼接和argmax；跨BLAS末位的复算容差不是模型验收门槛。

## GDN step8：排除所选输入上的负尾消减假设

使用此前已认证的GDN8–10共享BF16 norm权重和实际操作数，完成九组softplus诊断、十二个core/z输入替换组合。原11个job文件、6个源/模型文件以及核心清单前后SHA绑定；服务器52测试通过、零跳过。

九组实际`x=a+dt_bias`范围为[-11.25,0.9375]，没有出现runtime softplus消减成0、稳定公式仍大于0的元素。原FP32 `log(1+exp(x))`与稳定公式仍有末位差异；所选输入中最大`|Δg|=1.4901e-6`、`|Δdecay|=8.9407e-7`。不能据此排除其他步触发，也没有重跑递推证明最终影响为零。

| 层 | 只替换core的relative-L2 | 只替换z | 同时替换 | 实际norm TP2/TP1 |
| --- | --- | --- | --- | --- |
| 8 | 0.18213660 | 0.10036903 | 0.23176405 | 0.23165841 |
| 9 | 0.27778815 | 0.31694149 | 0.42422982 | 0.42414439 |
| 10 | 0.65643644 | 0.50049724 | 0.82139074 | 0.82117559 |

这些替换保留了输入的历史传播差异及norm、SiLU、乘积的交互，不能相加为因果比例，不能据此把根因归给某个算子。每个rank用自己的输入计算，九组最终BF16候选都精确匹配实际norm输出。[全部条件替换与逐head统计](../evidence/homework2_20261027/gdn_sensitivity.json)

本地不导入生产者数学模块，以纯NumPy从183个存档数组重算公式、四臂及逐head统计；十二臂FP32跨平台最大绝对差4.7684e-7，最终九组BF16候选仍精确相同。存档数组的离散指标/max-abs全部一致，relative-L2指标最大差4.44e-16。本地重读11份raw文件SHA，但未独立从`.pt`重导出操作数，也未在本地重读大checkpoint；raw→NPZ解释仍依赖绑定的服务器loader，范围如实保留。

## 规范化输出存储精度的单因素对照

R7–R9都仍把decoder规范化输出存成BF16，未测试此处的精度。R10相对R9仅取消这个写回：规范化及残差保留FP32，列投影接收FP32、按原R9方式计算后仍存BF16；行投影、权重、GDN/FA缓存及每模型输出头保持原候选合同。4B输出头仍为BF16，9B仍为FP32。它是外部数学覆盖，不是原生产数学，也没有提交为修复。

| 模型/实现 | TP2/TP1最大relative-L2 | 最大max-abs | 同前沿argmax一致 | 离线温度1最大TV |
| --- | --- | --- | --- | --- |
| 4B 原实现 | 0.30816710 | 6.046875 | 17/18 | 0.03122613 |
| 4B R9 | 0.13989776 | 2.386719 | 18/18 | 0.03118824 |
| 4B R10 | 0.26028707 | 4.750000 | 18/18 | 0.02105594 |
| 9B 原实现 | 0.38528013 | 12.304101 | 9/10 | 0.34141633 |
| 9B R9 | 0.27689970 | 9.842627 | 10/10 | 0.29836522 |
| 9B R10 | 0.23122590 | 10.140221 | 10/10 | 0.41334279 |

各最大值可来自不同步。温度1 TV只是离线诊断变换，不是原greedy采样分布。R10对9B的最大relative-L2下降，但最大TV上升；4B的最大relative-L2相对R9上升。不能据argmax更多一致而认为向量差异已解决，亦不能以其替换原失败，故未采纳。没有根据结果调整阈值或挑选样例。

R10最终四项在同card3串行成功、退出0、自身上下文释放；每rank每forward核对65个规范化边界、64个行投影和64个列投影的真实dtype及完整前缀，TP1的ReplicatedLinear路径也覆盖。norm日志没有输入dtype和显式step号，forward归属由严格连续日志块和固定前沿推导，不能写成额外观测字段。硬件工具服务器66测试/零跳过，分析器两端32测试/零跳过；controller与实际采集parent PID及baseline文件额外逐项交叉绑定。

前两次尝试保留：首次因原空闲chip新出现后台任务，在启动worker前选卡保护退出；第二次前三项成功、9BTP2因新的foreign进程进入而被共享guard中止并释放。最终重新执行完整四项，不拼接部分成功；没有杀死他人服务或放宽保护检查。

614份证据共264814036字节已归档、下载并逐SHA核对，包括两次中止。CPU分析导出原CPU、原R2、R9、R10共280个完整向量；本地独立NumPy重算588项配对指标，非浮点及top10完全一致，浮点字段最大差3.36e-15。复算容差1e-12只用于跨平台公式一致性，不是引擎验收门槛。[逐步结果](../evidence/homework2_20261027/norm_precision_r10.json)、[工具/退出/失败身份](../evidence/homework2_20261027/norm_precision_r10_provenance.json)、[本地复算](../evidence/homework2_20261027/norm_precision_r10_local_verification.json)

## 全文本层参数身份与原结果复现

原局部认证没有覆盖所有层，也没有认证旧step9任务的完整head矩阵。因此另在原生产数学上重复4B TP1/TP2、9B TP1/TP2，不采用R10，完整读取所有32个文本层、embedding/head及实际派生权重。每个rank观察412项：24个GDN层各13项、8个FA层各12项，加最终norm两项和embedding/head两项。其中81项是实际Gemma gamma、24项是实际conv_weights view；4B tied head虽与embedding别名，仍独立列出目标。这不是412个独立checkpoint参数。

每个大矩阵按256行分块只读复制到CPU，与checkpoint逻辑段及rank分片逐块字节比较，分别累积完整SHA，不导出矩阵。实际shape/dtype、4B tied/9B untied关系、norm epsilon、GDN模式和所有非visual学习参数对象覆盖也检查；checkpoint相关文件前后全SHA对齐冻结artifact。视觉分支实际已加载，4B每rank297个、9B每rank333个named参数条目逐名记录为未认证；每个armed文本forward给视觉根及全部子模块加拒绝hook并在finally移除，不能据此宣称视觉参数或多模态能力已认证。

四任务在card3串行退出0、各自资源释放。六个rank观察共2472项全部exact；84个原完整logits的uint8字节与原R2基线相同，case/step/rank/输入/位置/原dtype及采样记录保持。完整模型仍复现原4B/9B差异，因此本批证据排除的是文本学习参数/派生视图的加载、精度投影与切分错误，不能自动排除算子、状态或传播问题。

9B新TP1实际head的FP32字节SHA为`2804d09fe06a4da7cf9e876959cc8f15399cd77ad46a88e8621a4ca6351521d0`，与此前独立CPU计算的checkpoint期望FP32 head一致。它认证新重复执行的实际参数，不能追认为旧step9实例曾导出完整矩阵或有参数认证。

工具服务器29测试/零跳过。首次R1因把visual参数纳入全模型覆盖要求而fail closed，退出-10、自身释放，未得数值结果；工具范围错误和失败日志保留，R2才明确文本scope，不混拼。新429份证据共103976100字节全下载SHA核对。额外CPU分析交叉绑定collector PID、baseline和guard，并重新核对核心源前后；本地纯NumPy重算六组84向量，与另一独立导出器已绑定的原R2数组逐uint32精确相同，重新核对2472项ledger、配置和checkpoint artifact映射。实际NPU参数字节与checkpoint的比较在服务器完成，本地没有重读大checkpoint或从raw PT重新导出值。

[摘要与范围](../evidence/homework2_20261027/full_text_parameter_identity.json)、[完整2472项ledger和分析源](../evidence/homework2_20261027/full_text_parameter_ledger.zip)、[工具/归档/失败身份](../evidence/homework2_20261027/full_text_parameter_provenance.json)。没有新增模型通过线，原greedy失败及独立语义质量问题均保留。

## 数据与复现

[工具与归档身份](../evidence/homework2_20261027/cached_step9_provenance.json)记录142份实际采集文件及51份CPU诊断/失败日志的归档，均已下载到项目工作区并逐文件核对。公开目录只保留去除私有服务器绝对路径的指标与工具；原张量、完整NPZ和失败日志保留在工作区归档，公开摘要不能代替原输入。

四份工具zip分别对应实际观测、严格step9分析、输出头R2、GDN敏感性；各自包含工具manifest。解压后先核对SHA、运行其pytest，再按入口`--help`显式传入还原归档和源快照。输出头矩阵复算与step9张量分析需要匹配的CPU Torch；`independent_numpy_checks.zip`中的本地检查只需NumPy，GDN检查支持显式bundle/raw/source/output参数；head检查沿用归档在原工作区的目录布局。独立脚本及原结果身份见[复算记录](../evidence/homework2_20261027/independent_checks_provenance.json)。

上述局部复核未修改生产实现；原4B/9B失败、作业1冻结门槛、既有共享性能和Profiler的源码归属均保持。
