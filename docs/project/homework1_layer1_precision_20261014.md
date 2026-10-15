# 作业1：第1层同输入数值重放

## 协议

固定0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`、128-token BF16输入、FP32 SSM state，在空闲logical4独立进程重放layer1。使用已存HF输入和临时GDN+MLP+MHA组合输入，各执行HF与临时candidate两条路径；不修改生产实现、不测性能、不更改原完整logits合同。

原始结果在`results/raw/homework1_layer1_replay_r2_20261014/`。两条对应原捕获的路径完整输出均精确复现：HF输入/HF实现对HF layer1输出；candidate输入/candidate实现对临时引擎layer1输出。各源的HF手工组件组合也与直接调用HF decoder完全一致。

## 相同输入下的实现差异

两套输入分别对照时，input norm、QKV/Z/B/A投影、卷积+SiLU和beta均全元素精确一致。

| 边界 | HF输入 relative-L2 | candidate输入 relative-L2 |
|---|---:|---:|
| g | 6.1831e-8 | 6.2024e-8 |
| GDN core | 3.9188e-6 | 3.6024e-7 |
| FP32最终state | 2.5021e-7 | 2.5611e-7 |
| gated norm | 5.3192e-6 | 2.7648e-6 |
| GDN输出投影 | 6.0556e-5 | 3.6147e-5 |
| post-attn norm | 5.5735e-5 | 3.9436e-5 |
| MLP输出 | 0.000508322 | 0.000474521 |
| 完整layer1输出 | 0.000334863 | 0.000323311 |

这里candidate只代表已采集临时兼容路径，不能把较小误差冒称生产版本数值验收通过。core块大小、归约/SFU路径等仍不同；有限精度差异小不代表所有输入、连续Decode或全部层已验证。

## 输入传播与本层新增差异分离

采用完整向量分解，而非比较norm摘要：

`candidate(candidate_input) − HF(HF_input)`

`= [HF(candidate_input) − HF(HF_input)] + [candidate(candidate_input) − HF(candidate_input)]`。

FP64分析中此向量等式逐元素精确成立。统一用HF原输入输出的norm作分母：

| 边界 | 总差异 relative-L2 | 仅输入变化效应 | 同输入实现效应 |
|---|---:|---:|---:|
| input norm | 0.000369143 | 0.000369143 | 0 |
| GDN core | 0.000888055 | 0.000888055 | 0.000000360 |
| GDN输出 | 0.001788430 | 0.001788272 | 0.000036147 |
| MLP输出 | 0.003833034 | 0.003826426 | 0.000474564 |
| 完整layer1输出 | 0.002476196 | 0.002471464 | 0.000323325 |

两个分量的norm不能相加，也不能转成总误差的可加百分比；raw记录包含交叉项cosine。该样例中本层总差异主要表现为输入扰动经过同一HF计算的放大，并不支持将0.002476直接归因本层GDN算法错误。不能据此自动豁免原全模型1%失败。

## 失败与复现边界

第一次诊断脚本未向HF Decoder.forward传必需的position_embeddings，调用前失败，未采张量。完整检查已安装Transformers 5.12实现后，只对断言为linear_attention的层传None，Norm读取实际HF属性eps；生产代码未改。首次失败目录`homework1_layer1_replay_20261014/`保留原脚本和failure.json，修正版采用新独占目录，不覆盖失败。

本次22956退出0，参考实现使用现有torch fallback，未安装可选依赖。新鲜设备查询确认四物理NPU无运行进程。原完整模型、连续Decode、最终性能/HBM热点和阶段报告仍需验收，四模型Agent总目标不变。

归档终态：限速同步19992退出0，成功目录65文件及失败目录2文件，共67文件跨端SHA-256全部一致。本机项目Conda课程测试415通过、13跳过（8.17s），不是本实验或全模型NPU验收的替代证据。
