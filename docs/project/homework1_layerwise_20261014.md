# 作业1：共同前缀逐层诊断

本实验细化Task3，不改变完整logits验收合同、模型dtype或性能配置。固定native128输入、最后位置127/ID271，0.8B固定revision、BF16/TP1，单logical2引擎；HF参考在独立logical4。只作数值诊断，不用于性能，不采Profiler。

## 第一轮边界证据

原始目录`results/raw/homework1_layerwise_20261014`。临时opt-in脚本在既有位置/输入ID guard内采引擎24层输入和输出：TP1按原dtype重组hidden+residual，不把未相加MLP输出误作完整decoder输出。另保存最后一个token的final norm。HF对应24层输入/输出、final norm及完整logits共50个边界。每边界保存原shape/dtype、展开FP32数组及SHA-256。

49个配对边界shape均一致；引擎完整logits与上一轮Gamma/QK修正版逐字节相同，HF完整logits与原参考逐字节相同。这说明此128-token采集未改变最终输出，不证明所有场景无扰动，更不能外推时延。

| 边界 | 全部元素relative-L2 | 最后token relative-L2 | max-abs |
|---|---:|---:|---:|
| layer0 input | 0 | 0 | 0 |
| layer0 output | 0.00615889 | 0.00829201 | 0.0078125 |
| layer1 output | 0.00775765 | 0.01122113 | 0.015625 |
| layer3 output | 0.01034528 | 0.01542477 | 0.0126953125 |
| layer22 output | 0.02395081 | 0.02678185 | 0.234375 |
| layer23 output | 0.02688764 | 0.02709725 | 1.015625 |
| final norm | 0.02803021 | 0.02803021 | 0.5 |

layer0输入131072/131072元素exact，差异已在layer0输出出现；这排除了该输入的embedding内容不一致，但不能区分layer0 GDN、MLP或中间舍入。不能从累计误差曲线指认具体kernel。

## 采集问题与保护

首个客户端检查发生在gateway尚未ready时，连接拒绝；服务仍在初始化，未重启。随后错误地直接访问gateway `/generate`且无model字段，返回500 Tokenizer model unknown，未进入引擎；改为项目既有`TokenSpeedGenerateClient`通过control port加1调用，成功采集49边界与输出8160。两次失败都没有产生可用模型数据，不能计入正确性样本。

HF第一轮50个数组保存后，metadata读取`model.config.text_config`失败（实际是flat Qwen3_5TextConfig），原空`protocol.json`保留。随后逐文件核对manifest及原参考logits，补`protocol_recovered.json`，不重复计算或覆盖数组；恢复记录明确元数据失败。第二轮测量脚本使用`get_text_config()`，原失败不是模型forward失败。

第一轮服务仅停止自有PID92033，设备确认无运行进程后才启动第二轮新目录；不清理历史raw，不安装新依赖。最终跨端校验及第二轮结果继续补充，不提前称报告完成。

## 第二轮边界设计

新目录`results/raw/homework1_layerwise_r2_20261014`，保持输入、dtype、单chip/配置相同，补layer0 input norm、GDN输出、post-attention norm、MLP输出四个边界。新增hooks在调用后移除，只在guard armed时保存。两项本机无依赖测试先RED再GREEN，验证guard、残差重组、tuple结果和边界命名；本机测试不证明硬件结果。HF测量脚本和临时hook均归档到raw，以实际数据定位最早差异。

第二轮采集成功：引擎53边界、HF54边界、53组shape一致且逐文件校验manifest；两边完整logits再次与既有各自参考逐字节相同。HF测量脚本退出0，模型attention implementation按实际protocol记录；没有把第一轮metadata失败掩盖为成功。

| layer0内部边界 | 全部元素relative-L2 | 最后token relative-L2 | max-abs |
|---|---:|---:|---:|
| input norm | 0 | 0 | 0 |
| GDN输出投影后 | 0.00465781 | 0.00425934 | 0.0078125 |
| post-attention norm | 0.00587684 | 0.00767288 | 0.0625 |
| MLP输出 | 0.00958273 | 0.01139011 | 0.0048828125 |

layer0输入及input norm均131072/131072 exact。最早已观测到的差异在GDN模块内或其输出投影，不能指认为MLP独立错误，因为MLP输入已经不同；input norm在该输入中已排除。下一步进一步检查layer0 GDN投影、卷积、core递推和gated norm的边界，对齐实际张量，而非直接修改多个舍入位置。

第一轮112个文件跨端SHA-256核验一致；新增诊断脚本的两个无依赖测试包含在本机全套415通过、13跳过（8.15s）中。第二轮跨端终态核验在服务安全释放后补齐，未把仍变动的service.log哈希当最终值。

结束核验：第二轮自有服务PID95628及父95301均已退出，启动命令句柄16674退出0；新鲜`npu-smi info`四个物理NPU均无运行进程。停止后第二轮120个文件与远端SHA-256全部一致，包括终态service.log、模型源码快照、HF测量脚本和数组。全模型数值合同仍未通过，无新生产实现修正；下一轮可从保存的layer0 GDN输入开始细分诊断，不需要重新采集已经完成的49/53边界。
