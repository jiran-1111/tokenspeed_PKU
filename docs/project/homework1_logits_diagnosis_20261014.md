# 作业1：完整logits诊断与局部精度修正

## 结论与边界

共同前缀128 tokens的首个输出完整logits尚未通过[采集前合同](homework1_logits_contract_20261013.md)。局部Norm/QK修正有独立真机证据，但不是完整模型修复。未改合同、未删除失败记录；旧性能不能标成修正后性能。

模型为Qwen3.5-0.8B，revision `2fc06364715b967f1860aea9cf38778875588b17`，BF16/TP1，Ascend910 V1单个logical2；输入复用native精确128记录，末位置127、末输入ID271。向量248320维；保存FP32只是展开BF16值。HF参考用相同输入、模型、dtype，NPU Torch fallback。诊断会同步D2H，不能用于性能统计。

| 版本 | max-abs | relative-L2 | 固定合同结果 |
|---|---:|---:|---|
| 原计算实现 | 0.3515625 | 0.0240672510 | 不通过（L2超过0.01） |
| Gemma FP32 gamma + Q/K计算修正 | 0.375 | 0.0241289008 | 不通过（L2超过0.01） |

原引擎前两候选8160与71093保存logit均23.25；修正后均23.375。HF两候选71093/8160为23.375/23.25。两次引擎实际argmax均8160，返回采样ID一致。并列时`topk`排序并不稳定，不能把一次topk顺序当成实际argmax，也不能用近并列豁免完整向量误差。

## 独立确认的实现缺陷与修正

固定安装版本HF `Qwen3_5RMSNorm.forward`先在FP32归一化并乘`1 + weight.float()`，最后转换输入dtype。原NPU运行时预先以BF16做`weight + 1`，且残差分支把gamma转换回BF16。加载值0.003的确定性测试先复现gamma dtype/数值失败，再修正有效gamma为FP32；残差在原存储精度相加，单独调用norm，保持原residual对象及BF16输出。

该修正导致原Q/K fallback的末端乘法提升为FP32，下一步BF16 V的注意力内核无匹配组合；中间启动失败已保存。Q/K norm改为FP32归一化与gamma相乘后显式转换回原Q/K dtype。不能把中间失败删除后称首次启动即成功。

审查发现显式allreduce fusion辅助分支仍调用BF16输入/残差与FP32 gamma的`npu_add_rms_norm`。新增真机回归复现AclNN dtype不支持；修正仅在NPU且gamma与输入dtype不同时使用原位残差相加与非融合norm，其他平台/普通同dtype路径不变。该测试只替换通信为输入乘二，实际norm仍在NPU执行；它不证明真实多rank HCCL或TP模型验收。

## 测试与数据版本

- 原gamma定向RED、随后74项模块GREEN：原目录`results/raw/homework1_logits_20261013`。Q/K定向RED与12项旧kernel回归GREEN位于`homework1_logits_gammafix_20261014`。这些早期模块结果不替代本次整模型失败。
- `allreduce_gamma_red.xml`：1失败、2通过，真实失败为mixed-dtype fused add norm。
- `review_green_r2.xml`：76通过，包含3项canonical gamma/辅助分支、70项实际Norm/MRoPE、3项Q/K。Norm参考更新为canonical FP32 gamma；packed非连续Q/K有确定性精确BF16检查，残差身份也有检查。固定绝对/相对包络未放宽。
- 首次组合测试错误地引用不存在的`test_gemma_rmsnorm.py`，未执行任何测试，记录`review_green.xml`保留；正确入口是`test_gemma_rmsnorm_npu.py`。该空测试结果不算通过证据。
- 本机项目测试最新413通过、13跳过；本机无Torch，不能据此声称NPU runtime测试在本机执行。
- 两个新raw目录合计16文件与远端SHA-256一致；5个修改源码/测试按仅归一化CRLF核验跨端一致。原目录10文件此前单独核验。后修正模型目录保存当时runtime/kernel源码，避免用后来的allreduce修改冒称采集版本。
- 自有服务81653正常退出0，已确认四个物理NPU均无运行进程。退出日志含服务取消期间的`asyncio.CancelledError`，不是采样数值失败；警告如实保留。CANN共享目录所有者警告以及pytest xunit2 properties兼容性警告亦保留，XML实际76 tests及properties可读取。

## 下一步

先逐层对齐同一原生输入的HF与引擎中间结果，寻找最早扩大的差异；关注GDN的BF16舍入边界、gate/norm和投影，但它们当前只是源码差异候选，不能当已证明根因或直接批量修改。完整模型、连续Decode、最终HBM峰值及热点报告仍待验收。原四模型Agent项目不删除；作业1完成后继续其下一步。
