# 9B候选LM head输出精度

## 证据与目标

原八条固定native输入、两组eager单变量试验：BF16 head对同一新采HF BF16参考为7/8，FP32 head为8/8。原long-context首差处，真实hidden/权重的独立FP64点积偏好6278，但两候选在BF16输出中舍入成30.25并列；同投影CPU FP32/eager也偏好6278。该证据支持保留最终投影输出精度，不证明所有模型或所有长度均正确。

目标是在总项目候选配置中显式启用该路径，保持原baseline和HW1已发布提交不动。用户已授权继续实施、实验及源码交付；本项是可逆的候选实现，不改变冻结判据或选择验证样例。

## 方案取舍

1. 采用：复用现有 `--hf-overrides`，9B候选配置显式给出 `npu_lm_head_dtype: float32`。权重完成原BF16加载后精确展开为FP32，仍使用原logits matmul和采样路径。
2. 不采用动态tie refinement：每步候选数与控制流会变化，需要处理graph及批不变性，复杂度高，也不保留整个词表的输出精度。
3. 当前NPU栈未提供可直接调用的 `torch_npu.npu_mm` FP32输出入口；不引入未知依赖或假定存在的API。完整head FP32额外约1.895GiB权重存储，须实测系统HBM/性能。

## 接口与边界

- 课程配置可带可选字典 `hf_overrides`，启动器逐项原样序列化到已有CLI参数，类型错误明确拒绝。
- override缺省表示保持现有模型行为；显式 `float32` 才启用新路径。不新增全局默认或改变baseline配置。
- 当前实现仅支持NPU上未量化、非共享embedding/head的dense Qwen3.5文本推理。其他设备、dtype值、模型、encoder-only/无LM head、量化和共享权重组合明确报错，不能忽略已提供选项。
- 原Parameter对象、loader元数据与已投影数值保持；只替换其精确FP32表示。转换在WeightLoader的weights内存生命周期区域完成，在缓存预算、warmup和graph捕获之前。
- 不新增第二条执行/采样路径，不查看参考token，不进行逐样例或逐位置选择。

## 验证与交付

先测试配置传递及错误输入，CPU真实BF16近并列投影复现、FP32排序恢复、参数/数值保持、unsupported组合拒绝、weights区域内集成。服务器验证正式源码的course集合、9B原八条native输入、graph与Session；并补无Profiler性能/HBM。所有失败保留，0.8B门槛不用于9B。候选不能在数据完成前命名为最优配置。
