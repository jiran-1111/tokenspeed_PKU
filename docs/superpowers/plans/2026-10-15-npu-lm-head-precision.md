# NPU LM Head Precision Implementation Plan

> **For agentic workers:** Use executing-plans to implement sequentially; this session does not delegate further work.

**Goal:** 显式9B候选配置保留LM head FP32输出，基线与已发布HW1不变。

**Architecture:** 复用hf-overrides；在WeightLoader的weights区域内进行同值展开，沿用既有matmul、通信与采样。

**Tech Stack:** Python、PyTorch/torch_npu、既有TokenSpeed配置和pytest；不新增依赖。

- [x] 为 `test/course_project/test_npu_lm_head_precision.py` 添加配置、拒绝组合、参数同值/排序及内存生命周期测试，RED确认后GREEN；服务器13项通过。
- [x] 在 `benchmarks/course_project/core.py` 验证/传递可选hf_overrides，不丢弃调用者参数。
- [x] 在 `python/tokenspeed/runtime/execution/weight_loader.py` 加最小的显式策略及weights区域调用，不改变缺省路径。
- [x] 新建 `configs/candidates/qwen35_9b_fp32_head.yaml`，只在baseline上添加明确override。
- [x] 专项及完整软件回归，完整格式检查通过后固定R5清单1398项；服务器586通过/零skip，部署前后逐项一致。
- [x] 正式源码9B原8输入回归、graph、Session，排除诊断hook，保留所有失败。
- [x] 3轮交叉成本矩阵性能/HBM：6有效臂96请求/252采样通过审计，首轮解析及末臂退出采样失败保留，后者仅补跑该臂并公开原因；数据92项逐项SHA通过。
- [ ] 报告明确混合精度、额外内存、覆盖边界；与独立HW1发布分开按allowlist交付。
