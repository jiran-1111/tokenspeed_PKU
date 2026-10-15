# Baseline 分析

当前基线定义为固定 TokenSpeed 上游提交上的最小 Ascend 推理路径。continuous batching 属于 scheduler 基础能力，不能通过课程配置关闭；E0 关闭 mixed-batch 同轮调度、overlap scheduling、chunked prefill 与 prefix caching，保留上游 scheduler。

课程服务器提供的是 Ascend910 V1；课程原要求为 910C，课程方是否接受替代尚未确认，因此不能写成 910C 实测。Qwen3.5-0.8B、4B、9B 均已完成 TP=1 的首轮普通并发基线：目标输入/输出 128/32 tokens，并发 1/4/8/16，每档 3 轮、每轮 32 请求；每模型共 384 条请求，使用独立 prompt，prefix-hit ratio 为 0。4B 和 9B 详细中位数与容量限制见 [`../project/final_results.md`](../project/final_results.md)；0.8B 基线的定义、逐档数据与限制见 [`../project/baseline.md`](../project/baseline.md)。

这些数据只描述本次 Ascend910 V1、TP=1 和固定工作负载，不构成优化收益或普适容量结论。4B/9B 在 concurrency 8/16 时 KV 峰值达到 1.00，且服务 `max_num_seqs=8`，高并发请求存在排队；三模型首轮历史 summary 的 ITL 仍为 null，但 0.8B/4B/9B 均已单独完成运行时 ITL 口径修正后的复测，定义和结果见 [`../project/baseline.md`](../project/baseline.md)。27B 本机与远端 11 个权重分片均已通过固定 manifest 大小/SHA-256 校验并晋升到正式模型目录；NPU bring-up、正确性和性能验证仍未完成。
