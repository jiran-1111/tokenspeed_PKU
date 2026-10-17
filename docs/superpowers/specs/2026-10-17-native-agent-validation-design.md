# 原生16轮双Session与工具等待验证

当前基线是已发布R6 `3f4906aa`；本功能先验证4B，再根据实际结果扩展其余模型，不把旧结果当新源码实验。固定工作负载：2个独立Session、各16轮、每轮8个输出token，Session-0在第8轮输入前执行一次受控2s工具调用。工具实际计算sum(range(31))，将结果作为明确标注的脚本化工具输入，不宣称模型自主选择或生成了工具调用。初始上下文用固定重复文本达到至少512 tokens，增长过程每轮记录实际长度。

## 选择与边界

沿用原生Engine和C++ SessionIdle生命周期，Session保留原始生成IDs，下一轮只追加ChatML用户输入及从真实模板抽取的generation prefix。相比每轮重渲染全部消息，这不会裁剪空白或旧reasoning；相比增加另一套调度器/WAITING_TOOL枚举，不改生产调度路径。旧两轮helper及历史脚本保留，新增多轮验证入口。

每轮同时提交Session suffix和独立stateless完整原始历史对照，输出ID必须完全一致，元数据严格符合current-frontier（保留最后采样token尚未物化）。两路对照共享NPU资源，时延只用于验证记录，不作为Session独立性能加速结论。任一分叉保留完整输入/输出及已完成记录并报错，不能调容差或替换已冻结输入。

工具阶段在用户侧执行受控等待，但服务端使用真实原生Session持有KV；采集Prometheus LCM块计数及其他会话实际请求完成时间，随后检查该Session恢复的reused_context_tokens。此证据明确称“SessionIdle期间KV保留及其他会话进展”，不等于新增服务端WAITING_TOOL状态，也不把客户端mock wait单独当KV证据。块数是物理LCM parents，active包括仍持页的idle Session，不等于正在计算的请求数或HBM字节。

## 成功判据与实施约束

- 固定32轮配对全部输出逐ID一致，32组context检查通过，原始历史只追加，全部输入在4096限制内。
- 工具真实结果465、等待期间至少一个其他Session请求实际完成，恢复后复用长度正确且大于0；记录原始事件和采样，不能由软件fake宣称硬件通过。
- 请求完毕而Session仍打开时观察running=0、持有LCM块>0；关闭全部Session后检查active归零（prefix cache/KVStore关闭）。指标缺失、异常或超时明确失败，不填0。
- 软件测试覆盖真实prefix提取、尾空白/闭合reasoning/EOS、不支持模板、16轮原始历史追加和失败清理；实际设备运行前固定源码/workload清单并做服务器全量匹配。

所有环境和输出留工作区，固定同一物理卡、实验串行、每次实时映射/空闲及端口检查，保存失败。不增加依赖，不复写已发布HW1/R5工作树。工具等待2s不是CPU性能数据，受控工具也不是外部网络真实性测量。
