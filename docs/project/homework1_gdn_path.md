# 作业1：0.8B GDN模型到算子路径

本记录依据固定0.8B模型配置，2026-10-14对照当前源码更新，TP1，描述计算/状态合同；不是新的性能归因结论。

1. `Qwen3_5LinearDecoderLayer.forward`通过CommManager处理输入残差与Gemma RMSNorm，向linear_attn传hidden[N,1024]及ForwardContext。TP1不需要跨chip归约；不把CPU调度成本算作GDN算子时间。
2. 模型linear attention的 `_forward_input_proj` 用上游合并线性层投影QKVZ/BA。0.8B Q/K/V/Z各宽2048，BA为32；`fused_qkvzba_split_reshape_cat_contiguous`拆成mixedQKV[N,6144]、Z[N,16,128]及a/b[N,16]。布局不是full-attention的Q/gate按head交错布局，二者不能共用拆分假设。
3. `ctx.attn_backend.forward`传入mixedQKV、卷积weight、a/b、head维度、layer_id及状态池，Mamba backend按forward mode选择extend/decode。卷积Prefill采用packed序列边界元数据，Decode采用slot索引更新；conv width4意味着每个slot保留3个历史位置，形状[P,6144,3]。
4. 卷积后的Q/K/V拆分为Prefill[B,T,16,128]或Decode[B,1,16,128]。GDN执行Q/K L2归一化、decay与delta-rule递推。Prefill明确使用 `beta=b.float().sigmoid()`（NPU）及`fused_gdn_gating`，调用`gdn_chunk_prefill(...,scale=128**-0.5,output_final_state=True)`。最终状态必须写回当前请求对应的state槽；不能只验证输出finite。
5. Decode `_prepare_gdn_decode_state_path`接收read/write indices，随后 `gdn_decode_step` 使用 `initial_state_indices` 和独立 `output_state_indices`。读页代表n−1状态、写页代表n状态，不能覆写另一个请求/共享前缀的输入页。SSM存储形状[P,16,V=128,K=128]，与CPU参考常用[K,V]矩阵方向相反；转置是数值对照必须显式处理的布局变换。
6. Decode内核返回[B,1,16,128]，backend转置为[1,B,16,128]；模型将输出与Z flatten到[N×16,128]做gated RMSNorm，然后reshape[N,2048]，通过out_proj回到[N,1024]并进入残差/MLP路径。

## 同步与状态边界

状态/page metadata由scheduler发布、backend消费；是否读写不同slot取决于实际计划，不能仅用Python会话名证明隔离。诊断时的CPU拷贝/`.item()`/top-k保存会引入同步，因此诊断运行不得混入性能基线。常规Prefill元数据路径注释明确避免每层同步D2H；不能由此声称整个服务没有同步。以实际时间线为准。

完整chunk可形成状态checkpoint；最后生成token尚未forward时，其KV/SSM并未物化，current-frontier Session语义必须如实说明。HTTP全历史重发与原生Session KV留存是两种不同协议，均需说明重置及状态归属。

## 复用与本组修改

2026-10-14明确Norm适配：进入GDN前的GemmaRMSNorm以`1 + weight.float()`形成NPU有效gamma；残差在原BF16对象原位更新，再调用非融合norm，避免将gamma先舍入BF16。实际调用是既有NPU RMSNorm，并非CPU计算。full-attention另走`_apply_qk_norm`→NPU `qk_rmsnorm`，FP32中间计算后恢复Q/K输入dtype；这不是GDN Q/K L2归一化，不能混称同一模块。最新76项Norm/QK回归及原完整logits失败见[局部修正诊断](homework1_logits_diagnosis_20261014.md)。临时GDN/MLP/MHA中间舍入组合不属于这里的生产路径。

模型结构、线性层、GDN/conv派发与scheduler为上游复用。当前项目已有Q/gate布局、D256 MHA、NPU Norm兼容、GDN精度及状态路径、缓存容量/上下文边界等适配；最终阶段报告需引用具体diff和数值测试，不能将整个GDN实现称为本组从零开发。2026-10-13真实16-head多步/slot隔离与卷积NPU测试10项通过，见 `homework1_numerical_validation_20261013.md`；这仍不替代完整模型/引擎请求隔离，误差依据及其他真实模块形状的剩余限制见 `homework1_numerical_contract.md`。

源码：`models/qwen3_5.py`、`layers/attention/backends/state/mamba.py`、`layers/attention/configs/linear_attn.py`、`layers/attention/kv_cache/recipes/qwen35.py`、`layers/attention/linear/causal_conv1d.py`与`layernorm_gated.py`，均位于`python/tokenspeed/runtime/`下。
