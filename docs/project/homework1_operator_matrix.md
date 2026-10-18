# 作业1：0.8B计算模块与验证矩阵

2026-10-13从远端本地模型 `models/Qwen3.5-0.8B/config.json` 读取text_config，固定revision `2fc06364715b967f1860aea9cf38778875588b17`；2026-10-14对照当前源码和归档XML更新支持/覆盖状态。单逻辑chip/TP1/BF16，实际SoC为Ascend910_9362，V1是工具版本字段，详见[硬件身份](hardware_identity_20261014.md)。下表描述已查源码支持与现存测试；**不代表最终冻结版本所有测试均已在NPU复验**。

## 几何与记号

hidden=1024、MLP intermediate=3584、24 decoder layers，每四层最后一层full attention，共18 GDN/6 full。Full attention：Q heads=8、KV heads=2、D=256。GDN：key/value heads=16、K=V=128、conv width=4。vocab=248320、RMS eps=1e-6。

N为本次flatten后的实际计算token数，B为Decode batch，T为单会话Prefill token数，P为分配的状态/page槽位数。内核边界可能采用[B,T,H,D]或扁平布局，不能把N误作固定最大上下文。下表权重矩阵以数学方向表示，不保证等同实际张量stride/存储方向。

| 模块/功能 | TP1主要输入→输出/状态 | 精度 | 实现来源及Ascend支持 | 现有测试/需补覆盖 |
|---|---|---|---|---|
| Embedding/LM head | IDs[N]→[N,1024]；采样hidden[B,1024]→logits[B,248320] | 权重/hidden BF16；已捕获首token logits为BF16，保存时展开FP32 | 上游Qwen3.5及并行embedding/linear，已参与0.8B NPU推理 | native-ID多长度记录和完整logits已有；[首token全向量](homework1_logits_diagnosis_20261014.md)失败原1%工程合同，不称端到端通过 |
| Dense/MLP投影 | [N,1024]→gate/up各[N,3584]→[N,1024] | BF16输入/权重，底层累积精度由实现决定 | 上游并行linear与激活dispatch，NPU矩阵运算；不声称自写GEMM | 最终版本模块参考及代表N=1/128/1024复验 |
| Gemma RMSNorm/残差 | [N,1024]及可选残差→同形状；Q/K norm末维256 | BF16输入/输出/残差存储；有效gamma在FP32计算1+weight，归约FP32 | runtime `layers/layernorm.py` Gemma构造/loader和NPU residual路径已修改；调用既有NPU RMSNorm，不自写归约内核 | `test_homework1_norm_rope_npu.py`真实256/1024与rows0/1/129/1024；`test_gemma_rmsnorm_canonical_npu.py`验证FP32 gamma、残差对象和mock归约辅助路径；[76项最新回归](homework1_logits_diagnosis_20261014.md)不等于TP通信独立验收 |
| Q/K Norm、RoPE、Q输出gate | packed投影数学宽度5120；Q[N,8,256]、K/V[N,2,256]及Q gate[N,8,256] | Norm FP32中间计算后恢复原BF16；实际MRoPE使用rotary64、未旋转192维 | `qwen3_5.py::_split_q_gate_by_head`及`_apply_qk_norm`→NPU `ops/layernorm.py::qk_rmsnorm`→runtime MRotaryEmbedding；已修正Q/gate拆分和Q/K末端dtype | Q-gate布局、D256 MHA回归、实际MRoPE；`test_qk_norm_fp32_gamma.py`覆盖rows1/129及packed非连续Q/K，数值/输出dtype单独检查 |
| Paged MHA Prefill/Decode | Q[B,T,8,256]与KV[B,T,2,256]→[B,T,8,256]；KV每逻辑prefix块[128,2,256] | KV BF16，计算细节以kernel为准 | `backends/paged/mha.py`→NPU `ops/mha.py`；本组补D256兼容及paged/graph路径 | 原`test_mha.py`回归；新增`test_mha_homework1.py`实际D256、8Q/2KV、127/128/129跨页Extend/Decode六NPU项及三CPU解析项通过，独立CPU FP32参考，七个输入精确不变；不外推D256 graph |
| GDN输入投影/拆分 | [N,1024]→QKVZ[N,8192]、BA[N,32]；mixedQKV[N,6144]、Z[N,16,128]、a/b[N,16] | BF16输入/权重 | `qwen3_5.py::_forward_input_proj`及split/reshape；上游Triton-Ascend可执行路径 | 源码几何确认，真实输入布局数值复验需补 |
| 短因果卷积 | mixedQKV[N,6144]→同宽；weight[6144,1,4]；slot conv state[P,6144,3] | BF16 state/input；实际输入诊断记录FP32累积及激活量化边界，不外推任意数据 | runtime `linear/causal_conv1d.py`中的Triton实现，通过项目Triton-Ascend加载；已参与模型推理 | `test_causal_conv1d_homework1_npu.py`通道16/6143/6144、width4、两slot各8步及未选slot；**CUDA-only通用conv测试不算NPU验收**；[实测](homework1_numerical_validation_20261013.md) |
| GDN gate与递推 | Q/K/V[B,T,16,128]、a/b[B,T,16]→output同V；SSM池[P,16,128,128]（V,K顺序） | 模型BF16；NPU Prefill beta显式FP32 sigmoid；SSM dtype由配置/环境指定，现有启动FP32 | `backends/state/mamba.py`→`tokenspeed_kernel.ops.attention.gdn` dispatch；本项目NPU精度/状态路径适配，不重写整个算法 | `test_gdn_homework1.py`真实16-head、Prefill1/8/127/128/129、slot交错各8步与读写分离；[真实训练输入FP32/FP64递推](homework1_gdn_trained_oracle_20261014.md)另补单样例core/state，不替代全模型 |
| Gated RMSNorm/输出投影 | GDN输出和Z flatten[N×16,128]→gated norm→[N,2048]→[N,1024] | BF16输出，norm/gate FP32内部运算 | runtime `linear/layernorm_gated.py` NPU路径及model out_proj；现有本组适配 | `test/runtime/layers/test_layernorm_gated_npu.py`：PyTorch FP32公式参考；需真实宽度/组设置复验 |
| Greedy采样/输出 | logits[B,248320]→IDs[B]→detokenization | 整数IDs，logit精度单独记录 | 上游greedy sampler；日志/token对照保留近并列分叉 | 重复推理、stop口径、共同前缀logits；不将BF16近并列自动豁免 |

## 测试入口与边界

真实路径补充：0.8B `get_rope`返回runtime MRotaryEmbedding，head256、rotary64、theta1e7、sections11/11/10、interleaved=true；模型NPU `_project_qkv_rope`先分别Q/K Norm再调用该模块。并非通用npu_mrope入口，旧head128通用测试不能替代。新 `test_homework1_norm_rope_npu.py` 直接测试此路径、文本/三轴不同位置及未旋转192维精确不变，连同width256/1024的standard/Gemma Norm、FP32收紧容差，共70项NPU通过，数值见验收记录。主机矩阵乘/CPU参考不被当作模型主计算设备。

2026-10-13已完成新GDN/实际runtime卷积10项、既有kernel30项、runtime/布局12项复验，见[数值记录](homework1_numerical_validation_20261013.md)。2026-10-14最新Norm/QK组合回归为76项：70项实际Norm/MRoPE、3项canonical gamma/辅助分支、3项Q/K。本轮重新解析四份归档XML确认76/10/30/12 testcase均无failure/error/skip，分别76/92/0/0条properties；没有重新执行硬件测试。局部覆盖和有限Session隔离已有，不将当前源码所有模块或最终版本复验标成全通过。

## 生产修改与临时诊断边界

后续审计补充：旧GDN Prefill测试及真实输入oracle使用clamp归一化，而生产Prefill/Decode与固定HF都用平方和加eps。旧通过记录保留，近epsilon的参考边界仍需纠正/补测；不把历史通过当作两种定义完全相同。已有真实输入的加eps CPU FP64只读核对更接近既存NPU core/state，见[参考语义审计](homework1_gdn_reference_audit_20261014.md)；没有修改生产或测试，也不是新校准正式实验。

2026-10-15后续：两个测试参考已按sum+eps纠正，四epsilon比例CPU解析式及13项NPU正式复验共17项全部通过，原包络/生产不变。上段“仍需纠正/没有修改”仅指原审计时点，不是当前状态；详见同一语义审计的终态记录。

当前Gemma路径在加载后保留FP32有效gamma；有残差时先在原BF16存储原位相加，再调用NPU非融合RMSNorm，保持residual身份。NPU allreduce辅助分支仅在gamma/input混合dtype时同样拆开残差与norm，其他同dtype/平台路径保留；其定向单测mock通信，仅norm实算NPU。Q/K路径在FP32归一化并乘gamma后恢复输入dtype，以避免FP32 Q/K与BF16 V不兼容。两份当前Norm源码的SHA与当前性能基线归档源码快照一致，本轮未改生产代码。

临时GDN/MLP/MHA BF16中间边界组合仅存在于独立诊断采集，不计入生产适配或优化收益。它们仍失败原完整logits合同；不能为了描述工作量将诊断候选写成已交付修复。模型结构、线性层、主要GDN/卷积算法和调度器继续标明上游来源，新增验证辅助程序也不冒充新生产算子。

在远端项目目录使用既有Conda，选择一个确认空闲的逻辑chip，设置 `ASCEND_RT_VISIBLE_DEVICES` 为该chip。示例为未来复验入口，不是本次已执行的命令：

```bash
source scripts/use_project_env.sh
python -m pytest tokenspeed-kernel-npu/test/test_mha.py \
  tokenspeed-kernel-npu/test/test_layernorm.py \
  tokenspeed-kernel-npu/test/test_rotary_embedding.py \
  tokenspeed-kernel-npu/test/test_gdn_prefill.py \
  tokenspeed-kernel-npu/test/test_gdn_decode.py \
  tokenspeed-kernel-npu/test/test_gdn_solve_tril.py \
  test/runtime/test_qwen35_q_gate_layout.py \
  test/runtime/layers/test_gemma_rmsnorm_npu.py \
  test/runtime/layers/test_layernorm_gated_npu.py \
  -q --basetemp .tmp/homework1_operator_validation
```

复验前确认所选device/端口空闲，以及示例basetemp尚不存在，避免pytest清理旧目录；运行日志注明skip和实际NPU执行范围。原GDN0.03仅作旧回归，新增真实头数/多步包络采用[模块合同](homework1_numerical_contract.md)，不能拿宽旧容差替代已固定新判据。有限Session隔离见[Session记录](homework1_session_acceptance_20261013.md)，最终版本与任意长序列不由这些样例证明。单chip不需要HCCL作为主模型计算依赖；CPU网关/调度/分词不违反主要计算在NPU执行的要求。


## 2026-10-15最终源复验补充

本表上方历史缺项以本补充及[最终报告](../report/homework1_compute_subsystem.md)为准：最终源码9个关键模块/状态测试文件同源部署后108passed、0failure/error/skip，记录203properties，包括70 Norm/MRoPE、GDN17（13NPU/4CPU）、MHA9（6NPU/3CPU）、canonical Norm3、QK3、conv3、gatedNorm2及真实Session1。测试源SHA、原XML SHA和最终源码清单见[公开验证摘要](https://github.com/jiran-1111/tokenspeed_PKU/blob/course/final-20261018/docs/evidence/hw1_20261015/verification.json)。这是这些明确形状及行为的复验，不外推任意shape或TP通信。

Stage B原16条固定输入及续接的272完整向量通过冻结有限经验双门槛；旧1%失败与greedy差异保持独立记录，不能由模块或Session结果替代。未新增独立Dense GEMM任意形状的证明，不把复用上游矩阵层称作从零开发。
