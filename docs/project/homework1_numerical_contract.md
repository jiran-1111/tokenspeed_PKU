# 作业1数值验证合同（新增GDN覆盖，实测前固定）

## 后续参考语义勘误（不修改原容差或历史记录）

2026-10-14对照当前生产与固定HF源码确认：GDN Prefill和Decode均使用平方和加1e-6；本合同下方“Prefill采用sum clamp”描述旧测试参考，不是生产定义。2026-10-15已纠正两个测试参考，近epsilon四级解析式CPU RED→GREEN及正式17项（13NPU+4CPU）全部通过、52条误差指标归档，原NPU容差和生产计算不变。旧脚本/结果保留，来源见[语义审计](homework1_gdn_reference_audit_20261014.md)，不替代全模型验收。

当前版本在2026-10-13新增NPU实验执行前固定。此合同不是全部算子的最终验收报告，尚未证明下述新覆盖已在NPU通过。

## Norm与实际MRoPE补充合同（首次实测前固定）

新增 `test/runtime/test_homework1_norm_rope_npu.py`：standard/Gemma Norm宽度256/1024，rows0/1/129/1024，BF16/FP32及有/无残差；CPU FP32均方、rsqrt、gamma参考，输入与gamma从同量化dtype出发，Gemma有效gamma=1+weight，残差同dtype先相加。BF16输出atol0.002/rtol0.02；FP32额外必须满足atol=rtol=1e-5（明显比原0.02收紧，用于正常幅值与epsilon1e-6的归约/SFU工程包络，不称严密误差上界），残差更新精确相等。空输入只检查形状/finite，不用空max伪造误差。

固定模型配置确认MRoPE head256、partial factor0.25即rotary64、theta1e7、sections11/11/10、interleaved=true。实际model NPU路径是runtime `MRotaryEmbedding.forward`，不是通用 `npu_mrope` dispatcher；测试直接调用前者。CPU参考独立构造频率与各axis选择，系数按实际dtype量化后FP32计算旋转，BF16输出atol=rtol=0.02，额外FP32断言1e-5；未旋转192维精确不变。文本1D/三轴同位置及三轴不同位置覆盖0/1/127/128/4095。BF16旋转会在两个乘法及加减间舍入，对近零抵消项保留0.02绝对包络；不是放宽已测失败，不从旧head128通用测试推导head256模型验证。

以上阈值于新实测前固定。finite/max-abs/relative-L2记入XML，失败必须定位，不事后扩大判据。

补充覆盖仍在首次NPU实测前固定：GDN读slot1/写slot3数值对照，读页和其余槽精确不变；`test/runtime/test_causal_conv1d_homework1_npu.py`直接调用runtime卷积更新，width4、channels16/6143/6144，双slot各8步，以CPU FP32逐channel点积+bias+SiLU为参考，输出BF16比较atol0.002/rtol0.02，历史移位与未选槽精确不变。6143用于向量尾边界，6144为0.8B真实宽度。卷积state只是输入拷贝/移位，不进行归约，要求精确相等有直接依据。

新入口 `tokenspeed-kernel-npu/test/test_gdn_homework1.py`：真实0.8B 16 heads、K/V=128；Prefill长度1/8/127/128/129，BF16 QKV、FP32 gate和SSM state；Decode两会话slot1/3交错各8步，另3个slot作未选中哨兵。参考从与内核完全相同的量化后输入及初始state出发，在CPU FP32显式计算归一化、衰减、delta与输出。state参考使用[K,V]，池为[V,K]，对照时明确转置。Prefill采用sum clamp、Decode采用sum+epsilon，分别与已有测试参考保持一致。

新增覆盖固定逐元素判据：BF16输出 `atol=0.002, rtol=0.02`，FP32 state `atol=0.0003, rtol=0.01`；必须全部finite。未选择slot要求 `atol=rtol=0`，不能用普通浮点容差掩盖串扰。每次对照记录max-abs与relative-L2到pytest JUnit properties，relative-L2分母下限1e-12；不得只报告state norm之差。

依据：BF16 unit roundoff约0.00390625，输出相对包络预留数次舍入及归约次序影响，绝对包络控制近零值；state以FP32保存，应比已有单步BF16-state测试的0.03包络更紧，容许BF16输入投影与不同递推实现累积差异，但不代表FP32 machine epsilon级一致。此为明确的工程验收包络而非严密最坏误差上界，需通过多形状/多步实际误差报告及敏感性验证说明合理性，失败后不可为了通过而放宽。八步覆盖不是任意长序列稳定性证明。

现有MHA/Norm等测试的阈值与参考来源须另行汇总，不能将本合同移用于所有算子。端到端自回归token差异须在共同前缀上核验logits/state并结合margin解释，不用这些单元阈值自动豁免模型输出差异。

## 既有参考与判据审计（2026-10-13复验后整理）

下表记载原测试实际合同，不修改既有阈值，也不将所有宽容差当作已充分验证的最终数值上界。

| 模块及入口 | 参考计算 | 原测试atol/rtol | 判据解释及剩余限制 |
|---|---|---|---|
| `test_mha.py` | PyTorch QK缩放、causal mask、softmax、V聚合，显式GQA head复制 | 0.02/0.02 | BF16输出的量化和归约次序允许小差异；D256、单token、paged、graph更新已通过。该包络不证明FP32机器精度一致，不能豁免页索引错误 |
| `test_layernorm.py` | x转FP32、均方/rsqrt/gamma再量化；残差另对照x+residual | 0.02/0.02；残差0/0 | BF16归约/量化包络；真实hidden1024已测。原测试也覆盖FP32，但同0.02阈值对FP32较宽，需更紧精度/实际误差证据后才能声称FP32路径充分验收 |
| `test_rotary_embedding.py` | 显式cos/sin旋转与NeoX/interleaved重排 | 0.02/0.02；out-buffer模式原输入0/0 | 原参考按输入dtype计算，不是独立CPU FP32高精度参考；BF16/FP16、rotary64/128已通过，不代表模型head256所有设置都已覆盖 |
| `test_gemma_rmsnorm_npu.py` | 残差后转FP32，归约、有效gamma=1+weight，再量化 | 0.01/0.01 | 基于同量化输入区分gamma语义及BF16舍入；当前hidden16不能代替真实1024/256宽度数值验收 |
| `test_layernorm_gated_npu.py` | CPU FP32均方归一化×gamma×SiLU(z)，再量化BF16 | 0.02/0.02 | gate与归约舍入包络；width128与超大row/width8覆盖已通过，真实模型组数布局须与调用路径共同核对 |
| `test_gdn_solve_tril.py` | CPU FP32 `solve_triangular(I+A,I)` | 0.01/0.01 | 随机小幅strict-lower A、32×32、BF16输出；不是任意条件数/块尺寸稳定性证明 |
| 原GDN Prefill/Decode | CPU FP32归一化与delta递推，按对应state dtype量化 | 0.03/0.03 | 只作原有回归；新增真实头数/FP32-state采用本合同更紧阈值，不使用宽包络代替状态隔离 |

此汇总不把复验通过误写为所有模块最终验收闭环。BF16舍入量级提供工程依据，但累计舍入、归约和条件数无法仅靠一个unit roundoff解释；尤其FP32 Norm和模型实际RoPE/Norm形状还需补证据。任何新增判据仍须在新增实验前固定，保留失败记录，不为通过临时放宽。

## 2026-10-14补充源码覆盖核对（不改变判据）

逐行读取`tokenspeed-kernel-npu/test/test_mha.py`确认：原参考`_attention`直接接收NPU BF16 Q/K/V，QK与V聚合仍在NPU上执行，仅softmax中间转FP32；这是独立公式/调用路径对照，不是独立CPU高精度参考。D256覆盖单token及长度2/3的Prefill；paged extend/decode和graph长度更新测试使用D64、cache lengths5/70与12/80。它们虽有两列page table，实际长度均小于page_size128，不能证明跨页读取正确，也不能据此声称实际D256缓存路径已作同样的数值测试。历史六项通过记录保持原样，剩余需实际D256、127/128/129等跨页边界及独立CPU参考证据；新覆盖的合同须在采集前固定，不拿原0.02机械作为充分误差依据。

2026-10-15该MHA缺口已补：新增独立CPU FP32公式与手算自检，实际8Q/2KV/D256/page128、两个非连续页序列、Extend/Decode×127/128/129六NPU项全部通过，原0.02/0.02不变。最大observed maxabs=0.0021448731422424316、relative-L2=0.0022215167991816998（来自不同case），全部finite、七个输入精确不变；两内存变异分别错误页序/上左causal均被拒绝。见[MHA跨页证据](homework1_mha_boundaries_20261015.md)。有限边界实测支持该工程包络，不把它当任意输入理论上界，原graph覆盖不因此扩大为D256 graph验收。

`test_gdn_solve_tril.py`确用CPU FP32三角求解，但只覆盖seed23、32×32、随机0.05幅度的strict-lower矩阵，未报告条件数或方程残差。原BF16输出0.01/0.01是该样例工程回归包络，不证明任意实际块/条件数稳定性；真实GDN core/output/state参考仍独立保留，不将此一项通过外推为完整模块证明。

`test/runtime/layers/test_layernorm_gated_npu.py`的CPU FP32参考从同量化输入出发，覆盖rows2×width128和rows65536×width8，均采用norm-before-gate、SiLU、weights_independent=True；模型构造及forward对应相同语义，width128匹配真实head维度。超大row测试证明循环grid边界的有限样例，不代替真实heads flatten后的代表row数、近epsilon输入或训练权重敏感性验证。当前原XML未逐张量记录这些既有项的max-abs/relative-L2，因此不可从“通过assert_close”反推精确误差数字。

以上只补充覆盖与证据限制，不修改生产/测试、不采新NPU或B校准、不宣布Task2最终完成。

未来命令（本次未执行硬件测试）：

```bash
source scripts/use_project_env.sh
python -m pytest tokenspeed-kernel-npu/test/test_gdn_homework1.py -q \
  --basetemp .tmp/homework1_gdn_validation \
  --junitxml=results/raw/homework1_gdn_validation.xml
```

运行前确认该结果路径不存在，记录实际device、软件/源码版本；避免与当前27B性能测量并行。阈值变更必须说明新科学依据并保留旧失败结果，不能覆盖已有输出。
