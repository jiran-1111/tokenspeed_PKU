# 独立端到端 CPU 参考复核

复核日期：2026-10-28。复核方式：独立只读代码/数学检查、本地 CPU 单元测试及另写的内存标量计算。未 SSH、未运行 NPU、未修改参考实现或部署源码。部署目录 HEAD 为 `70265dac54378475f91ff2945fad12d4fc7e8ef6`，复核时其工作树干净。

## 结论

在已阅读的 `independent_ops.py`、`independent_model.py`、`prefill_oracle.py`、`reference_math.py`、`checkpoint_reader.py` 和 `run_reference.py` 中，未发现使新参考与当前部署的声明数学合同明确不匹配的阻断问题。当前实现可以作为从 checkpoint/IDs 自行传播的 CPU 数学参考继续验证；这不等于原 4B/9B 数值验收通过。

这份复核不要求逐位模拟 NPU 内部 FMA、exp、规约和 chunk scan，也未把这些未模拟细节自动视为实际误差均可接受的证明。CPU TP2 是互补权重分片和局部结果存储/相加的模拟，不是实际 HCCL。

## 已核对的数学与边界

| 项目 | 复核结果与具体依据 |
| --- | --- |
| FA 分片与 Q/gate | `column_sequence` 逐逻辑 Q投影/K/V 段切片；Q投影自身按完整 head 范围切片，`split_qkv` 再按 `[head,2,head_dim]` 解出逐 head Q/gate。与部署 `QKVParallelLinear`、`_split_q_gate_by_head` 一致。 |
| GDN 分片 | Q/K/V 从 checkpoint 的 `in_proj_qkv` 先拆逻辑段，再分别与 Z/B/A 逐段 TP 分片；conv 同样分别切 Q/K/V 通道。A_log/dt_bias 按本 rank value-head 范围切片，gated norm 权重不分片。没有对整个 packed 张量直接二等分。 |
| MLP 与 row 输出 | gate/up 各自切列，局部 SiLU乘up，再按 down/out/o 投影的输入列范围切 row 权重；每个局部输出先按存储合同存储，再求和并存储。 |
| 规范化与残差 | Gemma 的有效 gamma 为已加载 BF16 raw weight 升 FP32 后加 1，避免把有效 gamma 再压回 BF16；BF16 residual 相加先存储后归一化。部署 NPU `GemmaRMSNorm` 采用同一顺序。GDN gated norm 用 raw norm weight，无额外 +1。 |
| RoPE | 表由位置和配置自建；旋转仅覆盖 partial rotary dim。BF16 模式对四次乘法各自存储，再分别加/减并存储，符合部署 `MRotaryEmbedding` 的 eager Torch `_apply_rotary_emb`。文本同轴位置不需要引入不同 T/H/W 位置数据。 |
| FA 历史 | 首块严格因果 Prefill，后续每次一个 query；KV 由参考自身归一化/旋转和投影结果建立，包含当前 token，位置连续累计，无实际缓存注入。 |
| GDN 历史 | conv 保存输入投影的最近 3 行而非卷积输出，四 tap oldest→newest；SSM 为 K-last `[Hv,V,K]` FP32，先 decay、再 prediction/delta、最后用更新后状态读输出，完整传递到 Decode。 |
| Ascend GDN 精度合同 | 部署普通 prefill `ChunkGatedDeltaRuleFunction.forward` 在 NPU 上先把 Q/K 升 FP32 再 L2；`chunk_gated_delta_rule_fwd` 也把主要 scan 输入升 FP32；runtime beta 是 `b.float().sigmoid()`。因此参考没有再将 normalized Q/K 或 beta 压 BF16，并非遗漏这个部署的明确存储边界。Decode 也在 FP32 寄存器中 L2/更新状态。 |
| 参数加载 | checkpoint reader 只接受文本参数名/显式精度，按需行选择并返回 owned FP32 representation，A_log/dt_bias 保留 checkpoint FP32，其余按实际加载 BF16 值。9B head 虽输出 FP32，权重的 BF16→FP32 投影符合既有参数身份合同。 |
| 独立性 | 模型入口只有 config、case IDs 和 checkpoint read；算子依赖图只到 NumPy 数学函数。runner 在完成参考前不读实际 hidden/QKV/conv/SSM/logits，仅读身份清单、配置、checkpoint、IDs；捕获比较应保持在后续独立阶段。 |
| 完整传播 | 每层处理 Prefill 和后续各 Decode，再把所有 chunk 传播到下一层；在每层状态严格因果的前提下，这种 layer-major 排序与逐步模型执行的依赖关系一致。保存 64 个层规范化边界和 final norm、完整 logits 和自身最终状态。 |

## 已发现并已修复的问题

初次本地执行 20 测为 19 passed、1 failed：NumPy 1.26.4 对 BF16 RNE helper 的 0d scalar 发生整数提升，`rounded.view(np.float32)` 因 itemsize 不同失败，触发于 RoPE 的辨别性标量测试。已及时反馈主代理；主代理在新 `stored` 包装层将输入展平为一维后调用冻结 helper，再还原形状。随后本地重跑为 **21 passed**。该修复不改变实际矩阵路径的数学。

本地验证命令：

```text
python -m pytest .tmp/hw2-independent-e2e-20261028/test_independent_ops.py .tmp/hw2-independent-e2e-20261028/test_independent_model.py -q
```

环境为 Python 3.12.7、NumPy 1.26.4，无 Torch。checkpoint reader 和 Torch/HF 交叉测试已静态阅读，本复核不冒称在本机执行过这些测试；服务器执行及日志由主代理另行认证。

## 额外独立数学对照

为避免整段/分段或 TP1/TP2 共用同一错误公式仍能通过，本复核额外在内存中写了独立逐 token、逐 head 标量计算，未调用被审的 conv、scan、RMSNorm、GQA、RoPE 或 BF16 helper 来生成期望值，BF16 标量存储用 Python struct 单独实施。

- GDN 使用 `Hk=2, Hv=4, K=2, V=3`、四个不同卷积 tap、Prefill3 + Decode1 + Decode1。FP32 输出最大绝对差 `1.043081283569336e-7`，SSM 差 `1.1175870895385742e-8`；BF16 输出逐值相等，SSM 差 `3.725290298461914e-9`；两种模式 conv history 都逐值相等。
- FA 使用 `Hq=4, Hkv=2, D=4, rotary_dim=2`、同样的 3+1+1 续接，显式独立实现逐 head Q/gate 拆分、因果 softmax 和当前 KV。FP32 输出最大绝对差 `3.3527612686157227e-8`；BF16 输出和 KV 均逐值相等。

以上数值只记录小例子的公式核对结果，不是为原完整模型新增验收阈值，也不是对原 NPU 差异的容许结论。

原 GDN 测试使用 K=V=2 和四个相同 tap，只能检验共用公式的分段自一致；本复核反馈后，主代理已加入非方形 K/V、不同 tap、独立 recurrent 数学的测试。另已阅读新增的 eager Torch BF16 RoPE 和 HF tiny 全模型 FP32 Prefill+Decode 交叉测试，其比较对象为独立模型实现的 head_input/logits，比纯 TP 自一致具有更强辨别力。

## runner 身份接线与尚待完成的认证

`run_reference.py` 对原 1531 项 core、artifact manifest、checkpoint 文件大小/hash、revision、原 128 输入与 17/9 续接、输出几何进行了约束；checkpoint/core 前后复核、参数选择 hash、异常状态保存和零 production-runtime/NPU 导入检查均已阅读，未发现接线阻断项。

最后阅读 runner 时，本地尚未生成 `reference_protocol.json`、`tool_manifest.json` 和两个 `case_*.json`，故本复核尚未核对这些最终冻结文件的具体值。主代理需在启动正式参考前完成其冻结、核对原 case IDs/revision，以及服务器 Torch/HF 测试。比较器本次尚未提供，不在已审范围内；需要继续维持参考完成后才读取实际捕获的边界。

最终报告必须保留原 4B/9B greedy 失败和所有非零差异，分别报告 engine1/ref1、engine2/ref2、ref1/ref2 与 FP32 control；不能由“参考也分叉”直接推出部署所有误差可接受。

## 冻结协议与控制器续审（2026-10-28）

续审对象为 `tools_r1/` 中最终冻结的 14 个 manifest 条目，包括协议、两个案例、模型/算子、checkpoint reader、runner、controller 和测试；`tool_manifest.json` 本身另计。此节替代上节关于“协议、tool manifest 与 case 尚未落盘”的待办状态，比较器仍不在已审范围。

### 身份核对

- 独立计算 `tool_manifest.json` SHA256 为 `cf09ecac218c32702b757b6010f70b3f80125cc18ffec19290827c9bf1e534af`，与主代理冻结身份相同；其中 14 个条目的字节 hash 全部匹配。
- 冻结 `case_4b.json` 与 `.tmp/hw2-fullpath-20261028/tools_r3/case_4b.json` **逐字节相等**，SHA256 为 `4b2449ff2691548ef9228c532dfa85bf04c3aa9958dc1609f00bec72d4c06095`；128 输入、17 固定续接、18 向量。
- 冻结 `case_9b.json` 与原 tools_r3 对应文件 **逐字节相等**，SHA256 为 `827d28ace1f59561ee3a967e682c597fb4f83807505e7339f78b399e8b54a8c1`；128 输入、9 固定续接、10 向量。
- 协议中的两项 case hash、revision 均与实际冻结文件、原 TP1/TP2 配置和原 model artifact 清单一致；原 TP1/TP2 使用同一本地 checkpoint 路径。
- core manifest SHA256 为 `2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`，artifact manifest SHA256 为 `169e3069d4fdb70e57aaaffdee9f1a84b3f0001aceb4fe4a007ed66f9eeda1c6`，均与协议一致。另独立逐项核对本地 1531 个 core 文件，**0 项不匹配**。

case 原样保留了 `original_graph_tp1_next` / `original_graph_tp2_next` 等旧诊断 metadata；静态确认 `run_model` 只把 `input_ids` 和 `continuation_ids` 作为 token 操作数，不消费这些 next-token metadata。参考 argmax 仅在计算完成后记录，未用来改写后续输入；所有 TP/policy 因而比较同一条固定轨迹。

### BF16 与 FP32 control 配合

协议明确以同一实际投影权重、FP32 A_log/dt_bias、同一 case IDs 和自身状态传播运行两种策略。`fp32_control` 不改变 checkpoint 投影为 BF16 这一实际加载值身份，而是移除 hidden、局部 GEMM 输出/求和、residual、Q/K norm、RoPE、KV、conv/core/norm 与 head 输出的 BF16 激活存储边界；4B 的 control head 输出也为 FP32，符合“移除存储边界”的控制定义。GDN SSM 两种策略均保持 FP32。

协议保留 naive FP32 softplus、threshold20；没有按已见模型输出选择公式。`numerical_acceptance_threshold` 为 null，control 明示不能作为原 runtime 验收通过结论。当前代码实现与协议相符。

### runner / controller 接线

`reference_controller.py` 按 `9b/4b × bf16_contract/fp32_control × TP1/TP2` 固定遍历八组；每组使用同步 `subprocess.run`，返回后才启动下一组。每个子进程 CPU 线程为 8，屏蔽 NPU/CUDA 设备与 Torch 设备后端自动加载；runner 以 CPU safetensors reader 读取权重，并在成功前检查 production runtime / NPU 没有被导入。旧队列 stop 标记及无 current 条件在启动前检查，代码未恢复旧队列或结束外来任务。

单组只有在输出几何、checkpoint/core 前后身份和零生产/NPU导入检查后才标记 `collected`；控制器检查返回码、该状态和 `actual_runtime_intermediates_read=False`，记录 state/arrays 的 hash。仅在八组全部登记后生成 `reference_complete.json`；任意失败不会生成该完成标记。控制器和 runner 均不加载 R3 实际捕获。

比较器后续应以精确八组键、`collected` 状态及记录的 state/arrays hash 认证此完成标记，再读取实际捕获；不能仅凭完成标记文件存在就接受损坏或不全的参考。这是对尚未提供比较器的核对要点，并非当前冻结工具中的已发现错误。

### 验证与结论

对冻结 `tools_r1` 的 NumPy 模型/算子测试，本地再次执行为 **21 passed**。主代理另报告服务器 Torch/safetensors/HF 测试 **31 passed、0 skipped**，包括完整 HF tiny FP32 模型 Prefill+Decode 的 TP1/TP2 head_input/logits 比较和 eager Torch BF16 RoPE；本续审未 SSH，未冒称自行重跑或独立读到服务器日志。

**续审结论：协议/IDs/工具冻结及 runner/control 配合没有新增阻断项，可以继续已启动的 CPU 八组参考。** 正式结果尚未在本续审中完成，故此结论仅认证方法与接线；最终比较器和八组结果完整性仍需后续认证，原 greedy 失败不受本结论改变。

## 冻结比较器续审（2026-10-28）

已只读审查 `comparison_r1/{compare_reference.py,comparison_contract.py,test_comparison_contract.py,analysis_binding.py,parameter_identity.py}`。独立复算 manifest SHA256 为 `8976fe1f432691c47bca9a4e6097c255d3ebee07a233b79fff9fc8f9ee5ff261`，5 个条目全部匹配。`analysis_binding.py` 与上一轮 `operations_analysis_r1` **逐字节相等**，`parameter_identity.py` 也与该轮逐字节相等，没有悄然放宽旧 binder。

### 八组认证与读取顺序

`validate_complete` 要求精确的两模型×两 policy×两 TP 八组键，cohort 已 collected、CPU only、未读实际捕获、无 current、参考工具 manifest 为已审的 `cf09ecac...`。比较入口随后检查完成标记与 state 一致、全部参考工具字节、每组 state hash、model/policy/TP、core/protocol/case/revision、原 IDs、独立性声明、NPZ hash、参数 ledger hash；直到全部八组完成这些认证之后才调用旧 `load_model` 读取任何 R3 实际数值捕获。

cohort 内 state hash 绑定了包括各 state metadata 在内的完整单组记录，NPZ hash 绑定所有 norm/logits/head_input/状态数组，ledger hash 绑定每次参数选择及其 FP32 representation hash。之后还将各组 checkpoint 指纹与旧实际 capture binder 认证的 checkpoint 文件 hash 对接，并在比较结束再次核对参考 state/NPZ/ledger 和 HF 输入文件。

另用纯内存合成字典执行抽取的比较入口，故意篡改第八组参数 ledger 的 hash：入口按预期抛出 `Reference parameter read ledger changed`，**未调用 capture loader，也未创建输出目录**。该测试不读取原捕获或实际参考结果。

### 旧实际捕获 binder

复用 binder 仍要求原 R3 collector 工具 manifest、原 case、32 层/8 FA/24 GDN 覆盖、完整参数身份 412 项且 actual/checkpoint hash 精确一致、checkpoint 前后及原 artifact 一致、guard/core 认证、每步位置和 prefix 长度、duplicate metadata 一致、264 模块和 65 norm 覆盖。每步输出还与原 R2 frontier 的输入/位置/rank/step/case/dtype/raw greedy ID 及 logits 字节逐项对接；结束调用 `verify_unchanged`。比较器未删减这些现有约束。

### HF 完整 FP32 参考绑定

比较器使用既有完整 CPU/eager FP32 参考，而非 BF16 HF 缓存路径。检查其 collected/source/core/revision、原 logits payload SHA 和既有 projected-weight 全局 SHA；读取后再核 Prefill IDs。

- 4B 原参考为 32 步 greedy：只取前 18 个 logits，要求前 17 个已生成 IDs 与固定 case 的 17 个续接 IDs 完全相等，故这 18 个 logits 对应同一条轨迹。
- 9B 原参考为 10 步 forced-frontier：要求已认证 state 内的 9 个固定续接 IDs 与 case 完全相等，故没有把不同 greedy 轨迹混作同前缀参考。

还阅读了旧两份 CPU 生成器和项目 `project_weights`：其权重合同将 A_log/dt_bias 保留 FP32、其余按原投影值计算；这一点与新 FP32 control 的参数身份一致。比较并不因此强制宣称两条实现的有限算术全部相同。

### 指标、覆盖与状态方向

`compare(left,right)` 保存 `left-right`，相对 L2 以 **right 的范数** 为分母，符合 engine/ref、ref2/ref1、BF16/control、control/HF 的明确标签方向。`vectors` 逐步比较全部 248320 词表；`boundary_metrics` 对 65 个边界各自同时计算每步所有行和末行：Prefill slice 为 `[0:128]`，Decode 为各单行；没有用仅末行代替 Prefill 全行。

独立合成测试确认：left=2、right=1 的相对 L2 为 1，反向为 0.5；仅改 Prefill 第 1 行时，128 行指标记录差异而末行指标保持相等，Prefill 全行元素数确为 `128×hidden`。

logits/norm 比较以每种 TP 的 rank0 共同模型输出为对象；最终状态则遍历 TP1 rank0、TP2 rank0 和 rank1。GDN 取最后一步 `after`，与参考的 K-last `[local_Hv,V,K]` FP32 SSM 和 `[local_C,3]` conv history 直接同形比较；FA 取最后一步 `after` 的完整有效 chronological prefix `[128+continuations,local_Hkv,256]` K/V，包含当前 token，与参考自身 KV 同形，无额外 batch 轴、转置或截掉当前 token。所有这些比较都保留有限非零差异，没有新增模型通过阈值。

### 比较器自身执行身份

曾反馈：比较器内部只认证参考工具 manifest，summary 自身不直接记录比较器 manifest。随后已阅读实际启动层 `start_comparison_r1.py`：它在 Popen 前严格核对 `8976fe1f...` 和全部 5 个工具 hash、preparation passed，并要求八组参考完成；本地远程执行 receipt 显示该启动脚本成功。故此次运行的入口身份由启动层保证，不需要修改已冻结/已运行源码。

主代理已安排独立最终执行审计，须再核比较器 manifest/5文件，绑定启动脚本 SHA、执行 receipt、实际 PID/开始 tick/结束状态与最终 summary SHA，并随公开 provenance 交付。该最终审计完成前，本复核不把结果与比较器版本的交付链条标作已经全部闭环。

### 验证与结论

冻结比较 guard 的 7 项本地测试全部通过；另有上述指标和第八组 ledger 篡改的合成执行验证。主代理报告服务器同 7 项为 0 skipped，本复核未 SSH、未运行 NPU，也未提前打开实际 R3 数值捕获。

**比较器续审结论：未发现会造成错误比较方向、前缀/续接混用、128 行遗漏、状态转置或提前读取捕获的阻断问题。** 剩余要求是最终执行身份审计和实际结果认证；不能由代码复核、单个 argmax 序列重现或任何有限残差直接推出原 4B/9B 全部数值可接受或原失败已修复。
