# 第一次作业：计算子系统——Qwen3.5-0.8B 模型适配与正确性

## 成员职责与 AI 工具使用

小组成员：张晟祺、李书砚、计然。下表是小组责任边界和成果归属安排，按实现、实验核验、文档三类交付尽量均衡；不是个人实操或工时统计。

| 成员 | 负责的成果范围 | 与其他成员的接口 |
|---|---|---|
| 张晟祺 | 环境/依赖与启动配置；算子支持矩阵；GDN、Norm、注意力等计算路径及适配来源说明 | 交付固定模型、配置和计算实现，供正确性与性能实验共同使用 |
| 李书砚 | 数值参考及冻结判据；典型/边界形状、多步状态和请求隔离验证；失败记录与误差审计 | 交付可复查的通过项、失败项和覆盖边界，不以修改性能配置或阈值规避失败 |
| 计然 | TTFT/Decode/HBM 基线；Profiler 证据与分析；复现入口、报告整合和源码交付 | 只分析匹配源码的数据，保持原始记录、实验口径和发布版本可追溯 |

**实际执行说明：** 本阶段的代码编辑、实验调度、数据处理和报告起草由 Codex AI 工具辅助代为完成；未记录三位成员各自亲手执行的操作或个人工时，因此不虚构个人实操贡献。上述职责用于界定材料理解和答辩责任，不能代替学生对实现及结论的理解。教学网提交由用户本人处理。

## 1. 范围、环境与固定版本

本阶段对应《作业1.md》，截止日期为 2026-10-27，仅验收 0.8B、TP1、单 chip 的计算路径、分层数值/状态验证与性能基线。四模型 Agent 总项目仍未完成；4B/9B/27B 的结果不替代本报告的 0.8B 证据。

| 项目 | 本次条件 |
|---|---|
| 硬件 | Huawei Atlas 800I A3；运行时 SoC `Ascend910_9362` / `Ascend910_93`；单 chip 64 GiB HBM；Kunpeng920 系列主机 |
| 系统与软件 | openEuler 24.03 / aarch64；项目 Conda Python 3.11；PyTorch/torch_npu 2.9.0；CANN 8.5.0；Transformers 5.12.0；Triton-Ascend 3.2.1 |
| 模型 | `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17` |
| 推理 | BF16、TP1；最大上下文4096、block128、chunk4096、max_num_seqs16、max_total_tokens65536；prefix cache、KVStore、mixed batch、overlap 关闭 |
| 源码身份 | 独立交付分支 `course/hw1-20261015`；数值/Profiler共用1396文件源码清单，SHA256 `f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e` |

不能只凭工具的 `Chip Version: V1` 字段判定设备代际；SoC、板型和 CPU 证据见[硬件身份核实](../project/hardware_identity_20261014.md)。实验使用逻辑NPU2→物理NPU1/chip0；每次运行前实时检查映射及进程，一次只运行一个 NPU 实验，未终止物理NPU0的既有任务。环境、缓存、模型、临时文件和结果均留在工作区，未安装全局依赖或重置已有 dirty worktree。

运行环境来自课程既有 Ascend 栈的项目内 Conda 副本；运行前使用 `source scripts/use_project_env.sh` 加载项目环境和 CANN。完整依赖、模型版本及环境边界见[环境说明](../project/environment.md)。不把旧 setup 脚本中的建议版本冒充本次实际版本。

## 2. 计算路径与适配范围

模型包含24个 decoder layers，其中18层 Gated DeltaNet、6层 full attention；hidden=1024、MLP intermediate=3584、vocab=248320。Full attention 为8个Q头、2个KV头、head_dim256；GDN为16头、K/V维128、卷积宽4。

| 模块 | 主要形状与精度 | 来源与本阶段工作 |
|---|---|---|
| Embedding、线性层、MLP、LM head | IDs→hidden[N,1024]；gate/up[N,3584]；logits[B,248320]；BF16 | 复用上游并行层及 NPU 矩阵运算，未自写 GEMM |
| Gemma RMSNorm、残差、Q/K Norm | hidden宽1024，Q/K末维256；BF16输入/输出，FP32中间计算 | 保留FP32有效gamma；BF16残差原位更新；Q/K归一化后恢复输入dtype，避免与BF16 V不兼容 |
| RoPE、Q/gate、paged MHA | Q[8,256]、KV[2,256]；rotary64、192维不旋转；page128 | 修正Q/gate布局；复用实际MRotaryEmbedding与NPU注意力接口，补D256及跨页验证 |
| GDN投影、短卷积 | mixed QKV[N,6144]；Z[N,16,128]；conv state[P,6144,3] | 复用上游Triton-Ascend路径，验证卷积实际NPU执行、多步与slot隔离 |
| GDN、gated norm、输出投影 | Q/K/V[B,T,16,128]；SSM[P,16,V=128,K=128]；输出投影回[N,1024] | NPU Prefill beta采用FP32 sigmoid，并保留关键WY/状态计算FP32；检查状态读写槽及末端输出dtype |

完整功能、输入输出、dtype、实现来源、Ascend支持状态和测试方法见[算子矩阵](../project/homework1_operator_matrix.md)。模型结构、scheduler和大部分算子直接复用 TokenSpeed；来源遵循原仓库 MIT 许可。本组适配集中于精度/布局、D256注意力和有状态计算兼容，不把整个引擎或 GDN 算法称为从零开发。

### 关键模块：GDN 从模型调用到状态更新

1. Decoder先处理残差与Gemma RMSNorm；`_forward_input_proj`投影QKVZ/BA，拆出mixed QKV、Z及a/b。GDN拆分不能套用full attention的Q/gate按头交错布局。
2. `ctx.attn_backend.forward`进入Mamba backend；Prefill按packed序列边界执行卷积，Decode按slot索引更新3个历史位置。
3. 卷积后Q/K/V整理为[B,T,16,128]；归一化采用平方和加eps，执行decay及delta-rule。NPU Prefill显式从FP32 b计算sigmoid，并输出最终状态。
4. Decode分别传入初始状态读索引和输出写索引。状态池按[V,K]存储，CPU递推参考通常按[K,V]存储，对照时明确转置；不能覆盖别的请求的输入页。
5. 输出与Z整理为[N×16,128]进行gated RMSNorm，再投影回[N,1024]。请求结束时按scheduler生命周期回收状态；Session的current-frontier不把尚未forward的末个采样token写入KV/SSM。

详细调用位置与布局见[GDN路径](../project/homework1_gdn_path.md)。D2H、`.item()`及保存logits会同步并改变时延，所以数值诊断与性能测量使用不同进程。最大的适配障碍是混合注意力的布局/缓存语义和多处BF16舍入边界，不能仅凭输出finite或首token一致证明正确。

## 3. 分层正确性与请求状态

最终源码在服务器新进程执行课程软件集合 **569 passed、0 skipped**。关键模块及真实Session组合复验 **108 passed、0 failure/error/skip**，记录203条误差或状态属性；其中101项包含NPU计算，7项为CPU解析检查。完整向量诊断、模块误差与Session行为分别验收，不互相替代。

| 验证层 | 本次覆盖与参考 | 结果与限制 |
|---|---|---|
| Norm / 实际 MRoPE | 宽度256/1024、典型及边界rows、rotary64及未旋转192维；CPU FP32公式；canonical gamma与Q/K额外检查 | 70+3+3项通过；辅助归约分支的通信使用mock，不当作真实TP通信验收 |
| GDN Prefill / Decode | 真实16头；Prefill长度1/8/127/128/129；多步双slot、读写槽分离；CPU FP32/FP64递推与epsilon边界 | 17项通过，其中4项CPU解析；原数值容差未放宽；历史clamp参考勘误保留 |
| D256 paged MHA | 8Q/2KV、KV长度127/128/129、非连续页表、Extend/Decode；独立CPU FP32 GQA/causal参考 | 6项NPU+3项CPU通过；输入不变性检查保留，不外推所有图执行形状 |
| 卷积 / gated norm | 通道16/6143/6144、每slot连续8步及未选slot精确不变；gated norm宽128及大rows边界 | 3+2项通过；覆盖有限形状 |
| 原生 Engine Session | 两会话交错、stateless对照、关闭后拒绝、重开及context元数据 | 1项集成测试内8组输出/context配对、4次关闭拒绝及重开复现通过；不外推HTTP多并发Session性能 |

测试退出码为0、设备释放，但Session清理仍有pending-task及关闭日志流错误记录，因此不称完全无警告退出。关键算子精确判据及依据见[数值合同](../project/homework1_numerical_contract.md)；GDN参考语义勘误见[审计记录](../project/homework1_gdn_reference_audit_20261014.md)。

## 4. Stage B：冻结的完整模型数值验证

先完成校准组四种独立精度/设备参考、缓存/重算敏感性审计，再由用户批准冻结经验判据；本次没有按验证组结果调阈值、换样例或删除失败。该判据仅用于固定配置下的有限回归，不是普遍误差上界或模型质量充分证明。

验证集为原manifest中16条validation输入：长度1/8/127/128/129/512/1024/2048各两条，使用原固定16-token续接。文件SHA256为`23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb`，canonical SHA为`f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936`。主参考为同投影learned weights的CPU FP32/eager缓存路径，threads1；320参数共同值SHA为`c2084362ae79d1d0f3d61a9781598296fae0944efe4e6fd04888a48692c57884`。

引擎为BF16/TP1单chip，诊断显式eager。每条保存末端Prefill及连续Decode step1–16的248320维完整logits，校验实际ID、绝对位置与三轴MRoPE位置、shape、finite和缓存前沿；仅在原采样完成后替换输出token以控制下一步输入，绝不修改保存的logits。

误差在CPU FP64计算：relative-L2=`||actual-reference||₂ / max(||reference||₂, 1e-12)`，max-abs为全部元素最大绝对差。每个向量必须同时满足两项门槛。

| 阶段 | 向量数 | 冻结relative-L2上限 | 观测最大relative-L2 | 冻结max-abs上限 | 观测最大max-abs |
|---|---:|---:|---:|---:|---:|
| Prefill | 16 | 0.04160680933973707 | 0.021534648897684416 | 0.4934459924697876 | 0.32581615447998047 |
| Decode step1–16 | 256 | 0.027781371644764415 | 0.019179627022419887 | 0.4458632469177246 | 0.4235382080078125 |

**272/272向量通过冻结协议。** 最终源码重采与上一轮的272个原始张量文件SHA逐项一致；NumPy使用独立平方和公式复算全部向量，门槛判定、max-abs及argmax均一致，relative-L2最大重算差约2.50e-15。两次重复不当作32条新的独立样本，也不计算总体覆盖概率。

仍须单列：本批向量在原额外1%工程门槛下有 **68个失败**，固定输入前沿下argmax有 **13处不同**；历史1%失败及自由greedy的长上下文分叉保持原结论。强制共同续接下的条件argmax不等于自由生成轨迹，不能把本次272向量通过改写成greedy全一致、Session全覆盖或普遍正确性证明。[冻结记录](../project/homework1_frozen_numerical_gate_20261015.md)与[Stage B执行、失败及终态](../project/homework1_stage_b_20261015.md)保留完整依据。

## 5. 最终源码的可比较性能与 HBM

同一最终源码、同一服务配置完成96条正式请求/12份摘要。四条件各24请求，实际输出均为32或128，无提前结束。目标输入128/1024经网关构造后实际为130/940，不能称精确128/1024的native性能测量。

| 实际输入/输出tokens | TTFT P50均值(ms) | 客户端TPOT P50均值(ms) | 服务端ITL mean均值(ms) | E2E P50均值(ms) | 输出吞吐(tok/s) |
|---|---:|---:|---:|---:|---:|
| 130/32 | 206.133 | 10.868 | 10.862 | 543.207 | 58.919 |
| 130/128 | 206.211 | 10.806 | 10.793 | 1577.973 | 81.179 |
| 940/32 | 722.418 | 10.872 | 10.963 | 1060.191 | 30.101 |
| 940/128 | 719.266 | 10.893 | 10.887 | 2103.297 | 60.891 |

12轮均有服务端metrics，ITL observation count分别为8×31=248及8×127=1016。长输入主要增加首token前延迟，稳态Decode指标变化较小；这一阶段差异不单独证明全部新增耗时来自哪类kernel。服务退出码0，开始/结束全部源码清单一致。

| HBM状态 | 目标chip设备占用(MB) |
|---|---:|
| 启动前 | 3112 |
| health通过后 | 8459 |
| 两档输出负载采样最大 | 8461 |
| 退出后 | 3113 |

共121个查询样本，返回码均为0；每次查询后等待1秒，最大查询耗时1.147秒、最大相邻查询起点间隔2.148秒，覆盖启动、两档输出负载及退出。峰值包含条件预热且可能遗漏短尖峰；退出后的基础占用不判为模型泄漏。历史基线保留，不据新旧测量之差宣称优化收益。

TTFT为客户端首token延迟，包含服务、主机、HTTP等；客户端TPOT按生成总时间计算，服务端ITL来自Prometheus直方图sum/count差分，三者不混用。每个条件预热3次后，正式三轮、每轮8请求；表中P50先按每轮计算再取三轮均值，不是合并请求分位数。目标输入长度与实际token数分别报告。所有计时排除logits诊断和Profiler。

HBM采用独立全设备周期采样、按本次physical1/chip0（Phy-ID2）解析；设备总占用不是模型有效tensor或KV专属占用，采样最大值也不是瞬时峰值保证。旧benchmark helper每轮硬编码读取physical0，其快照不用于本次目标设备资源结论。

## 6. Prefill 与 Decode 热点

最终Decode Profiler在Stage B通过、源码固定之后串行执行。native输入128 tokens，三个8-token预热请求后，单请求生成256 tokens；首个输出之后启动、累计128个输出时停止采集，再继续完成请求。实际metadata确认Level1及`ACL_AICORE_PIPE_UTILIZATION`。762次full-attention调用对应6层×127步，GDN Decode为18×127次，无GDN chunk Prefill算子，支持该窗口为连续Decode而不是混合Prefill/Decode。

| Decode算子类别 | 记录任务数 | 全部Task Duration累计占比 |
|---|---:|---:|
| 卷积状态更新 `_causal_conv1d_update_kernel` | 2286 | 36.944% |
| 矩阵乘法 MatMulV2 | 12319 | 21.107% |
| ViewCopy | 3302 | 13.334% |
| fused GDN Decode状态更新 | 2286 | 3.223% |
| Cast | 18415 | 2.840% |

分母为108331行算子记录的累计Task Duration（981676.691µs），不是端到端延迟或互斥墙钟分解。只看AICore计数会得到MatMul约92.15%，却漏掉卷积更新等无该计数的kernel，因此不能据此宣称MatMul占整个Decode的92%。

任务时间线包络1.531812秒、记录区间并集0.996449秒，未覆盖0.535363秒，最大间隙约1.406ms；未覆盖不等于设备空闲，也不能全部归因Host调度。MatMul记录的搬运计数明显高于MAC计数，且存在大量Cast/ViewCopy，支持继续检查权重读取、状态布局转换和小算子调度；这些流水计数可重叠，不相加成“访存占比”。

本阶段最值得优化的Decode环节是**卷积状态更新及相关布局拷贝**，其次检查矩阵投影和小算子发射开销，而不是仅凭GDN名称先优化递推核心。Prefill历史混合负载中，MatMul、GDN chunk输出和W/U重算居前；该历史结果只用于解释阶段差别，不冒充本次纯Prefill测量或最终源码优化收益。完整资料见[历史资源分布](../project/homework1_profile_resource_share_20261014.md)及本次公开证据。

Profiler会扰动时延，停止后的导出/解析还引入长暂停，因此其请求E2E不能写入第5节无Profiler基线。本阶段不宣称已完成严格计算/带宽/Host的全部因果分解或新的性能优化收益。

## 7. 复现与适用边界

- 运行入口、固定配置：`scripts/use_project_env.sh`、`configs/baseline/qwen35_0.8b.yaml`；模型主要计算在NPU执行，CPU仅作离线参考和分析。
- 分层测试：上述9个测试文件；全软件集合为 `python -m pytest test/course_project`，须使用项目内新basetemp。
- 固定输入：`configs/validation/homework1_inputs.json.gz`无损解压后必须得到上述原文件SHA；不得重新分词后替换验证集。
- 完整模型：`collect_homework1_validation_reference.py` → `run_homework1_stage_b.py` → `analyze_homework1_stage_b.py`；输出目录必须不存在，既有失败数据保留。
- Profiler：`run_homework1_decode_profile.py`要求272向量全部通过、源码清单一致及无诊断环境；先检查实时设备映射和空闲，串行运行。
- 性能：`serve.py`配合`run_benchmark.py`，短/长输入与两档输出、c1、三轮各8请求；具体最终配置、数据、源码清单和复现命令见交付证据目录与README。

该阶段完成的是固定0.8B配置的计算适配、分层验证、有限完整logits回归和基础测量。尚未解决的总项目事项包括9B自由greedy长上下文分叉、四模型最终统一版本/负载、并发Session与服务端工具等待语义、完整消融和最终演示答辩；不以本报告宣称四模型Agent项目完成。
