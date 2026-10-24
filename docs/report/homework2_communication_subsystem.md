# 智能计算系统课程第二次作业实验报告（阶段稿）

**题目：Qwen3.5 四模型的同卡双 chip 张量并行与通信分析**

**小组成员：张晟祺、李书砚、计然**

**版本：2026-10-24。状态：硬件实验待补齐，尚不具备提交终稿的条件。**

本报告按作业2的四项任务组织，研究对象是在一张 Ascend 910C 的两个 chip 上，分别以 TP=2 执行 Qwen3.5-0.8B、4B、9B、27B。正文说明已具备的切分实现、验证方法和实际取得的结果；未取得的实验数据明确记为“未测”，不以配置存在或软件测试通过代替设备验收。截至本稿采集状态时刻（2026-10-24 18:59，北京时间），本轮串行实验队列仍在等待同卡空闲双 chip，完成任务数为0，尚无本轮 TP1/TP2 配对结果。[状态快照与范围](https://github.com/jiran-1111/tokenspeed_PKU/blob/homework2/in-progress/docs/evidence/homework2_20261023/report_state_20261024.json)

## 小组分工

| 成员 | 作业2主要责任与交付成果 | 与作业1的接续及协作边界 |
| --- | --- | --- |
| 张晟祺 | 双 chip 适配与实现说明：四模型 TP 配置、权重分片和进程/设备绑定；典型层张量形状、FA/GDN 状态及集合通信点；对应第1章及第3.1节 | 延续作业1的环境与模型适配责任。交付固定源码、配置和张量接口，由另外两项工作据此验证、测量；不凭启动成功判定正确性 |
| 李书砚 | 并行正确性与运行证据：核对权重/精度/输入身份，检查双 rank 参与、四模型连续推理、TP1/TP2 输出配对及大模型参考；保留失败与定位材料；对应第2章及运行证据清单 | 延续作业1的独立正确性验证责任。以固定实现为输入，向实现工作反馈可复现差异；不以速度或 Session 自一致性替代并行正确性 |
| 计然 | 性能与通信实验：两长度×两并发矩阵、逐 chip HBM、大模型性能/容量、双 rank Profiler 的计算/通信/等待区分、分块策略实验和复现材料；对应第3.2节、第4章及附录 | 延续作业1的性能与瓶颈分析责任。使用固定实现和已核对负载，正式计时与 Profiler 分开；发现差异后另建实验，不在采集中改变参数 |

三项工作分别形成实现、验证、测量三个可独立检查的交付包，均包括方法、证据和对应章节；以此分配相近的成果责任，不虚构等量工时。总论与任务覆盖核对由小组共同负责。**代码编辑、实验执行、数据处理和报告整理由 Codex AI 工具在本组授权下完成；上表是成果责任分工，不表示三人亲自执行了相应操作，也不将尚未运行的实验写成已完成。**

## 1. 任务一：完成双 chip 适配

### 1.1 环境、模型与实现范围

设备记录为 Huawei Atlas 800I A3，运行时 SoC 为 `Ascend910_9362`，CANN 系列为 `Ascend910_93`；已有身份核对支持 A3/910C 软件系列判断。当前账号可见4个 NPU ID，每个含两个 chip；本作业每次只选同一个 NPU ID 下的两个空闲 chip，不把跨卡两个设备当作同卡，也不要求四个模型同时驻留。每次实验均重新读取映射和进程占用，历史编号不作为当前可用性的依据。[硬件身份记录](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/hardware_identity_20261014.md)

本轮依赖记录为 Python 3.11、PyTorch `2.9.0+cpu`、torch_npu 2.9.0、Transformers 5.12.0、Triton-Ascend 3.2.1；CANN 沿用服务器已有8.5.0。PyTorch 的版本字符串按实际包元数据记录，不能仅凭字符串判断设备路径是否执行。硬件执行是否成立须由实际 NPU 日志与进程记录证明。[依赖清单](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/runtime_dependencies.json)

| 模型 | 固定 revision | hidden / MLP 宽度 | 总层 / FA / GDN |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | 1024 / 3584 | 24 / 6 / 18 |
| Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 2560 / 9216 | 32 / 8 / 24 |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 4096 / 12288 | 32 / 8 / 24 |
| Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 5120 / 17408 | 64 / 16 / 48 |

模型配置及权重文件身份分别见[模型几何](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/model_geometry.json)和[模型文件校验清单](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/model_artifacts.json)。四模型均采用 BF16 主计算；显式固定 GDN 的 SSM 状态为 FP32，9B 的输出头另为 FP32，并在 TP1/TP2 两侧保持一致。9B 不是全模型 FP32。

实现复用 TokenSpeed 已有统一 Engine、C++ 调度器、分片权重加载器和 HCCL 后端；本阶段新增成对启动配置、设备保护、测量与审计工具。并行执行所用运行时仍为同一条路径，不另起一套独立模型实现。`configs/homework2/qwen35_*_tp{1,2}.yaml` 每对仅改变 `tp_size`；TP2 配置由框架启动两个 worker、设置 rank 并绑定所选逻辑设备。两个 rank 各持有互补权重分片，通过集合通信合成同一模型的结果；两个独立完整模型副本不能替代该过程。[配置与实现范围](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/homework2_status_20261023.md)

### 1.2 典型4B层的切分与通信

下表给出由模型配置及加载器推导的逻辑形状，尚未冒称本轮设备抓取值。权重记为 `[输出维,输入维]`；T 是当前调度实际计算的行数，可能受 Prefill 分块、批次或图填充影响。两个 rank 的形状相同，分片内容不同。

| 数据或计算步骤 | rank0 | rank1 | 数据分布与通信 |
| --- | --- | --- | --- |
| 输入 hidden、残差 | `[T,2560]` | `[T,2560]` | 复制相同完整 hidden |
| FA 融合 Q/gate/K/V 权重 | `[5120,2560]`，各段前半 heads | `[5120,2560]`，各段后半 heads | 每个逻辑投影沿输出切分 |
| FA 投影后 Q/gate；K/V | 各 `[T,8,256]`；各 `[T,2,256]` | 同形状、互补 heads | 注意力在本地 heads 上计算 |
| FA 输出投影 | 权重 `[2560,2048]`，部分和 `[T,2560]` | 同形状、互补输入通道 | SUM all_reduce 恢复完整 hidden |
| GDN 六段融合权重 | `[6176,2560]` | `[6176,2560]` | Q/K/V/z/b/a 各自切分 |
| GDN 投影后张量 | Q/K 各 `[T,8,128]`；V/z 各 `[T,16,128]`；b/a 各 `[T,16]` | 同形状、互补 heads | 本地卷积与递归更新 |
| GDN 输出投影 | 权重 `[2560,2048]`，部分和 `[T,2560]` | 同形状、互补输入通道 | SUM all_reduce 恢复完整 hidden |
| MLP gate/up 与激活 | 融合权重 `[9216,2560]`；激活 `[T,4608]` | 同形状、互补通道 | gate/up 各切分一半中间通道 |
| MLP down 与输出 | 权重 `[2560,4608]`；部分和 `[T,2560]` | 同形状、互补输入通道 | SUM all_reduce 后恢复完整 hidden |

FA 为完整注意力，GDN 为门控 DeltaNet 线性注意力。一层只选择其中一种注意力，再接 MLP，不会把表内两类注意力串行各执行一遍。4B 中8层为 FA、24层为 GDN。FA 的 Q/gate/K/V 分段切片也不能简化为把整个融合 checkpoint 按字节二等分。[形状与代码位置](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/homework2_tp_layout.md)

### 1.3 推理状态、词表与四模型扩展

FA 的 K/V cache 沿 KV heads 分片；GDN 卷积和 SSM 状态沿本地 heads 分片。请求 ID、位置与逻辑块编号保持一致，用于两个 rank 对齐同一请求的执行，但编号一致不代表缓存内容全部复制。下表为**每 rank、每层、每个逻辑缓存槽**的形状，不包含槽数量。

| 模型 | FA 单个 K 或 V 块 | GDN 卷积状态（BF16） | GDN SSM 状态（FP32） |
| --- | --- | --- | --- |
| 0.8B | `[128,1,256]` | `[3072,3]` | `[8,128,128]` |
| 4B | `[128,2,256]` | `[4096,3]` | `[16,128,128]` |
| 9B | `[128,2,256]` | `[4096,3]` | `[16,128,128]` |
| 27B | `[128,2,256]` | `[5120,3]` | `[24,128,128]` |

四模型的 FA head_dim 均为256，GDN K/V head_dim 均为128，卷积核宽度均为4。`A_log`、`dt_bias` 沿 V heads 分片，保留 FP32。状态池、FA cache、权重、图与运行时开销共同占用 HBM，不能只用单槽大小推算实测总 HBM。

词表大小248320，embedding 和输出头沿词表分片，每 rank 为 `[124160,H]`。embedding 局部结果经 all_reduce 合成为完整 hidden；局部 logits 经 all_gather 拼回完整词表。0.8B/4B 的 embedding 与输出头共享权重，9B/27B 不共享；9B 的普通 HCCL gather 保留 FP32 输出头结果。

当前已完成配置、切片与状态的静态核对及软件测试，但**本轮四模型 TP2 的设备运行验收尚未完成**。双 rank 启动、运行期参与、连续生成及逐 chip 占用须由后续原始记录补齐，不能由上述形状表代替。

## 2. 任务二：验证并行正确性

### 2.1 对照原则与验收证据

对能够运行 TP1 的模型，使用相同 revision、权重文件、dtype、tokenizer、输入 IDs 和采样参数，对照 TP1/TP2。主协议为无 Session、关闭 prefix cache 与 KVStore、`temperature=0`、`ignore_eos=true`、输出固定32 tokens；同模型两侧使用完全相同的输入序列。若 TP1 配置确实失败，先保留失败原因；采用其他参考时必须另外注明参考实现、权重与精度，课程认可性不能靠报告自行宣称。

验收分三部分。第一，执行记录核对两个 rank 均属于同一 TP group、位于同卡两个 chip，且从运行日志、通信记录与占用记录确认实际参与计算。第二，逐请求核对连续推理是否结束、输出长度和状态是否符合协议，保留异常与部分输出。第三，严格逐 token ID 比较 TP1/TP2 的有限 greedy 结果；差异列出请求位置、首个不同 token 及两侧输出，不删除失败样例。

有限 greedy 一致只支持被测输入和长度下的序列一致；它不等于完整 logits 数值等同，也不证明任意输入的生成质量。若发现差异，需要在共同输入前沿下比较 logits/中间状态，定位切分、集合通信或舍入影响，并保留原自由输出差异。作业1冻结的误差门槛不能自动充当作业2的跨 TP 验收标准，Session 自一致性也不能替代 TP1/TP2 对照。[配对审计规则](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/homework2_status_20261023.md)

### 2.2 目前取得的证据

| 模型 | 已有历史硬件起点 | 本轮固定源 TP1/TP2 与双 rank 运行结果 |
| --- | --- | --- |
| 0.8B | 已有作业1单 chip 结果 | 未测；不能由 TP1 外推 TP2 |
| 4B | 2026-10-18 旧源具有 TP1、TP2 固定会话轨迹运行及各自输出审计 | 未测；历史 Session 协议不等于本轮无 Session 配对协议 |
| 9B | 已有 TP1、FP32 输出头的运行记录 | 未测；FP32 头须两侧一致 |
| 27B | 2026-10-15 旧源 TP2 对独立 HF NPU BF16 参考，固定8条各64输出，raw 与 stop-aligned 均8/8匹配 | 未测；旧参考结果不能重标为本轮源验证 |

4B 的历史两臂各含192个正式请求和64个预热请求，分别通过其固定轨迹的协议/质量审计；这不等于当前配对审计已经通过。[4B TP1审计](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/course_final_20261018/tp4b_tp1_baseline/audit.json)、[4B TP2审计](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/course_final_20261018/tp4b_tp2/audit.json)。27B 的8条对照属于2026-10-15的1396文件旧源，其原始记录及范围见[27B参考验证](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/final_source_27b_correctness_20261015.md)。不同日期和源身份的结果不拼成一份新源“全通过”结论。

本轮固定源已通过本地课程软件测试878项（33跳过）、服务器 CPU 软件测试903项（8跳过，阻止 NPU 导入）、隔离编译的 scheduler Python 测试259项（无跳过），并核对 parent/spawn/forkserver 的导入来源。这些结果证明采集、审计和启动准备经过检查，**不证明四模型已在 TP2 上正确运行**。[软件验证收据](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/software_verification.json)

27B 历史默认 TP1 的失败分类为：缓存/调度配置无法接纳请求，报 `num_lcm_blocks=31`；并非已证实权重放不下。旧95%内存预算候选曾启动，因此不写“27B一定无法单 chip 运行”。本轮先测默认配置；若失败，再以预先固定的95%预算 TP1/TP2 成对测试，保留默认失败，候选不替换默认基线。[旧失败记录](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/course_final_20261018/tp27b_tp1/failure.json)

## 3. 任务三：分析通信路径及计算、通信耗时

### 3.1 通信位置、传输内容与集合类型

当前默认 attention TP 与 dense TP 均为2，主路径如下。表内是代码合同，不是尚未采集的耗时结果。

| 位置 | 每 rank 提供的数据 | 集合通信与目的 |
| --- | --- | --- |
| 词表分片 embedding 后 | 本 rank 词表对应的局部 hidden，BF16 `[T,H]` | SUM all_reduce，恢复完整 hidden |
| 每层 FA 或 GDN 输出投影后 | 互补输入通道形成的部分和，BF16 `[T,H]` | SUM all_reduce，使两个 rank 获得相同完整 hidden |
| 每层 MLP down 后 | 互补中间通道形成的部分和，BF16 `[T,H]` | SUM all_reduce，恢复下一层输入 |
| 输出头后 | 局部词表 logits `[S,124160]`；S 为实际需要输出 logits 的位置数 | all_gather，拼回 `[S,248320]`；9B 为 FP32，其余为 BF16 |

4B 的32层各有两处层内 all_reduce，形成64个逻辑通信点；embedding 和词表输出通信另计，实际调用次数须以 trace 核对。默认路径不能写成 reduce-scatter/all-gather：并行宽度等条件不同的其他分支才可能采用那一组合。[通信实现说明](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/homework2_tp_layout.md)

例如4B 的 BF16 hidden 为 `[T,2560]`，单个逻辑缓冲区为 `5120×T` 字节。T=1和4时分别为5和20 KiB；T=128和2048时分别为0.625和10 MiB。这是输入缓冲区的形状换算，不是链路总字节、实际带宽或端到端耗时；分块、批处理、图填充与集合算法均会影响真实通信。小 T 时启动/同步成本可能占比较高，大 T 时计算量和消息规模同时增长，具体收益必须测量。

### 3.2 耗时分析方法及当前结果

代表模型选4B。普通性能测试关闭 Profiler；在相关正确性核对后，以固定源码另行采集两个 rank 的 Level1 trace。Profiler 协议为输入128/2048、并发1/4、输出16 tokens，每格4次预热及4个正式请求，不将其延迟填入输出32 tokens 的正式性能表。[Profiler协议](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/configs/homework2/profile_protocol.json)

每个 rank 使用自己的原生时间戳和明确窗口，区分设备计算、经确认的设备集合通信、未分类设备活动以及可见同步/等待。host 上的 HCCL enqueue 是提交调用时间，不能直接当作设备通信耗时；未确认名称的 device 任务保留 unknown，不强行归为通信。

设窗口内计算区间并集为 C，通信区间并集为 H，二者重叠为 `C∩H`，则计算或通信覆盖时长为 `|C∪H|=|C|+|H|−|C∩H|`。不能把每个任务持续时间直接相加，也不能把 rank0、rank1 时间相加当作端到端延迟。窗口内未被活动覆盖的部分还可能包含调度、采集缺失或空闲，只有可识别的同步依赖才能进一步解释为等待，不能把全部空白归因于 HCCL。

| 本轮4B通信分析项目 | rank0 | rank1 |
| --- | --- | --- |
| 计算、设备集合通信及 unknown 区间 | 未采集 | 未采集 |
| 计算/通信并集、交集与覆盖比例 | 未计算 | 未计算 |
| 同步/等待事件与跨 rank 关系 | 未分析 | 未分析 |

目前已完成分析工具及其软件测试，但尚无本轮真实双 rank trace。因此本稿不能给出“通信占比多少”“哪一层最慢”或“重叠隐藏了多少通信”的实测结论。

## 4. 任务四：性能对比与通信相关策略实验

### 4.1 已有4B历史结果及适用范围

2026-10-18 的旧源1415文件清单下，4B 在固定会话负载完成 TP1、TP2 实测。负载为2条会话、每条16轮、每轮固定8个输出 tokens，正式3轮；同时含 Session 与无状态两种执行方式。以下仅列 **Session 开启一侧**各轮结果，用于展示已有 TP 对照起点，不与本轮无 Session 矩阵混算。

| TP | 正式轮次 | 输出吞吐（token/s） | TTFT P50（ms） | E2E P50（ms） | 客户端 TPOT P50（ms） |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 28.57 | 114.90 | 325.99 | 25.08 |
| 1 | 2 | 29.05 | 123.23 | 300.66 | 24.99 |
| 1 | 3 | 28.28 | 106.95 | 369.35 | 25.06 |
| 2 | 1 | 35.01 | 111.08 | 245.06 | 18.09 |
| 2 | 2 | 34.53 | 112.19 | 245.34 | 18.29 |
| 2 | 3 | 34.35 | 111.32 | 244.71 | 18.12 |

数据直接取自第2.2节引用的两份审计 JSON。旧协议吞吐的分母包含工具等待，客户端 TPOT 按 `(E2E−TTFT)/7` 计算，不能当作设备 Decode kernel 时间。TP1 的全运行逐 chip 已用 HBM 观察峰值为 chip0 16986 MB；TP2 为 chip0 11375 MB、chip1 11037 MB。它们包括预热和臂间阶段，不是某一表格行的 HBM 峰值，也不是瞬时最大值。

在该有限负载中，TP2 的三轮吞吐均高于 TP1，单 chip 观察到的 HBM 较低；两 chip 合计占用没有因此减半。两臂使用不同 Engine、固定执行顺序，未随机化共享主机背景负载，也未拆解 HCCL 耗时，因而仅作描述性比较，不宣称通用加速比例或通信优化的因果收益。该负载没有独立控制两档输入长度和两档并发，不能替代作业2要求的完整矩阵。

### 4.2 本轮两长度×两并发的 TP1/TP2 矩阵

主代表模型为4B，使用原始输入 IDs 精确构造128与2048两档输入，设置并发1与4；输出固定32 tokens。每个“长度、并发、TP”测量格含4次预热、3轮各8个正式请求，因此每个 TP 配置共有96个正式请求、16个预热请求。预热从性能统计中排除，保留每轮实际顺序。四个模型的 TP1/TP2 运行均使用这一主协议；4B 是重点分析对象。[主协议](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/configs/homework2/matrix_protocol.json)

吞吐取一轮正式请求的总输出 tokens 除以该轮墙钟时长；客户端 TTFT 从提交到首个非空输出，E2E 从提交到流结束，近似 TPOT 另按 `(E2E−TTFT)/(输出数−1)` 解释，均不冒称设备算子时间。逐请求报告延迟分布，保留每轮结果；每轮8个请求的 P95 仅作有限样本描述，不作总体分位数保证或显著性推断。

| 4B 输入长度 | 并发 | TP1吞吐/延迟/HBM | TP2吞吐/延迟/HBM |
| --- | --- | --- | --- |
| 128 | 1 | 未测 | 未测 |
| 128 | 4 | 未测 | 未测 |
| 2048 | 1 | 未测 | 未测 |
| 2048 | 4 | 未测 | 未测 |

本轮 HBM 按 chip 分别保存查询原表及起止时间，目标间隔0.25秒；查询本身有耗时，不能把间隔当作瞬时连续监测。测量格的峰值仅使用查询起止均落在格内的样本；没有完整覆盖时记为未观测，不填0。全运行峰值单列，不能把别的测量格或预热阶段的值搬入当前格。

### 4.3 大模型 TP2 性能与容量

大模型选择27B，按相同主协议采集 TP2 的吞吐、延迟和两 chip HBM，并以独立容量协议执行4095输入加1输出，核对4096上下文边界实际可用。配置中的 `max_model_len=4096` 不是容量测试结果。[容量协议](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/configs/homework2/capacity_protocol.json)

已有旧源27B TP2 固定会话实测的两 chip 全运行已用 HBM 观察峰值分别为34828 MB、34489 MB，Session 侧三轮吞吐为15.68、15.16、15.55 token/s，负载仍为2会话×16轮×8输出；这些来自1409项源码清单的原生 Agent 性能旧轮次，与第4.1节4B的1415项旧源身份不同，也不是新矩阵或容量探针。[旧性能说明](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/project/native_agent_performance_20261018.md)、[27B旧审计](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/native_performance_20261018/27b/audit.json)

本轮27B 的 TP2 矩阵、逐格 HBM 和4095+1容量结果均未产生，TP1 默认与95%预算候选也未取得本轮结果。故不能计算本轮27B并行加速比，不能宣称4096边界已在本轮固定源复验通过。

### 4.4 通信相关实验：Prefill 分块粒度

**实验假设。** 在4B TP2、输入2048、并发4时，将 Prefill 分块上限从4096减至512，可能使调度分段和集合通信调用次数增加、单次消息变小，改变同步与等待开销，也可能改善请求推进的粒度。吞吐或尾延迟的变化方向不预先假定。

**改变的因素。** 仅改变 `chunked_prefill_size` 为4096/512，保持模型 revision、权重、精度、TP=2、设备对、输入、输出32、并发4及其他配置相同。两臂各4次预热、3轮各8个正式请求，另用独立 Profiler 采集策略时间线。[策略协议](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/configs/homework2/strategy_protocol.json)、[512候选配置](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/configs/homework2/candidates/qwen35_4b_tp2_chunk512.yaml)

**测量与解释。** 正式运行比较吞吐、TTFT/E2E 分布及逐 chip HBM；时间线核对实际分块、通信次数、消息形状、计算/通信覆盖与等待。虽然只改一个参数，它影响的是 Prefill 调度与每段的计算/通信规模，不能只报告“某参数更快”而省略机制。小 chunk 不保证每次消息恰为512 tokens，实际批次和调用应以记录为准。候选与基线还须逐项检查有限输出一致性，不能用速度提升掩盖输出差异。

| 策略 | 普通性能结果 | 双 rank 通信与计算结果 | 可作出的结论 |
| --- | --- | --- | --- |
| chunk4096 | 未测 | 未采集 | 尚无实测结论 |
| chunk512 | 未测 | 未采集 | 尚无实测结论 |

本稿尚未完成这一策略实验，不能把第3.1节按形状计算的消息规模当成实测结果，也不能先写优化收益再等待实验佐证。

## 5. 当前结论与任务覆盖

静态实现表明，现有统一执行路径具有 FA、GDN、MLP 和词表的张量切分与 HCCL 通信合同；配对配置、采集与独立审计工具已有软件检查。旧源4B和27B硬件结果为后续对照提供起点，但不足以证明本轮四模型在同一固定源上完成作业2。

| 作业2要求 | 对应章节 | 当前证据状态 |
| --- | --- | --- |
| 四模型同卡 TP2，权重切分、启动、通信及连续推理 | 第1、2章 | 静态说明与配置已具备；本轮硬件未验收 |
| 典型层的双 rank 形状、FA/GDN 与状态分布 | 第1.2、1.3节 | 已按实际模型几何列出；待运行记录核对 |
| 两个 rank 启动/运行、设备映射和逐 chip HBM | 第2.1节、第4章 | 协议与工具已具备；本轮记录待补 |
| 同权重/精度/负载的 TP1/TP2 正确性 | 第2章 | 保留历史结果；本轮配对未测 |
| 至少一个模型的计算、通信与等待分析 | 第3章 | 静态通信点已说明；实测时间线待补 |
| 两长度×两并发×TP1/TP2；大模型性能与容量 | 第4.2、4.3节 | 协议已固定；本轮结果待补 |
| 至少一项通信策略的假设、因素、结果与结论 | 第4.4节 | 假设及因素已固定；结果与结论待补 |

因此，**当前阶段稿已按任务书组织内容，但尚未满足作业2全部实验要求，不作为提交终稿。** 终稿需要补齐四模型本轮连续推理与正确性记录、4B完整矩阵、27B性能/容量、双 rank 时间线和分块策略结论，再逐项复核源码与报告一致性。

## 附录A：复现与证据入口

本轮待执行实验的代码基点为 `536e54c034c2aeecb5213375593a919046fc623d`，分支为 `homework2/in-progress`。1531项运行源清单 SHA256 为：

```text
2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3
```

[源清单](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/source_manifest.json)覆盖运行时、内核、scheduler、配置、脚本、基准和课程测试。旧数据保留各自身份：第4.1节4B性能为1415项清单，SHA256 `d05a47d932b519b6e50ed0c668725935d93b744fffffa68e95f37a0227f6c6ea`；第4.3节27B性能为1409项清单，SHA256 `2c9e47616eb3c460a740e2542e776ab655934a9ad7ed0ac0c1401a008a97d712`；第2.2节27B正确性另为1396项清单，SHA256 `f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e`。这些不能混用。仅新增本报告的文档提交不会把历史实验追认为新源实测。

从已准备环境的仓库根目录运行，先检查映射及整卡进程：

```bash
source scripts/use_project_env.sh
npu-smi info -m
npu-smi info -t proc-mem -i CARD_ID
```

其中 `CARD_ID` 必须替换成当次确认的同卡 NPU ID。仅当该卡两个 chip 均无其他任务时运行实验。以下是4B TP2的**手动复现模板，不是本轮已成功执行记录**；映射、源清单位置及新输出目录必须按实际环境填写，不可与自动队列同时启动：

```bash
export TOKENSPEED_MAMBA_SSM_DTYPE=float32
export ASCEND_RT_VISIBLE_DEVICES=LOGICAL0,LOGICAL1
python scripts/run_tp_matrix.py \
  --config configs/homework2/qwen35_4b_tp2.yaml \
  --protocol configs/homework2/matrix_protocol.json \
  --source-manifest docs/evidence/homework2_20261023/source_manifest.json \
  --npu-id CARD_ID \
  --chip-logic-ids 0:LOGICAL0,1:LOGICAL1 \
  --output-dir results/raw/homework2_NEW_RUN/4b_tp2 \
  --profile-mode off
```

实际自动队列另核对固定的启动导入钩子、scheduler 二进制、模型文件和退出清理收据，详见[准备与固定源证据](https://github.com/jiran-1111/tokenspeed_PKU/blob/536e54c034c2aeecb5213375593a919046fc623d/docs/evidence/homework2_20261023/README.md)。所有依赖、临时文件、权重与采集结果留在项目工作区；实验串行，不终止其他用户进程，不覆盖既有结果，不在采集后修改协议。大体积原始 trace 不随源码提交，公开证据保留文件校验和；未实际公开的原始文件不能只凭校验和推定为可下载。
