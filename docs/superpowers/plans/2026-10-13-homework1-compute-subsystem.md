# 作业1计算子系统增量实施计划

> **For agentic workers:** Use executing-plans or subagent-driven-development task-by-task. 保留当前工作区与历史结果，不提交或重置无关修改。

**Goal:** 在2026-10-27前交付0.8B单chip计算适配、分层数值/状态验证、可比较基础性能数据及成员贡献开头的第一阶段报告。

**Architecture:** 复用现有Ascend模型/算子实现、项目Conda、测试与测量工具。新增的是第一阶段证据索引和欠缺的验证覆盖，不重建引擎；原四模型Agent总项目继续保留。

**Tech Stack:** 当前项目Python/PyTorch/torch_npu/Transformers/Triton-Ascend/CANN；不新增下载或共享环境安装。文件都留在项目内。

**执行责任（用户补充确认）：** 三人分工是文档职责安排，不是把工作交还成员执行。助手继续完成作业1的项目实现、真实硬件实验和报告，随后继续原大项目下一步；不得仅完成分工/草稿即停止。成员实际贡献按真实参与核对，助手生成或执行的内容不得冒称成员已亲自完成。

## Task 1：保护在运行实验与整理计算路径

- [x] 读取作业1并建立逐项对照：`docs/project/homework1_requirements_audit_20261013.md`。
- [x] 27B E0正常退出0，核验15 summaries/480 requests、30硬件快照，全部实际479/128；41个文件同步本机并跨端SHA-256一致，生成summary.csv。仅停止本次服务并确认无NPU进程后启动作业1新数值测试；详情 `docs/project/27b_e0_workload_a_20261013.md`。
- [x] 新建 `docs/project/homework1_operator_matrix.md`，从实际固定revision的0.8B配置提取几何参数；列功能、symbolic及实际主要输入/输出、dtype、源码来源、NPU支持状态、九个测试入口。未实测/覆盖不足项明确标记，CUDA-only卷积测试不算NPU证据；最终版本NPU复验仍属于Task2。
- [x] 新建 `docs/project/homework1_gdn_path.md`，追踪Qwen模型→线性投影/卷积→GDN→gated norm，解释QKV/head布局、conv与SSM state方向/索引、Prefill与Decode状态边界和诊断同步。依据当前源码，不声称新数值验证或性能归因完成。

## Task 2：分层数值和有状态验证

真实尺寸补充：新 `test_homework1_norm_rope_npu.py` 在预定合同下70项NPU通过，包含Norm width256/1024及实际runtime MRotaryEmbedding head256/rotary64路径；FP32额外1e-5，未旋转部分/残差精确不变。70 cases/76 properties XML同步哈希一致。原通用RoPE测试仅作回归，不替代此actual路径；完整模型数值仍属Task3。

最新实测（2026-10-13）：新增GDN与runtime卷积共10项在logical chip0通过；既有6个kernel测试文件共30项通过，runtime层Norm/backend及Q/gate12项通过（含CPU布局检查）。新增XML含92条输出/state误差properties、无skip/error/failure，源文件及三个XML跨端哈希一致，预定阈值未修改。`docs/project/homework1_numerical_validation_20261013.md`记录数值和警告。下方旧准备状态保留为历史；模块整体判据与完整引擎验证未完成，不把Task2整体提前标通过。

准备状态（2026-10-13）：新增 `tokenspeed-kernel-npu/test/test_gdn_homework1.py`，含真实16-head Prefill长度1/8/127/128/129、FP32 state参考及slot1/3交错各8步Decode；未选slot精确不变。新增 `docs/project/homework1_numerical_contract.md` 在实测前固定输出/状态判据、解释依据及限制。已做AST语法检查，未执行NPU测试；不提前标记下面实测/覆盖验收完成。

随后补上GDN独立read/write slot测试及 `test/runtime/test_causal_conv1d_homework1_npu.py`，后者直接验证runtime卷积通道16/6143/6144、width4、两slot各8步，与CPU FP32点积+SiLU对照，历史及未选slot精确不变。两文件AST语法检查通过，未运行NPU；避免拿CUDA-only通用卷积测试替代Ascend验证。

- [x] 复查并运行上述6个现有kernel测试文件及runtime Norm/backend/Q-gate测试：30+12项通过，无skip/error/failure，单chip0、项目Conda及项目`.tmp/`，两个XML同步本机并跨端SHA-256一致。含CPU布局检查，不称全部42项为NPU kernel。
- [x] 在独立 `test_gdn_homework1.py` 补齐原GDN测试覆盖不足：真实0.8B头数与head维度；长度1/8/127/128/129；多步Decode输出和最终FP32 state；两个非相邻slot交错更新及独立读写槽、未选slot精确不变。使用可信CPU FP32递推，7项NPU通过；无新增生产实现修正或阈值放宽。
- [x] 确认通用卷积测试CUDA-only不足，新增实际runtime卷积NPU参考测试，channels16/6143/6144两slot交错各8步，三个参数项全部通过；历史/未选槽精确不变，不用CPU-only测试证明NPU支持。
- [x] 新建 `docs/project/homework1_numerical_contract.md`，在新增GDN/卷积及实际Norm/MRoPE实测前固定判据，解释BF16舍入、FP32 state、近零值和隔离要求；新增结果含max-abs、relative-L2及finite/state-isolation断言。合同存在和对应模块通过不代表全部模块或完整模型验收。
- [ ] 对其余既有模块的dtype/参考/判据依据完成最终审查；不得机械沿用GDN0.03或为观察到的失败扩大阈值。端到端B方向新规格独立评审，不覆盖既有合同失败。
- [ ] 源码覆盖核对发现MHA原参考仍为NPU BF16矩阵计算；D256仅短Prefill，D64缓存长度均未跨128页边界。补独立CPU参考和实际D256的127/128/129等缓存边界，新增合同先固定再采集；solve_tril单32×32随机回归不外推条件数稳定性，gated norm代表row/近epsilon与原数值指标缺项单独核验。见模块合同2026-10-14补充审计；当前只记录缺项，不改旧门槛或运行新测试。
- [ ] 后续源码审计确认旧GDN Prefill参考clamp与生产/固定HF sum+eps不同；保留原记录/容差，在参考定义评审边界下纠正测试并补近epsilon形状，再新目录复验。已有真实输入CPU FP64只读核对通过两项旧包络，不替代这一欠项或B正式校准；见`docs/project/homework1_gdn_reference_audit_20261014.md`。

## Task 3：端到端与状态隔离

独立参考控制（2026-10-14）：保留既有HF-NPU BF16 1%工程合同失败，不事后改门槛。对同native128输入，在CPU独立参考进程以16线程分别测HF BF16/SDPA、FP32/SDPA、FP32/eager，记录实现/精度/完整向量。此为参考敏感性校准，不是通用CPU serving fallback、不替换NPU模型计算、不宣称CPU控制通过即可替代原验收。控制前固定组合，先看参考自身差异及FP32实现一致性，再决定需要的进一步数值定位；不继续无依据叠加临时兼容算子。

逐层诊断细化（2026-10-14）：固定此前128 native IDs、BF16/TP1单个空闲logical2、新目录独占。只在guarded Prefill中临时采24层输入/输出的hidden+residual（按输入dtype相加）及final norm；HF同输入同dtype逐层hooks对齐，保存全128行，而非只比较norm摘要。全logits与此前版本复核，防止hook意外改变输出。TP1边界重组不外推TP通信；不改生产模型、不采Profiler、不用于性能。不通过原数值合同则保留失败，下一步按最早差异诊断。

2026-10-14进展：已按`homework1_logits_contract_20261013.md`捕获native128共同前缀首token全向量，与同revision HF对照。原模型relative-L2=0.024067251；NPU Gemma FP32 gamma、Q/K FP32计算恢复BF16修正后=0.024128901，均未通过预定0.01，不能因top2近并列或单元74+12项通过而验收。中间启动dtype失败保留。必须继续定位；最终性能须标明修正前后版本，不能覆盖旧raw。

执行细化（在已授权增量范围内）：精确native两档使用固定chat template的前N−64 tokens加末64 tokens，N128/1024，记录构造来源而非冒称自然文本恰好该长度；各32/128输出已完成四条原始ID记录。不同长度请求交错四次与各自独立baseline逐ID对照，是stateless请求证据，不代替Session。新增硬件验收测试 `test/runtime/test_homework1_session_npu.py` 复用现有Session helper：同一0.8B Engine中两会话各两轮对照独立全历史，关闭拒绝旧ID；重开新会话再跑，比较输入/输出与前一轮完全相同。两次测试round共8个Session/control配对、4个closed rejection和4个重开跨round配对。current-frontier不额外物化尾token，仍用原合同验证reused/new tokens。不存在新生产引擎修改，测试失败后才考虑最小修正。

- [x] 用固定0.8B revision、BF16、greedy，对实际输入128/1024两档连续生成32/128两档，四条均达到生成上限，保存native IDs、实际长度和构造来源；与性能矩阵使用同一单chip/TP1固定服务。参考对照仍单独待做，不因完成采集标为正确性全通过。
- [ ] 对共同前缀对比参考logits/输出、多步GDN状态；保留strict token对照和分叉解释，按Task2预定数值判据验收，不仅凭“近并列”判通过。
- [x] 原生Engine两个Session并行各两轮，8组Session/control、4次closed rejection、4组关闭重开跨round对照通过，context metadata正确；第三次XML同步且跨端哈希一致。首两次template辅助脚本失败保留，最小修正采用实际前缀与append-only闭合think边界；不修改引擎，53项聚焦与全套411/13回归通过。范围与警告见 `docs/project/homework1_session_acceptance_20261013.md`，不是任意长会话证明。

## Task 4：统一可比较性能矩阵

2026-10-14补证据：当前Gamma/QK修正后生产源码、无诊断/Profiler，新进程完成96条正式请求/12 summaries和24轮次快照，actual130/940×32/128；九源码hash结束后未变、46 raw文件cross-SHA一致。121个HBM样本覆盖startup/两输出负载/teardown，loaded8461MB、两负载采样peak8465MB、exit3117MB；最大采样起点间隔2.414s，不称瞬时峰值。服务/编排正常退出0、四卡无进程。见`homework1_baseline_hbm_20261014.md`。现有六Level1采集按全部正AICore任务计数归组得三轮一致资源Top3，但历史混合阶段不能代替当前稳态Decode/完整因果归因。全模型数值与最终版本复验仍保留。

执行细化：复用 `scripts/run_benchmark.py`，四条件各三轮、每轮8请求、每条件预热3请求，c1；新目录 `results/raw/homework1_baseline_20261013/out32` 与 `out128`，合计96正式请求/12 summaries。输入128/1024是目标，须核对chat/template后服务端actual长度；不冒称native精确128/1024。Task3的native-ID两档精确长度验证独立保留。测量不与reference/NPU测试并行。

- [x] 固定单chip、0.8B/TP1/BF16配置完成target128/1024×输出32/128，四条件各三轮、每轮8条、c1、预热排除；96条正式请求/12 summaries。actual输入130/940、输出32/128无提前EOS，39文件跨端SHA一致。不是native精确128/1024验收；结果及计时口径见 `docs/project/homework1_model_baseline_20261013.md`。
- [x] 当前生产版本首token客户端TTFT、Decode客户端TPOT与服务端ITL直方图口径分开记录；HBM加载/负载采样峰值/退出快照齐全，限定采样周期及尖峰遗漏，不把driver保留HBM称有效KV。完整模型尚未数值验收，后续若生产计算改动需复验，不提前称最终交付基线。
- [ ] 分离Prefill与Decode测量；复用已有Profiler热点及单请求窗口，必要时采同协议Prefill/Decode算子数据。比较计算、访存、转换、Host调度的证据，未确定因果的候选明确标注。
- [ ] 对既有数据先核对配置/实际长度/重复数是否能覆盖四条件；覆盖则复算，不为新计划机械重跑。新raw/analysis目录独占创建且不覆盖旧数据。

## Task 5：第一阶段报告与验收

2026-10-14资料核对：算子矩阵与GDN路径已对照当前源码修正Gemma gamma/残差、Q/K恢复BF16、真实形状/多步验证状态；重新解析76/10/30/12项归档XML均无fail/error/skip，两份当前Norm源SHA匹配当前性能归档。临时精度组合仍明确为诊断而非生产修复；这里只补文档来源与覆盖，不完成Task2/3/5整体验收或启动新实验。

准备状态（2026-10-13）：用户提供张晟祺、李书砚、计然三位成员，要求工作量相近、边界合理且尽量解耦。报告草稿 `docs/report/homework1_compute_subsystem.md` 已建立：张晟祺负责环境/计算适配与路径，李书砚负责数值/状态/端到端验证，计然负责性能/数据/报告整合；每人负责本块核验和报告段落，最终共同复核。此为职责安排，不冒称已完成贡献或真实工时统计；提交前按实际完成情况核对。草稿明确新NPU测试、统一2×2性能及最终验收待完成，因此下方交付复选框保持未勾选。

- [ ] 完成 `docs/report/homework1_compute_subsystem.md`，开头写经成员核对的真实主要贡献；已有职责草稿不能代替实际贡献。正文含环境/启动、算子矩阵、GDN路径、分层验证及误差依据、2×2性能、热点证据、最大适配障碍、复用/修改边界和限制。
- [ ] 从干净的新进程按README命令复验单chip0.8B启动、关键模块测试和连续推理；核对报告链接、原始数据/图表和源码版本。自动化测试通过不替代真实NPU结果。
- [ ] 更新README、submission_status及总计划，第一阶段与完整课程验收分开；教学网提交留给用户，除非用户另行授权外部提交。

成功标准以作业1全文及对照表为准。所有任务未完成前，不宣告阶段交付完成；后续27B/多模型/Agent要求不删除。
