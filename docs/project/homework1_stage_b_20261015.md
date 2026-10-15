# HW1 Stage B：冻结验证组执行记录

状态：已完成本冻结协议的16条轨迹/272向量验收；全部通过有限经验双门槛。原1%失败、条件argmax差异及失败尝试单列保留。具体终态见末节；下方实施记录按发生时点保留。

## 固定协议

沿用[用户批准的冻结记录](homework1_frozen_numerical_gate_20261015.md)：原manifest中validation组16条输入，原顺序、原16-token续接；输入长度1/8/127/128/129/512/1024/2048，各两条。文件SHA为`23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb`，canonical SHA为`f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936`。

模型Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`，NPU BF16/TP1/单chip。主参考为同投影learned weights的CPU FP32/eager连续缓存路径，CPU线程1（在验证引擎数值采集前固定；校准单例线程探针对比见下文）。每条保存末端Prefill及Decode step1–16，共272个248320维logits。误差使用CPU FP64计算，逐向量同时应用冻结relative-L2/max-abs两项门槛，任一超限即失败，保留全部失败。

## 实施边界

- `collect_homework1_validation_reference.py`复用原采集器的完整manifest检查、模型加载、权重投影和身份哈希；明确注入缓存路径采集函数。原Stage A入口仍明确选择原来的缓存/重算及状态比较函数，原始Stage A数据不变。
- `homework1_stage_b_capture.py`只在诊断进程显式安装：在ModelRunner返回后、NaN guard/采样前保存完整原始向量及实际输入ID/绝对位置；运行原采样器并校验argmax，随后仅替换输出token以控制下一步输入。绝不修改logits。
- 引擎诊断显式eager，避免图回放绕过Python捕获；prefix cache、KVStore及overlap均关闭，每次串行一个完整请求。该模式与图执行性能样本分别记录，不纳入性能统计。
- 原始argmax是固定输入前沿下的条件greedy选择，不冒充自由生成的greedy轨迹；历史raw/stop-aligned greedy、模块状态、Session和旧1%工程失败分别保留。
- `analyze_homework1_stage_b.py`逐条检查身份、frontier、shape、finite和采样记录，再应用原冻结门槛；单独列原1%工程判据结果。

## 已有证据与运行记录

原始目录：`results/raw/homework1_stage_b_20261015/`。

- 初始服务器参考选组及原采集器测试38passed；随后捕获/防篡改等组合44passed，无skip。
- 增加缓存专用采集后，组合首次50passed/1failed：原异常保留测试的mock签名未包含新增必填回调参数，属于测试契约不匹配；已精确更新mock并保留故意失败及不覆盖原始文件的断言。修正后`cached_preflight_fixed.xml`为51passed、无skip。
- 首次CPU尝试保留在`cpu-fp32-eager/`，因复用Stage A的全前缀重算/全部状态审计导致额外开销，核实进程身份后仅终止本次CPU进程。保留日志、身份、源文件和`cpu_interrupted.json`，没有删除样例、改门槛或产生可供挑选的引擎误差。
- 主参考新运行目录`cpu-fp32-eager-cached-threads1/`，退出码另存`cpu_cached_threads1_exit.txt`。16线程缓存尝试保留；校准单例仅改变CPU线程1/16，Prefill为2.33/22.93秒，全向量exact（`cpu_thread_probe.json`），因此采引擎前固定线程1以减少CPU并行开销，不改变数学路径、样例、续接或门槛。
- 初始1376个相关源码/配置/测试文件跨端核对：request_handler仅行尾差异；scheduler测试缺少一个既有准入测试。先保存两源到`remote_before/`，再同步本机内容；后续新增源纳入重新生成的manifest。
- 硬件只读检查发现物理NPU0现存任务，保持不动；物理NPU1空闲。每次硬件实验前重新核查逻辑2→物理1/chip0映射及无进程条件。

## 后续顺序

CPU参考完成并核对身份→重新验证源码manifest和设备空闲→单NPU采集16条连续轨迹→272向量完整分析及失败保留→数值终态后固定源码并串行Profiler→报告/源码匹配复验与GitHub allowlist交付。当前不能宣称HW1或四模型课程项目完成。

交付准备：独立分支`course/hw1-20261015`以本地HEAD为基点；目标main与本地历史分叉，不覆盖main、不强推。完整pre-commit已运行并修正格式；额外一个MLA测试仅增加isort要求的空行。格式化后的本机course为541passed/23skipped。一次交付回归误把basetemp放在另一checkout下，保留1failed/4errors记录；修正调用路径后全套通过，无测试断言改动。


## 2026-10-15 Stage B终态（接续更新）

主参考CPU FP32/eager缓存路径16/16、threads1，320投影参数共同SHA与Stage A一致。最终交付源码1396文件跨端对齐；三轴MRoPE诊断修复后服务器course为569passed/0skip。引擎R2全部16条、272个248320维向量完成，退出0，physical NPU1释放；R1因诊断位置展平错误在第一向量前失败，目录/退出/修复RED→GREEN均保留。

冻结双门槛272/272通过、0失败：Prefill最大relative-L2=0.021534648897684416、maxabs=0.32581615447998047；Decode最大relative-L2=0.019179627022419887、maxabs=0.4235382080078125。原1%工程门槛在本次仍68向量失败，固定前沿条件argmax有13处不同；均独立保留，不称自由greedy轨迹一致。NumPy独立公式复算全部272向量、0不一致，L2最大重算差2.501471252358556e-15。

完整分析为`results/analysis/homework1_stage_b_20261015_r2/`。第一分析已写summary后因服务器缺冻结记录而在provenance步骤失败，保留旧目录，补齐原冻结文件后在新目录复算成功；未改数值/源码/门槛。源码manifest为raw根`source_manifest_r2.json`，交付分支为`course/hw1-20261015`（隔离目录`delivery/hw1-20261015`，尚无commit/push）。下一步串行最终Decode Profiler，然后报告/最终匹配复验与GitHub发布；总课程项目未完成。


## 最终交付源R3复验

Profiler增量流修复只改变采集脚本/测试；生产模型、engine、kernel未改。1396文件R3清单SHA为`f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e`，服务器course569passed/0skip。为匹配最终交付又重采16轨迹，退出0；全部272个PT文件与R2逐一SHA相同，冻结门槛仍全过，旧1%68失败/条件argmax13差异不变。对R3实际raw又用NumPy独立公式全量复算，0不一致。

最终分析为`results/analysis/homework1_stage_b_final_20261015_r3/`。其后成功完成[最终Decode Profiler](homework1_final_decode_profile_20261015.md)、同源码关键模块/Session108项实测以及96请求2×2性能/HBM。完整最终809文件本机与服务器逐项SHA一致；R3张量体只在核对服务器canonical SHA后复用已同步、逐字节相同的R2文件内容，元数据及其余原始产物实际归档同步，不补造观测。公开标量和复现入口见[交付证据](../evidence/hw1_20261015/README.md)。
