# 作业1独立参考：采集终态与分析审计

## 当前结论

阶段A四组参考均完成16条固定轨迹，每条末端Prefill加16次固定续接；每条同时保存缓存和完整prefix重算路径。结构、身份、finite和分析计算检查通过。针对最早BF16差异完成局部受控复核后，独立审查允许提出有限经验判据；用户已确认并冻结，见[冻结记录](homework1_frozen_numerical_gate_20261015.md)。验证组引擎向量仍未采集；旧额外1%工程检查失败保留。通过该有限门槛不等于普遍正确性证明。

分析包：`results/analysis/homework1_reference_calibration_20261015/`。原始包：`results/raw/homework1_reference_calibration_20261014/`。完整张量不上GitHub。

原始数据SCP正常退出0后，本机已逐一重哈希144/144，与远端冻结provenance全部一致。两SVG通过XML解析；这不是图形视觉审查。

## 来源与覆盖

- 固定模型revision：`2fc06364715b967f1860aea9cf38778875588b17`。
- 输入manifest文件SHA256：`23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb`。
- 四参考320个learned parameters展开后的共同值SHA256：`c2084362ae79d1d0f3d61a9781598296fae0944efe4e6fd04888a48692c57884`；不是直接将所有原检查点参数一律BF16化。
- CPU FP32/eager、CPU FP32/SDPA、CPU BF16/SDPA、NPU BF16/SDPA；同IDs、同续接、同绝对位置和完整输入前沿。
- 64个PT，每个两条17×248320 logits与36对最终GDN状态；各参考loading的四类异常列表均为空。
- NPU采集包装退出0；CPU原包装退出码未恢复，不补写为0。CPU结果完整性另由实际读取张量和身份审计确认。

## 独立只读审计

独立审计者没有运行模型、初始化NPU、修改原始数据或查看验证组引擎数值。检查了144个raw文件、8个选定来源和归档分析源的SHA；全部匹配provenance。

3808条vectors、39168条states的组合主键唯一；全部JSON的step/prefix/state键匹配CSV。28组summary均按16条轨迹的阶段最大误差重新计算，均值、样本SD、最大值和top1计数匹配。17个依赖步骤没有充当17次独立实验，不给出置信区间、p值或总体覆盖概率。

为避免共享`compare_tensor`实现风险，另用NumPy的`sqrt(dot(delta,delta))/sqrt(dot(reference,reference))`抽查case02和case15、step0/16的FP32实现、NPU BF16对主参考、NPU缓存对重算三类比较；relative-L2、maxabs和argmax全部匹配。四参考这两个case共288对最终状态的L2/maxabs独立重算也匹配。这是明确范围抽验，不冒称所有向量均独立重算。

纯CPU审计第一次import torch触发torch_npu自动加载而因libhccl不可见失败；随后仅该进程设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`完成。无共享环境修改，无NPU forward。

## 观测与尚未解决的判断

FP32 SDPA/eager及缓存/重算的最大relative-L2均约数个1e-6，所有对应top1一致，支持本批输入的实现一致性；共享HF实现家族不能证明绝对真值。BF16路径出现百分比量级差异和少量Decode top1变化，必须与FP32一致性分开解释，不能自动豁免引擎误差。

仅从CPU/NPU BF16各自对CPU FP32/eager、汇集cached/recomputed，可算出以下**未批准的观测最大值**，不是充分正确性门槛或理论上界：

| 阶段 | relative-L2最大值 | max-abs最大值 |
|---|---:|---:|
| Prefill | 0.04160680933973707 | 0.4934459924697876 |
| Decode | 0.027781371644764415 | 0.4458632469177246 |

Prefill两个最大值由CPU BF16、calibration/1/0、step0产生；Decode L2来自CPU BF16重算、calibration/2048/1、step5，maxabs来自NPU BF16缓存、calibration/512/1、step2。不同来源最大值构成矩形允许域，可能包含未实际观察的误差组合；固定样本最大值没有泛化覆盖保证。

首次汇总时，尚未解释的参考差异暂时阻止了候选判据形成；后续局部受控实验和只读批处理审计把最早分歧定位到BF16 `rsqrt` 的调用尺寸效应。独立复审认为这足以支持“可提出有限经验判据”，不构成全模型误差分解或普遍误差上界。用户随后明确批准冻结原观测最大值并开展隔离验证；阈值未根据验证组结果修改。

## 后续静态定位：缓存状态舍入已证实

固定HF模型源SHA `395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6`、cache源SHA `2783b2702910dbb6e70aa05a093f71e1c51c110d7fa73c904fc38d12e5882545`。模型523–548行在单token缓存分支调用recurrent、整段重算调用chunk（默认64）；两个fallback归一化在原dtype执行，然后核心升FP32，最终state仍FP32。cache926/937行按conv输入dtype分配recurrent缓冲，986–989行copy将核心FP32 state存为BF16；模型下一步456–458/530行读回该量化state。于是cached每forward边界量化，full仅在当前完整prefix结束后量化。这是已核对源码事实，但不是全部误差的定量解释。

新校准长度1/source0、step1、第0层CPU BF16 conv状态完全相同，而recurrent relative-L2=0.003291905784112513、maxabs=0.00390625；两个FP32同位置仅6.874997241905618e-8/1.7881393432617188e-7。全部16例step1的CPU BF16第0层conv均相同、recurrent均非零。机制与最早分歧相容，尚未证明其解释最大logits差异或所有top1换序；不以此宣布参考可信性通过。

## 局部受控实验：舍入并非完整解释

后续CPU-only probe对calibration/1/0执行prefill1、cachednext1、fullprefix2三次短forward，原模型/权重/代码身份不变，不采engine/验证组。全部原core调用逐位重放，actual cached initial与prefill核心FP32 state的BF16回转逐位相同。实际cached/full末token的query/key/value/g/beta五类输入全部逐位相同，conv相同，recurrent差异精确复现上述StageA数值。

固定cached step所有输入，仅将initial state从prefill FP32改为其BF16回转：核心state relative-L2=0.000562519055109435，输出=0.0004881682006999061。随后对原captured两token BF16 Q/K 做逐阶段离线CPU审计，发现square、sum、epsilon逐位一致；差异出现在32元素整批与每次16元素调用的 `torch.rsqrt`，query有7/32、key有11/32个结果不同，最大绝对差分别为0.015625和0.03125。FP32 rsqrt后转BF16与整批BF16 rsqrt逐位相同；固定已经归一化的Q/K后，整段recurrent与chunk的输出均逐位相同、state relative-L2约7.52e-8。原始整批归一化/逐token归一化下的较大state差异仍被完整保留，不据机制定位将其清零或豁免。

原始目录`results/raw/homework1_reference_state_rounding_20261015/`；四文件同步后本机SHA与远端一致：metrics.json `785846041f083ac242ae914295553c2a8a799fd081460cbef77e382d33ede257`、probe_source.py `5ac36c1974da52d7a3a457cbc7687b683e8554277c36140232d2dc912b4f2cd9`、protocol.json `54dc835969d06a80f9670349f35b15cf938f839268f431077cefeda6638f3e2c`、tensors.pt `75a84226bdb1d42632863670afee33b41a8b3483832c48eb59b9063be10211b4`。进程退出0；该结果是单层单样例受控诊断，不是新参考准入门槛或完整模型通过。

只读批处理审计`results/analysis/homework1_hf_norm_batching_20261015/`在服务器退出码0；输入张量、Transformers源码及cache源码的SHA均匹配冻结身份。本机复制后的protocol、audit source与metrics逐文件SHA和服务器一致（分别为`9ab898184e1fdb9302bd4d495090043e676ce0952a1abb79207f54aca3c73118`、`8f75ecabb193f0533ff3d0ec3b208cc698f065d9339a372eec4f22c7a4c3abc7`、`17bd1a12aae98673e2b23ab612c7c283590acda7a80a21e9484a9689a6aba731`）。该审计只读取已冻结18MB捕获张量，没有加载模型、执行forward或使用NPU；全部发现只适用于该单例局部机制。
