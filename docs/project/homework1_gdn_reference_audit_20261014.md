# GDN归一化参考语义审计（2026-10-14）

## 发现与影响

当前生产Prefill与Decode、固定安装的HF fallback均以`s=sum(x*x)`、`eps=1e-6`计算`x/sqrt(s+eps)`。旧Prefill测试参考及真实激活独立递推脚本采用`x/sqrt(max(s,eps))`，不具有相同数学定义；此前将二者说成生产Prefill/Decode有不同epsilon约定，是参考说明错误，而非已证实的生产内核错误。

旧结果及预定模块容差全部保留。本次没有修改生产、测试或旧合同，没有新NPU forward，没有开始待评审的独立校准实验。仅审计当前源码、核对既有归档输入，并在CPU上按正确加法epsilon重算同一真实Prefill样例。完整模型1%失败不变。

## 源码与版本证据

- 生产`_triton/l2norm.py`两个kernel均使用平方和加eps；默认eps为1e-6。
- `_triton/chunk.py::ChunkGatedDeltaRuleFunction.forward`在NPU上先将Q/K展开FP32，再调用上述l2norm，记录路径的`output_h=False`。
- `gdn/triton.py`Decode归一化明确为平方和加1e-6。
- 固定远端Transformers5.12.0的`models/qwen3_5/modeling_qwen3_5.py::l2norm`同样加eps；已完整读取其l2norm、chunk和recurrent fallback函数，不依赖网上最新版替代固定安装源码。
- `test_gdn_homework1.py::_step(prefill=True)`、旧`test_gdn_prefill.py`及`.tmp/homework1_gdn_trained_oracle_20261014.py`使用clamp。随机长度边界并不覆盖接近epsilon的norm边界，因此旧测试通过不能证明这些两种定义等价。

本轮本机/远端三份生产源码SHA逐字节相同：

| 文件（相对GDN目录） | SHA-256 |
|---|---|
| `_triton/l2norm.py` | `42b8d827846667afcd01d0d5c2d87e1536bb052b272f444c2022948800912fff` |
| `_triton/chunk.py` | `722d96f0937d7b03a096539b94804d9884aa80317f643a24a7f3e4461f01012e` |
| `triton.py` | `355ebe1dc4e3be53f82025697ec462d61835c190fc5a41be32a0f1da57046c5a` |

远端固定HF模块SHA为`395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6`；未安装/更新Transformers。

## 既有真实输入的只读CPU核对

输入仍为`results/raw/homework1_gdn_replay_20261014/`中engine conv/g/beta/core/state，128 tokens、16 heads、K/V=128、零初态；5个文件逐一按既有manifest核对SHA。CPU FP64递推与旧脚本相同，只将归一化改为平方和加eps；状态方向、scale、门控输入和逐token更新不变。旧FP64输出/state两个文件另按其comparison.json哈希核对，不重算或覆盖旧归档。

2048个Q head向量的最小平方和0.007588594，K的最小值0.001936572；均大于1e-6，没有任何一个落在epsilon附近。该数据下clamp与加eps归一化向量relative-L2分别约7.559e-6、2.710e-5，说明差异虽小却不为零。近零输入仍未在本次核对中验证。

| NPU既有张量 vs 加eps CPU FP64参考 | max-abs | relative-L2 | 原模块包络 |
|---|---:|---:|---|
| 输出（参考后舍入BF16） | 9.536743e-7 | 3.614816e-7 | 通过，atol0.002/rtol0.02 |
| FP32 state | 6.568204e-7 | 1.213531e-7 | 通过，atol0.0003/rtol0.01 |

两者各262144元素、全部finite；输出262135个元素精确相等，不声称全向量逐位相同。只进行了FP64参考的两项原包络检查，不把它误写为又跑了一套FP32/FP64四项检查。

加eps参考与旧clamp FP64参考的输出relative-L2约1.39349e-5，state约4.42884e-6；旧state的差异主要受这项参考定义影响。但只覆盖一个layer0实际Prefill输入，不能推断全部层、连续Decode或全模型误差来源。本样例的新同输入证据不支持修改生产GDN归一化算法。

## 证据成熟度与后续

### 后续已实施：参考纠正与epsilon CPU自检

旧clamp参考已按固定数学合同修正，未改生产kernel或任何原NPU容差。两个测试源的当前SHA为40de22f7297e34d592ed82db587a8385a3a0e90e8ae6864a5daf58c703285fb7（homework1）及7ef0c0d3761f01b3ab446ef525a68ba770b0cf442bae11d56c7de4d088f4c6f9（旧prefill）。SPEC与quality均实际读取源码并通过。新增四ratio0/.5/1/2，以实际BF16量化后的16×128 Q/K及FP32零初态，CPU FP64解析式独立检查输出/state。

服务器从实际测试源AST只提取helper及CPU测试函数，不导入NPU算子：旧clamp源四ratio分别pass/fail/fail/fail，纠正后全部pass、GREEN进程退出0。RED及GREEN源、逐ratio原始断言JSON保存到[新独占结果目录](../../results/raw/homework1_gdn_epsilon_20261014/)，四文件已同步本机并与远端SHA一致。CPU自检紧容差只用于解析式，不替换原NPU包络；这是参考函数数学验证，不是NPU执行或全模型通过。

2026-10-15正式参考采集完成后，在physical1/chip0（logical2）串行执行两文件完整pytest，实际17 passed（13 NPU、4 CPU解析式）、0 failure/error/skip、8.326秒、52条误差properties。包装任务退出0；结束后两源SHA仍与上述冻结值一致，fresh npu-smi确认physical1无运行进程，未碰NPU0他人服务。XML位于`results/raw/homework1_gdn_epsilon_20261014/npu_tests_20261015/tests.xml`，两端SHA256为`cfd00e5261a49591aa588464d9c0b174d64c4c5e9cf0eeec6785786c8b3b14f5`。原NPU输出/state包络不变；这完成本模块参考纠正复验，不是全模型验收。下面旧“本轮未实现”仅表示原审计轮次；原训练输入FP64只读观察的范围及归档限制保持不变。

CPU只读核对进程正常退出0，所有张量由CPU数据构造，不调用NPU计算；CANN共享路径所有者警告保留。新数值来自本轮工具结果，已摘录在本文件；未保存新的参考二进制或独占raw目录，不声称有新的cross-SHA结果包，也不作为最终交付的完整可复现产物。

旧Prefill测试参考及合同描述需要纠正并补norm接近epsilon边界，但本轮未执行这一实现修改或重新测试。必须保留原通过/失败归档和原容差，在明确参考定义的评审边界下再补新的验证。新校准规格应明确主参考数学定义及中间dtype，不能将旧clamp脚本未经审计直接当作完整模型可信参考。

勘误传播后7份文档61个本地链接均存在，diff检查退出0；未新跑软件测试或NPU测试，不以文档检查替代数值验收。
