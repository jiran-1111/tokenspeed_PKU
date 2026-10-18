# 原始生成历史的保留与跨模型Session复验

HW1已发布提交保持 `1d113e54f789a22b14cbbdccbc6230c3f4d0e574`。下述修复属于总项目分支 `course/agent-20261015`，只改实验helper及配套测试，没有改变引擎模型、算子或采样。

## 真实失败及最小修改

4B双Session各32-token输出的首轮，两组Session/stateless输出及context元数据均一致；下一轮在 `_next_suffix` 的历史前缀检查报错。实际模型输出以空格结尾，而Qwen聊天模板重新渲染时strip这些生成尾空白，导致旧token不能保持。原始目录 `results/raw/session_final_source_4b_20261015/` 与退出日志保留。

辅助脚本原有的append-only路径只处理generation prefix中已闭合thinking的情况。现在对于可确认的Qwen模板尾空白或assistant内容中闭合reasoning造成的重写，也保留全部真实历史ID，只追加缺失的ChatML边界与下一轮。不重编码或删掉旧生成内容；仍要求历史严格等于初始模板IDs加实际output IDs，其他历史变动继续拒绝。

新增空白尾部与生成中闭合reasoning的EOS有/无测试，先复现4失败/4通过，再修复；加上不可改写历史的反例后，Session专项57项通过。真实4B tokenizer对两条原始失败重放：旧helper均失败、新helper均保留原历史且构造出合法suffix。

## 新源码服务器结果

1396文件新清单SHA `b80a4e7db1426c53ec803d9932115dec84ce1aed0fcafbf1750b74d7001afc6c`，相对HW1原清单只改helper和一个测试文件。服务器完整course为573 passed/零skip；本机548 passed/25 skipped，环境缺项不冒称通过。最早本机运行目录及basetemp选择错误的失败日志保留，修正为工作树内basetemp后运行通过。

三模型同一新源码、原baseline配置，仅隔离端口；原生Engine双Session、同时提交stateless控制，每Session两轮32输出tokens，再关闭重开重复。采用current-frontier，关闭generic prefix/KVStore/overlap/mixed batch；每次NPU运行前实时映射/空闲检查，串行运行。

| 模型 | 结果 | 覆盖与限制 |
|---|---|---|
| 4B TP1 | 通过 | 两次打开共8组输出/context配对、4次关闭ID拒绝；重开输入及输出一致 |
| 9B TP1 | 失败，保留 | 第一轮两组配对通过；第二轮session-0在输出index18首次分叉，context94/new20/reused74检查通过；不能由元数据通过推断数值状态完全正确 |
| 27B TP2 | 通过 | 两次打开共8组输出/context配对、4次关闭ID拒绝；重开输入及输出一致 |

9B失败不由聊天模板修复豁免，也不套0.8B冻结门槛。尚需区分状态数值与最终输出舍入；不把本表当跨模型HF正确性或性能证明。

原始目录为 `results/raw/session_final_source_{4b,9b,27b}_20261015_r2/`。Engine上下文退出后的即时快照仍观察到调用进程占用，原记录 `target_released=false` 保留；驱动进程结束后的独立 `post_exit_device.txt` 均确认物理NPU1无进程。没有杀死其他设备任务，也不把即时占用改写为“无警告退出”。

27个原始文件/源码清单/备份/软件测试日志XML已同步逐项SHA通过，归档SHA `e822c375fbd7ef75388de2edddf05d173776309b2c2da2cd7c919850b44aa877`。这些新结果尚未上传HW1分支，待总项目证据完整后单独发布。

## R5四模型同源补充（2026-10-17取回核对）

显式输出头精度候选引入后，R5清单1398项SHA为 `ef7b3e028233517746147f36da92efea679a29ba50471df4574e3cac9c103a27`。四模型均使用该清单再次按上述两Session/两轮/32输出/关闭重开协议运行：0.8B、4B、27B用原baseline，9B用显式FP32 head候选，均各8对输出/上下文及4次关闭拒绝通过，重开重复输入/输出一致。4B/27B原八条对既有HF参考的同输入复验也均8/8，并非重新采集HF参考。

旧9B BF16失败不删除。此补充不代表16轮Agent或任意并发/工具等待验收；0.8B项目扩展也不替代HW1固定快照。即时退出flag仍按原始false保留，独立进程完全退出后确认对应目标chip空闲。

新增4B/27B回归41项证据逐项SHA通过，归档SHA `b9faa5d199f7ab3439c749e9d64bab6bc5e2b99dfefae3eedbaf3f7555d7c7f8`；包含首次端口占用导致启动前拒绝的记录。取回时重新核对服务器全部1398项源零差异。公开结果见[Session汇总](https://github.com/jiran-1111/tokenspeed_PKU/blob/course/final-20261018/docs/evidence/agent_20261015/session_summary.json)及[同源回归](https://github.com/jiran-1111/tokenspeed_PKU/blob/course/final-20261018/docs/evidence/agent_20261015/uniform_regression.json)。
