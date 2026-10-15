# 作业1：0.8B最终版本模型与基础基线记录

## 固定服务

实际Ascend910 V1单chip，`ASCEND_RT_VISIBLE_DEVICES=0`、BF16、TP1，固定revision `2fc06364715b967f1860aea9cf38778875588b17`。服务由 `bash scripts/serve.sh configs/baseline/qwen35_0.8b.yaml --host 127.0.0.1` 启动，项目Conda/offline权重，无新依赖。HTTP8000、native control8001、metrics39142。配置prefix=false、disable_kvstore=true、overlap=false、chunk4096、block128、max_num_seqs16、max_total_tokens65536、context4096；SSM实际为FP32。

新进程ts-serve PID4144609完成权重加载、两轮graph capture、warmup并health200；启动日志 `results/raw/homework1_model_20261013/service.log`（远端）。可见cache token_capacity65536、max_input4095、cache arena2654994432 bytes。日志device memory分类属于运行时估算，不替代npu-smi HBM；0.8B服务不能与旧缓存容量配置的历史结果混为同一版本。

启动前确认设备无进程并探测四个端口可bind。首个端口检查因PowerShell/SSH内嵌引号产生Python SyntaxError，没有启动服务；删除不必要的输出字符串后正确检查并启动。中断后复核同一PID和health200，未重复启动。

## 固定样例native-ID采集

`scripts/run_correctness.py`以max_new_tokens128、greedy完成8条原生记录，输出 `results/raw/homework1_model_20261013/correctness_max128.jsonl`。本机与远端SHA-256一致：`36be5a8acc0d472e02924b6e0fef911a442d21b619bfe13da2ccec64677660a0`。

| 条件 | 实际输入tokens | 实际输出tokens |
|---|---:|---:|
| chinese_beijing | 19 | 38 |
| english_capital | 25 | 7 |
| math_arithmetic | 26 | 11 |
| code_python | 26 | 128 |
| long_context_recall | 979 | 74 |
| multi_turn_memory turn1 | 21 | 2 |
| multi_turn_memory turn2 | 49 | 9 |
| multi_turn_memory turn3 | 76 | 10 |

较短输出属于真实生成结果，不补写到128；参考比较与EOS/stop口径仍需核验。八条采集完成不是自动正确性通过，不证明精确128/1024输入的两档native验证完成；此runner的多轮为HTTP全历史重发，不是Session KV复用验收。

## 统一2×2性能测量协议与运行

使用同一服务和现有 `scripts/run_benchmark.py`，c1、目标输入128/1024×max_tokens32/128，各三轮、每轮8请求，条件预热3请求且排除，共96正式请求/12 summaries。原始根目录 `results/raw/homework1_baseline_20261013` 在启动前要求不存在，分out32/out128，不覆盖既有数据。目录provenance保存8个配置/脚本/关键runtime文件；这是测量所选源码，不是完整仓库快照。

测量已启动，out32完成后同一命令串行启动out128。不得在测量期间启动reference/NPU测试或变更这些测量源码。每轮记录设备前后快照，另保存loaded_device.txt；out128后半程增加约每2秒设备信息采样 `out128_hbm_usage_sampled.txt`，标记时间，不声称捕获任意瞬时峰值，也未覆盖out32全程。首次 `npu-smi info -t memory` 的 `out128_hbm_sampled.txt` 仅提供容量/频率/温度，不能用于HBM已用量；保留该记录并改用完整info采样，不能把65536MB容量写成占用。输入长度以actual usage为准，chat template可能使actual不同于target。

在完整run退出并核验96 raw、12 summaries、actual tokens、跨端哈希前不写最终性能数字。server ITL若缺失如实标记，不以SSE块间隔替代Token ITL；需要区别客户端TTFT/TPOT、native延迟与算子事件时间。

## 最终测量核验

同一基准命令正常退出0，96条raw、12 summaries、24轮次前后快照；四条件各三轮，每轮8条。全部实际输入为短130/长940，输出为32/128且无提前结束。39个数据/快照/测量源码文件跨端SHA-256一致。原始目录保持不变；`results/analysis/homework1_baseline_20261013/summary.csv`由现有报告工具复算。首次报告命令用了错误工作目录，未执行；在repo cwd重放正确命令。目录类别不属于默认baseline，因此工具未生成baseline专用图，不篡改raw类别。

下表是三轮run级指标的算术均值，P50为先每轮计算再平均，不是合并请求后分位数。客户端TTFT含排队/HTTP/采样等，不称纯Prefill kernel耗时。TPOT定义来自客户端整段时间与生成长度；ITL mean独立来自服务端直方图sum/count差分。

| actual 输入/输出 | TTFT P50均值(ms) | TPOT P50均值(ms) | 服务端ITL mean均值(ms) | E2E P50均值(ms) | 输出吞吐(tok/s) |
|---|---:|---:|---:|---:|---:|
| 130/32 | 205.940 | 10.611 | 10.621 | 535.091 | 59.778 |
| 130/128 | 208.775 | 10.674 | 10.695 | 1564.550 | 81.640 |
| 940/32 | 727.318 | 10.826 | 10.836 | 1062.742 | 30.079 |
| 940/128 | 721.130 | 10.764 | 10.784 | 2088.486 | 61.186 |

全部12轮server_metrics_available=true，ITL histogram observation count精确等于8×(output_tokens−1)：32输出每轮248，128输出每轮1016。P50/P95是histogram估计，不冒称逐token timestamp精确分位数。

加载后npu-smi记录chip0 HBM 8732/65536 MB，24份轮次前后chip0 usage rate均13%（百分比取整，不能反推精确MB）。HBM-capacity温度采样不是占用证据；后补usage sampler在启动时基准已结束、文件为0 bytes，不能作为峰值证据。当前只能报告加载占用和轮次端点占用，仍缺有效负载期HBM峰值与退出后快照，不假装完整HBM验收。

本协议输入由130增至940时TTFT显著增加，但两档输出的ITL mean仍约10.6–10.8ms。这支持本条件下主要新增延迟出现在首token前，不直接证明是计算/访存/转换/Host中的哪一项；具体归因还需Profiler或模块计时。未据此声称优化收益。

## 本轮模型参考对照

性能结束后确认物理NPU1无进程，将reference单独置于逻辑chip2（其进程内为npu:0），使用同一revision、BF16 greedy Transformers参考；可选fast-path库缺失，按已有torch fallback运行，未安装依赖。对照文件 `reference_max128.json` 已同步且跨端哈希一致。初次带 `-i 1` 而无type的npu-smi检查被工具拒绝，reference未启动；改完整info后确认空闲才运行。

八条raw exact为1/8，共同EOS/stop截断后exact为7/8。七条raw差异主要是停止边界，不能与正文分叉混同；`long_context_recall`正文首错仍在index32。当前状态不是所有样例完整正确性通过，后续共同前缀数值及状态验证仍保留。Session隔离、native精确128/1024长度协议、模块真实形状补证据尚未完成。

## 精确native长度与无状态复现补充

最新覆盖本节旧待测状态：Session最终append-only辅助脚本修正后 `session_r3.xml` 通过8组Session/control、4次closed rejection和4组重开对照；前两次失败保留，详情 `homework1_session_acceptance_20261013.md`，三个XML跨端SHA一致。日志与退出快照也已哈希核验。退出pending asyncio task警告未隐藏，最终设备检查四个物理NPU均无进程。

native128首tokenHF诊断 `native128_hf_prefix_logits.json` cached/recomputed相同top2为71093/8160，logits23.375/23.25，engine此前选择8160。margin0.125但尚无engine完整logit误差，不自动豁免。探针对index0空prefix先删除prefill result产生UnboundLocalError；零/非零prefix回归RED→GREEN后最小修改仅在非空prefix删除，再实测完成，JSON同步且哈希一致。原探针代码项目内备份，未覆盖失败数据；一次scp cwd错误纠正后成功。

新增 `native_exact_lengths.jsonl` 四条记录均实际128/1024输入、实际32/128输出，沿用本节同一revision、服务配置与greedy。输入构造为用户文本 `Context: ` + 600次 `TokenSpeed supports efficient inference. ` + `Write a Python function that sums a list of numbers. Explain how it works.` 经当前tokenizer chat template后，取前N−64和末64 tokens拼接；切分发生在重复上下文中，保留模板尾部及指令。记录注明 `fixed_chat_template_prefix_plus_tail64` 和原始模板长度，不冒称自然文本恰好该Token数。

四条完整input/output IDs已同步并跨端SHA-256一致。native interface保证输入长度不受网关再分词影响；此处E2E不是TTFT或纯Decode计时。

随后按A128/B1024/A128/B1024交错发送max128 stateless请求；四条output IDs分别与该长度独立baseline完全相同，保存 `stateless_interleave.json`，同步且跨端哈希一致。请求完成后state回收范围内复现通过，不据此证明Session KV状态隔离或关闭Session API。

四条精确长度reference正在另外一个空闲chip对照。HTTP服务完成上述采集后，仅SIGTERM自有PID4144609；退出设备快照 `service_exit_device.txt` 显示chip0 HBM3122/65536MB且四个物理NPU均无进程。加载8732MB到退出3122MB为driver视角，差值不等于全部模型有效tensor或KV，不将驱动基础占用当泄漏。日志与退出快照已同步；其哈希还需最终核验。

Session验收改用新的独立Engine进程、同一0.8B baseline单chip配置，运行 `test/runtime/test_homework1_session_npu.py`，Junit路径 `results/raw/homework1_model_20261013/session.xml`。首次本地pytest因本机缺numpy在runtime conftest加载时失败，未执行测试；不安装依赖，改为AST语法检查和远端既有Conda执行。真实硬件结果待核验，不提前写通过。
