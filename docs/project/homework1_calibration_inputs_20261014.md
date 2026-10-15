# 作业1：独立参考输入冻结验收

日期2026-10-14。已完成输入工具和实际tokenizer清单冻结，不是独立参考数值校准完成，也不是作业1全部验收。

## 服务器实际结果

先按已确认规格和[输入子计划](../superpowers/plans/2026-10-14-homework1-calibration-inputs.md)实现纯标准库builder和离线CLI。Task1/2分别经过规格、质量审查；主线程本机完整回归448 passed/14 skipped（10.73s），其中CLI路径符号链接测试因Windows权限跳过，没有冒称已通过。

四新增文件服务器目标不存在才SCP部署，不覆盖生产或历史文件；源码逐字节SHA一致。服务器项目Conda/source环境实跑两文件共34 passed（3.44s），XML中34 testcase、0 failure/error/skip，Linux符号链接防逃逸case实际通过。原始[XML](../../results/raw/homework1_calibration_inputs_20261014/input_tests.xml)保留，不用本机跳过代替Linux验证。

真实本地模型tokenizer生成[inputs.json](../../results/raw/homework1_calibration_inputs_20261014/inputs.json)：校准、验证两组各16条，1/8/127/128/129/512/1024/2048长度各两条，32组native输入IDs唯一、每条16续接IDs。固定内容family按气象数学/Python与历史清点/SQL植物分开，来源文本、原始IDs和各自SHA保存。短native输入可能只包含标题前缀，不宣称它是完整自然语言；这是一组有限覆盖，不作统计独立总体样本声称。

整份内容canonical SHA：`f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936`；inputs文件SHA：`23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb`。另一个新项目内临时路径重新运行CLI，内容逐字节相等，两进程都退出0。清单在新数值采集前冻结，后续失败不得默默换输入。

原始目录四文件（inputs、XML、audit、tokenizer_identity）已同步本机，跨端SHA全部相同。独立对32条长度/分组/合法模型词表/三个逐条hash和整份hash核验通过；见[audit.json](../../results/raw/homework1_calibration_inputs_20261014/audit.json)。生成后物理NPU1/2/3无运行进程，NPU0其它任务不干预；tokenizer加载有既有CANN目录owner警告，未修改共享环境权限。

## 身份与范围

后续阶段A权重前置检查发现整个检查点为452个BF16张量、36个FP32张量（GDN A_log及gated norm权重）；本输入工具没有加载这些模型参数。参考必须按引擎实际loader/parameter生效精度对齐，而不是把全checkpoint一律降为BF16再展开。该映射与回转检查属于下一采集计划，不把embedding的BF16头部当成全部权重精度证明。

固定模型标签revision `2fc06364715b967f1860aea9cf38778875588b17`，配套config、index及tokenizer文件哈希已记录；不把CLI revision字符串单独当作所有权重来源的密码学证明。实际Qwen2Tokenizer、Transformers5.12.0、tokenizers0.22.2及tokenizer实现/vocab/merges哈希见[tokenizer_identity](../../results/raw/homework1_calibration_inputs_20261014/tokenizer_identity.json)。分词器length248077，模型embedding词表248320，合法ID按模型词表检查。

该入口没有模型forward/引擎生成；不据此宣称模型数值正确性通过。原1%工程失败仍保留，候选引擎判据未建立/未批准，验证组引擎数值未采集。接下来实施阶段A四种固定参考组合、同量化权重、Prefill+16步缓存/重算和参考自身可信性校准；阶段B只在候选判据独立确认和哈希冻结后执行。

## 复现入口

服务器项目目录内先选择新的结果路径，避免覆盖此清单：

```bash
source scripts/use_project_env.sh
python -m pytest test/course_project/test_calibration_inputs.py \
  test/course_project/test_calibration_inputs_cli.py -q \
  --basetemp .tmp/calibration-input-check-NEW_RUN_ID
python scripts/prepare_homework1_calibration_inputs.py \
  --model models/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output results/raw/calibration-input-NEW_RUN_ID/inputs.json
```

NEW_RUN_ID须换为新的唯一标识，basetemp不得复用旧目录。文件组织保持benchmarks/scripts/test原布局；大raw/模型/环境不上GitHub，服务器验证对应发布源码后才上传指定fork。
