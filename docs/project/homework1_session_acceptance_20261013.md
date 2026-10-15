# 作业1：0.8B原生Session隔离与关闭重开验收

2026-10-13，Ascend910 V1单个logical chip0，固定0.8B revision、BF16/TP1、FP32 SSM，baseline配置prefix=false/KVStore=false/overlap=false。自有HTTP服务已停止并验证设备无进程，再启动独立Engine验收；测试不是性能数据，不把其他chip上的reference采集混进基线。

## 实测范围与结果

`test/runtime/test_homework1_session_npu.py`：两会话同时存在，各两轮；每轮Session请求与同一完整原始Token历史的stateless控制请求并行。按用户已选current-frontier，最后采样token留给下轮forward，不额外物化。关闭两会话后请求旧ID应返回closed-session abort；再打开不同新ID重复协议，比较跨round完整输入与输出。

第三次硬件运行 `session_r3.xml`：1个硬件测试通过，35.16秒；内含8组Session/control逐ID一致、8组context metadata通过、4次旧ID拒绝、4组关闭后重开跨round输入/输出完全一致。两个round分别记录完整input/output、metadata和关闭结果到JUnit properties。解析确认2个round及各4个turn，无fail/skip/error。原始目录为 `results/raw/homework1_model_20261013`，三次XML均已同步且跨端SHA-256一致。

首轮两会话context45/41、reused0；第二轮context68/65，newly-prefilled21/22、reused47/43。重开后的对应记录与初次相同，符合current-frontier：前轮历史length减1为reused。独立slot逐步不变的数值验证由GDN/卷积测试另外提供，不以输出一致推导所有内部state逐位相等。

这证明所测两个提示、两轮和关闭重开范围的原生Session/状态隔离行为，不证明任意长对话或全部并发情况，也不替代模型对Transformers数值验收。

## 两次失败与最小辅助脚本修正

`session.xml`首轮2组Session/control一致，第二轮在tokenizer历史前缀guard失败；没有绕过guard，不指认为引擎串扰。原4B helper硬编码 `<think>\n`，而0.8B初始generation prefix是 `<think>\n\n</think>\n\n`。

第一候选从实际template有/无generation prompt的差分提取前缀，相关51项回归通过，但 `session_r2.xml`仍在相同guard失败。完整阅读固定tokenizer的模板后确认：含`</think>`的旧assistant消息会被模板去掉reasoning部分。因此，重建整段文本改变了旧Token，不能直接复用缓存。

最终辅助脚本仅在已确认Qwen ChatML闭合think前缀、且history精确等于初始IDs加真实output IDs时，采用append-only新轮次：补齐缺失EOS/换行、追加新user块和实际assistant generation prefix。不删除/重新编码旧history。Session与独立控制都使用同一完整原始IDs，因此没有用更短控制输入使测试通过。原4B open-think路径保留；无效/不保留history的模板仍报错。

新增无依赖回归覆盖open/closed thinking及闭合reasoning被模板删除、输出有/无EOS：先复现失败，再作最小修正；本地/远端53项聚焦回归通过，本机全套course_project测试411通过、13跳过。该修正只改变实验helper，不改Engine/GDN/MHA算子，不重写已测性能数据。

修改前后hash核对确认远端原文件与本机反推旧文件完全相同，才同步；原文件与prefix-only候选保留于 `.tmp/homework1_prefix_fix_20261013/`。原始失败XML不覆盖，三个run各有独立pytest temp及XML路径。

## 警告与复现

原Engine测试退出仍提示pending asyncio tasks被销毁；没有隐藏，测试assert通过不代表异步退出告警已修复。设备退出检查另行核验，不能据此宣称所有生命周期问题消失。共享CANN所有者及transformers deprecation警告同样保留。

确认所选chip空闲、端口39142等无冲突、结果路径不存在后，从repo根目录运行：

```bash
source scripts/use_project_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest test/runtime/test_homework1_session_npu.py -q -s \
  -o junit_family=legacy --basetemp .tmp/homework1_session_fresh \
  --junitxml=results/raw/homework1_session_fresh.xml
```

第一次本地runtime pytest因为缺numpy在conftest失败，未执行测试；没有为此安装依赖，先AST检查再用远端项目Conda实测。此记录不把本地静态检查当硬件通过。
