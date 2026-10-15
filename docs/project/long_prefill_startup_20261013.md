# Long Prefill 正式启动检查（2026-10-13）

用户已批准正式 Long Prefill 测量，Profiler 仍需另行确认。本次未启动 Profiler。

## 已核实的结果

- 启动前 NPU0 两个 chip 的计算利用率及 HBM bandwidth 为 0，无设备进程；API 8000、控制 8001、Prometheus 39141、TokenSpeed metrics 39142 均可绑定。
- 正式 runner 创建远端项目内目录 `results/raw/benchmark/long_prefill_decode/20261013T020048366473Z-89f62e4fa65649479491f37bdd06a0d9/`。
- 启动 SSH 连接曾被重置；重新检查确认 runner PID 4036747 当时仍存在，因此没有根据连接中断重复启动。
- 后续检查确认 runner 已退出。第一组 `service-round-01-chunk-4096.log` 为 `startup_status=failed`、`exit_status=1`；requests/trials 均为 0 行。失败目录保留，不纳入正式配对结果。
- 独立启动诊断只复现相同服务配置，没有发送测量请求；报错发生在 `ServerArgs.validate_cache_options()`，模型加载之前。

## 根因与待确认修正

当前远端 `python/tokenspeed/runtime/utils/server_args.py` 的 `_handle_kvstore()` 在没有 `--disable-kvstore` 时默认启用 KVStore。`validate_cache_options()` 第 957 行明确拒绝 KVStore 开启且 prefix caching 关闭的组合：

> KVStore and disabled prefix caching are mutually exclusive

现有 Long Prefill 配置变体关闭 prefix caching，但没有关闭 KVStore，因而无法启动。建议两个 chunk 变体都增加 `disable_kvstore=true`，保留其余条件及配对顺序。该修正改变原规格控制项，尚未实施，需用户确认；不建议修改运行时校验或重新开启 prefix caching 来绕过。

诊断结束后重新检查 NPU0：两 chip 无设备进程，计算利用率和 HBM bandwidth 均为 0。没有正式延迟样本或新的比较结论。

## 本地证据核验

失败目录已完整复制到本地项目相同的 `results/raw/benchmark/long_prefill_decode/<run_id>/` 路径。逐文件对照远端 `sha256sum` 与本地 SHA-256，六个文件全部一致，包括两个空 JSONL。没有覆盖历史实验或生成配对统计。当前等待用户批准关闭两组 KVStore；先前正式测量授权不视为对该控制项修正的授权。

## 修正授权

用户后续明确回复“批准”，授权两组共同关闭 KVStore 并重新正式测量。仅修改 Long Prefill 配置变体，原 bring-up 配置及运行时校验不变；新增控制项需计入配置快照与实验结论边界。Profiler 不在本次授权内。

## 第二次启动及端口探测修正

run `20261013T030821237999Z-e4ba674feff4474abe765af68fcebfad` 的 round 1/chunk 4096 完成两项预热及五项正式条件，均为 valid；实际输入 256/2048/3584，Decode 输出 128，Prefill 输出 1。服务正常退出（exit_status=0），但下一组端口探测失败，runner 停止。此 run 不具备完整配对，保留为部分记录，不能代替三轮验收。

远端 `/proc/net/tcp` 显示端口 8000 仅剩状态 06（TIME_WAIT）、inode 0，连接探测为 ECONNREFUSED；NPU0 无设备进程。新增真实 socket 回归：活动监听拒绝测试通过，关闭监听后的 TIME_WAIT 探测测试在修复前失败。修复只在非 Windows 平台为探测 socket 设置 SO_REUSEADDR，仍不使用 SO_REUSEPORT，仍实际 bind 检查占用；Windows 原有行为不变。

修复后远端 Long Prefill 测试 145 passed/1 skipped，本机 140 passed/6 skipped。本机首次从工作区父目录运行导致两个导入收集错误，改在项目目录执行后通过，未修改导入逻辑。随后启动新的完整 run，既有部分目录不覆盖、不续写。
