# Long Prefill/Decode 远端预检与软件验证（2026-10-12）

本记录是正式测量前的准备证据；没有启动模型、推理服务或 Profiler。

## 连接与只读预检

- Windows OpenSSH agent 原为 Stopped，启动此前已授权的服务后，原有 ed25519 身份可用；SSH 非交互认证恢复。
- 远端项目：`/path/to/tokenspeed`；验证使用其现有 `.conda-env/bin/python`。
- 4B 配置 revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；模型目录存在，包含两个 safetensors 分片。本次只检查存在性，没有重复校验权重内容。
- 仅查询 NPU ID 0：chip 0/1 的 NPU、Aicore、Aivector 与 HBM bandwidth 利用率均为 0%；HBM 使用率均为 4%。该快照只代表检查时刻。
- socket bind 检查通过：API `127.0.0.1:8000`、控制端口 `127.0.0.1:8001`、Prometheus `0.0.0.0:39141`、TokenSpeed metrics `127.0.0.1:39142`。

## 独立远端软件验证

远端原项目尚无本轮 Long Prefill 实现，且 core/runner/telemetry 与本机哈希不同。因此先在项目内创建独立目录：

`/path/to/tokenspeed/.tmp/lp_validate_20261012_81a467`

该目录包含当前 course_project 模块、本轮两个脚本、两个测试文件和 4B 配置。15 个同步源文件的本机/远端 SHA-256 全部一致。未覆盖远端原项目文件。

现有远端 Conda 执行两份 Long Prefill 测试：**140 passed，1 skipped**。pytest 临时目录位于上述验证目录内。

dry-run 退出码为 0，打印三轮交替顺序 `4096→512`、`512→4096`、`4096→512`。结束后 raw/analysis 目标目录仍不存在。

## 尚需完成

整合进展：已审阅 core/runner/telemetry 全部差异，均为本轮接口新增。三个原文件保存在验证目录的 `pre_sync_backup/`；同步八个实现/测试文件并逐一校验哈希。实际项目根目录回归为 **347 passed，1 skipped**，dry-run 通过。远端测试集合与本机不同，此计数不能替代本机 396/6 的验证。

真实 tokenizer 检查发现阻断：seed `preflight-20261012` 构造 256-token 完整 chat 输入失败，最近值为 254。只读长度扫描显示候选 245/250 的完整输入均为 254，而 255 为 265，存在不可达区间。共享 `make_token_sized_text` 按单次 seed 的 token 数估算重复次数，但拼接后实际分词长度可更短，导致源 token 不够且切片无法补足。尚需用回归用例验证并修正本轮构造器，再检查 256/2048/3584 三种真实长度；正式测量尚不具备条件。

1. 已完成输入构造修复及本机/远端回归；正式运行前重检空闲状态和端口。
2. 按已批准计划报告具体运行配置并取得正式测量的单独确认；预计六次服务启动，4B/TP1、prefix cache 关闭、2048/3584-token Prefill、256/128 Decode。
3. 正式启动前重新检查 NPU0 与端口；本记录不能代替运行时预检。

真实测量与 CSV/SVG 硬件结果尚未完成，27B 仍按用户选择暂缓。

## 输入构造修复验收

- 在本轮构造器内倍增重复文本，直到实际 encode 长度至少达到目标，再复用原截断函数；循环和模板搜索均有界。共享 runner 未改动。
- 新增模拟 seed 边界 token 合并的三个回归用例，修复前均失败，修复后通过。提示词配方版本升为 `long-prefill-v2`。
- 本机完整课程测试：399 passed、6 skipped；远端完整现有课程测试：350 passed、1 skipped。两端测试集合不同，分别记录。
- 真实 4B tokenizer 在离线模式下满足 256/2048/3584 三个完整 chat-template 长度；按正式 run ID/round/condition/warmup 格式构造的 21 组代表输入也全部精确匹配。
- 两个修复文件本机/远端 SHA-256 一致，远端原件保存于验证目录 `prompt_fix/original_module.py` 与 `original_test.py`。
- 独立代码复核通过，复核者的 prompt focused tests 为 9 passed。上文 tokenizer 阻断已解决。
- 正式测量仍等待已批准实施计划所要求的单独确认；本轮没有启动推理服务或 Profiler。

## Benchmark 源码身份记录

远端源码目录没有 Git 元数据，因此 `run.json` 新增 `benchmark_source_sha256`，记录实际源码根目录中的 Long Prefill runner/workload、公共 core/runner/telemetry 五个文件的 SHA-256。测试使用独立 fake project 存放配置与结果，同时确认哈希来自实际源码文件，避免把 fake 配置根目录当作运行代码位置。

新增回归断言先因缺少该字段失败，修复后本机 Long Prefill 两份目标测试为 139 passed/5 skipped；远端实际项目为 143 passed/1 skipped。两个修改文件已在比较原版后同步，原件存入 `.tmp/lp_validate_20261012_81a467/source_provenance/`。

该字段只记录 benchmark 实现身份，不能替代完整推理 runtime/内核源码归档、依赖版本记录与最终发布快照。正式测量尚未执行。
