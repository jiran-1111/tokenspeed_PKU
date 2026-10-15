# 单请求 TTFT Profiler 远端软件验证（2026-10-12）

对应已批准的单请求时间窗设计与实施计划。本轮完成实际远端 Conda 环境的软件验证，没有启动推理服务或采集 Profiler。

## 当前代码与验证位置

远端实际项目尚无 `benchmarks/course_project/profiler_analysis.py` 与 `scripts/analyze_profiler_window.py`；现有 `run_agent_benchmark.py` 为较早版本。因此将当前模块、入口、配置和测试放到独立目录：

`/path/to/tokenspeed/.tmp/profiler_window_validate_20261012`

目录内含 course_project 模块、六个脚本、五份测试与 0.8B 配置，22 个源文件的本机/远端 SHA-256 一致。使用实际项目现有 `.conda-env/bin/python`；pytest basetemp 与 TMPDIR 都限定在上述目录。

## 测试结果

五份测试 `test_agent_profiler_capture.py`、`test_profiler_analysis.py`、`test_profiler_control.py`、`test_profiler_runner.py`、`test_benchmark_core.py` 共 **65 passed**。

覆盖流式首块时间戳/回调、单请求采集条件、控制调用和失败处理、合成 Torch/CANN 时间窗与时钟拒绝条件、报告输出冲突及项目内路径边界。测试使用 fake client/control 和合成 trace；不能证明真实 Torch/CANN 时钟一致或设备采集成功。

`scripts/analyze_profiler_window.py --help` 退出 0，说明 raw/profile/analysis 参数及项目内输出限制。

首次测试为 61 passed/4 failed，原因是独立目录未复制测试依赖的 `run_benchmark.py` 与 `serve.py`；补齐后上述 65 项全部通过。没有修改测试断言或安装依赖。

## 下一步

1. 已完成实际远端 Agent runner 差异审阅、原件备份、单请求分支与分析模块部署，并在实际项目根目录复验。后续运行前重新校验具体配置与设备状态。
2. 保留已批准计划的真实采集单独确认门；正式采集前检查 0.8B 配置/revision、服务与 profile control、NPU0、唯一输出路径。
3. 真正采集后才验证单 token usage、start/stop 配对、Torch/CANN 产物、1 ms 时钟核验、CSV 和哈希，并更新 Profiler 实测报告。

当前仅完成独立目录的软件验证；真实采集与请求时间窗重叠证据仍缺失，不能作请求到算子的因果归属。

## 入口差异审阅与时间戳修复

已审阅实际远端旧 Agent runner 与当前版本的完整差异：主要新增显式单请求分支及对应参数/输出路径约束。实际 core/runner 已是 Long Prefill 整合后的版本。

审阅发现 `request_record.update(timestamps)` 会把已记录的 request_send/stream_drain epoch/monotonic 时间覆盖为初始化的 None。新增端到端 fake-client 回归断言先复现 `None != 1800000000070000000`；修复后只有非空回调时间或原记录中尚无的字段才合并，从而保留正常请求的四个边界值。

修复后本机相关测试 60 passed、完整课程测试 402 passed/12 skipped；远端独立目录五份测试为 66 passed。本机六项 Prometheus 发布/接线用例因依赖缺失跳过，已在远端 KV 独立目录通过。没有安装依赖或执行真实采集。

修复已同步独立验证目录；以下部署记录取代当时的未部署状态。

## 实际远端部署验收

- 审阅入口完整差异及测试方法差异，没有移除远端独有测试方法。core/runner/profiler 公共依赖与已验证版本字节一致。
- 部署前再次校验三个现有文件的原哈希，三个新目标确认不存在；保存原文件至独立验证目录 `deploy_backup/` 后同步六个入口/模块/测试文件，逐个校验与 stage 字节一致。
- 实际项目根目录完整课程回归：**378 passed，1 skipped**。现有远端集合仍与本机不同，不能把此数值当作发布快照一致性证据。
- `run_agent_benchmark.py --help` 显示单请求 TTFT 开关；`analyze_profiler_window.py --help` 正常，均退出 0。
- 新 KV runtime/telemetry 统计保持在独立目录验证，未混入本次实际项目部署。
- 没有启动服务、调用真实 Profiler 控制或发出推理请求。真实采集仍受已批准计划的单独确认门约束。
