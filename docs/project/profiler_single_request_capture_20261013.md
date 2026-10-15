# 单请求 TTFT Profiler 采集状态（2026-10-13）

用户明确批准后执行 0.8B/BF16/TP1、仅 NPU0 的已审核八轮流程。Run ID：`20261013T065315242014Z`。27B 未运行。

请求记录 status=success、output_tokens=1、stream_completed=true；Profiler start/stop 与首块 callback 确认均为 true，无清理 stop 重复调用。带 Profiler 的客户端 TTFT 为 260.863890 ms，不能用于无 Profiler 基线性能对比。

指定 profile 目录最初只有 Torch trace。只读搜索定位本次 CANN 输出于项目内 `.tmp/ascend_work/profiling_data/5664fbce7f43_4093333_20261013065351734_ascend_pt/`，随后完整复制到本次 profile 目录；没有重复采集，没有改写原始 trace。

分析 CSV 已生成：clock_status=verified，max_clock_error_ns=800；TTFT 窗内 Torch 事件数 41871、CANN interval 数 3296。per-event overlap sum 分别为 798636.98 us 和 138195.29 us，区间重叠会重复计时，不能视为互斥耗时分解或算子因果归属。

本次拥有的服务正常收到 SIGTERM，SSH 服务会话退出码 0，NPU0 两 chip 无设备进程。关闭时 Uvicorn 输出 CancelledError，未影响先前采集；不隐去该关闭信息。

Raw 位于 `results/raw/profiler/single_request_ttft/<run_id>/`，profile 位于 `results/profiles/qwen35-0.8b/single_request_ttft/<run_id>/`，CSV 位于 `results/analysis/profiler_single_request_ttft_<run_id>/`。本地同步已经启动，尚需同步完成后两端哈希、实际 Level1 配置及 CSV 重生成验收；当前不宣告完整采集验收或项目完成。

后续验收：Raw/profile/analysis 共 145 个文件的本地/远端 SHA-256 全部一致，本机重生成的窗口 CSV 与远端逐字节一致。实际 profiler_info_0.json 的 experimental_config 为 Level1、ACL_AICORE_PIPE_UTILIZATION，而非仅靠 provenance 的声明。该单请求时间窗限定实验已完成产物/时钟验收；整体项目仍未完成，不能由单次采集推导统计性能或 Top3 因果瓶颈。
