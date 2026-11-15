# 作业3复现与验收

工作分支为 `homework3/host-offloading`，基于 `homework2/completed` 的
`88f67e8b`。实验模型为 Qwen3.5-0.8B 和 Qwen3.5-4B，固定沿用课程配置的权重版本。

## 两级缓存与混合状态

设备端和 Host 都复用 TokenSpeed 的缓存块布局。缓存标识包含命名空间、
组、内容哈希和块内页编号。每个 TP rank 保存自己的权重分片对应状态。
Host 按 `--kvstore-ratio` 或 `--kvstore-size` 建池，后者覆盖前者，
参数作用于每个 rank，报告中另列所有 rank 的总和。

Attention K/V 按 128-token 的前缀粒度发布。GDN 递归状态和卷积状态保存已物化的
Prefill 检查点，二者使用同一上下文位置。Decode 期间的工作状态继续在设备更新。
卸载优先保存可恢复的 Prefill 检查点，恢复时选三类状态都完整的最近边界。
边界后的原始 token 历史按序重算，其中包括上一轮最后一个尚未计算的采样 token。
每轮实际位置记录在 trace 的 `restore_positions`，前向长度记录在 `input_lengths`。

只有没有在途推理结果的空闲会话可被回收。写出和恢复分别使用现有的传输流，
写出完成事件使复用前的清零和前向等待旧数据读出。普通后台写出继续持有源块，
直到完成确认。各 rank 的缓存完成确认经过既有集合通信求交后才推动镜像调度器。
Host 淘汰优先选择未保护、未使用的条目，然后按访问时间与固定次序选择，
已被命中过的条目具有既有框架定义的保留优先级。传输中的源、目标均受引用保护。
Host 未命中沿用普通前缀探测或完整历史重算。

## 编译与运行

先激活项目环境并加载 CANN。在独立源码目录中编译当前调度器，
将生成的扩展复制到该源码目录自己的 `tokenspeed-scheduler/python/tokenspeed_scheduler/`。

```bash
cmake -S tokenspeed-scheduler -B .tmp/hw3-build \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DOPENSSL_ROOT_DIR="$CONDA_PREFIX" -DPython_EXECUTABLE="$(command -v python)"
cmake --build .tmp/hw3-build -j6
cp .tmp/hw3-build/tokenspeed_scheduler_ext*.so tokenspeed-scheduler/python/tokenspeed_scheduler/
ASCEND_RT_VISIBLE_DEVICES=6,7 bash scripts/run_homework3_matrix.sh results/homework3/new-matrix
```

可见设备按当时实际映射选择。共享运行需记录常驻服务，保持一次一个实验。
矩阵脚本覆盖两模型 TP=2、高／低复用、Off、Host 比例 0.3 和 2，
每格三会话交替访问四轮。首轮为预热，剩余九轮为请求耗时样本。
所有请求都通过正常 `Engine.open_session/generate/close_session` 入口。
高复用重复访问同三会话，低复用每轮建立三个新会话。

每格生成独立的完整历史参考执行，关闭 Host 与前缀缓存，逐 token 比较。
主实验的 Off 与 On 都启用前缀缓存，使用相同设备块预算和其他配置。
每次运行记录源码和调度器二进制 SHA-256，结束时复核源码没有变化。
启动脚本处理已安装环境的旧调度器导入重定向，并在运行前核对实际扩展来源。

## 统计口径

`TOKENSPEED_CACHE_TRACE_DIR` 启用每进程 JSONL。默认关闭该证据采集。
`scheduler` 记录 rank、设备活动和可用块、Host 块、命中、未命中、淘汰、
会话回收、每次前向的起始位置与块表。
`host_allocation` 记录紧凑 Host 池预留字节与每个物理块大小。
`transfer` 只在完成事件到达后记录方向、载荷字节、源／目标块和设备事件耗时。

Host 实际占用根据已占用物理父块乘块大小计算，另列预留池容量。
设备 HBM 采样保留 `npu-smi` 原文。预分配缓存池回收后仍可保留总 HBM 占用，
以会话间块编号的实际重新分配证明容量可复用。
恢复耗时包括 H2D 设备事件耗时和续轮请求端到端延迟，后者包含必要重算与生成。
搬运耗时允许各流、各 rank 重叠，因此不能累加为墙钟时间。

汇总可使用 `benchmarks.course_project.offloading.summarize_run`。
该汇总将逐 token 匹配和实际卸载恢复证据分别列出。高复用大预算的恢复验收要求每个 rank
都有回收、D2H、H2D 与跨会话块复用，恢复位置达成一致，并通过预算淘汰及重复循环测试。
