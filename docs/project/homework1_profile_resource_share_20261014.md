# 既有Level1计算资源分布：描述性复算

不启动新Profiler、不修改服务，离线读取已有六份8-turn Level1 CANN `op_summary`。固定0.8B revision、BF16/TP1，c1/c4各三轮；原采集、时间戳与负载见`docs/report/ascend_profile.md`。这是历史生产快照，不与当前关闭prefix cache的性能测量混作同一配置。

复算脚本：`.tmp/homework1_profile_resource_share_20261014.py`。仅累计正且finite的`aicore_time(us)`，按CSV的`OP Type`归组；每个capture的正时间行数及全量累计值均与既有审计JSON一致。不是只从旧Top10摘录，尾部也进入分母。所有六份输入CSV的SHA保存在新`results/analysis/homework1_profile_resource_share_20261014/summary.json`；`per_capture.csv`保存每轮每类型的行数、累计值、排名和占比。

| 并发 | 算子类型 | 三轮平均记录时间占比 | 三轮范围 | 每轮名次 |
|---|---|---:|---:|---|
| 1 | MatMulV2 | 43.938% | 43.832–44.133% | 1/1/1 |
| 1 | chunk_fwd_kernel_o | 23.443% | 23.386–23.479% | 2/2/2 |
| 1 | recompute_w_u_fwd_kernel | 14.005% | 13.910–14.061% | 3/3/3 |
| 4 | chunk_fwd_kernel_o | 34.464% | 34.145–34.861% | 1/1/1 |
| 4 | MatMulV2 | 24.435% | 23.575–25.695% | 2/2/2 |
| 4 | recompute_w_u_fwd_kernel | 19.967% | 19.417–20.469% | 3/3/3 |

矩阵乘法按算子类型合并不同kernel，避免把同一计算类别拆开后与单个GDN kernel比较。分母是记录到的累计AICore任务计数，不是整个设备墙钟、不等于端到端延迟；不同任务可重叠。这些百分比不能与Host gap或HBM计数直接相加。

在上述负载中，投影矩阵乘法、GDN chunk输出和W/U重算是三轮重复一致的主要计算资源候选。因此后续计算侧优化可以优先检查GDN Prefill的输出/重算以及矩阵投影路径；这不表示三者已经被证明是客户端TTFT长尾的前三因果瓶颈。并发4工作量更多，绝对累计值不能当效率或加速比。该混合Prefill/Decode采集也不能冒充稳态Decode单独分析；未校准HBM带宽单位、未形成可信请求到算子关联，计算/访存/Host之间的完整归因仍待补齐。

本次脚本正常退出0，未运行NPU计算或安装依赖。新描述性汇总不改变原始trace，不放宽模型正确性合同，也不将历史Profiler数据称作当前生产版本优化收益。

远端复算也已正常退出0，得到同一六capture及上述数值。原CANN仍位于项目`.tmp/ascend_work/profiling_data`；在确认目标缺失后同步已有分析CLI，并建立可供标准遍历读取的项目内文件链接索引（1516个链接），没有复制或删除原始数据。链接依赖源目录保留，本机archive为完整文件副本。最初缺工具/归档目录、目录链接不被Python3.11 rglob递归等失败均记入notes；只替换了本次新增六目录链接，没有动历史数据。
