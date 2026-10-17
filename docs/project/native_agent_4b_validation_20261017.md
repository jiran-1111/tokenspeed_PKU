# 4B原生16轮验证：配对通过，工具窗口未通过

4B/BF16/TP1/图模式在固定双Session、每Session16轮、每轮8输出的实验中，32组原始输出逐ID一致、32组current-frontier上下文元数据一致。独立重算历史拼接、实际输出数与长度边界均无差异，最大实际上下文968 tokens。**整组实验仍为failed**：2秒工具窗口没有另一Session请求开始或完成，也没有窗口内KV采样，不能称完整Agent验收通过。

## 固定条件与结果

源码R7/r4共1403项，SHA `dc3142dba99acac6e874d82b55de58343da067dd0befa6c1658bbe620b68a394`；workload字节SHA `26cd9cf606589da095f064d9985ccf30a35a7014c711c73b09398a6aeb15a794`。各次修订未改该workload、2s窗口或判据。原生Session与独立stateless完整原始历史对照同时提交，prefix cache/KVStore关闭，沿用基线配置、独立端口。所有模型计算经同一NPU路径，无强制参考token。

Session-0第8轮前的受控工具实际返回sum(range(31))=465，等待2.00006638s；恢复时实际reused_context_tokens=715。下表是同一进程的客户端单调时钟相对工具开始时刻，单位秒；不解释为NPU算子时间线。

| 观测 | 开始 | 完成 |
|---|---:|---:|
| 工具窗口 | 0 | 2.000 |
| Session-1第8轮原生请求，完整上下文735 tokens | -0.005 | 6.421 |
| 同轮stateless完整历史对照 | -0.005 | 6.421 |

该请求跨越整个窗口，下一轮也未在窗口内开始。并发对照额外进行了完整历史重算，当前点测量包含这部分干扰；它不能单独证明原生Agent无对照时的进展、服务端WAITING_TOOL阻塞或NPU idle。

工具前采样为running=2、LCM total/active/cached/empty/allocated=105/23/9/73/32。当前采样在另一Session请求完成后触发，因此该短窗口内无采样；缺失保持缺失，没有填0。两次close API成功、结束后目标chip空闲，尚无引擎存活时的关闭后LCM快照，不能用进程退出替代这项生命周期观测。

## 实施与失败记录

新增历史构造只追加新的ChatML输入和实际模板generation prefix，保留所有旧生成IDs，覆盖尾空白、闭合reasoning、EOS及16轮前缀。脚本复用既有原生stream、context检查和Prometheus解析，无新调度路径。软件最新为本机575 passed/34 skipped、服务器609 passed/零skip；完整pre-commit native round6通过。

早期软件fake的短sleep及Windows15.625ms单调时钟边界造成时序失败，改为事件及跨tick观测后通过；真实runner/workload当时未改。首次NPU运行在工具综合判据失败后，又因完整指标解析中的tuple键导致JSON落盘失败，因此不验收该次32组原始数据。其26项证据保留且逐项SHA核对，归档SHA `3af0001f115e4b8434b2cf69907ee13d04817cd0770c50d0094c9c3c87daa65a`。

使用真实Prometheus解析结果的软件测试先复现同一JSON错误，修复为只记录协议所需六个计数后21项通过；同workload复采保留完整结果。最新19项原始/分析证据全部SHA一致，归档SHA `9e2388774cf0566ddb8097d1cee86f0dd4461ccb2e6c683d932ae1319599dbdd`。进程退出1、未触发420s超时；独立后验1403源零差异，目标chip空闲。

## 下一步

把工具期间采样与请求完成解耦；另设明确的原生执行阶段，在其结束后逐条完整历史重算校验，避免质量控制本身混入Agent进展窗口。原并发点继续作为负结果保留；新的阶段仍须全部逐ID/元数据检查，不能以改变控制时序取消质量校验或改称原实验通过。独立性能对比、四模型扩展、TP及完整消融仍待完成。

公开[审计与复现资料](../evidence/native_agent_20261017/README.md)省去不透明的引擎Session/请求ID，仅保留模型输入/输出token与必要标量。实际实现、实验和材料由AI工具辅助完成；HW1及R5/R6既有发布不变。
