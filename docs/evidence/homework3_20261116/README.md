# 作业3原始证据与汇总

两模型 TP=2 的十二组实验均退出 0，144/144 轮匹配独立完整历史参考。
源清单在十二组中相同，逐组运行前后均有源码复核。

- [统计汇总](summary.json)保存每 rank 的块、Host 字节、搬运、命中、淘汰及请求输出率。
- [逐轮验收](acceptance.json)保存双 rank 恢复位置核对、跨会话块复用实例、重算区间及共享 HBM 峰值。
- [矩阵退出码](status.tsv)、[设备映射](mapping.txt)、[运行前共享进程](before.txt)、[运行后共享进程](after.txt)保留环境边界。
- [压缩档案 SHA-256](archive_sha256.json)用于核对以下原始记录。

每个档案包含 results.json、trace、reference_trace、源码及调度器二进制的哈希清单、各轮 NPU 采样。
trace 内的 scheduler、host_allocation、transfer 分别是实际块分配、Host 池及已完成传输事件。

- [0.8b-tp2-high-off](raw/0.8b-tp2-high-off.tar.gz)
- [0.8b-tp2-high-on-large](raw/0.8b-tp2-high-on-large.tar.gz)
- [0.8b-tp2-high-on-small](raw/0.8b-tp2-high-on-small.tar.gz)
- [0.8b-tp2-low-off](raw/0.8b-tp2-low-off.tar.gz)
- [0.8b-tp2-low-on-large](raw/0.8b-tp2-low-on-large.tar.gz)
- [0.8b-tp2-low-on-small](raw/0.8b-tp2-low-on-small.tar.gz)
- [4b-tp2-high-off](raw/4b-tp2-high-off.tar.gz)
- [4b-tp2-high-on-large](raw/4b-tp2-high-on-large.tar.gz)
- [4b-tp2-high-on-small](raw/4b-tp2-high-on-small.tar.gz)
- [4b-tp2-low-off](raw/4b-tp2-low-off.tar.gz)
- [4b-tp2-low-on-large](raw/4b-tp2-low-on-large.tar.gz)
- [4b-tp2-low-on-small](raw/4b-tp2-low-on-small.tar.gz)
