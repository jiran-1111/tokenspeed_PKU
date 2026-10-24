# Qwen3.5双rank切分与通信实现

本页解释作业2所用代码路径，形状由本地模型配置和权重加载器推导，尚不代替服务器运行证据。启动配置位于 `configs/homework2/`。TP=2时两个rank共同执行同一个模型，各持有一部分权重；它们不是两份独立模型副本。

## 四模型几何

配置读取来源、revision和字段见[模型几何记录](../evidence/homework2_20261023/model_geometry.json)。H是隐藏宽度，I是MLP中间宽度；FA为完整注意力，GDN为门控DeltaNet线性注意力。

| 模型 | H / I | 总层 / FA层 / GDN层 | FA Q heads / KV heads | GDN K heads / V heads |
| --- | --- | --- | --- | --- |
| 0.8B | 1024 / 3584 | 24 / 6 / 18 | 8 / 2 | 16 / 16 |
| 4B | 2560 / 9216 | 32 / 8 / 24 | 16 / 4 | 16 / 32 |
| 9B | 4096 / 12288 | 32 / 8 / 24 | 16 / 4 | 16 / 32 |
| 27B | 5120 / 17408 | 64 / 16 / 48 | 24 / 4 | 16 / 48 |

四模型FA head_dim均为256，GDN K/V head_dim均为128，conv kernel为4，词表248320。每四层中的最后一层是FA。以下写权重为 `[out_features,in_features]`，T为该次调度实际计算的token数；Prefill的T受分块和批次影响，Decode的T通常与当轮活动请求数有关。

## 典型4B层：rank0与rank1的对应关系

两个rank使用相同的形状，但各自持有不同head或通道切片。

| 阶段 | rank0 | rank1 | 复制/切分与通信 |
| --- | --- | --- | --- |
| 输入hidden与residual | `[T,2560]` | `[T,2560]` | 输入与残差复制 |
| FA Q/gate/K/V融合权重 | `[5120,2560]`，相应head前半 | `[5120,2560]`，相应head后半 | 每个逻辑投影分别沿输出切分，不是将整个checkpoint扁平二等分 |
| FA投影输出 | Q/gate各`[T,8,256]`，K/V各`[T,2,256]` | Q/gate各`[T,8,256]`，K/V各`[T,2,256]` | 本地heads完成各自注意力 |
| FA输出投影 | 权重`[2560,2048]`，部分和`[T,2560]` | 权重`[2560,2048]`，部分和`[T,2560]` | 按输入通道切分；SUM all_reduce后得到复制的完整hidden |
| GDN融合权重 | `[6176,2560]` | `[6176,2560]` | Q/K/V/z/b/a六逻辑段分别切分；本地K heads=8、V heads=16 |
| GDN投影输出 | Q/K各`[T,8,128]`，V/z各`[T,16,128]`，b/a各`[T,16]` | 同形状、不同head内容 | conv及递归更新均在本rank本地heads上执行 |
| GDN输出投影 | 权重`[2560,2048]`，部分和`[T,2560]` | 同形状、互补通道 | SUM all_reduce后恢复复制hidden |
| MLP gate/up融合权重 | `[9216,2560]` | `[9216,2560]` | gate/up各取4608通道；激活为`[T,4608]` |
| MLP down权重与输出 | `[2560,4608]`，部分和`[T,2560]` | 同形状、互补通道 | SUM all_reduce后得到复制hidden |

一层采用FA或GDN中的一种，再接MLP。不会在同一层把两种注意力都执行一遍。默认attention TP和dense TP均为2，主要通信是注意力输出与MLP输出的SUM all_reduce。通道切分可以让部分矩阵乘法和本地状态减少，但每层仍需交换部分和，因而双chip未必对所有小负载都更快。

## 各模型的GDN状态与词表切分

下表是每rank、每个逻辑缓存槽的形状，不包含槽总数或层数。缓存池实际HBM不能只用单槽大小推算，还包括FA缓存、权重、图、运行时及分配器开销。

| 模型 | GDN融合权重 | conv权重 / 单槽conv状态 | 单槽SSM状态 | GDN输出投影权重 |
| --- | --- | --- | --- | --- |
| 0.8B | `[4112,1024]` | `[3072,1,4]` / `[3072,3]` | `[8,128,128]` | `[1024,1024]` |
| 4B | `[6176,2560]` | `[4096,1,4]` / `[4096,3]` | `[16,128,128]` | `[2560,2048]` |
| 9B | `[6176,4096]` | `[4096,1,4]` / `[4096,3]` | `[16,128,128]` | `[4096,2048]` |
| 27B | `[8240,5120]` | `[5120,1,4]` / `[5120,3]` | `[24,128,128]` | `[5120,3072]` |

conv状态为BF16；本阶段驱动显式固定SSM状态为FP32，维度次序为`[本地V heads,V dim,K dim]`。A_log与dt_bias同样沿V heads切分并保留FP32。FA每个K/V缓存块的形状为 `[128,KV_heads/2,256]`。两个rank的请求、位置和逻辑块编号需要保持一致，但实际head数据是切分的，不能把一致编号解释成缓存内容全部复制。

embedding及输出头沿词表分片，每rank权重为 `[124160,H]`。embedding的局部结果通过all_reduce形成完整hidden；输出头局部logits通过all_gather拼回完整词表。0.8B/4B共用embedding与头权重；9B/27B不共享。9B的显式FP32输出头对TP1/TP2保持一致，普通HCCL gather保留FP32，不将它称为全模型FP32。

## 实现位置与后续实测核对

- `python/tokenspeed/runtime/models/qwen3_5.py`：FA、GDN投影和分段权重加载；`qwen3_5_moe.py`的dense路径：MLP切分。
- `python/tokenspeed/runtime/distributed/comm_manager.py`：注意力与MLP输出通信。只有并行宽度等条件不同的路径才使用reduce-scatter/all-gather，不能把默认TP2写成该模式。
- `python/tokenspeed/runtime/distributed/comm_backend/hccl.py`：HCCL集合通信；`execution/distributed_initializer.py`：进程组和设备绑定。
- `python/tokenspeed/runtime/layers/attention/configs/linear_attn.py`及`kv_cache/recipes/qwen35.py`：状态形状、dtype和缓存字段。
- `python/tokenspeed/runtime/layers/vocab_parallel_embedding.py`与`logits_processor.py`：词表分片、入口all_reduce和输出all_gather。

后续需用两个rank运行日志与时间线核对上述实际调用、消息大小、状态和耗时。每个all_reduce的主要hidden载荷为BF16 `[T,H]`，还存在控制通信、输出词表收集及运行时等待；仅凭静态调用点不能推导总通信时间。计算、通信可能重叠，需按rank内区间并集和交集分析；host enqueue时间与device任务时间分开，两个rank时间不相加为请求端到端延迟。
