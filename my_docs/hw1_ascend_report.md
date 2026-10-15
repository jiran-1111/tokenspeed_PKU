# 第一次作业：Qwen3.5-0.8B 单 chip Ascend 适配与正确性

实验日期：2026-10-15。本文是基于实际运行结果整理的阶段报告草稿。

**小组成员与主要贡献：提交前请填写真实姓名及各自实际完成的工作。**
本次自动化协助完成了环境核查、离线计算路径实现、分层验证、性能测量和本报告整理；不能据此替小组成员认领贡献。

## 1. 已完成范围与边界

在课程仓库 `tokenspeed_PKU` 内新增了单 chip、单请求、纯文本的完整推理入口，执行 Qwen3.5-0.8B 的全部 24 层文本计算，支持 Prefill、连续 Decode、同进程连续请求和分块输入续算。模型主要计算和状态均驻留 NPU，没有将 GDN 等不支持的运算转到 CPU。CPU 仅用于权重读取、tokenizer、控制流程及独立正确性参考。

**当前入口为 `python -m tokenspeed.experiments.qwen35.run`，尚未接入 `tokenspeed serve` 的生产调度器。** 本次没有宣称已完成 LCM 分页缓存、服务端批处理、ACL Graph、请求回撤、多模态或推测解码。现有生产服务路径保持原有设计；后续服务集成必须遵循仓库的缓存与状态导出约定，详见 `docs/design/qwen35-offline-ascend.md`。

未重建 nanoGPT，未修改共享 `aics`，未停止其他用户进程。保留了原有 `my_docs/my_log.md` 的未提交内容；首次实验完成时未提交或推送；后续备份版本以 Git 历史为准。

## 2. 环境、权重和运行方式

| 项目 | 实际记录 |
| --- | --- |
| 远程项目 | `/home/u2601213425/hw1/tokenspeed_PKU` |
| 分支与起点 | `jr`；`f494dac3f2fa644632196a112300b3b9b3ff23e5` |
| 硬件 | Ascend910_9362，单 chip HBM 65536 MiB；课程标称 910C |
| CANN / 驱动 | `/usr/local/Ascend/cann-9.0.0` / npu-smi 25.5.2 |
| 共享基础环境 | Python 3.11，PyTorch 2.9.0+cpu，torch_npu 2.9.0，Transformers 5.17.0，triton-ascend 3.2.0 |
| 个人环境 | `/home/u2601213425/hw1/.venv-qwen35`，使用 `--system-site-packages` 复用基础包 |
| 新增开发依赖 | pytest 9.1.1、einops 0.8.2、black 26.3.1、isort 8.0.1 及其小型依赖；均在个人环境 |
| 模型目录 | `/home/u2601213425/hw1/models/Qwen3.5-0.8B` |
| 权重来源 | ModelScope 官方 `Qwen/Qwen3.5-0.8B`；Hugging Face 直连不可用 |
| 模型权重版本 | `05f056712f8c582c8708395156518d963b46fb29`；模板等文件有各自 revision |
| 权重文件大小 | 1,746,942,600 字节 |
| safetensors SHA256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |

完整文件 revision、SHA256 和大小见 `hw1_results/weights_manifest.json`。加载器严格检查 320 个文本权重的键和形状；168 个非文本权重被明确排除。使用 tied embedding 作为输出头，没有跳过任何文本 decoder 层。

开始实验时 chip 0 已被占用，chip 7 无进程。基础验证和性能基线在物理 chip 7 上完成。之后 chip 7 出现其他服务，启动脚本自动拒绝继续运行，后续阶段计时和回归使用重新检查为空闲的 chip 5。每个进程仅设置一个 `ASCEND_RT_VISIBLE_DEVICES`，在进程内映射为 `npu:0`；芯片检查是即时占用检查，不是集群调度器的独占预约。

从仓库根目录运行，参数 `5` 表示物理 chip，启动前脚本会再次检查其占用。若占用则选取另一颗已确认空闲的 chip。

```bash
# 新增算子测试，以及原有 Ascend RMSNorm/RoPE 回归
bash test/experiments/run_qwen35.sh 5 test \
  tokenspeed-kernel-npu/test/test_layernorm.py \
  tokenspeed-kernel-npu/test/test_rotary_embedding.py

# 完整推理；可通过 --prompt 指定文本，默认关闭 thinking，greedy 解码
bash test/experiments/run_qwen35.sh 5 generate \
  --dtype bfloat16 --gdn-algorithm chunk --chunk-size 64 \
  --output my_docs/hw1_results/generate_bf16.json

# 整模型参考对照；float32 与 bfloat16 分别运行
bash test/experiments/run_qwen35.sh 5 validate \
  --dtype bfloat16 --gdn-algorithm chunk --chunk-size 64 \
  --output my_docs/hw1_results/validation_bf16.json

# 四组性能配置，每组三次；固定输出长度，计时不因 EOS 提前终止
bash test/experiments/run_qwen35.sh 5 benchmark \
  --dtype bfloat16 --gdn-algorithm chunk --chunk-size 64 --repetitions 3 \
  --output my_docs/hw1_results/benchmark_chunk.json

# 阶段计时包含额外同步开销，不能当作正常推理基线
bash test/experiments/run_qwen35.sh 5 profile \
  --dtype bfloat16 --gdn-algorithm chunk --chunk-size 64 \
  --output my_docs/hw1_results/profile_chunk.json

# 完整递归/分块 GDN 比较
bash test/experiments/run_qwen35.sh 5 gdn-benchmark \
  --repetitions 5 --output my_docs/hw1_results/gdn_microbenchmark.json
```

脚本依次执行用户提供的 conda/CANN 初始化，再激活个人 venv，并保留 CANN 设置的 `PYTHONPATH`。单独覆盖该变量会导致 `tbe` 导入失败，因此必须追加源代码路径。

## 3. 计算路径和支持情况

记批大小 B=1，输入长度 T，隐藏维度 D=1024，MLP 中间维度 I=3584。模型有 18 层 GDN 和 6 层完整注意力，非量化推理。

| 模块/功能 | 主要输入 → 输出/状态形状 | 精度 | 实现来源与 Ascend 支持 | 验证 |
| --- | --- | --- | --- | --- |
| Embedding、LM head | `[1,T] → [1,T,1024]`；末位置输出 `[1,1,248320]` | BF16，验证另跑 FP32 | 复用 ATen embedding/linear，经 torch_npu 执行；绑定输出头权重 | 严格权重加载、完整 logits 对照 |
| QKV/门/MLP 投影 | `[1,T,1024] → QKV/Z/A/B`；MLP 到 3584 再回 1024 | BF16 或 FP32 | 复用 ATen linear、SiLU、逐元素乘 | 整模型对照 |
| Zero-centered RMSNorm | 输入最后维 1024 或 256；同形输出 | FP32 归约及权重偏移，转回输入精度 | 新增组合算子，使用 `1+weight`，避免误用普通 RMSNorm | CPU FP64 公式及原有 RMSNorm 回归 |
| 因果深度卷积 | `[1,T,6144]`，权重 `[6144,1,4]`；窗口 `[1,6144,3]` | FP32 累加，卷积结果转回 BF16 后 SiLU | 新增四项移位累加，避免依赖 depthwise Conv1D 专用 kernel | CPU grouped Conv1D、逐步窗口及完整输入对照 |
| GDN Q/K 归一化与门 | Q/K/V `[1,T,16,128]`；g、β `[1,T,16]` | Q/K L2、g 为 FP32；β 由输入精度 sigmoid 得到 | 新增 ATen 组合；Q 额外乘 `128^-0.5` | 官方 CPU recurrent GDN |
| Gated DeltaNet | `[1,T,16,128] → [1,T,16,128]`；状态 `[1,16,128,128]` | 递归状态与分块中间值 FP32，输出转回输入精度 | 新增完整递归及分块三角代入，全部在 NPU | 常见/边界长度、多步状态、相关 key/慢衰减回归 |
| Gated RMSNorm | core、z `[1,T,16,128]` → 同形 | FP32 归约/SiLU，按参考顺序回写精度 | 新增组合，先归一化后乘 gate | 全模型参考验证覆盖 |
| QK norm、RoPE | Q `[1,8,T,256]`；K `[1,2,T,256]`；旋转前 64 维 | FP32 频率，sin/cos 转回输入精度 | 新增文本 partial RoPE；纯文本 T/H/W 位置相同 | partial RoPE 公式、原有 Ascend RoPE 回归 |
| Causal GQA + output gate | Q 8 heads，KV 2 heads；历史长度 L；输出 `[1,T,2048]` | QK/AV 输入精度，softmax FP32 | 复用 ATen matmul，新增因果 mask、KV 拼接、gate；NPU 执行 | 独立 CPU SDPA，含 offset 续算和长输入 |
| 请求状态 | 绝对 position、KV、卷积窗口、GDN 状态 | KV/卷积 BF16，GDN FP32 | 请求显式持有，禁止复用旧请求状态 | A/B/A 交错请求、继续 Decode、完整/分块前缀对照 |

Runtime 实验入口仅通过 `tokenspeed-kernel` 调用计算组合，没有直接依赖第三方 GPU kernel。另修复了 `tokenspeed_kernel` 初始化时强制加载可选 GPU profiling/compiler 的问题：仅在既有 profiling 环境开关开启时加载，开关开/关均有测试。现有 Ascend 专用 RMSNorm/RoPE 测试仍通过，但本实验数值路径主要复用了 ATen/torch_npu，而不是把所有现有注册 kernel 接入新的模型入口。

### GDN 从模型调用到状态更新

`TextModel.forward → gdn → gated_delta`。先进行 QKV/Z/A/B 投影；QKV 经过宽度 4 的因果卷积，拆成 16 个 128 维头。卷积状态只保留原始卷积输入的最后 3 个位置，不能保存激活后的结果。

定义 `g=-exp(A_log)*softplus(a+dt_bias)`、`β=sigmoid(b)`。Q/K 在 FP32 中做 L2 归一化；对每个位置，完整递归为：

```text
S_decay = exp(g_t) * S_previous
delta   = beta_t * (v_t - k_t^T * S_decay)
S_t     = S_decay + k_t * delta^T
y_t     = (q_t / sqrt(128))^T * S_t
```

分块实现将 `[B,T,H,K]` 转成 `[B,H,T,K]`，每块最多 64 Token。块内建立 unit lower-triangular 更新系统，用递归分块三角代入求解；使用前块的 FP32 状态计算本块输出，并写出下一块状态。最后不满 64 的块按真实长度处理，无 padding token 的虚假更新。

模型连续调用只更新当前 `RequestState`。新请求使用新对象；算子以新张量返回状态，旧 tensor snapshot 不被原地污染。同一 NPU stream 保证先后依赖；正常 forward 内不插入主机同步，计时在 Prefill 和每一步 Decode 边界同步。

## 4. 分层正确性验证

参考模型为环境中的**未修改 Transformers 5.17.0 `Qwen3_5ForCausalLM`**，CPU FP32、eager attention，加载相同 checkpoint 的原始数值。没有给参考实现替换为本次适配的 GDN。模型比较使用固定重复英文短句 token 序列，长度 1、17、65、128、512；每个输入比较 Prefill 最后位置及后续 4 步 logits。Decode 使用参考模型选出的 token 做 teacher forcing，避免两条序列先分叉后失去可比性。

| 层级 | 方法与判据 |
| --- | --- |
| GDN FP32 | 对官方 CPU recurrent 实现，`atol=3e-5, rtol=3e-3`；最终状态同样校验 |
| GDN BF16 | 输出 `atol=2e-3, rtol=3e-3`；FP32 最终状态仍用 `atol=3e-5` |
| 相关 key 回归 | key 近乎平行、β=0.99、g=-1e-8；T=64/65/129；`atol=3e-5, rtol=3e-4` |
| 卷积 | T=1/3/4/17/65，对 CPU FP32 grouped Conv1D；窗口状态严格相同；分步与一次计算一致 |
| Attention | CPU SDPA；FP32 `atol=2e-5, rtol=2e-4`；覆盖 offset 和 T=65/128 |
| 全模型 FP32 | max abs≤0.004，relative RMSE≤0.0003，cosine≥0.99999 |
| 全模型 BF16 | max abs≤0.5，relative RMSE≤0.02，cosine≥0.999 |
| 请求隔离 | A 后运行 B，再重跑 A，并比较 A 后续一步；logits 逐位相同 |
| 模型分块连续性 | 同一 17 Token 前缀一次输入与 3+14 分块，比较末位置 logits；沿用对应精度误差门槛 |

BF16 的 7 位尾数及多层矩阵运算会累积量化误差，因此不要求与 FP32 逐位相同；同时约束最大绝对误差、相对 RMSE 和方向相似度，避免仅凭 top-1 相同判定正确。接近零的分量以绝对误差约束，整向量尺度由 relative RMSE 补充；这些阈值是本实验的接受准则，不是对任意输入的误差证明。

**测试结果：57 passed**，其中新增 37 项，原有 Ascend RMSNorm/RoPE 20 项；见 `hw1_results/final_tests.log`。FP32/BF16 全模型验证结果及每个位置的指标见 `validation_fp32.json`、`validation_bf16.json`，最终摘要见本节后续的“实测数值摘要”。

### 实测数值摘要

两种精度分别检查 5 档长度 × 5 个位置，共 50 组 logits；全部通过，所有检查位置的 top-1 Token 一致。

| 精度 | 最大绝对误差（所有位置最大） | relative RMSE 最大 | cosine 最小 |
| --- | ---: | ---: | ---: |
| FP32 | 0.0000934601 | 0.00000540460 | 0.999999999986 |
| BF16 | 0.245296 | 0.0184857 | 0.999850290 |

请求隔离为逐位一致。完整前缀与 3+14 分块的 relative RMSE：FP32 为 `8.61283e-7`，BF16 为 `0.0101110`；两者 top-1 均一致。

### 定位并修复的问题

最初分块求逆使用有限矩阵幂展开。独立随机输入测试通过，短输入生成也正常；但 65/128 Token 整模型 FP32 相对 RMSE 达到约 0.16–0.39。逐层定位发现第 6 层 GDN 首次明显偏离，真实 key 高度相关，矩阵高次幂中间量与最终结果的差异导致严重数值抵消。

改为稳定的分块三角代入，并添加相关 key 的专门回归。修复前结果保存在 `validation_fp32_before_fix.json`，没有通过放宽阈值消除失败。这个例子说明：能够生成文本、随机算子测试通过、甚至 top-1 经常一致，都不能替代长输入的逐层数值检查。

## 5. 性能基线

物理 chip 7，B=1，BF16 权重/激活，FP32 GDN 状态，chunk=64，greedy。短输入 32 Token、长输入 512 Token；输出分别 8/32 Token。每种输入先运行一次 warm-up（生成 4 Token），每组测 3 次。输入来自固定英文句子的精确 token 截取，不含 tokenizer 和权重加载耗时。

TTFT 包含完整 Prefill、末位置 LM head、argmax 和同步；第一个输出 token 由 Prefill 产生，所以 Decode 平均值分别统计 7/31 次单步 forward。性能测量固定输出长度，不因 EOS 停止；交互生成模式会停止于 EOS。

| 输入 Token | 输出 Token | TTFT 中位数 ms | Decode ms/Token 中位数 | 峰值 allocated MiB | 峰值 reserved MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 49.014 | 40.115 | 1518.283 | 1562 |
| 32 | 32 | 49.124 | 38.415 | 1518.283 | 1562 |
| 512 | 8 | 173.275 | 37.644 | 1567.159 | 1628 |
| 512 | 32 | 175.217 | 37.957 | 1567.159 | 1628 |

逐次原始结果和全部生成 token ID 保存在 `hw1_results/benchmark_chunk.json`。HBM 列使用 torch_npu allocator 峰值：allocated 是活跃 tensor 内存，reserved 包括缓存池；二者不包含驱动/运行时及其他进程开销，不能解释为整个 chip 的 npu-smi 总占用。共享主机仍可能有调度抖动，三次结果用于建立课程基线，不是严格的性能置信区间。

### 瓶颈证据与优化尝试

重新确认 chip 5 空闲后，对 512 Token Prefill + 2 步 Decode 做阶段计时。每段前后同步，因此此数据用于定位，不能代替上表正常推理时间。各段累计时间：

| 阶段 | Prefill ms | Decode 两步合计 ms |
| --- | ---: | ---: |
| GDN | 184.689 | 30.495 |
| 线性投影 | 13.741 | 19.983 |
| RMSNorm | 11.530 | 18.095 |
| 因果卷积 | 9.601 | 14.877 |
| Gated norm | 3.992 | 6.340 |
| 完整注意力 | 2.391 | 3.422 |
| RoPE | 1.790 | 2.588 |
| LM head | 0.588 | 0.916 |

GDN 占所记录 Prefill 阶段时间约 80.9%。Decode 的 GDN 占约 31.5%，其余时间分散在大量小操作；归一化、卷积、布局/类型转换及主机发射值得进一步融合。阶段时间包含主机调度与设备执行，**本次没有有效的硬件带宽/占用率采样，不能给出精确的计算/访存/主机等待百分比**。原生 torch_npu profiler 的 CANN timeline/relation parser 失败，记录保存在 `failed_native_profiler.log`。

对完整 GDN 的同设备微基准使用 `[1,T,16,128]`、BF16 输入、FP32 状态、5 次测量中位数，并在计时前核对两种算法的输出和最终状态：

| T | 逐 Token 递归 ms | 分块三角代入 ms | 比较 |
| ---: | ---: | ---: | --- |
| 1 | 0.566 | 0.903 | 小输入递归更快，可作为 Decode 后续优化方向 |
| 32 | 5.881 | 1.541 | 分块约 3.82 倍 |
| 512 | 88.798 | 10.511 | 分块约 8.45 倍 |

这证明将大量递归小操作合并成块内矩阵运算对长输入有收益；所有状态更新仍完整执行。该比值是算子微基准，不是端到端模型加速比。一个递归测量出现明显离群值，已保留原始数据，使用中位数而非删掉离群值。

## 6. 文件与后续工作

- 模型与运行入口：`python/tokenspeed/experiments/qwen35/`。
- 算子：`tokenspeed-kernel/python/tokenspeed_kernel/ops/experimental_qwen35.py`。
- 测试、环境/芯片检查和微基准：`test/experiments/`。
- 所有实测 JSON、测试日志、环境及权重清单：`my_docs/hw1_results/`。
- 实验范围与缓存边界设计：`docs/design/qwen35-offline-ascend.md`。

下一步若要求走标准 TokenSpeed 服务，应把已验证的卷积/GDN 算子接入已有 kernel registry 和 GDN backend，沿用 LCM 状态块及 checkpoint/retraction 语义，再验证服务调度、并发请求、缓存复用、回撤和图执行。当前 `RequestState` 仅适用于离线验证入口，不应直接作为生产私有槽位缓存。

参考资料：仓库 `AGENTS.md`、`tokenspeed-kernel-npu/README.md`、`docs/recipes/models.md` 的 Ascend 小节、缓存及数值设计文档；[Qwen 官方模型仓库](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B)、[Hugging Face 模型配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)，以及本机安装的 Transformers 5.17.0 Qwen3.5 参考源码。实际复现以 SHA256 清单和环境快照为准。
