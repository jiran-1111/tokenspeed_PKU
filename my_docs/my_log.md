# jr的实验日志

### 20261015
#### 0、准备操作

```bash
git clone
cd tokenspeed_PKU

# git提交
git branch
git switch jr
git add .
git commit -m "docs:"
git push -u origin jr
```

#### 1、配置环境

```bash
# 激活conda
source /opt/miniconda3/etc/profile.d/conda.sh
# 激活ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 激活aics环境
conda env list
conda activate aics
python -m pip show torch torch-npu

# 测试环境是否正常运行
python - <<'PY'
import torch
import torch_npu

print("PyTorch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("Visible NPU count:", torch.npu.device_count())
PY

# 结果显示如下
PyTorch: 2.9.0+cpu
torch_npu: 2.9.0
NPU available: True
Visible NPU count: 8
```

#### 2、运行现成的nanoGPT

```bash
# 检查启动参数和已有权重
sed -n '1,130p' /workspace/nanoGPT/sample.py
ls -lh /workspace/nanoGPT/out-shakespeare-char
ls -lh /workspace/nanoGPT/data/shakespeare_char

# 尝试nanoGPT推理
cd /workspace/nanoGPT
python sample.py \
  --out_dir=out-shakespeare-char \
  --device=npu:0 \
  --num_samples=1 \
  --max_new_tokens=64 \
  --compile=False
```

#### 3、这个实验要干什么

让 TokenSpeed 能调用昇腾 NPU，正确执行 Qwen3.5-0.8B 的完整推理；然后用测试证明结果正确，并测出运行速度和显存占用。

任务一：在单 chip 上启动 Qwen3.5-0.8B 并完成连续推理

| 子任务 | 具体做什么 | 完成标志 |
|---|---|---|
| 1.1 确认基础环境 | 激活环境，验证 NPU 计算 | **你已经完成：矩阵乘和 nanoGPT 均成功** |
| 1.2 确定 TokenSpeed 启动方式 | 查看仓库的 Ascend 启动示例，确认需要的依赖 | 能进入 TokenSpeed 启动流程 |
| 1.3 准备模型 | 获取 Qwen3.5-0.8B 权重、配置、tokenizer，固定版本 | 能正确读取模型文件 |
| 1.4 加载模型并运行短输入 | 在单 chip 上加载权重、执行 Prefill | 得到第一个新 Token |
| 1.5 跑通连续生成 | 执行多步 Decode，补齐遇到的 NPU 支持问题 | 能生成一段文本 |
| 1.6 跑通连续请求 | 同一进程依次处理多个输入 | 不崩溃，各请求正常完成 |
任务二：运行关键算子或模块测试，以及模型端到端正确性测试
这个任务可以与任务一并行：每适配一个关键模块，就测试一个。
子任务	具体做什么	完成标志
2.1 选定参考实现	固定参考代码、权重、计算精度和输入	明确“与谁比较”
2.2 测关键模块	对新增或修改的 Attention、Gated DeltaNet、卷积等关键路径做数值对照	典型形状和边界形状通过
2.3 测多步状态	对照连续多步计算的输出和状态	状态更新正确
2.4 测请求隔离	比较“先运行 A 再运行 B”与“全新状态运行 B”	B 不受 A 的残留状态影响
2.5 测完整模型	固定配置，在多个输入长度下比较模型输出，并运行连续生成	端到端测试通过


不要求把整个项目所有算子重新测试一遍。重点是你们新增或修改的关键模块，以及完整推理所依赖的重要路径；已有测试可以复用，但要说明覆盖范围。
**本任务交付：**测试命令或脚本、通过结果、参考实现说明、误差指标与阈值依据。

任务三：提供已测得的基础性能数据，解释主要计算开销


#### 4、三个实验推进

|实验|做什么|负责人|
|---|---|---|
|已有NPU算子测试|运行仓库的注意力、归一化、RoPE 测试|jr|
|Qwen3.5 计算路径梳理|阅读模型代码，追踪 Attention、Gated DeltaNet、卷积及状态更新|张三|
|最小模型推理|加载权重，短输入 Prefill，然后逐步 Decode|李四|
