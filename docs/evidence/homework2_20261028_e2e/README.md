# 作业2：独立端到端CPU参考

本批补充从checkpoint和原token IDs自行计算的完整32层参考，不再把部署的中间hidden、Q/K/V、卷积或SSM喂给参考程序。两种模型各使用原128输入及原17/9续接；每模型比较TP1/TP2数学分片与BF16存储/FP32控制，共8组、112个完整logits向量。它们对应两模型各一条固定轨迹，共用同一个独特128-token Prefill输入，不是112个独立样例。

**本批全部在CPU运行，TP2是互补分片的数学模拟，不是真实HCCL或新增NPU性能实验。**实际双chip运行与通信证明沿用同1531项核心的既有实测。原4B/9B自由输出差异、旧失败和全部非零误差保持，未新增模型通过阈值。

## 结果与复现材料

| 文件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 原服务器完整摘要，包含9组两两比较、末步分数及独立embedding核对 |
| `4b_comparison.zip`、`9b_comparison.zip` | 原服务器完整数值JSON，逐字节无损压缩；包括全部logits指标、65边界的全128/1行及末行比较、各rank最终状态对照和原输入清单 |
| `reference_tools.zip` | 原实际执行14项工具/测试/协议/case及原manifest；独立模型、算子、checkpoint读取器和CPU串行驱动 |
| `comparison_tools.zip` | 原实际执行5项比较工具/测试及manifest，复用上一批严格R3数据绑定器 |
| `local_verifier.zip` | 实际本地独立NumPy复算脚本，不依赖Torch |
| `launch_and_audit_tools.zip` | 客户端发送的启动与后验审计脚本快照；启动时和结束后均核对比较器身份 |
| `publication_tools.zip` | 本地图像及公开材料生成脚本 |
| [execution_audit.json](execution_audit.json) | 比较器启动源摘要、客户端执行凭证摘要、前后工具指纹及最终summary绑定；不冒称完整PID/starttick历史链 |
| [local_verification.json](local_verification.json) | 独立重算20272组指标、重核768个自身状态数组载荷；范围明确不包括本地重提取NPU PT |
| [tie_diagnostics.json](tie_diagnostics.json) | 参考精确最大值集合的零epsilon诊断；不替代原canonical argmax比较或建立新通过门槛 |
| [boundary_curves.zip](boundary_curves.zip) | 无损压缩CSV，含三组跨分片比较全部5460个末行指标；图像只展示Prefill |
| [provenance.json](provenance.json)、[archives.json](archives.json) | 源码、案例、归档、测试及全部ZIP成员SHA |

所有公开文件小于500 KiB。原CPU参考53项约1.208 GB、分析16项约227 MB均已下载当前工作区并逐文件校验；大数组留在本地归档，不提交Git或包含checkpoint权重。

## 独立计算合同

参考自己计算完整原128行Prefill及每次单行Decode，各层分别建立自己的因果KV、卷积历史和FP32 SSM。layer-major组织只改变CPU计算顺序，不使用未来信息；软件测试同时检查未来续接不能影响前面输出。

BF16合同保留实际投影权重值、FP32 A_log/dt_bias/有效Gemma gamma、BF16列投影结果、行投影部分结果及求和、残差存储、FA缓存、卷积/core/gated norm等明确存储点。MRotary的每次eager乘法及加减分别存储；4B head结果BF16，9B head结果FP32。完整细节及原IDs在压缩包的`reference_protocol.json`和case中冻结。

FP32控制使用相同投影权重，取消BF16激活/缓存/中间结果存储，用于检查分片代数和数值敏感性。它没有部署到NPU，不是对原BF16实现的修复或替代验收。CPU参考不逐位仿真NPU全部FMA、exp、规约和chunk内算术。

参考仅在八组全部完成后写`reference_complete.json`。比较器必须核对精确八组键、每组state/NPZ/参数读取ledger摘要、case/revision/核心及工具，才读取原R3捕获；不存在边读部署中间量边产生参考的路径。所有原始运行核心保持`2a1a2196dba54928cc0bde375eb8412df0c222f8cf98cf4ff94c9f66c13f2af3`。

服务器参考31项、比较约束7项软件测试均通过且零跳过，包含独立HF小模型完整Prefill/Decode与eager Torch BF16 RoPE交叉检查。RED记录以及NumPy1.26标量兼容失败保留在工作区；修正只在新增参考的存储包装层，未改部署。软件测试不是模型数值验收通过的替代证据。

## 复现方法

在工作区解压参考工具，使用已验证的1531项快照及原revision checkpoint。依赖为该快照已有的NumPy、Torch、safetensors；全部装在工作区环境，禁止全局安装。以单项为例：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=/workspace/reference_tools \
/workspace/source_final/.conda-env/bin/python -B /workspace/reference_tools/run_reference.py \
  --source /workspace/source_final --model 9b --tp 2 \
  --storage-policy bf16_contract --output /workspace/results/9b_bf16_tp2
```

路径均为占位示例。`--tp`、存储策略和输出目录必须显式指定，输出目录不得已存在。运行前后的工具、checkpoint和核心身份不匹配即拒绝，不根据参考argmax自由续接。完整八项可按原`reference_controller.py`的任务目录布局恢复，保持CPU串行；比较还需要已核对的原R3数据，不能以缺数据时的跳过当成功。

解压或引用前按`archives.json`逐包和逐成员校验SHA；两个完整数值JSON的ZIP成员与服务器实际输出逐字节相同。

## 结果边界

独立BF16参考重现9B的25389/35378分叉，FP32控制两分片均选25389。4B参考两分片末步都把198/271压成21.25并列，canonical都选198；部署TP2自身271为唯一最大，不能把差异仅称为不同tie-break。其271落在参考的精确最大集合只是额外诊断，原17/18不变。

实际BF16对独立BF16完整向量仍明显偏离：4B两种TP最大relative-L2为0.18898931/0.16481254，9B为0.08653788/0.32681534。即使argmax相同，分数向量也可能不同；这些结果不能被隐藏或自动认定可接受。独立参考显示低精度存储与分片本身足以产生明显差异，提供更强的数值敏感性证据；本批没有确认可直接采用的部署修正，也未证明全部原误差正常。
