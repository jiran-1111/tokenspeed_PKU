# Environment

## 本地开发主机

- OS：Windows（仅用于源码检查、文档和不依赖 NPU 的静态/单元测试）
- Conda prefix：`../.conda-env`
- Python：3.11.15
- 上游起点：`cf4a4f89ae5f9423a9af490d8f3a0613adfba3fd`；HW1最终交付分支`course/hw1-20261015`及源码清单见[证据](../evidence/hw1_20261015/README.md)
- TokenSpeed Python requirement：`>=3.10`

项目外层的 `.condarc`、`scripts/setup_env.ps1` 和 `scripts/use_project_env.ps1` 将新下载与生成缓存约束在课程工作目录内。远端 `scripts/use_project_env.sh` 还将 Ascend Profiler 的 `ASCEND_WORK_PATH` 指向项目内 `.tmp/ascend_work`，使 CANN 原始采集文件留在项目目录。

## Ascend 实际测试栈（Ascend910_9362，Ascend910_93系列）

2026-10-14纠正：不能仅凭`Chip Version V1`认定为非910C。新鲜运行时和本机CANN配置确认`Ascend910_9362`/`Ascend910_93`；一手软件来源映射到Atlas A3/910C系列，整机DMI和CPU标识补证据如下。详见[身份核实与来源](hardware_identity_20261014.md)。旧报告“V1”是工具字段简称，不改变已测设备或原始数据。

后续只读核实：DMI产品名为Huawei Atlas800I A3；640条CPU记录均为implementer0x48/part0xd02，结合华为该型号规格与util-linux维护者源码，支持Kunpeng920系列判断，具体CPU子型号未直接读取。当前可见4个NPU ID，每个含2个计算chip，共8个逻辑device；0.8B TP1用单逻辑device，27B当前TP2 pair2/3用NPU ID1的两chip，详情与编号表见上述身份记录。此资源可见范围不能等同官方整机满配或当前空闲状态。

### 模型版本

2026-09-27 通过 Hugging Face 官方模型 API 核验的完整 commit：

| 模型 | revision |
|---|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` |

来源：`https://huggingface.co/api/models/Qwen/Qwen3.5-{size}` 的 `sha` 字段。
四份 bring-up 配置必须指定 `model_revision`。服务启动传入 `--revision`，
正确性、Benchmark、Agent、消融及 Profiler 客户端的 tokenizer 使用同一 revision，
并在结果记录中保留该值。这固定了下载版本，尚不证明目标设备已加载或通过验收。

### 软件版本

- Python：3.11（远端项目 Conda 环境）
- PyTorch：2.9.0
- torch_npu：2.9.0
- CANN：8.5.0
- Transformers：5.12.0
- Triton-Ascend：3.2.1
- 安装参考：`test/ci_system/install_triton_ascend.sh`（当前复用服务器预装 CANN，不重装共享环境）

实际远端为 openEuler 24.03 / aarch64，`npu-smi` 报告 `Ascend910`、NPU Name9362、Chip Version V1，每芯片64 GiB HBM；4个NPU ID各含2个芯片，运行时SoC身份见上文。当前项目专属Conda环境由预装`aics`克隆，位于远端项目目录；共享环境保持只读。运行前需要source `/opt/miniconda3/Ascend/cann-8.5.0/set_env.sh`和项目`scripts/use_project_env.sh`。Triton-Ascend smoke test通过；该验证不等于模型兼容或性能验收。所有结果保留真实SoC和逻辑device记录，不再凭V1单独判断硬件不符合910C要求。TokenSpeed引擎指标端点使用本项目端口39142，SMG网关指标使用39141。历史run后的空闲快照不证明服务器当前空闲，任何硬件实验均需新鲜检查并避开其他用户任务。
