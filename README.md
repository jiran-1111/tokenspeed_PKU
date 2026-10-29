# TokenSpeed Ascend Agent

面向 Ascend 的大模型推理课程项目，基于 [TokenSpeed](https://github.com/lightseekorg/tokenspeed) 开发。复用统一推理引擎、C++ 连续批处理调度器及多设备算子接口，增加 Qwen3.5 四模型配置、原生 Session KV 实验、分层正确性验证与性能分析。代码采用 [MIT 许可](LICENSE)，第三方组件遵循各自许可。

## 当前分支与项目状态

**当前分支：`homework2/completed`。按[作业2要求核对](docs/project/homework2_requirements_audit_20261028.md)，四项任务的实现、实验与[报告](docs/report/homework2_communication_subsystem.md)材料已齐备。0.8B/27B在有限协议下与TP1输出一致，4B/9B原短输入差异如实保留，不宣称四模型输出全部一致。**

[实机证据](docs/evidence/homework2_20261025/README.md)覆盖共享条件下两长度×两并发性能、每chip HBM、双rank时间线、27B容量与通信粒度实验。[当前状态](docs/project/homework2_status_20261023.md)以任务书为边界；完成分支对应作业2任务书范围，完整大项目状态单独记录。

| 入口 | 状态 | 用途 |
| --- | --- | --- |
| `homework2/completed` | **作业2已完成；对照差异如实报告** | 本分支；作业2报告、同卡TP2、性能与通信证据 |
| `project/develop` | **大项目继续开发** | 四模型运行时、原生Session/KV、Agent实验、开源维护与后续整合主线 |
| [`homework1/completed`](https://github.com/jiran-1111/tokenspeed_PKU/tree/homework1/completed) | **作业1已完成** | 独立保留0.8B单chip验证、基线、Profiler、补充实验和报告 |
| `jr` | **其他成员所有，本次未修改** | 保留独立工作；不跨分支覆盖或自动合并 |
| `main` | **上游参考/原默认分支** | 保留原历史，当前课程开发请使用本分支 |

[BRANCHES.md](BRANCHES.md)记录完整归档表及恢复命令。R5、R9、课程最终、旧HW1和旧维护分支已转为同名附注标签，原源码、证据和历史GitHub链接继续保留。

整理前大项目基线为标签 `project-baseline-20261023`（`415983de`）；原课程四模型实验仍属于 `course/final-20261018` 标签（`d41b6798`）。维护修复不追认成历史设备实验使用的源码。已有阶段材料和有限协议验收不等于整个项目最终验收或未知统一评测已通过；本分支专门补齐作业2，整合主线仍为 `project/develop`。

- [课程阶段交付及冻结复现入口](docs/project/final_delivery_20261018.md)：四模型配置、精确命令、性能和TP候选边界。
- [课程总报告](docs/report/course_project_final_20261018.md)、[作业1统一报告](docs/report/homework1_compute_subsystem.md)、[答辩材料](docs/presentation/README.md)。
- [阶段验证证据](docs/evidence/course_final_20261018/README.md)、[HW1原证据](docs/evidence/hw1_20261015/README.md)、[HW1补充证据](docs/evidence/hw1_supplement_20261022/README.md)。失败、旧门槛和适用范围一并保留。
- [维护与协作核查](docs/project/oss_maintenance_20261018.md)：维护修复与其他成员分支的关系。

原生Session API与普通HTTP请求的行为边界不同；工具等待实验使用固定工具结果。报告中的硬件、数值和统计限制以各自证据为准。教学网由用户本人提交，实际项目工作如实披露AI工具辅助。

## 仓库结构

| 目录 | 职责 |
| --- | --- |
| `python/tokenspeed/` | 引擎、服务接口、模型、缓存和设备执行编排 |
| `tokenspeed-scheduler/` | C++ 调度器与 Python 绑定 |
| `tokenspeed-kernel/` | 统一算子 API、注册和选择；设备算子的公共边界 |
| `tokenspeed-kernel-npu/`、`tokenspeed-kernel-amd/`、`tokenspeed-mla/` | 各设备或算子实现包 |
| `benchmarks/course_project/` | 配置构造、实验协议、客户端、采样与分析公共逻辑 |
| `scripts/` | 安装、环境激活、服务、验证和实验命令入口 |
| `configs/` | `final` 最终配置；`baseline`、`candidates`、`validation` 等保留实验角色 |
| `test/` | 上游测试；`course_project/` 为课程软件测试 |
| `docs/` | 设计、当前入口、历史过程、报告、公开证据和演示材料 |

控制面负责请求与调度；执行层负责设备计算；运行时通过统一内核接口调用设备算子。修改前阅读 [贡献指南](CONTRIBUTING.md)、[AGENTS.md](AGENTS.md) 和对应的 [设计文档](docs/design/)。

## 开发机：运行 CPU 软件测试

需要 Conda。以下操作只建立仓库内环境，不安装推理运行时、Torch 或 NPU 依赖。

Windows PowerShell，从仓库根目录执行：

```powershell
./scripts/setup_env.ps1
. ./scripts/use_project_env.ps1
python -m pytest test/course_project -q --basetemp .tmp/pytest-course
```

如已有独立的项目 Conda 环境，可向激活脚本显式传入 `-EnvironmentPrefix`。首次安装脚本可能需要网络访问包源；不要把包源凭据写进仓库。

Linux/macOS CPU 开发机：

```bash
export CONDA_PKGS_DIRS="$PWD/.conda-pkgs"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
conda create --prefix "$PWD/.conda-env" python=3.11 pip -y
conda run --prefix "$PWD/.conda-env" python -m pip install -r requirements/course-test.txt
PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python" \
conda run --prefix "$PWD/.conda-env" python -m pytest \
  test/course_project -q --basetemp .tmp/pytest-course
```

没有 Torch 或设备时，相应测试会明确跳过；软件测试不代替模型数值与硬件验收。新建输出目录运行实验，保留失败记录。自动检查见 [课程 CPU 工作流](.github/workflows/course-project.yml)。

## Ascend：环境与启动

要求 Linux、匹配的驱动/固件/CANN、Conda 及可用的 Ascend 设备。源码构建 scheduler 还需要兼容 C++20 的编译器和 OpenSSL 开发头文件及库，由目标主机预先准备。实际已测软件栈和固定模型 revision 见 [环境记录](docs/project/environment.md)。首次安装需要访问包源与模型源。CANN 由管理员预装，脚本不安装驱动或重装共享 CANN。

从仓库根目录执行，并将示例替换为本机已安装的 CANN：

```bash
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
bash scripts/setup_env.sh
source scripts/use_project_env.sh
python scripts/serve.py configs/final/qwen35_4b.yaml --dry-run
```

确认设备映射和空闲后再启动。`ASCEND_RT_VISIBLE_DEVICES` 必须按当前设备映射选择；0.8B、4B、9B 的最终配置为 TP1，27B 为 TP2。服务示例使用逻辑设备 0：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
bash scripts/serve.sh configs/final/qwen35_4b.yaml
```

四模型分别使用 `configs/final/qwen35_{0.8b,4b,9b,27b}.yaml`。9B 最终配置显式选择 FP32 输出头。普通服务提供 OpenAI 兼容接口，原生 Session 复用须通过 Engine 会话 API；完整实验和分析命令见 [最终交付入口](docs/project/final_delivery_20261018.md)。

## 产物、协作和维护

模型、缓存、临时文件和原始结果保存在仓库内的 `models/`、`.cache/`、`.tmp/` 和 `results/`，默认不提交。可审阅的标量证据、校验清单、图和报告放在 `docs/evidence/` 等公开目录；大型原始采集数据不随源码分发。公开证据中的原始 SHA 用于比对已保存的实验数据，不表示 Git 包含全部原始 trace。

`docs/project/` 和 `docs/superpowers/` 中带日期的记录保留当时状态；当前入口以本页和上方最终交付说明为准。开发使用独立分支和明确文件清单，通过 PR 审阅；不要跨分支全量覆盖运行时或重写他人的历史。

代码和文档贡献通过 [本 fork 的 Pull requests](https://github.com/jiran-1111/tokenspeed_PKU/pulls) 审阅，在 PR 中说明问题、修改及验证。仓库当前未启用 Issues。安全问题按 [安全策略](SECURITY.md) 私下报告；参与协作遵循 [行为准则](CODE_OF_CONDUCT.md)。
