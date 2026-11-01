# 当前大项目交付与复现入口

主线为`project/develop`，被测实现固定提交`1d6099da5909e38fdbc737fe77bcb4176c978de1`。本页及当期报告/演示在后续材料提交补充，1535项被测源码清单保持不变。最终材料版本以本页所在Git提交为准；不要把旧20261018命令配上当前源码后跳过来源检查。

- [当期完整报告](../report/course_project_final_20261101.md)、[八项运行证据](../evidence/project_agent_20261031/README.md)、[机器可读验收汇总](../evidence/project_agent_20261031/final_verification.json)。
- [演示PPTX](../presentation_20261101/course_defense.pptx)、[演示PDF](../presentation_20261101/rendered/course_defense.pdf)、[讲稿与答辩问题](defense_outline_20261101.md)。
- [历史阶段报告](../report/course_project_final_20261018.md)保留原身份；作业1和作业2完成分支不改。

当前四模型各32输出/32context配对、各192正式/64预热性能请求通过，均为固定轨迹及共享条件；不是未知统一评测或任意输入模型质量通过。每个模型独立部署，至多使用一张卡的两个chip，一次一个本组NPU实验。教学网由用户提交。

## 1. 代码、环境和模型准备

用新短路径检出材料版本，例如公共课程工作区下`.tmp/pj-repro`。不要覆盖原服务目录或HW2冻结快照。下列路径变量是读者需要替换的示例，执行前使用真实绝对路径：

```bash
WORKSPACE=/path/to/course-workspace
SOURCE="$WORKSPACE/.tmp/pj-repro"
TASK="$WORKSPACE/.tmp/agent-repro-unique"
git clone --branch project/develop https://github.com/jiran-1111/tokenspeed_PKU.git "$SOURCE"
cd "$SOURCE"
git switch --detach MATERIALS_COMMIT
```

`MATERIALS_COMMIT`替换为本页所在固定完整提交；当前运行实现为上述1d6099da。先按[环境说明](environment.md)在项目目录建立环境，已有环境复用时提供其已核验激活脚本绝对路径；不安装全局依赖、不升级共享服务环境。Linux新环境入口为`scripts/setup_env.sh`，激活入口为`scripts/use_project_env.sh`，`CANN_ENV`显式指定本机CANN的`set_env.sh`。当前实测版本见[软件与环境记录](../evidence/project_agent_20261031/software_current.json)和[运行环境](../evidence/project_agent_20261031/environment_current.json)。可见资源不是课程预算。

将四模型固定revision文件放在当前检出内`models/Qwen3.5-{0.8B,4B,9B,27B}`，文件身份见[model_files.json](../evidence/project_agent_20261031/model_files.json)。可对已经核验的只读模型逐文件复制或硬链接；不能以目录软链接逃出当前检出、修改原权重或下载浮动revision。本轮测量核对55文件完整SHA，复现准备也重新核对，而非只看文件大小。revision如下：

| 模型 | revision | 最终配置 |
|---|---|---|
| 0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `configs/final/qwen35_0.8b.yaml`，TP1 |
| 4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `configs/final/qwen35_4b.yaml`，TP1 |
| 9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `configs/final/qwen35_9b.yaml`，TP1/FP32头 |
| 27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | `configs/final/qwen35_27b.yaml`，TP2 |

隔离scheduler扩展必须实际位于当前检出的`tokenspeed-scheduler/python/tokenspeed_scheduler/tokenspeed_scheduler_ext.cpython-311-aarch64-linux-gnu.so`。本轮复用已核验的同源隔离构建，SHA为`3931f548933ea65f3449fa032c1e9ff4c71610bdb26ce26872d77cc0afffbeae`。可以复制该已验证构建；重新编译时应使用当前scheduler源码和项目环境，在本地`.tmp/wheels`构建wheel并仅提取其扩展，不覆盖其他环境安装。示例（已激活项目Python且具备pyproject声明的构建依赖）：

```bash
mkdir -p .tmp/wheels
python -m pip wheel --no-deps --no-build-isolation --wheel-dir .tmp/wheels ./tokenspeed-scheduler
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root = Path.cwd()
wheels = list((root / '.tmp/wheels').glob('tokenspeed_scheduler-*.whl'))
assert len(wheels) == 1
target = root / 'tokenspeed-scheduler/python/tokenspeed_scheduler/tokenspeed_scheduler_ext.cpython-311-aarch64-linux-gnu.so'
assert not target.exists(), 'do not overwrite an existing extension'
with ZipFile(wheels[0]) as archive:
    names = [n for n in archive.namelist() if n.endswith('/' + target.name)]
    assert len(names) == 1
    with target.open('xb') as output:
        output.write(archive.read(names[0]))
PY
```

不同工具链构建的二进制SHA可能变化，必须记录新SHA和导入检查，再将其结果标为新实验；不能宣称复制了本轮完整二进制身份。配置、输入或源码清单变更时也需要另立清单和完整复验，不能沿用当前通过结论。

## 2. 新机器的软件与身份记录

先运行完整`pre-commit run --all-files`；依赖与缓存放当前工作区。准备脚本不伪造新的pre-commit执行：它只继承被测1535项字节相同的历史成功收据，并在ready中写明这一来源。本次新增复现工具本身由最终材料提交的完整pre-commit另外检查。

```bash
ENV_SCRIPT=/absolute/path/to/verified/scripts/use_project_env.sh
source "$ENV_SCRIPT"
cd "$SOURCE"
export PRE_COMMIT_HOME="$SOURCE/.cache/pre-commit"
export TMPDIR="$SOURCE/.tmp" TMP="$SOURCE/.tmp" TEMP="$SOURCE/.tmp"
pre-commit run --all-files
python docs/evidence/project_agent_20261031/runners/prepare_reproduction.py \
  --repo "$SOURCE" --workspace "$WORKSPACE" --task-dir "$TASK" \
  --environment-script "$ENV_SCRIPT"
```

该步骤不启动NPU实验。它拒绝已有任务目录/不同bootstrap，核对完整1535清单，重新运行课程CPU测试、两个自身进程清理用例以及parent/spawn/forkserver导入，核验55模型SHA并生成本机stat。测试临时目录位于被测仓库内；原始失败日志保留，不覆盖重试。准备成功后生成新的`model_state.json`和`ready_current.json`；不能复制旧机的PID、inode、ready收据作为本机验证。

`import_bootstrap.txt`是本轮实际使用bootstrap的逐字节快照，用来把旧环境的scheduler editable重定向放到普通源码路径之后；启动探针仍强制核对真正导入位于当前源码内，不靠PYTHONPATH文字推断。准备脚本与它的5项纯CPU回归属于后补复现工具，清单独立于实测8工具。

## 3. 设备与串行运行

激活环境后重新查看`npu-smi info -m`、`npu-smi info`以及目标chip进程。不要照抄本轮的NPU1/逻辑2、3。下例参数必须替换为本机新鲜观察的物理卡号和chip:逻辑device配对；共享运行须保留至少8192MB/chip安全余量，准入还参考旧峰值加余量。其他服务由原所有者管理，不能为本实验清理未知进程。

```bash
NPU_ID=YOUR_PHYSICAL_NPU_ID
CHIP_LOGIC_IDS=YOUR_CHIP0_LOGIC_AND_CHIP1_LOGIC_PAIRS
python "$TASK/controller.py" \
  --repo "$SOURCE" --task-dir "$TASK" \
  --model-state "$TASK/model_state.json" \
  --source-manifest "$SOURCE/docs/evidence/project_agent_20261031/source_manifest.json" \
  --python "$(command -v python)" --environment-script "$ENV_SCRIPT" \
  --jobs "$TASK/jobs.json" --ready "$TASK/ready_current.json" \
  --npu-id "$NPU_ID" --chip-logic-ids "$CHIP_LOGIC_IDS"
```

配对格式例如`0:2,1:3`，数字只能来自实时检查。控制器保持四质量→四性能串行，按TP1/TP2使用所选chip；质量失败则阻止相应性能。全部输出为不存在的新目录，默认名称见`jobs.json`。复现使用全新源码检出避免已有`results/raw/project_agent_20261031/`冲突；不得删除失败目录“重跑过关”。没有资源时等待，不恢复历史独占队列。控制器`finished_pending_independent_audit`只表示采集结束，仍需检查每项状态和跨端复算。

退出/超时只清理经过PID、UID、启动时间及会话/进程组核验的自身任务。共享模式要求本次进程释放、背景身份仍可信，不要求原有服务退出。守卫轮询无法预留整卡，性能仅对应记录条件。

单项采集入口也支持`--resource-mode idle|shared`，但完整退出后审计由外层控制器生成，不建议为最终验收省略外层。TP1若要选chip1，直接入口显式传`--chip-logic-ids 1:当前逻辑ID`；控制器按其明确作业协议从提供的配对中选较小chip编号，不将这一选择隐藏为采集器限制。

## 4. 独立复算和证据检查

每项`run.json`关联原始设备表、阶段、请求、配置及`post_exit_verification.json`；驱动在每次结束后立即审计。跨端下载应对整个原始文件清单及tar SHA逐字节核对，然后在同源码再次独立运行：

```bash
python docs/evidence/project_agent_20261031/runners/audit_native.py \
  RUN_JSON NEW_ANALYSIS_DIR --source-root "$SOURCE" \
  --source-manifest docs/evidence/project_agent_20261031/source_manifest.json
python docs/evidence/project_agent_20261031/runners/analyze.py \
  RUN_JSON ORIGINAL_TRACE_JSONL NEW_ANALYSIS_DIR --source-root "$SOURCE" \
  --source-manifest docs/evidence/project_agent_20261031/source_manifest.json
```

`RUN_JSON`、`NEW_ANALYSIS_DIR`和模型对应原轨迹由读者替换，输出目录不得存在。轨迹路径及SHA已经固定在性能协议内，不能换失败输入或重生成参考。新摘要必须核对退出0/无超时、源码与模型不变、设备审计、32组质量配对和工具/close、192正式/64预热及3配对、两端分析一致，不能只看一个布尔值。

本轮公开`summary.json`、逐请求CSV和`raw_manifest.json`；原始归档本地保留，SHA与公开摘要关联，未发布主机账号、凭据或不透明Session IDs。原始PID祖先链、UID与模型stat是当时外层观察，离线设备表不能独自重新证明这些事实。

本轮共享性能、资源和当前源码质量详见当期报告；未知统一评测、规定主机预算、任意输入质量和大并发/长上下文边界仍未验证。旧HW1/HW2失败及27B旧TP1容量拒绝继续保留。最终源码发布不自动等于课程最终验收。
