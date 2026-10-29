# 分支与版本使用说明

## 应该选择哪个分支

| 分支 | 标签/状态 | 用途 |
| --- | --- | --- |
| [`homework2/completed`](https://github.com/jiran-1111/tokenspeed_PKU/tree/homework2/completed) | **作业2已完成** | 从project/develop a77f25c3建立；四模型同卡TP2、性能、HBM、通信及报告已取得。4B/9B短输入差异未解决，不宣称全部数值通过。 |
| [`homework1/completed`](https://github.com/jiran-1111/tokenspeed_PKU/tree/homework1/completed) | **作业1已完成**；冻结标签 `hw1-completed-20261022` | Qwen3.5-0.8B单chip、作业1报告、原正确性/性能/Profiler及补充实验。实现保持冻结，文档可继续说明。 |
| [`project/develop`](https://github.com/jiran-1111/tokenspeed_PKU/tree/project/develop) | **大项目继续开发**；整理前基线 `project-baseline-20261023` | 四模型Agent项目的开发、维护和后续整合入口。已有阶段实验与报告不等于全部最终验收已通过。 |
| [`jr`](https://github.com/jiran-1111/tokenspeed_PKU/tree/jr) | **其他成员分支，受保护不清理** | 保留作者独立工作。本次不修改、不重命名、不删除、不合并该分支。 |
| [`main`](https://github.com/jiran-1111/tokenspeed_PKU/tree/main) | **上游参考/原默认分支** | 保留原历史，不把它当作当前课程开发成果入口。本次不改其内容。 |

只需查看作业1成果时选择 `homework1/completed`；作业2实验使用 `homework2/completed`；继续整个项目时选择 `project/develop`。Git标签是固定快照，开发分支可以继续前进。这里的“作业1已完成”指任务书范围内已记录的实验与报告，不表示教师评分或自由生成逐Token一致；“大项目继续开发”不抹除已经完成的阶段材料。

## 已退役分支及归档

下面旧分支已由上述两个入口替代，原名称保留为**同名附注标签**。因此旧GitHub `tree`/`blob`链接仍能解析原快照；完整提交和失败记录没有被删除。标签不会随日常开发移动。

| 原分支 → 同名归档标签 | 冻结提交 | 保留原因 |
| --- | --- | --- |
| [`course/agent-20261015`](https://github.com/jiran-1111/tokenspeed_PKU/tree/course/agent-20261015) | `c7f7694a42f717611f28b82f952570ee0b2e5ea5` | R5阶段；已包含于大项目后续历史 |
| [`course/agent-20261017`](https://github.com/jiran-1111/tokenspeed_PKU/tree/course/agent-20261017) | `82788a85eca26a631222d6ca0d03ac245aff6d9a` | R9四模型Session性能阶段；已包含于大项目后续历史 |
| [`course/final-20261018`](https://github.com/jiran-1111/tokenspeed_PKU/tree/course/final-20261018) | `d41b6798c737f923e36b9a3876cd3b276c9074ff` | 四模型课程阶段固定源码/配置和实验复现基线 |
| [`course/hw1-20261015`](https://github.com/jiran-1111/tokenspeed_PKU/tree/course/hw1-20261015) | `1d113e54f789a22b14cbbdccbc6230c3f4d0e574` | HW1原始1396项计算源码、数值/性能/Profiler证据 |
| [`course/hw1-supplement-20261022`](https://github.com/jiran-1111/tokenspeed_PKU/tree/course/hw1-supplement-20261022) | `abad5130bc9add029b258bc7e8e9ea8194885f33` | HW1最终补充实验、1407项快照和统一报告 |
| [`fix/oss-project-20261018`](https://github.com/jiran-1111/tokenspeed_PKU/tree/fix/oss-project-20261018) | `415983de06e918a1901505ed20b9f93b6e7175a3` | 环境与开源维护历史基线；后续由project/develop接续 |

归档前已核对R5、R9和课程最终提交均是大项目维护基线的祖先；HW1原始提交是HW1补充快照的祖先。HW1补充计算诊断保持独立快照，维护分支已同步同一份报告与证据，不把不同模型实现的测量结果混合。归档不会把旧数值或性能结果改称为新代码测量。

旧维护PR #1作为历史审阅记录保留并关闭，不合并到 `main`；其原head与base均有归档标签，维护内容完整接续到 `project/develop`。其他成员的 `jr` 不受此次清理影响。

## 获取和恢复版本

```bash
git fetch origin --tags
git switch --track origin/project/develop
# 或查看已完成的作业1：
git switch --track origin/homework1/completed

# 严格复现冻结HW1补充快照（脱离分支）：
git switch --detach refs/tags/hw1-completed-20261022
# 严格复现原课程四模型快照：
git switch --detach refs/tags/course/final-20261018

# 如确需恢复某个旧阶段为一个新分支：
git switch -c restore/r9 refs/tags/course/agent-20261017
```

已有同名本地分支时直接 `git switch project/develop` 或 `git switch homework1/completed`。切换前先保存个人改动，不要用reset/clean清理未提交工作。历史本地工作树可以继续保留，不要求为远端分支整理删除实验目录。

## 报告和复现入口

- [作业1统一报告](docs/report/homework1_compute_subsystem.md)、[HW1原证据](docs/evidence/hw1_20261015/README.md)、[HW1补充证据](docs/evidence/hw1_supplement_20261022/README.md)。
- [大项目开发入口](https://github.com/jiran-1111/tokenspeed_PKU/tree/project/develop)、[四模型阶段交付和冻结复现](https://github.com/jiran-1111/tokenspeed_PKU/blob/project/develop/docs/project/final_delivery_20261018.md)。

分支整理仅改变引用和说明文件，不改变模型、算子、调度、配置、输入样例或数值门槛。报告里的设备实验仍归属于各自原始commit/标签；后续计算实现变化必须重新验证。
