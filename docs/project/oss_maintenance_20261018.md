# 开源维护与协作核查（2026-10-18）

## 修复基线和范围

维护分支 `fix/oss-project-20261018` 从课程最终提交 `d41b6798` 建立。
原开发工作树、`main`、HW1、R5、R9 和课程最终分支保留原样。
修复环境安装与激活入口、Windows 独立克隆复现、当前文档、课程 CPU CI
和临时产物忽略规则，不修改模型、调度器、内核、配置或数值接受门槛。

- Linux 环境入口使用同一个 CANN 根目录，传递给下级安装脚本；不存在或
  相互矛盾的 CANN 配置在安装依赖前拒绝。隔离回归使用假的 Conda/CANN
  命令验证实际 Shell 行为，不能替代真实驱动和包安装验证。
- Windows 的安装和激活脚本位于仓库 `scripts/` 内，CPU 测试依赖在
  `requirements/course-test.txt`，不再引用克隆目录外的文件。
- 当前入口集中到 README；历史状态文档明确按日期解释。Python 包元数据
  引用的 `python/README.md` 已补齐。
- 课程 CPU 工作流使用标准 runner 和独立环境，不安装 Torch、模型或 NPU
  栈；缺少可选依赖的测试在输出中明确列出。
- `results/` 和旧 `.pytest_tmp*` 目录默认忽略，不删除已有实验数据。
  公开证据继续存于 `docs/evidence/`，发布使用明确文件清单。

项目结构继续遵守 [AGENTS.md](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/AGENTS.md) 和已有设计边界；本轮没有
为统一目录而重写推理实现。Python 修改参考
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)，
格式与检查以仓库完整 pre-commit 配置为准。

## 其他上传内容和冲突

核查时远端有六个分支，完整 SHA、文件清单和只读模拟结果保存在
[协作核查记录](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/docs/evidence/oss_maintenance_20261018/collaboration_audit.json)。
`main` 为 `d7091b25`；课程最终、HW1、R5 和 R9 引用与交付时一致。
未发现远端 PR head 引用；这不覆盖尚未 push 的个人本地工作。

另一个 [jr 分支](https://github.com/jiran-1111/tokenspeed_PKU/tree/jr)
包含 `f494dac3` 和 `202838e6` 两个个人提交，基于另一版上游 main。
其个人提交涉及 38 个路径，与课程交付改动没有直接路径交集；但从双方
共同祖先 `cf4a4f89` 计算，jr 有 367 个单侧提交，整分支改动 1449 个路径，
与课程交付有 46 个路径重叠。只读模拟合并实测 22 个文本冲突，涉及
Engine 输出、NPU 执行、Qwen3.5、GDN 和 C++ scheduler。
这些分歧不能归咎为个人提交直接覆盖了课程代码。

个人补丁也不能原样应用到当前运行时：它依赖上游已经进行的 kernel
公共导出迁移，`git apply --check` 在 kernel 初始化文件失败。用 jr 的整个
初始化文件覆盖当前文件，会丢失当前运行时仍引用的公共算子名称。
因此本轮保留该分支及其作者归属，增加导航和核查记录，不合并或覆盖它。

该分支的 ATen 实验明确属于单请求、离线执行，使用调用者自有状态，
未注册替换生产算子；保持独立时不会覆盖当前生产运行时。它的 0.8B
模型 revision、输入和固定续接、误差指标、依赖版本及阶段计时口径均与
HW1 冻结验证不同，不能替换本项目的 Stage B、最终 Decode Profiler
或软件测试计数。两个分支的报告应各自按自己的源码和协议解释。

部分外部分支日志和脚本包含机器私有绝对路径。核查文件仅保存分类、
相对文件名和行号，不复制真实值；本轮没有把这些日志转存到维护分支。
静态扫描未命中所检查的凭据模式不代表证明不存在任何秘密。

未来如需集成，应逐项列出其个人补丁文件、兼容公共 kernel API、移除
机器路径依赖并保留作者归属，再对新的源快照进行软件和设备验证。
不能通过直接合并整分支顺带升级上游，也不能将独立实验结论混入既有证据。

## 源码和实验归属

原 1415 项清单和历史证据不修改。维护改变了其中两个环境 Shell 入口
及一个环境软件测试；模型、scheduler、kernel 和实验配置保持字节不变。
新增 Windows 入口、依赖清单、CI 和包说明属于维护源，使用单独的
[维护清单](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/docs/evidence/oss_maintenance_20261018/source_manifest.json)。

历史命令中指定旧 `source_manifest.json` 时，应使用冻结的课程最终提交。
在维护分支运行新的验证时，显式指定本页维护清单；校验失败不能绕过。
本轮仅做软件、Shell 路径和交付验证，没有追加 NPU 数值、性能或 Profiler
测量，也没有把旧快照的硬件测试改称维护分支新测结果。

验证结果及可复核命令见
[维护验证记录](../evidence/oss_maintenance_20261018/README.md)。
