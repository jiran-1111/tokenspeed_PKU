# 作业1：诊断脚本正式化与执行前状态快照

用户2026-10-15已明确回复“按建议修复诊断脚本”，确认本规格所述方案：正式化现有可选诊断脚本，补实际执行前状态复制，验证原地更新不污染快照；不改推理计算或容差。

## 根因与范围

当前测试从未发布的`.tmp/diag_capture/sitecustomize.py`导入不存在的`_clone_state_page`，干净环境全course回归失败。旧356行hook只输出执行后状态，不能证明有执行前记录；不通过复制临时文件、跳过断言或新增无人使用helper掩盖问题。

将既有hook整理到正式`scripts/diagnostics/sitecustomize.py`，保留opt-in arm、position、input ID和layer0限制及旧输出。空诊断环境导入不得加载Torch或安装hook。生产路径、旧raw、参考采集器、阈值不改。

仅在已启用的layer0 GDN诊断、原backend调用前，从真实输入页保存conv和recurrent的`detach().clone()`；原backend结束后将该快照与执行后状态实际接入原raw输出，名字明确区分before/after。不能把输出页当输入页，不能用view替代独立存储。无效页明确处理，不默默采错误页。

测试改为正式路径，保留alias断言；补真实wrapper行为测试：模拟backend对同页原地更新，before输出保留旧值、after输出为新值，输入输出页不同时按真实索引读取。不得仅mock一个无人调用helper。未启用时无Torch导入/输出/patch。

## 验证与发布

先观察修正后正式路径/缺前态行为的RED，再最小实现GREEN；专项与全course回归、SPEC→quality、根代理独立复验，然后部署同SHA到服务器验证。保留原1failed/523passed/18skipped记录。未经服务器验证不上传GitHub，不提交/推送或清理无关文件。
