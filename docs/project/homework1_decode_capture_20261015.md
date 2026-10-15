# 作业1诊断脚本：正式来源与状态快照验收

## 根因与修复

本机全course原结果1failed/523passed/18skipped；旧测试从未发布的`.tmp/diag_capture/sitecustomize.py`导入，而旧hook本身也没有所需快照helper。此为未完成诊断契约，不是已证明的引擎状态污染。用户2026-10-15批准正式化并补实际执行前快照。

新正式入口`scripts/diagnostics/sitecustomize.py`保留opt-in arm/position/input目标及原poststate、raw记录。只在命中目标的layer0 GDN调用前，从实际输入页复制conv/SSM；执行后实际写入`conv_state_before`/`ssm_state_before`，与原poststate区分。无效输入页记录capture_error，不用负索引采错页，也不阻止原backend执行。非0层不克隆、不增加before字段，旧post行为保留。未启用环境导入不加载Torch或安装hook。

未修改生产模型、kernel、数值容差、正式参考或旧raw。新脚本只用于诊断，不计入性能样本。

## 验证

- 初始真实wrapper RED：4failed/4passed，失败为缺before/helper及无效页记录；不是只观察FileNotFound。
- SPEC指出before误覆盖所有层，补非0层测试实际RED 1failed/8passed，再最小限制layer0；复核通过。quality无Critical/Important。
- 根代理本机全course：530passed/18skipped，13.29秒；跳过不是数值通过。
- 服务器对应源码专项：9passed、0fail/error/skip，0.58秒；包含同页原地修改、输入输出不同页、未启用/目标guard、非0层和负输入页行为。
- 服务器真实Torch2.9.0 CPU张量另验：BF16 conv形状5×6144×4、FP32 recurrent形状5×16×128×128。复制后原页加1，快照逐位保持原值且存储地址独立。此为真实Torch CPU语义，不称NPU/fullmodel执行。
- 服务器其余现存course测试509passed/1skipped，17.25秒。但测试文件清单与本机不同（缺5文件、3文件不同），**不等于同一最终快照的全量验收**；发布前仍须统一清单/对应源码并重验。

质量审查非阻断建议：新增test doubles的类型注解可补齐；正数越界页的额外覆盖可增强。旧hook的历史动态patch风格没有作为本次新生产算法改动。

## 来源

原始目录：`results/raw/homework1_decode_capture_20261015/`。远端旧测试部署前已备份且SHA为`c729401a236a522c69afb936013940ce9da6500880556749eb2bce91734f430f`；比较后确认新版本保留原有两个测试、增加诊断覆盖，没有覆盖远端独有逻辑。

| 产物 | SHA256 |
|---|---|
| 正式hook | 70d85bbeb8eefa917181f2fe8060d12092f5cbdfc5c9817d27e30284708d1f82 |
| 专项测试源 | 23ea9799e64f5cafd1128f7b791f64470f6d0c910cd140ee3cffab5fb621c3b7 |
| focused.xml | a944e14e26f9955df255240588731f1c28f1ac029083f891ff5adf30e77df4a5 |
| full.xml | 9e4258ed952ecbde80fe3a692cf5acdd8e6c7345840f8d67350da7bc9b468b90 |
| real_torch_snapshot.json | 5bd85c17a8818681bea3bdb3058f847ebbd04d2c0b981a03a5fefeec589e837e |

源码部署前后SHA相同；XML/JSON同步后本机三项SHA均匹配。后续已对齐测试与配套源并完成服务器549项、0skip软件回归，另见[快照对齐验收](homework1_snapshot_alignment_20261015.md)。没有Git提交或推送。
