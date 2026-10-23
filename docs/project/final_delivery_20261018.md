# 最终课程交付入口

当前完整项目后续开发入口为 `project/develop`，作业1完成版为 `homework1/completed`，见[分支与标签说明](https://github.com/jiran-1111/tokenspeed_PKU/blob/project/develop/BRANCHES.md)。下文 `course/final-20261018`、`course/hw1-20261015` 和 `fix/oss-project-20261018` 等旧名称现为同名冻结标签；严格复现请用 `git switch --detach refs/tags/<标签名>` 或指定commit，不跳过源码身份检查。

本页保存 `course/final-20261018` 的冻结复现协议。要逐字节复现历史 1415 项清单，请在该分支或提交 `d41b6798` 的独立检出中执行；不要在维护分支上把旧清单检查跳过或改成通过。`fix/oss-project-20261018` 修复了环境入口及软件测试，维护后源码清单和兼容性边界见[维护记录](oss_maintenance_20261018.md)。该分支没有追加设备性能结果。

最终分支`course/final-20261018`；完整材料提交`bb51066d5552db8d45569b77d28259a5c446881d`已发布并回读核实，后续状态补充保持1415项实现字节不变。原始课程要求对应的四模型配置、最终同源质量验证、独立性能、有限TP候选、总报告与答辩材料集中于本页。HW1仍固定在`course/hw1-20261015`提交`1d113e54f789a22b14cbbdccbc6230c3f4d0e574`；R9性能固定提交`82788a85eca26a631222d6ca0d03ac245aff6d9a`。本分支通过普通新分支上传，不覆盖fork main、HW1或历史阶段。

- [总报告](../report/course_project_final_20261018.md)：原始要求映射、系统结构、分层正确性、性能/资源/消融与成员责任/AI披露。
- [最终证据](../evidence/course_final_20261018/README.md)：1415源清单、四模型各32组最终质量/上下文配对、工具/阶段/close审计、TP候选与匹配基线、原始SHA。
- [主性能报告](native_agent_performance_20261018.md)：R9r2四模型各192正式/64预热、P50/P95、近似TPOT、吞吐、观察HBM、全部逐请求CSV与图。
- [HW1报告](../report/homework1_compute_subsystem.md)：四任务、冻结数值门槛、旧1%与argmax差异、最终Decode Profiler和短/长×短/长基线。
- [答辩讲稿与问题清单](defense_outline.md)、[演示材料与重生成说明](../presentation/README.md)。

最终1415源SHA `d05a47d932b519b6e50ed0c668725935d93b744fffffa68e95f37a0227f6c6ea`：原R9r2的1409项实现逐字节不变，只新增4份final配置及2个仅改tp_size的候选。各模型final配置逐字节等同已验证基线或9B FP32头候选，不加入无效Session开关。服务器课程637通过/零skip；本机603通过/34项因无Torch跳过。完整pre-commit及交付allowlist/Git blob/服务器字节检查按交付记录执行，不以文件存在代替设备验证。

## 环境与服务启动

先按[环境说明](environment.md)准备工作区内项目环境及四个固定revision模型，现有环境执行`source scripts/use_project_env.sh`。不修改全局依赖或其他用户服务。每次查看`npu-smi info -m`和对应物理NPU的进程状态，确认逻辑2/3当前仍是物理NPU1两chip且所需chip空闲；不把旧快照当当前空闲。

单个4B服务示例：

```bash
source scripts/use_project_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
bash scripts/serve.sh configs/final/qwen35_4b.yaml
```

0.8B、9B相应改为`configs/final/qwen35_0.8b.yaml`、`configs/final/qwen35_9b.yaml`，均用一个chip。27B改`configs/final/qwen35_27b.yaml`，可见设备改`2,3`，两chip必须空闲。9B头精度由配置中显式HF override选择；不依赖诊断hook或隐藏环境变量。原生Session复用须使用Engine会话API；普通HTTP请求不自动等同原生Session。部署/评测一次仅运行一个，停止自己的服务并确认资源释放后再更换模型。

## 最终原生质量复现

下面4B示例输出目录必须不存在；只在所有既有NPU实验退出且所需chip空闲时运行：

```bash
source scripts/use_project_env.sh
python -m pytest test/course_project -q --basetemp .tmp/reproduce_course_tests
pre-commit run --all-files
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_native_agent_validation.py \
  --config configs/final/qwen35_4b.yaml \
  --workload configs/validation/native_agent_16turn_afterrun_20261017.json \
  --source-manifest docs/evidence/course_final_20261018/source_manifest.json \
  --output-dir results/raw/reproduce_final_4b \
  --engine-port 18780 --metrics-port 39982
npu-smi info -t proc-mem -i 1 > results/raw/reproduce_final_4b/post_exit_processes.txt
python docs/evidence/course_final_20261018/runners/audit_native.py \
  results/raw/reproduce_final_4b/run.json results/analysis/reproduce_final_4b
python docs/evidence/course_final_20261018/runners/audit_phases.py \
  results/raw/reproduce_final_4b/run.json results/analysis/reproduce_final_4b/phase_audit.json
```

其余模型仅替换配置/新输出目录，并按TP设置可见chip。需同时核验进程退出0、无超时、run状态/source、32输出与32context、真实历史、工具窗口其他会话进展与正活动块、idle持页/close归零、控制阶段边界、退出后目标空闲。任何失败保留原目录，不能只看exit或某个审计布尔值，也不能自动覆盖重试。外置预算原采集为900秒，超时仅向核实属于本次的子进程发SIGINT，不触及其他任务。

## 性能与TP复现

使用同固定协议`configs/validation/native_agent_performance_20261018.json`、上述1415源清单与新输出目录，入口改为`scripts/run_native_agent_performance.py --config ... --protocol ... --source-manifest ... --output-dir ... --engine-port 18780 --metrics-port 39982`。每配置一次完整Engine运行包括双方预热和三轮交替次序，总256真实请求；实际输出必须逐ID匹配R8固定轨迹。完整参数示例及独立分析器见[R9证据README](../evidence/native_performance_20261018/README.md)，在本最终分支把清单改为本页1415源即可；R9原1409清单仍可检查公共实现字节，但新增final配置须用1415清单。

TP候选为`configs/candidates/qwen35_4b_tp2.yaml`（可见2,3）、`configs/candidates/qwen35_27b_tp1.yaml`（可见2）。它们仅改tp_size，未为候选调整内存/调度/输入或门槛。27B TP1本次在配置缓存容量检查拒绝，未形成有效性能结果；不能把该点说成所有TP1配置均不可能，亦不改成OOM或质量向量失败。候选/匹配基线使用独立Engine，不作随机跨进程显著性推断。

这些命令复现有限协议，不代表未知统一课程评测已通过。教学网由用户本人提交；实际操作和材料由AI工具辅助完成，成员责任与个人理解边界见总报告。
