# R9r2四模型Session独立性能证据

主结果为四模型各192正式/64预热：0.8B、4B为BF16 TP1基线；9B为FP32输出头TP1候选；27B为BF16 TP2基线。固定2会话×16轮×8输出，2秒脚本工具等待，逐次实际生成且逐ID匹配R8参考。协议与轨迹哈希均固定，无性能选优重跑。

- `verification.json`：源码/协议/运行哈希、软件检查计数、每模型原始归档SHA与有限范围。
- `source_manifest.json`：服务器固定1409项源码清单；清单文件SHA为`2c9e47616eb3c460a740e2542e776ab655934a9ad7ed0ac0c1401a008a97d712`。
- 每模型`audit.json`与`requests.csv`：全部预热/正式臂、逐请求输入规模/输出检查、客户端时间、采样覆盖、资源与关闭检查。
- 每模型`raw_manifest.json`：完整原始run、查询表、阶段与分析文件逐项SHA。完整私有原始归档保存在工作区，不公开服务器地址、帐号、Session不透明ID等。
- `paired_repetitions.csv`：12组配对结果，不把不同模型或同一Engine内重复当独立随机样本。
- `review_receipt.json`：两项失败路径问题、三项回归及复核范围；`history_r1/`仅保留修复前3模型历史结果，未计入主结果。
- `figures/`：PNG、可编辑SVG、输入哈希/图注；`runners/`为独立复算和绘图程序。

## 在规定Ascend主机复现

在仓库根目录使用项目本地环境，按环境文档准备固定revision模型与模型路径，先完成所有软件检查。必须每次检查设备映射及所需chip空闲，只串行运行一个实验。程序会再次检查映射、空闲和端口，不抢占其他任务。以下端口只用于单个实验，下一模型须等待当前完全退出。

```bash
source scripts/use_project_env.sh
pre-commit run --all-files
python -m pytest test/course_project -q
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_native_agent_performance.py \
  --config configs/baseline/qwen35_4b.yaml \
  --protocol configs/validation/native_agent_performance_20261018.json \
  --source-manifest docs/evidence/native_performance_20261018/source_manifest.json \
  --output-dir results/raw/reproduce_native_perf_4b \
  --engine-port 18780 --metrics-port 39982
```

输出目录必须全新。0.8B改为`configs/baseline/qwen35_0.8b.yaml`；9B用`configs/candidates/qwen35_9b_fp32_head.yaml`；27B用`configs/baseline/qwen35_27b.yaml`且可见设备为`2,3`，需两chip均空闲。每次使用不同输出目录。运行中不得修改源码、协议或轨迹。成功需同时满足run状态、所有256条实际输出/上下文、采样无错误、关闭active=0及退出后目标chip空闲；仅进程退出0不够。失败保留原目录与日志，不覆盖或自动重试。

## 独立复算与绘图

`analyze.py`仅用Python标准库，读取原始run及对应R8轨迹，重新检查配对、历史、实际输出、时间、工具与原始HBM表，输出全部请求CSV与审计。若审计`protocol_and_quality_passed`为false，不能作为通过结果使用。以下输入须为原始JSON，不能用audit代替run。

```bash
python docs/evidence/native_performance_20261018/runners/analyze.py \
  results/raw/reproduce_native_perf_4b/run.json \
  docs/evidence/native_afterrun_20261017/4b/native_trace.jsonl \
  results/analysis/reproduce_native_perf/4b
```

对其余模型同样复算。绘图需要项目本地环境的Matplotlib；正式图由Matplotlib 3.11.2、NumPy 2.4.6绘制，无全局依赖安装。

```bash
python docs/evidence/native_performance_20261018/runners/plot.py \
  --input-root docs/evidence/native_performance_20261018 \
  --protocol configs/validation/native_agent_performance_20261018.json \
  --source-manifest-sha256 2c9e47616eb3c460a740e2542e776ab655934a9ad7ed0ac0c1401a008a97d712 \
  --output-dir results/figures/reproduce_native_perf
```

输出目录必须不存在。图中每轮中位线和最小/最大带来自2会话×3重复，带不是置信区间。TTFT/E2E/近似TPOT均为客户端观测；吞吐wall含工具等待；HBM为间隔查询观察值，非瞬时峰值或仅KV。完整解释见[实验报告](../../project/native_agent_performance_20261018.md)。HW1原冻结数值门槛及旧1%失败、greedy质量、模块与Session验证分别保留，不互相替代。
