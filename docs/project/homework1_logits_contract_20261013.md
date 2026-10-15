# 作业1完整logits对照合同（采集前固定）

目的：补充完整模型在共同输入/前缀处的数值证据，不能只看两个候选或以近并列自动豁免差异。本合同先于本次引擎完整logits采集及误差分析固定；原strict raw/stop-aligned Token门槛保留为总项目独立检查。

第一步固定0.8B revision、BF16/TP1、原baseline计算配置和原生input IDs，选择精确128输入的首token位置127，最后输入ID271。引擎logit计算后、采样前保存完整248320维向量；仅用独立opt-in临时sitecustomize诊断目录，不改生产引擎。参考使用同固定revision、BF16、Transformers NPU torch fallback、相同input IDs，保存首token完整logits；缓存/重算参考一致性单独记录。输出向量存FP32只是无损展开BF16取值，不宣称模型改成FP32推理。

必须核对输入ID、位置、vocab长度、向量hash、finite、返回Token与logits argmax一致。完整向量报告max-abs、relative-L2、均方误差，以及双方top候选、margin。此前已知HF两个候选相差0.125；只凭该margin不能通过。

完整向量的工程包络预定为max-abs≤0.5、relative-L2≤0.01、全部finite；分母下限1e-12。依据BF16在20–30量级单ULP约0.125，绝对包络限定几ULP而非无界差异，relative-L2约束全词表累计差异；这是工程验收包络而不是对24层传播误差的严密上界。该限值不替代关键模块更紧判据，不允许事后放宽。必须同时解释排名差异，并继续检查连续Decode共同前缀/状态证据，不能把一个Prefill向量通过写为完整模型所有输入验收完成。

采集钩子带位置/输入ID guard，仅在arm文件存在时保存，使用独占二进制文件名和SHA-256元数据。采集D2H会同步并改变时延，故不得混入性能基线。旧性能raw/source snapshot保持不变。本次运行启动日志出现base Conda辅助进程加载sitecustomize时torch缺失警告，最终项目Conda进程的钩子有效性仍须通过实际记录核验；不因此安装base torch或污染环境。

后续将该合同扩展到其他前缀时，保持原阈值和原失败数据，注明eager/graph、cache与position边界；若观测无法支持验收，保留未通过状态并继续定位。
