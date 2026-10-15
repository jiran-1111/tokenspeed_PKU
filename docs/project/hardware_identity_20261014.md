# 硬件身份核实与早期结论纠正（2026-10-14）

## 结论与边界

早期把`npu-smi`的`Chip Version V1`直接解释为“第一代Ascend910、非910C”的判断依据不足，应撤回。当前运行时与安装平台配置确认SoC为`Ascend910_9362`、短系列为`Ascend910_93`。根据下列一手软件来源，可识别为Atlas A3/910C软件支持系列；这是本机观测加官方/维护者系列映射的推断，不是服务器采购型号、CPU型号或课程最终验收的证明。

现有实验仍在同一服务器、相同记录的逻辑chip上完成，没有换设备。旧文档中的“Ascend910 V1”保留为当时工具字段简称，不应再理解成已经证实的非910C营销型号。原始实验、失败、哈希、精度门槛、状态检查及性能统计不因命名纠正而变更。

## 新鲜只读设备证据

通过现有SSH密钥代理、BatchMode连接，在项目环境下查询；未输入凭据、安装依赖、运行模型、创建张量或调用`acl.init`。

- `npu-smi info -t board -i 1 -c 0`：NPU Name为9362，Chip Name为Ascend910，Chip Version为V1。此处不归档芯片唯一标识。
- 使用标准库ctypes加载既有`libascendcl.so`，仅调用无参数查询`aclrtGetSocName()`，返回`Ascend910_9362`，进程退出0。
- 读取现有`/opt/miniconda3/Ascend/cann-8.5.0/aarch64-linux/data/platform_config/Ascend910_9362.ini`，选择性字段为：

```text
SoC_version=Ascend910_9362
Short_SoC_version=Ascend910_93
AIC_version=AIC-C-220
CCEC_AIC_version=dav-c220-cube
CCEC_VECTOR_version=dav-c220-vec
```

查询使用项目`.conda-env`，先激活`scripts/use_project_env.sh`；没有修改CANN平台配置或生产源码。本机记录是本轮工具返回结果的摘录，未声称已生成远端原始证据目录或cross-SHA文件包。

## 一手来源与解释

以下页面于本轮实际打开读取，不仅使用搜索摘要；不是依赖个人论坛的硬件猜测。

1. [AscendCL get_soc_name接口](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0009.html)：查询运行环境芯片版本；C接口为`aclrtGetSocName`。本机执行的是既有CANN8.5对应符号，不由新版文档推断本机安装版本。
2. [Ascend官方fbgemm-ascend仓库](https://github.com/Ascend/fbgemm-ascend)：构建说明将A3对应到Ascend910_93系列。仅查资料，没有安装该项目或引入新算子。
3. [vLLM Ascend构建源码](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build_aclnn.sh)：匹配`ascend910_93`分支，维护者注释标识910C/A3系列。这提供软件系列映射，不证明本机整机产品名。
4. [华为官方A3介绍](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)：说明Atlas900 A3 SuperPoD使用910C；这是A3产品关系的辅助依据，不从该页面直接判定本机为Atlas900整机。

网页链接指向本轮看到的在线版本，未来可能变化；目前未建立这些网页的固定commit离线快照。第一轮查询只有aarch64/640逻辑CPU证据，后续取得的整机和CPU标识证据见下一节；不从指令集或数量认定指定CPU子型号。

## 后续只读核实：整机、CPU和逻辑device

本轮读取`/proc/cpuinfo`的实现者/part/架构字段，并只读取DMI的厂商、产品名及非序列号板型字段；没有读取或归档序列号、UUID、网络接口或凭据。

- `/sys/class/dmi/id/sys_vendor`为Huawei，`product_name`为Atlas 800I A3，`board_vendor`为Huawei。DMI直接支持该整机产品名称，不等于完整采购配置清单。
- 当前640条CPU记录均为implementer0x48、part0xd02、architecture8、variant0x0、revision0；640是可见逻辑CPU记录数，不直接当作物理核心或CPU插槽数。
- 当前`lscpu`来自util-linux2.39.1，之前返回model name unknown；未更新系统工具。
- [华为Atlas800I A3产品规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3)说明该型号采用鲲鹏920 CPU。[util-linux维护者当前源码](https://kernel.googlesource.com/pub/scm/utils/util-linux/util-linux/+/master/sys-utils/lscpu-arm.c)将HiSilicon implementer0x48的part0xd02映射到Kunpeng-920，并允许BIOS补充名称。两者结合本机标识支持Kunpeng920系列推断；未读取CPU BIOS完整子型号，不把板型尾缀当CPU SKU。旧lscpu的unknown不足以证明CPU不属于Kunpeng。
- 官方产品页的整机完整规格为8颗NPU及4颗CPU；本容器目前仅可见下表4个NPU ID。不将官方满配规格当作本账号实际拥有的全部资源，剩余资源是否被隔离未查明。

本轮`npu-smi info -m`退出0，直接确认可见映射：

| 可见NPU ID | chip0逻辑ID / 物理ID | chip1逻辑ID / 物理ID |
|---|---|---|
| 0 | 0 / 0 | 1 / 1 |
| 1 | 2 / 2 | 3 / 3 |
| 2 | 4 / 4 | 5 / 5 |
| 3 | 6 / 6 | 7 / 7 |

每个NPU ID另有chip2名为Mcu，无逻辑/物理计算ID，不能计为第三个推理device。此表表示当前驱动/容器可见编号，不保证设备可用或空闲；实验前仍要实时检查。

作业1既有0.8B测试使用TP1和单个逻辑device0，对应NPU ID0的chip0；当前27B TP2已测pair2/3对应NPU ID1的chip0/1。因A3的封装NPU与逻辑device粒度不同，报告明确写TP1/单逻辑chip，不把它冒称用了整个双device封装的计算/128GB内存。教师“单chip”在这一粒度下的正式口径仍需确认，但当前0.8B没有使用TP2，硬件查询不改变实验策略。

历史Profiler host CPU元数据仅标记HiSilicon且Type为空，不提供Kunpeng子型号；DeviceInfo中的控制CPU名称是NPU内部控制核，不能拿它当服务器主CPU。未将Profiler元数据缺失补成猜测的型号。

## 报告和后续影响

- 主要入口改用“Ascend910_9362（Ascend910_93系列，软件映射Atlas A3/910C）”，同时保留工具的V1字段，避免混淆。
- 不再把“V1替代910C需申请”作为已经证实的硬件不符合项。DMI已确认产品名Atlas800I A3，CPU标识与一手来源支持Kunpeng920系列；尚需核实完整配置/CPU子型号，以及教师所称single chip与逻辑device边界。
- 0.8B仍按TP1、单逻辑chip验证；27B仍按TP2对应已测pair运行，硬件命名纠正不授权扩大实验或绕过空闲检查。
- 完整数值验收、当前热点、四模型Agent负载/消融、最终复验与报告仍欠；本次身份核实不意味着项目完成。

web-access技能前置检查发现Edge调试端口，但代理连接等待授权后报告超时；没有操作浏览器标签或登录页面。公开资料通过网页检索/读取取得，未依赖浏览器登录态。检查日志限定在项目`.tmp/cdp-proxy.log`；没有为此下载安装工具或改写技能目录配置。

本轮修改后5份主要文档55个本地Markdown链接无缺失，`git diff --check`退出0，已有LF/CRLF提示保留。仅文档及查询记录变化，本轮未重新执行模型、性能或软件测试；上一轮415通过/13跳过不冒称本次新运行结果。

CPU/整机和映射补充之后，再次检查5份主要文档56个本地链接，均存在，`git diff --check`退出0。网页已实际读取，但本地链接检查不验证外部链接未来可用性；仍未运行新的推理/性能实验或软件测试。
