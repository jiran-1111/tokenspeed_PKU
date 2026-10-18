# 开源维护验证证据

本目录对应维护分支 `fix/oss-project-20261018`，基线为课程最终提交
`d41b6798`。修复说明和协作边界见[维护记录](../../project/oss_maintenance_20261018.md)。

- [源码清单](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/docs/evidence/oss_maintenance_20261018/source_manifest.json)：维护软件源及安装、测试依赖的实际 SHA-256；
  与原冻结清单的差异明确列出，不修改旧清单。
- [验证记录](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/docs/evidence/oss_maintenance_20261018/verification.json)：本机独立环境、服务器隔离目录的软件检查、
  文档构建、完整格式 hook、源码和远端核查的结果与边界。
- [协作核查](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/docs/evidence/oss_maintenance_20261018/collaboration_audit.json)：远端六分支、共同祖先、路径交集、
  22 个模拟文本冲突、独立实验的接口和数值口径。

课程 CPU 检查从仓库根目录执行；先按 README 建立环境和 `.tmp`，并为
每次验证选择新的临时目录：

```bash
python -m pytest test/course_project -q -ra --basetemp .tmp/pytest-maintenance
pre-commit run --all-files
```

使用工作区内的 Bun 1.3.0，从 `docs/` 执行以下命令验证公开文档站。
不关闭死链接检查、不删除历史实验记录，未发布原始数据使用文字归档路径：

```bash
bun install --frozen-lockfile --ignore-scripts
bun run docs:build
```

源码清单可用 Python 标准库核验，不加载模型或访问设备：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest = root / 'docs/evidence/oss_maintenance_20261018/source_manifest.json'
sources = json.loads(manifest.read_text(encoding='utf-8'))
assert all(
    hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for name, expected in sources.items()
)
print(f'{len(sources)} source files verified')
PY
```

服务器软件验证复制到既有项目内的独立临时目录，使用同一维护源清单；
不覆盖原已验证源码。环境脚本回归通过隔离命令验证流程，不代表在空白
Ascend 主机上实际安装全部运行依赖。服务器 CPU 测试需要正确加载现有
CANN 软件库以导入已安装的 Torch 扩展；没有运行 NPU 模型、性能或 Profiler。
首次测试准备失败的记录保留，修正准备后使用新的测试输出目录。

此处的新软件结果不替代课程最终快照的数值或性能证据。硬件结论继续引用
[课程最终证据](../course_final_20261018/README.md)及其固定源版本。
