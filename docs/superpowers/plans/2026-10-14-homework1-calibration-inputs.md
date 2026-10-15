# 作业1独立校准：输入清单冻结 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development task-by-task. 步骤使用checkbox；用户已授权当前main工作区，不新建worktree、重置或清理。提交暂缓至Git署名、服务器验证和发布检查完成。

**Goal:** 生成可复现且内容隔离的校准/验证输入清单，在任何新参考或引擎数值采集前冻结32条输入及续接IDs。

**Architecture:** 标准库模块负责确定性内容构造、IDs/身份/哈希校验及exclusive写入；CLI仅在调用时加载远端已有AutoTokenizer，不导入Torch/NPU运行模型。单独测试使用简单字符编码验证真实构造逻辑，不下载依赖；随后服务器用真实固定tokenizer验收。参考采集与数值分析另开依赖该清单的实施计划，本计划不跑阶段A/B模型。

**Tech Stack:** 项目Conda/Python3.11、标准库、既有pytest、服务器Transformers5.12.0/固定模型tokenizer。

## 文件结构

- Create `benchmarks/course_project/calibration_inputs.py`：输入构造与哈希，不导入torch/transformers。
- Create `scripts/prepare_homework1_calibration_inputs.py`：离线tokenizer入口，全部路径显式参数。
- Create `test/course_project/test_calibration_inputs.py`：长度/来源隔离、IDs、哈希、防覆盖。
- Create `test/course_project/test_calibration_inputs_cli.py`：入口必填参数、离线加载、项目路径限制、防覆盖，不修改Task1已审查文件。
- Modify 本计划/task_plan/notes仅记录实际结果，不改生产或旧门槛。

所有新Python文件复制仓库完整MIT头，绝对导入、显式行为参数，不增加其它配置/算法。

## Task 1：标准库输入构造及测试

- [x] 先添加以下测试；新module不存在时失败说明缺实现，不跳过失败。

```python
import hashlib
import json
import pytest
from benchmarks.course_project.calibration_inputs import build_manifest, write_manifest

def encode(text):
    return [ord(char) for char in text]

def test_manifest_matrix_and_group_disjointness():
    data = build_manifest(encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev")
    cases = data["cases"]
    assert len(cases) == 32
    for group in ("calibration", "validation"):
        rows = [row for row in cases if row["group"] == group]
        assert sorted(row["input_length"] for row in rows) == sorted([1,8,127,128,129,512,1024,2048]*2)
        assert all(len(row["input_ids"]) == row["input_length"] and len(row["continuation_ids"]) == 16 for row in rows)
    assert len({tuple(row["input_ids"]) for row in cases}) == 32
    assert {row["source_family"] for row in cases if row["group"] == "calibration"}.isdisjoint(
        {row["source_family"] for row in cases if row["group"] == "validation"})

def test_content_hashes_are_reproducible():
    first = build_manifest(encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev")
    second = build_manifest(encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev")
    assert first == second
    content = {key:value for key,value in first.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert first["manifest_sha256"] == expected

def test_write_refuses_existing_output(tmp_path):
    data = build_manifest(encode=encode, vocab_size=100000, tokenizer_revision="fixed-rev")
    output = tmp_path / "inputs.json"
    write_manifest(data, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(data, output)
    assert output.read_bytes() == original

@pytest.mark.parametrize("encoder", [lambda text:[], lambda text:[-1]*3000, lambda text:[True]*3000, lambda text:[100000]*3000])
def test_invalid_or_short_tokenization_rejected(encoder):
    with pytest.raises(ValueError):
        build_manifest(encode=encoder,vocab_size=100000,tokenizer_revision="fixed-rev")

def test_write_refuses_tampered_manifest(tmp_path):
    data = build_manifest(encode=encode,vocab_size=100000,tokenizer_revision="fixed-rev")
    data["cases"][0]["input_ids"][0] += 1
    with pytest.raises(ValueError):
        write_manifest(data,tmp_path/"inputs.json")
    assert not (tmp_path/"inputs.json").exists()
```

- [x] 在项目Conda用新的项目内basetemp跑RED；缺module导致ModuleNotFoundError，与未实现一致。

```powershell
$taskTemp = Join-Path '.tmp' ('calibration-input-red-' + [guid]::NewGuid().ToString('N'))
../.conda-env/python.exe -m pytest test/course_project/test_calibration_inputs.py -q --basetemp $taskTemp
```

- [x] 最小实现：`build_manifest(*,encode,vocab_size,tokenizer_revision)`、`write_manifest(manifest,output)`，行为参数无默认。实现以下确定性来源，不换seed声称独立，不导入/复用旧diagnostic prompt：

```python
families = {
    "calibration": (
        ("cal_weather_math", "甲校准气象数学记录", lambda i: f"站点{i}: 温度={(i*7)%31}，降水={(i*11)%97}；检查整数和{ i + i*i }。"),
        ("cal_python_bits", "Blue calibration Python bit laboratory", lambda i: f"record_{i}: value={i*13}; mask={i%17}; result=value ^ mask; assert result >= 0"),
    ),
    "validation": (
        ("val_history_counts", "乙验证历史清点档案", lambda i: f"档案{i}: 年份={1500+i%500}，卷数={i%23+1}，合计={(i%23+1)*3}；核对分类。"),
        ("val_sql_botany", "Green validation SQL botanical ledger", lambda i: f"plant_{i}: species=fern_{i%19}; leaves={i%29}; SELECT SUM(leaves) FROM garden_{i%7};"),
    ),
}
```

每个group/length/variant source为对应header+`input_length={length}`+i=1..1024的不同记录。encode后取前length个IDs，source不足则中止；保留完整source_text、source_family及source_text_sha256和实际input_ids_sha256。四个不同首文本用于1token边界，真实tokenizer仍须证明32条IDs不重复，不能默默改输入碰运气。后缀独立固定为对应family的`"\nContinuation: verify the ledger, then state one arithmetic check and a short explanation. 后续检查记录并给出一项计算。"`，encode后取前16IDs，记录continuation_text及16 IDs哈希。

ID必须为非bool整数且0<=ID<vocab_size；vocab_size为正整数、revision非空。长度矩阵固定不开放无关sweep。每条case_id为group/length/variant组合，整份清单含model_id固定Qwen/Qwen3.5-0.8B、revision参数、continuation_steps16、两组cases与canonical JSON SHA。`write_manifest`先重算去manifest_sha256后的内容SHA拒绝篡改，再以UTF8独占模式x写文件；不覆盖，失败不创建空文件。完整内容校验不使用hash相等替代输入构造测试。

- [x] 跑GREEN及course集合回归，新basetemp均项目内；完成spec再quality双审查，未通过不部署。

实际进展：implementer专项23pass、course438pass/13skip；主线程独立专项23pass（1.19s），spec reviewer独立23pass（1.18s）且source-only尾部非法控制拒绝，规格通过。quality review Approve，无Critical/Important，续接列表独立性检查通过；主线程完整回归438pass/13skip（9.56s），83381终态exit0。Task1完成，尚没有真实tokenizer/模型/NPU采集。source-only非法尾部永久回归为非阻塞minor，当前共享校验实现正确，记录后续补证据，不以minor掩盖必要硬件验证。

## Task 2：离线真实tokenizer入口

- [x] 先测试CLI参数必须显式、没有模型forward、现有目录拒绝与错误退出；真实tokenizer验证由服务器做，不用假编码证明真实IDs。
- [x] CLI最小内容（MIT头之后），不在import阶段加载Transformers：

```python
import argparse
import json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))
from benchmarks.course_project.calibration_inputs import build_manifest,write_manifest

def main():
    parser=argparse.ArgumentParser(description="Freeze homework1 calibration and validation native IDs; no model inference")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Output must remain inside the project")
    config=json.loads((args.model/"config.json").read_text(encoding="utf-8"))
    vocab_size=config["text_config"]["vocab_size"]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
    data=build_manifest(encode=lambda text:tokenizer.encode(text,add_special_tokens=False),vocab_size=vocab_size,tokenizer_revision=args.revision)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_manifest(data,args.output)
    print(f"Frozen {len(data['cases'])} inputs; SHA256={data['manifest_sha256']}")

if __name__=="__main__":
    main()
```

最终实现按既有Python风格展开，不添加默认revision、在线下载或模型计算模式。CLI还须拒绝output解析后位于项目外；不验证新文件存在就使用。本地集成测试在新项目临时目录执行，禁止共享系统临时目录。

2026-10-14服务器只读config与safetensors头确认vocab248320、embedding BF16[248320,1024]；合法ID范围取模型config，不取可能包含额外special IDs的len(tokenizer)。该只读证据不是新数值采集。

## Task 3：服务器生成与冻结验收

- [x] 只同步新module/CLI/test，远端同路径已存在则先比SHA，差异不覆盖；保护其它用户与旧数据。四目标fresh不存在，四次SCP均exit0，四源码SHA全部匹配，raw独占mkdir成功。
- [x] 服务器用`source scripts/use_project_env.sh`后跑聚焦测试，再使用固定模型路径/上述revision生成新`results/raw/homework1_calibration_inputs_20261014/inputs.json`。34pass3.44s，Windows跳过项Linux实跑通过，56984exit0。
- [x] 独立读取并验证两组16条、32不同IDs、精确8长度×2、16续接、合法vocab、来源分组隔离、三个逐行hash、全manifest hash；实际tokenizer实现/vocab/merges/文件SHA和版本归档，身份标签不冒称全部权重来源证明。
- [x] 四新增源码byteSHA一致、四raw本机/远端cross-SHA一致；另新项目.tmp目录重新生成byte-exact、68367exit0。旧目录不清理，输入sha已冻结。
- [x] 更新完成步骤与来源，接续[参考合同](2026-10-14-homework1-reference-contracts.md)和[参考采集](2026-10-14-homework1-reference-collection.md)计划：四固定参考组合、按实际mixed dtype投影的权重回转检查、每输入17vector、cached/recomputed、自身可信性与finite、状态记录、阶段A-only guard及失败保存。输入冻结已完成；真实参考采集和新阈值另计，不提前标通过。

## 自检与发布边界

本计划是已批准大规格的第一个独立子任务，不把输入清单存在当作作业正确性通过。模块、入口、服务器核验均有明确接口与测试。所有产物留项目；不commit/push至作者、格式检查、服务器最终快照验收满足。新进程/设备idle若受到影响必须记录，不将metadata导入自动视为无NPU副作用。目标fork结构不重排。
