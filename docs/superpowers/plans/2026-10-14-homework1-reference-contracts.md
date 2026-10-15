# 作业1阶段A：参考权重与连续前缀合同 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development task-by-task。继续用户授权的dirty main；不commit/push/重置，不安装依赖，不并行NPU工作。

**Goal:** 为已批准阶段A采集固定、可测试的权重投影规则与17步同输入窗口，避免混合权重或缓存前沿错误污染校准。

**Architecture:** 标准库模块只定义权重dtype规则和cached/recomputed请求窗口，无Torch、模型加载或数值容差。后续采集器调用此合同并用服务器实际Tensor核对权重回转/指纹，严格只采冻结清单的calibration组。

**Tech Stack:** 项目Python3.11/pytest/标准库；后续服务器已装HF5.12.0、PyTorch2.9、torch_npu2.9。当前不采集新模型向量。

## 前置证据与定义

检查点452 BF16/36 F32，F32为A_log和linear_attn.norm.weight各18。当前生产构造A_log/dt_bias FP32；gated norm config.dtype=BF16；default_weight_loader.copy_按目标dtype写入。Gemma Norm存的是FP32有效gamma=1+checkpoint_weight.float()，不能把这个派生gamma与HF原始checkpoint weight直接比较。

合同对**HF原始learned parameter**定义：A_log/dt_bias checkpoint投影为FP32，其余投影为BF16；FP32主参考将该投影展开FP32，BF16参考保留投影dtype。Gemma有效gamma仍由固定HF数学定义计算，不在参考中预先给raw weight加1。该规则基于当前0.8B实际源码/检查点，不泛化其他模型或未知F32参数。采集前需验证实际加载和预期投影同值，不把政策字符串当参数相等证明。

## Task 1：两个纯合同函数和TDD

验收记录：Task1所有步骤完成。RED为新模块缺失；最终两文件SHA分别0ca972169bab048f82419eb24308ce2eec78a07f6264c6adacb9aa1d15813bcc、c020f0fae12dab408effff66d9d8f9d43830eecddc10ee73d37f5329f3fd04f8。主线程focused44pass0.06s、全course492pass14skip10.48s/exit0，SPEC和quality各独立44pass0.05s。未commit，不修改生产计算。

Files：Create `benchmarks/course_project/reference_contracts.py`；Create `test/course_project/test_reference_contracts.py`。完整MIT头、绝对导入、类型标注、行为参数显式，不改已有输入/生产文件。

- [ ] 写以下失败测试；记录module不存在的RED，不跳过。

```python
import pytest
from benchmarks.course_project.reference_contracts import reference_weight_dtypes, forward_windows

@pytest.mark.parametrize("name", ["model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias"])
@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_decay_parameters_keep_fp32(name, compute_dtype):
    assert reference_weight_dtypes(name=name, compute_dtype=compute_dtype) == ("float32", "float32")

@pytest.mark.parametrize("name", ["model.layers.0.linear_attn.norm.weight", "model.norm.weight", "lm_head.weight"])
def test_projected_weights_expand_only_for_fp32_reference(name):
    assert reference_weight_dtypes(name=name, compute_dtype="float32") == ("bfloat16", "float32")
    assert reference_weight_dtypes(name=name, compute_dtype="bfloat16") == ("bfloat16", "bfloat16")

@pytest.mark.parametrize("length", [1,8,127,128,129,512,1024,2048])
def test_same_prefix_frontier_for_all_seventeen_vectors(length):
    prompt=list(range(length)); suffix=list(range(20000,20016))
    rows=forward_windows(input_ids=prompt,continuation_ids=suffix)
    assert len(rows)==17
    for step,row in enumerate(rows):
        assert row["step"]==step and row["prefix_length"]==length+step
        assert row["recomputed_input_ids"]==prompt+suffix[:step]
        assert row["recomputed_position_ids"]==list(range(length+step))
        assert row["cached_input_ids"]==(prompt if step==0 else [suffix[step-1]])
        assert row["cached_position_ids"]==(list(range(length)) if step==0 else [length+step-1])
    assert prompt==list(range(length)) and suffix==list(range(20000,20016))

@pytest.mark.parametrize("bad", [[], [-1], [True], [1.5]])
def test_invalid_native_input_fails(bad):
    with pytest.raises(ValueError):
        forward_windows(input_ids=bad,continuation_ids=list(range(16)))

@pytest.mark.parametrize("bad", [[],list(range(15)),list(range(17)),[True]*16,[-1]*16])
def test_invalid_continuation_fails(bad):
    with pytest.raises(ValueError):
        forward_windows(input_ids=[3],continuation_ids=bad)

def test_windows_do_not_alias_mutable_lists():
    prompt=[3,4];suffix=list(range(16));rows=forward_windows(input_ids=prompt,continuation_ids=suffix)
    rows[0]["cached_input_ids"][0]=999
    assert rows[0]["recomputed_input_ids"]==[3,4] and prompt==[3,4]
    rows[1]["recomputed_input_ids"][0]=888
    assert rows[2]["recomputed_input_ids"][0]==3

@pytest.mark.parametrize("name,dtype", [("","float32"),("  ","float32"),(None,"float32"),("x","float16"),("x",None)])
def test_invalid_weight_policy_selection_fails(name,dtype):
    with pytest.raises(ValueError):
        reference_weight_dtypes(name=name,compute_dtype=dtype)
```

- [ ] 运行RED：使用`../.conda-env/python.exe -m pytest test/course_project/test_reference_contracts.py -q --basetemp .tmp/reference-contract-red-<unique_guid>`，基目录必须新建项目内标识；不得写系统临时目录。
- [ ] 最小实现以下函数，不增加默认模式、泛型registry或模型参数修改；名称边界须按末字段判断，不因名字中包含A_log误判普通weight。

```python
def reference_weight_dtypes(*, name: str, compute_dtype: str) -> tuple[str,str]:
    if not isinstance(name,str) or not name.strip():
        raise ValueError("Parameter name must be nonempty")
    if compute_dtype not in ("float32","bfloat16"):
        raise ValueError("Reference dtype must be explicit float32 or bfloat16")
    projected="float32" if name.rsplit(".",1)[-1] in ("A_log","dt_bias") else "bfloat16"
    return projected, "float32" if compute_dtype=="float32" else projected

def forward_windows(*, input_ids: list[int], continuation_ids: list[int]) -> list[dict]:
    if not input_ids or len(continuation_ids)!=16:
        raise ValueError("Native prompt required with exactly16 continuation IDs")
    if any(type(i)is not int or i<0 for i in input_ids+continuation_ids):
        raise ValueError("IDs must be non-bool nonnegative integers")
    rows=[]
    for step in range(17):
        prefix=input_ids+continuation_ids[:step]
        cached=input_ids.copy() if step==0 else [continuation_ids[step-1]]
        positions=list(range(len(input_ids))) if step==0 else [len(prefix)-1]
        rows.append({"step":step,"prefix_length":len(prefix),"cached_input_ids":cached,
                     "cached_position_ids":positions,"recomputed_input_ids":prefix,
                     "recomputed_position_ids":list(range(len(prefix)))})
    return rows
```

整理成当前项目风格，加文档说明tuple第一项是checkpoint投影，**不是全部engine runtime parameter storage dtype**；特别Gemma派生gammaFP32不能预加给HFraw weight。ID词表上界由冻结清单/采集器核对，此窗口模块不新增任意vocab默认。

- [ ] 增补普通参数名`x.A_log.weight`不误判、tuple投影/计算dtype边界、输入/续接列表独立性的必要回归，拒绝无关功能。
- [ ] 跑GREEN聚焦+全course（唯一项目.tmp），spec再quality审查，记录确切通过/skip及SHA；不commit。

## Task 2：接续服务器权重实值及采集计划（不提前采集）

已完成合同部署、服务器测试和下一采集计划。Linux44pass0.06s/exit0，无skip/failure/error；独占raw/homework1_reference_contracts_20261014/tests.xml两端SHA d17dbea8ee159e4463db0733426b1540117a7d9330199423a027f8de6b204d48。两源部署前fresh不存在，服务器源码SHA与上述一致；未加载模型。下一计划[独立参考采集](2026-10-14-homework1-reference-collection.md)已保存，Task1采集器实现正在执行；真实权重回转/向量采集仍未执行，不把这些条目标通过。

- [ ] 把合同两新文件server freshguard部署、SHA一致，再CPU聚焦测试；该工具不导入Torch/NPU，不改变输入清单。
- [ ] 下一子计划固定采集器四个reference选择cpu-fp32-eager/cpu-fp32-sdpa/cpu-bf16-sdpa/npu-bf16-sdpa；路径/revision/inputmanifest/output/CPU线程均显式；NPU只在fresh idle/mapping检查后触碰指定device；不采Profiler或validation组引擎。
- [ ] 下一采集器先HF CPU FP32加载原检查点，按合同投影、再转参考paramdtype；核对每个learnedparameter投影回转相等、canonical FP32值指纹跨reference一致，记录config/weight shards/HF源码与合同源码哈希及缺失权重信息。Gemma数学gamma交由原模型计算，不预加1。
- [ ] 每条校准输入严格17步：cached前沿使用已生成窗口，recomputed每步从完整prefix零初态重建新cache；两路径均use_cache=True以获得可比较final状态，position显式、mask对应完整prefix、logits_to_keep1。每步记录完整logits/finite/排名/margin与18层conv/recurrent状态差异，保存最终状态原始张量；不能拿仅state norm当逐元素误差。
- [ ] 原始输出/失败/进程/设备/源码终态齐全再校准分析；无参考可信性依据则无法确定判据，不看旧engine失败反推门槛。独立验证需候选判据另确认。输入contract完成不能标作业完成。

## 自检

本计划实现第一个独立subtask，完整代码和测试已给出，先保证参数/前沿的纯合同，再写依赖实际HF接口的采集子计划。混合checkpoint与engine派生gamma边界明确。整体目标仍包括GDN参考纠正与边界、模型端到端、Prefill/Decode热点、报告和服务器最终发布；不以本子任务代替这些要求。
