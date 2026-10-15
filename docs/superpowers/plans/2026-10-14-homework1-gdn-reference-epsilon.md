# 作业1 GDN 参考语义纠正与 epsilon 边界 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。继续授权dirty main，不commit/push/安装依赖。正在进行的参考采集源不可修改；本任务仅两个kernel测试文件，不修改生产计算或容差。

**Goal:** 纠正已确认的clamp参考错误，加入接近epsilon的反例，并按原包络重新验证输出/state。

**Architecture:** 现有CPU递推helper始终使用生产和固定HF共同的sum squares+eps数学定义，不新增独立算法选项。解析式CPU检查先RED→GREEN；随后单chip硬件边界与既有回归串行执行，旧归档保留。

**Tech Stack:** 项目服务器已有Torch/torch_npu/pytest，BF16 QKV、FP32 state、CPU FP64解析式参考。

## Task 1：仅修改测试参考与新增边界

2026-10-14 Task1数值/审查记录：新增RED源SHA14f347f00ed689337a5d7d9b7ce9c3c8b0735140b4daf9c90b9758ceb4670ea3，服务器实际AST CPU helper/test四ratio0/.5/1/2为pass/fail/fail/fail，raw/cpu_red.json保留。最小修正后SHA40de22f7297e34d592ed82db587a8385a3a0e90e8ae6864a5daf58c703285fb7和7ef0c0d3761f01b3ab446ef525a68ba770b0cf442bae11d56c7de4d088f4c6f9，AST CPU GREEN4pass/exit0；SPEC、quality均实际源码/AST/SHA核验Approve。未运行NPU，正式17项是13NPU+4CPU解析式自检，不把17统称NPU计算。新raw/homework1_gdn_epsilon_20261014各独占，参考采集器/production与旧包络不变，硬件复验继续Task2。

Modify `tokenspeed-kernel-npu/test/test_gdn_homework1.py`、`tokenspeed-kernel-npu/test/test_gdn_prefill.py`。先读取完整两文件和`docs/project/homework1_gdn_reference_audit_20261014.md`。原OUTPUT/STATE atol/rtol不改变。已有源无MIT头不做无关重排；新test函数typed并用现有风格。

- [ ] 先在homework1文件新增以下CPU helper边界测试和NPU Prefill test；此时不改旧helper。此段完整代码中的prefill=True只供RED版本，修正时随移除flag一起删除。

```python
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0, 2.0])
def test_prefill_reference_norm_epsilon(ratio: float) -> None:
    amplitude = (ratio * 1e-6 / 128) ** 0.5
    q = torch.full((16,128), amplitude).to(torch.bfloat16)
    k = q.clone()
    v = torch.ones((16,128),dtype=torch.bfloat16)
    output, state = _step(q,k,v,torch.zeros(16,128,128),torch.zeros(16),torch.full((16,),0.5),prefill=True)
    q64, k64 = q.double(), k.double()
    q64 = q64 / (q64.square().sum(-1,keepdim=True)+1e-6).sqrt()
    k64 = k64 / (k64.square().sum(-1,keepdim=True)+1e-6).sqrt()
    expected_state = k64[:,:,None] * (0.5*v.double())[:,None,:]
    expected_output = (q64[:,:,None]*expected_state).sum(1) * 128**-0.5
    torch.testing.assert_close(output.double(), expected_output, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(state.double(), expected_state, atol=1e-7, rtol=1e-5)

@pytest.mark.parametrize("ratio", [0.0,0.5,1.0,2.0])
def test_real_heads_prefill_norm_epsilon(ratio: float, record_property) -> None:
    amplitude = (ratio*1e-6/128)**0.5
    q = torch.full((1,1,16,128),amplitude).to(torch.bfloat16)
    k, v = q.clone(), torch.ones_like(q)
    g, beta = torch.zeros((1,1,16)), torch.full((1,1,16),0.5)
    initial = torch.zeros((1,16,128,128),dtype=torch.float32)
    expected, state = _step(q[0,0],k[0,0],v[0,0],initial[0].transpose(-1,-2),g[0,0],beta[0,0],prefill=True)
    actual = gdn_chunk_prefill(q.npu(),k.npu(),v.npu(),g.npu(),beta.npu(),
        scale=128**-0.5,initial_state=initial.npu(),
        cu_seqlens=torch.tensor([0,1],device='npu',dtype=torch.int32),
        qk_l2norm=True,output_final_state=True,solution='triton')
    _check(actual.out,expected.reshape(1,1,16,128).to(torch.bfloat16),state=False,
           label='epsilon_output',record_property=record_property)
    _check(actual.final_state,state.transpose(-1,-2).unsqueeze(0),state=True,
           label='epsilon_state',record_property=record_property)
```

- [ ] CPU RED可由主线程服务器仅AST提取实际`_step`函数与新CPU test，提供torch/pytest并调用ratios；不导入NPU算子，不变更正式参考采集代码/输入、不在并发参考期间运行硬件。应在ratio0.5/1/2观察旧clamp解析式断言失败，保存新raw的RED证据；零比率通过是有限对照。
- [ ] 修正最小实现：`_step`删除prefill行为参数与分支，统一`q_norm,k_norm=q_norm+1e-6,k_norm+1e-6`；删除调用处prefill=True。旧prefill文件两处表达式分别改成`torch.rsqrt(q_t.square().sum()+1e-6)`及k对应，别动其他计算/容差。
- [ ] 新CPU解析式GREEN并AST语法核验，spec→quality独立审查；同条件随机原测试保持，不新增默认算法、helper模块或生产patch。

## Task 2：串行服务器验收与归档

- [ ] 正式参考采集完成或明确安全停止前不开展NPU测试。之后freshidle/mapping，使用单logical2/physical1chip0，source项目env，不碰NPU0他人任务。
- [ ] 部署前核对远端旧两源SHA，独占备份到项目新raw目录，差异中止；根代理才同步审核后新源码并两端SHA核对，不删除旧记录。
- [ ] 两文件全pytest新basetemp和XML，预期homework1原7+新8、prefill2=17cases；实际计数/skip/error/fail按终态为准，误差properties归档。既有多步双slot/独立读写slot和未选slot精确不变必须实跑，不因新增epsilon结果跳过。
- [ ] 保存protocol/旧源/newSHA/XML/errorproperties/退出0与设备释放证据，两端SHA。更新参考勘误、数值合同、报告来源与覆盖；如果仍失败保持原阈值/失败，不将本任务当全模型验收。全模型旧1%失败不改。

## 自检

修的是已证实数学定义错误，不以观察结果调整允许误差。四norm级别覆盖零/epsilon两侧，BF16实际输入再转FP64参照，状态方向仍[K,V]参考→[V,K]pool。生产源码/正式采集源不改，不并行硬件/性能，参考修正须新原始证据，不抹旧通过或失败。
