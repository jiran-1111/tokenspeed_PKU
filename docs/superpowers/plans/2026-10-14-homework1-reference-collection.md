# 作业1阶段A：独立参考向量采集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development逐项实施，用户已选此方式并授权dirty main。不commit/push/安装依赖/改变生产模型；不再重复询问是否继续。

**Goal:** 对冻结校准组16条输入，采集四种独立参考的同前沿17步完整logits和GDN状态，保留失败与身份，为后续判据评审提供证据。

**Architecture:** 一个离线脚本复用已测试reference_contracts，不另建框架。运行一次只加载一个参考；每条缓存路径持续递推、重算路径每步零初态。标准库输入/路径防护先于Torch导入；数值与状态保存使用服务器已有Torch/HF。判据分析另一步，不访问验证组引擎结果。

**Tech Stack:** Python3.11/pytest；服务器已有Torch2.9/torch_npu2.9、Transformers5.12.0/safetensors/numpy；本机标准库防护测试不需要安装Torch。

## Task 1：采集入口、TDD与双审查

Task1验收：已实施，TextConfig接口及HF5.12 loading set实际兼容修复均有RED→GREEN；最终scriptSHA b08a6dae106b5fa17f4031ce78a3229b8191c5e0854938987e7c4fef98adca80、testSHA c9b57bb741be028c883c3ba10d86e1e4d6a1c3ac2793ab64f1b38c956a47d223。主线程32pass4skip（本机Torch3项/Windows symlink1项），SPEC复审/quality同通过；服务器官方目录同SHA，全部合同+采集tests80pass10.45s、零skip/fail/error/exit0。仅工具验收，不是参考数值可信性通过。

Create `scripts/collect_homework1_references.py` 和 `test/course_project/test_homework1_reference_collection.py`。不修改其他源码。MIT完整头、绝对导入、类型标注、公开函数文档。以下代码给出完整计算流程；整理风格和必要日志不改变协议。

- [ ] 先写真实脚本导入测试（脚本未存在时RED），测试下列纯函数，不mock其实现：

```python
def test_frozen_input_identity():
    manifest = build_manifest(encode=lambda text: list(text.encode('utf-8')), vocab_size=256, tokenizer_revision='fixed')
    cases = calibration_cases(manifest, revision='fixed', expected_sha=manifest['manifest_sha256'], vocab_size=256)
    assert len(cases) == 16
    assert {c['group'] for c in cases} == {'calibration'}
    assert {c['input_length'] for c in cases} == {1,8,127,128,129,512,1024,2048}
    manifest['cases'][0]['input_ids'][0] += 1
    with pytest.raises(ValueError):
        calibration_cases(manifest, revision='fixed', expected_sha=manifest['manifest_sha256'], vocab_size=256)
```

实际冻结builder使用四源首ID可能重复，测试应使用现有test_calibration_inputs中的确定性不同头编码fixture，不能为了让测试过而改生产builder。增加完整断言：错误revision/SHA、非法词表ID（重新计算清单哈希以单测ID守卫）、组数/长度错误、后缀非16、duplicate输入、existing输出及symlink逃逸/项目外路径；badguard发生在任何Torch/Transformers导入之前。CLI四reference枚举、模型/revision/manifest/manifest-sha/output/threads均required，NPU选择时npu-device必须显式，CPU禁止误带device。测试所有required和CPU/NPU行为分支，不运行真实模型或读取验证组数值。

- [ ] RED命令 `../.conda-env/python.exe -m pytest test/course_project/test_homework1_reference_collection.py -q --basetemp .tmp/reference-collection-red-<new-id>`。
- [ ] 最小实现纯函数与CLI；校验选组前检查整个冻结清单哈希/身份，验证组仅输入完整性，不采向量。完整实现骨架如下：

```python
def calibration_cases(manifest: dict, *, revision: str, expected_sha: str, vocab_size: int) -> list[dict]:
    content = {k:v for k,v in manifest.items() if k != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if digest != expected_sha or manifest['manifest_sha256'] != digest:
        raise ValueError('Frozen manifest hash mismatch')
    if manifest['tokenizer_revision'] != revision or manifest['model_id'] != 'Qwen/Qwen3.5-0.8B':
        raise ValueError('Frozen model/tokenizer identity mismatch')
    if manifest['continuation_steps'] != 16 or len(manifest['cases']) != 32:
        raise ValueError('Expected frozen two-group 32-case manifest')
    seen = set()
    for case in manifest['cases']:
        forward_windows(input_ids=case['input_ids'], continuation_ids=case['continuation_ids'])
        if len(case['input_ids']) != case['input_length']:
            raise ValueError('Recorded length mismatch')
        if any(i >= vocab_size for i in case['input_ids']+case['continuation_ids']):
            raise ValueError('IDs exceed model vocabulary')
        ids = tuple(case['input_ids'])
        if ids in seen:
            raise ValueError('Duplicate frozen input')
        seen.add(ids)
    for group in ('calibration','validation'):
        lengths = sorted(c['input_length'] for c in manifest['cases'] if c['group']==group)
        if lengths != sorted([1,8,127,128,129,512,1024,2048]*2):
            raise ValueError('Frozen group/length coverage mismatch')
    return [c for c in manifest['cases'] if c['group']=='calibration']

def new_output_path(output: Path) -> Path:
    resolved = output.resolve()
    if not resolved.is_relative_to(REPO_ROOT) or resolved == REPO_ROOT:
        raise ValueError('Output must be a new directory inside repository')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output already exists')
    return resolved

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

def compare_tensor(actual, reference, *, torch) -> dict:
    a = actual.detach().float().cpu()
    b = reference.detach().float().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite or mismatched reference tensor')
    delta = a.double()-b.double()
    norm = torch.linalg.vector_norm(b.double()).item()
    return {'shape':list(a.shape), 'actual_dtype':str(actual.dtype), 'reference_dtype':str(reference.dtype),
            'max_abs':delta.abs().max().item(), 'relative_l2':torch.linalg.vector_norm(delta).item()/max(norm,1e-12),
            'reference_norm':norm, 'finite':True}

def collect(args, cases: list[dict], *, output: Path) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    torch.set_num_threads(args.threads)
    # Force fixed HF torch fallback before layer construction, process only.
    hf.causal_conv1d_fn = None
    hf.causal_conv1d_update = None
    hf.chunk_gated_delta_rule = None
    hf.fused_recurrent_gated_delta_rule = None
    hf.FusedRMSNormGated = None
    hf.is_fast_path_available = False
    device = 'cpu'
    if args.reference == 'npu-bf16-sdpa':
        import torch_npu
        torch.npu.set_device(args.npu_device)
        device = f'npu:{args.npu_device}'
    compute = 'float32' if 'fp32' in args.reference else 'bfloat16'
    attention = 'eager' if args.reference.endswith('eager') else 'sdpa'
    model, loading = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        local_files_only=True, dtype=torch.float32, attn_implementation=attention, output_loading_info=True)
    if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs']:
        raise ValueError('Incomplete checkpoint loading')
    weight_digest = hashlib.sha256()
    parameters = []
    for name, parameter in sorted(model.named_parameters()):
        projection, actual = reference_weight_dtypes(name=name, compute_dtype=compute)
        projected = parameter.detach().to({'float32':torch.float32,'bfloat16':torch.bfloat16}[projection])
        parameter.data = projected.to({'float32':torch.float32,'bfloat16':torch.bfloat16}[actual])
        if not torch.equal(parameter.detach().to(projected.dtype), projected):
            raise ValueError('Weight projection roundtrip mismatch')
        values = parameter.detach().float().contiguous().numpy().tobytes()
        digest = hashlib.sha256(values).hexdigest()
        weight_digest.update(json.dumps([name,list(parameter.shape),digest],separators=(',',':')).encode())
        parameters.append({'name':name,'shape':list(parameter.shape),'projection_dtype':projection,
                           'compute_dtype':actual,'canonical_fp32_sha256':digest})
    checkpoint_index = json.loads((args.model/'model.safetensors.index.json').read_text())
    selected_files = [args.model/'config.json',args.model/'model.safetensors.index.json']
    selected_files += [args.model/name for name in sorted(set(checkpoint_index['weight_map'].values()))]
    selected_files += [Path(hf.__file__),Path(transformers.cache_utils.__file__),Path(__file__),
                       REPO_ROOT/'benchmarks/course_project/reference_contracts.py']
    hashes = {str(p):sha_file(p) for p in selected_files}
    write_json(output/'identity.json', {'reference':args.reference,'revision':args.revision,
        'manifest_sha256':args.manifest_sha,'weight_value_sha256':weight_digest.hexdigest(),
        'parameters':parameters,'loading':loading,'torch':torch.__version__,'transformers':transformers.__version__,
        'device':device,'threads':args.threads,'attention':attention,'gdn':'fixed HF torch fallback','source_sha256':hashes})
    model.to(device).eval()
    for number, case in enumerate(cases):
        cached = None
        logits = {'cached':[], 'recomputed':[]}
        observations = []
        final_states = {}
        with torch.inference_mode():
            for window in forward_windows(input_ids=case['input_ids'],continuation_ids=case['continuation_ids']):
                results = {}
                for path in ('cached','recomputed'):
                    results[path] = model(input_ids=torch.tensor([window[f'{path}_input_ids']],dtype=torch.long,device=device),
                        position_ids=torch.tensor([window[f'{path}_position_ids']],dtype=torch.long,device=device),
                        attention_mask=torch.ones((1,window['prefix_length']),dtype=torch.long,device=device),
                        past_key_values=cached if path=='cached' else None, use_cache=True, logits_to_keep=1)
                    logits[path].append(results[path].logits[0,-1].detach().float().cpu().clone())
                cached = results['cached'].past_key_values
                rebuilt = results['recomputed'].past_key_values
                row = {'step':window['step'],'prefix_length':window['prefix_length'],
                       'logits':compare_tensor(logits['cached'][-1],logits['recomputed'][-1],torch=torch),'states':[]}
                for path in ('cached','recomputed'):
                    values, ids = logits[path][-1].topk(5)
                    row[path] = {'top5_ids':ids.tolist(),'top5_values':values.tolist(),'margin':(values[0]-values[1]).item()}
                for layer_idx,(left,right) in enumerate(zip(cached.layers,rebuilt.layers,strict=True)):
                    if isinstance(left,LinearAttentionCacheLayerMixin):
                        if not isinstance(right,LinearAttentionCacheLayerMixin):
                            raise ValueError('Cache layer type mismatch')
                        for field in ('conv_states','recurrent_states'):
                            a = left.conv_states if field=='conv_states' else left.recurrent_states
                            b = right.conv_states if field=='conv_states' else right.recurrent_states
                            row['states'].append({'layer':layer_idx,'field':field,**compare_tensor(a,b,torch=torch)})
                            if window['step']==16:
                                final_states[f'cached/{layer_idx}/{field}'] = a.detach().cpu().clone()
                                final_states[f'recomputed/{layer_idx}/{field}'] = b.detach().cpu().clone()
                if len(row['states']) != 36:
                    raise ValueError('Expected 18 GDN layers with two states each')
                observations.append(row)
                del results, rebuilt
        name = f'case_{number:02d}'
        torch.save({'case_id':case['case_id'],'input_ids':case['input_ids'],'continuation_ids':case['continuation_ids'],
                    'cached_logits':torch.stack(logits['cached']),'recomputed_logits':torch.stack(logits['recomputed']),
                    'final_gdn_states':final_states},output/f'{name}.pt')
        write_json(output/f'{name}.json',{'case_id':case['case_id'],'observations':observations})
        print(f'{args.reference}: {number+1}/16 completed',flush=True)
        del cached, final_states, logits
    if any(sha_file(Path(p))!=digest for p,digest in hashes.items()):
        raise ValueError('Selected source changed during collection')
    write_json(output/'result.json',{'status':'collected','case_count':16,'vectors_per_path':272,
        'reference_numerical_trust':'pending assessment','engine_validation':'not executed'})
```

CLI `main`: required explicit args above, positive threads; revision nonempty; CPU/NPU flags strict; read model config vocab248320 and frozen manifest, call new_output_path + calibration_cases BEFORE imports/output creation; use output.mkdir(parents=True,exist_ok=False), write protocol with reference/args/input-file-SHA and status collecting. try collect; except Exception save new failed.json with exception type/message/completed JSON/PT counts and re-raise (no false result); never dump full env/credentials. Include sys.path root setup used by existing CLI. Unexpected checkpoint keys record rather than silentlydiscard; only vision/known multimodal exclusion expected, investigate actual loading warning if text keyunexpected. Logit vector length must match model vocab, cache prefix length must match window, states rank/shape check against actual cached/rebuilt contracts. These guards must be implemented/tests included, not just documented. NPU post-transfer parameter dtype/value check must preserve mixed A_log/dt_bias; identity records actual model class/config attention and dtype, not just requested label. Preserve all completed/failed files, no overwrites.

- [ ] GREEN focused + fullcourse；server完整Torch tests补compare_tensor finite/shape/relative-L2分母/values、FP32/BF16投影回转identity、tiny deterministic fake model cached/recomputed17步/新增prefix position/mask/cache传递及两字段状态逐元素/36条/最后原始保存/异常失败保留。本机缺Torch的数值tests明确skip不能算server通过；主控制guard测试不skip。不得为测试增加通用framework/default模式。
- [ ] SPEC→quality独立审查无阻塞再部署；不commit。

## Task 2：服务器采集与证据核验

执行中：首CPU FP32/eager新独占目录已开始，own PID1011756、根线程句柄11915。实际HF模型加载missing/mismatch/unexpected/error皆空，320parameters投影/移动后身份核对成功，weight值总SHA c2084362ae79d1d0f3d61a9781598296fae0944efe4e6fd04888a48692c57884。16threads/cpu/eager实际记录。输入/源码未变；尚须16case终态/其他3reference/四值SHA一致、全部有限完整与跨端同步。旧RED/GREEN项目.tmp包保留，不上传大型raw。

- [ ] 合同/采集两源和两tests目标freshguard，不存在才SCP；存在须核对同SHA，冲突中止不覆盖。服务器项目环境运行对应全部tests，XML新路径，主线程核对0failed/errors/数值tests无skip。
- [ ] 固定manifest f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936、revision2fc06364715b967f1860aea9cf38778875588b17。新独占results/raw/homework1_reference_calibration_20261014下各reference独占目录。CPU线程16。2026-10-14执行调度修订：只读确认cgroup CPU quota160核、cpuset0–639、内存限1TB；三CPU参考允许不同进程/独占目录并行，48threads不超配额，数学协议/参考组合/输入/精度/门槛不变，无性能测量混入。每路径独立核验终态、身份和完整性，失败不忽略；三CPU全部结束后才单独开展NPU参考，不并行NPU/性能。原“逐个CPU退出才继续”是内部执行选择，现优化排程，不解释为数值标准改变。总16×17×2×4=2176完整向量、64对case文件；保存72finalstate tensors/case。
- [ ] NPU运行前npu-smi info确认physical1无任何进程，mapping核验logical2=physical1chip0；process-only ASCEND_RT_VISIBLE_DEVICES=2后使用npu-device0。每次guard未满足即停止，不能抢占NPU0。CPU参考不导入torch_npu/不设置NPUdevice；NPU不得与性能或其他参考并行。原1%和Greedy失败不覆盖。
- [ ] 每个执行单独capturestdout/stderr/exit/livePID，新进程命令/环境只列已知非敏感字段。SSH断开只探测own精确PID及产物、不自动重复。正常结束核对16JSON/PT、272cached/recomputed、有限状态、所选源码SHA不变、manifest未变、四weight_value_sha一致；NPU退出后新鲜无自有设备进程。失败保留原目录不续写success。
- [ ] 同步identity/protocol/result/逐步JSON、完整raw张量（带项目内路径、合理传输速率），两端SHA一致；生成阶段A观测CSV/报告另task，不宣称数值可信性通过或新门槛已冻结。如四reference权重identity不同，停止归因/判据形成，定位而不是放宽。

## 自检

已覆盖批准规格的身份/投影、校准隔离、17步同前沿、完整logits/状态、finite、停止/失败保留、无Profiler/无engine新阈值、服务器idle单chip与对应SHA。Task1给出计算流程及具体测试要求；采集实际接口须按已安装HF源码核对，不能用网络新版。阶段A采集不是作业完成；阶段B需要候选判据另确认。后续仍补关键模块边界/性能解释/最终报告与待发布快照复验。
