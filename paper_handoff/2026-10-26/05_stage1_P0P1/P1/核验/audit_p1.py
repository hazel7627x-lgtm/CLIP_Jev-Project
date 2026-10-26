from pathlib import Path
import json, hashlib, zipfile, importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/'p1_received'
P0=HERE.parent/'paper_closeout_20261014'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def close(a,b,tol=1e-10): assert np.allclose(a,b,atol=tol,rtol=0),(a,b)
def metrics(y,p):
    p=np.array(p,dtype=float); y=np.array(y,dtype=int)
    assert p.shape==(len(y),3) and np.isfinite(p).all() and (p>=0).all()
    close(p.sum(1),1,1e-6)
    pred=p.argmax(1); cm=np.zeros((3,3),dtype=int)
    for a,b in zip(y,pred): cm[a,b]+=1
    f=[]
    for k in range(3): f.append(2*cm[k,k]/max(int(cm[k,:].sum()+cm[:,k].sum()),1))
    confidence=p.max(1); correct=(pred==y); bins=np.minimum((10*confidence).astype(int),9)
    ece=0
    for k in range(10):
        mask=bins==k
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return dict(n=len(y),accuracy=float(correct.mean()),macro_f1=float(np.mean(f)),f1_E=f[0],f1_N=f[1],f1_C=f[2],
        nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12,1)).mean()),ece10=float(ece),confusion_matrix=cm.tolist())
def softmax(z):
    z=np.array(z,dtype=float); z-=z.max(-1,keepdims=True); ex=np.exp(z); return ex/ex.sum(-1,keepdims=True)

delivery=read(R/'DELIVERY_HASHES.json')
for rel,h in delivery.items(): assert sha(R/rel)==h,rel
assert set(delivery)=={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_HASHES.json'}
package=read(R/'PACKAGE_HASHES.json')
for rel,h in package.items(): assert sha(R/rel)==h,rel
assert read(R/'P1_STATUS.json')['complete'] is True
repair=R/'repair_records/processor_json_comparison_v1'
before=(repair/'p1_runtime_before.py').read_text(encoding='utf-8')
after=(R/'p1_runtime.py').read_text(encoding='utf-8')
assert after==before.replace("require(model.processor.image_processor.to_dict()==ref['processor'],'Teacher图像预处理变化')",
    "require(json.loads(json.dumps(model.processor.image_processor.to_dict()))==ref['processor'],'Teacher图像预处理变化')")
protocol=read(R/'p1_protocol.json'); ref=read(R/'reference/teacher_run.json'); actual=read(R/'teacher_predictions/run.json')
for key in ref:
    if key not in ['manifest_sha256','image_audit_sha256','suffix_example']: assert actual[key]==ref[key],key
for key,file in [('manifest_sha256','manifests/new_dev_holdout.jsonl'),('image_audit_sha256','manifests/image_audit.jsonl')]:assert actual[key]==sha(R/file)
for key in ['study.py','reliable_io.py']:assert sha(R/key)==actual['local_code_sha256'][key]
assert sha(R/'calibration.json')==protocol['calibration_sha256']
assert sha(R/'reference/student_KD_raw_20261010.pt')==protocol['student_benchmark']['checkpoint_sha256']
manifest=rows(R/'manifests/new_dev_holdout.jsonl');ids=[r['sample_id'] for r in manifest]; y=[r['label'] for r in manifest]
assert len(ids)==len(set(ids))==len(set(r['image_id'] for r in manifest))==200
source_manifest=P0/'received_features/evaluation_inputs/new_dev_holdout.jsonl'
assert sha(source_manifest)==sha(R/'manifests/new_dev_holdout.jsonl')
assert actual['suffix_example']==ref['suffix_example'].replace(rows(R/'reference/val_select.jsonl')[0]['hypothesis'],manifest[0]['hypothesis'],1)
image_audit={r['sample_id']:r for r in rows(R/'manifests/image_audit.jsonl')}
table=read(R/'reports/teacher_student_200.json')['rows']; newtable=[]; expected={}
maximum_metric_difference=0.
for name in ['base','jev']:
    records=rows(R/f'teacher_predictions/{name}.jsonl');byid={r['sample_id']:r for r in records}
    assert len(records)==len(byid)==200 and set(byid)==set(ids)
    records=[byid[k] for k in ids]
    for row,source in zip(records,manifest):
        for key in source: assert row[key]==source[key]
        assert row['image_sha256']==image_audit[row['sample_id']]['file_sha256']
    logits=np.array([r['logits'] for r in records]); raw=softmax(logits)
    close(raw,[r['probabilities'] for r in records],1e-12)
    assert [r['prediction'] for r in records]==[['E','N','C'][k] for k in raw.argmax(1)]
    expected[name]={r['sample_id']:r['probabilities'] for r in records}
    par=protocol['calibration_params'][name];cal=softmax((logits+np.array([par['b'],0,0]))/par['T'])
    for variant,p in [('raw',raw),('calibrated',cal)]:
        m=metrics(y,p); target=next(r for r in table if r['kind']=='Teacher' and r['model']==name and r['variant']==variant)
        for k,v in m.items():close(v,target[k]);maximum_metric_difference=max(maximum_metric_difference,float(np.max(np.abs(np.array(v)-target[k]))))
        newtable.append(dict(model=name,variant=variant,**m))
perseed=read(R/'reports/student_200_per_seed.json'); student_metrics=[]
for target in perseed:
    records=rows(R/f"reference/student_predictions/{target['method']}_{target['seed']}.jsonl")
    byid={r['sample_id']:r for r in records};assert len(records)==len(byid)==200 and set(byid)==set(ids)
    records=[byid[k] for k in ids];assert [r['gold'] for r in records]==y
    m=metrics(y,[r['probabilities'] for r in records])
    assert [r['prediction'] for r in records]==[int(np.argmax(r['probabilities'])) for r in records]
    for k,v in m.items():close(v,target[k]);maximum_metric_difference=max(maximum_metric_difference,float(np.max(np.abs(np.array(v)-target[k]))))
    student_metrics.append(dict(method=target['method'],seed=target['seed'],**m))
    if target['method']=='KD_raw' and target['seed']==20261010:expected['student']={r['sample_id']:r['probabilities'] for r in records}
assert len(student_metrics)==21
for target in table:
    if target['kind']!='Student': continue
    rr=[r for r in student_metrics if r['method']==target['model']];assert len(rr)==3
    for k in ['accuracy','macro_f1','f1_E','f1_N','f1_C','nll','ece10']:
        close(np.mean([r[k] for r in rr]),target[k]);close(np.std([r[k] for r in rr],ddof=1),target[k+'_std'])
    old=next(r for r in read(P0/'最终评价重放/summary.json') if r['split']=='new_dev_holdout' and r['method']==target['model'])
    for k in ['accuracy','macro_f1','f1_E','f1_N','f1_C','nll','ece10']:close(target[k],old[k+'_mean'],1e-6)

selection=read(R/'reports/cost_round_selected.json');rd=R/'cost_rounds'/Path(selection['round_dir']).name
assert sha(rd/'comparison.json')==selection['comparison_sha256']
comparison=read(rd/'comparison.json'); cost=[];max_prob_delta=0.
ranked=sorted(ids,key=lambda s:hashlib.sha256(('P1-cost-20261014|'+s).encode()).hexdigest())[:50]
assert ranked==protocol['cost']['sample_ids']
for name in ['student','base','jev']:
    records=rows(rd/name/'timings.jsonl'); summary=read(rd/name/'summary.json');warm=read(rd/name/'warmup.json')
    assert len(warm)==10 and len(records)==150 and summary['sample_ids']==ranked
    assert [(r['repeat'],r['sample_id']) for r in records]==[(i,k) for i in range(3) for k in ranked]
    for r in records:
        p=np.array(r['probabilities']);q=np.array(expected[name][r['sample_id']]);delta=float(abs(p-q).max())
        close(delta,r['probability_max_diff']);assert delta<=1e-4 and p.argmax()==q.argmax()
        assert r['latency_ms']>0 and np.isfinite(r['latency_ms'])
        max_prob_delta=max(max_prob_delta,delta)
    for r in warm:
        close(r['probabilities'],expected[name][r['sample_id']],1e-4)
    latency=np.array([r['latency_ms'] for r in records])
    measured=dict(median_ms=float(np.median(latency)),p95_ms=float(np.quantile(latency,.95)),throughput_items_per_second=150/(latency.sum()/1000),
        peak_allocated_mib=max(r['peak_allocated_bytes'] for r in records)/1024**2,
        peak_reserved_mib=max(r['peak_reserved_bytes'] for r in records)/1024**2)
    for k,v in measured.items():close(v,summary[k]);close(v,next(r for r in comparison['rows'] if r['model']==name)[k])
    assert summary['source_runtime_sha256']==sha(R/'p1_runtime.py') and summary['protocol_sha256']==sha(R/'p1_protocol.json')
    assert summary['batch_size']==1 and summary['requests']==150 and summary['warmup']==10
    for k,v in protocol['environments']['student' if name=='student' else 'teacher'].items():assert summary['environment'][k].split('+')[0]==v
    cost.append(summary)
assert len(set(r['gpu_uuid'] for r in cost))==1
parity=read(R/'student_full_pipeline_parity.json');assert parity['status']=='PASS' and parity['class_mismatches']==0
assert parity['max_feature_difference']<=1e-4 and parity['max_probability_difference']<=1e-4
archive=P0/'P0完整归档_20261014'
p0zip=Path('C:/Users/2006/Desktop/CLIP训练集准备_20261009/16_论文收尾与评价核验_20261014/P0论文归档与评价核验_20261014.zip')
assert sha(p0zip)=='d6b9e2af830071a47fca36b32f0fb225531f46a0260594f2e871420e214b16ae'
with zipfile.ZipFile(p0zip) as z:
    assert z.testzip() is None
    hashname=next(n for n in z.namelist() if n.endswith('/FILE_HASHES.json') or n=='FILE_HASHES.json')
    prefix=hashname[:-len('FILE_HASHES.json')]
    p0hashes=json.loads(z.read(hashname))
    for item in p0hashes:
        name=prefix+item['path'];assert z.getinfo(name).file_size==item['bytes'],name
        h=hashlib.sha256()
        with z.open(name) as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        assert h.hexdigest()==item['sha256'],name
p0audit=read(archive/'最终评价重放/replay_audit.json');assert p0audit['status']=='PASS' and p0audit['class_mismatches']==0
report=dict(status='PASS',delivery_files_checked=len(delivery),immutable_package_files_checked=len(package),teacher_metric_rows=4,
    teacher_predictions=400,student_metric_rows=21,student_predictions=4200,maximum_metric_difference=maximum_metric_difference,
    selected_cost_round=rd.name,complete_cost_models=3,measured_requests=450,measurement_probabilities_max_difference=max_prob_delta,
    same_gpu=True,configuration_patch_only_json_normalization=True,p0_archive_files_checked=len(p0hashes),p0_prior_replay=p0audit,
    scope='Local recomputation of saved predictions/timings and hash verification. No new GPU execution or training.',
    limitations=['200-image prior exposure caveat remains.','One GPU session and one full round; 150 requests per model are repeated measurements on 50 inputs, not 150 independent images.',
    'Student FP32 vs Teacher FP16 backbone/FP32 adapters and different historical software stacks.',
    'PyTorch allocated/reserved memory is not whole-device memory. Disk/download excluded; cached loading is not full cold start.'])
(HERE/'P1核验记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'复算Teacher成绩.json').write_text(json.dumps(newtable,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['p0_prior_replay','limitations']},ensure_ascii=False,indent=2))
print('COST',[(r['model'],r['median_ms'],r['throughput_items_per_second'],r['peak_allocated_mib']) for r in cost])
