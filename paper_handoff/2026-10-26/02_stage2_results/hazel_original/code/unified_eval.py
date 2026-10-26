"""Frozen Stage2 inference only. No optimizer, fitting, or checkpoint selection."""
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
import os, sys, io, json, hashlib, zipfile, subprocess, platform, traceback, datetime, uuid
import numpy as np
import pandas as pd

PROTOCOL_SHA = 'f4e5acb3ca81cf8a2c27bd347b337c29cfaec9e6717c0793574ccd0a659ffe4e'
VERIFIER_SHA = 'a40414480755899ccdf7a4a11622f2c2b531e46f25608a642fd3529f5c205962'
NAMES = ['Stage2_统一评价_协议与输入_20261026.zip', 'Stage2_Hazel_results_f54a0fef.zip', 'Stage2_Hazel_TextOnly.zip']
SEEDS = [20261010, 20261011, 20261012]
METHODS = ['CE', 'LS005', 'LS010', 'KD_raw', 'KD_calibrated', 'Text_CE', 'Text_KD_raw']
METRICS = ['macro_f1', 'accuracy', 'f1_E', 'f1_N', 'f1_C', 'nll', 'ece10', 'brier']
BOUNDARY = ('本轮为已查看过成绩的数据上的后续补充评价，不能解释为首次独立确认。'
            '补充200张中130张曾进入早期验证准备，历史1000张中601张曾进入早期测试准备；'
            '准备清单不证明每条实际运行过，包外完整使用史仍未完全确认。'
            '三种子概率未做集成，重复概率行不构成独立样本量。人工标注继续暂停。')

def check(ok, message):
    if not ok: raise ValueError(message)

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p, b):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.partial')
    tmp.write_bytes(b); tmp.replace(p)
def save(p, obj): atomic(p, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
def csv_save(p, df): atomic(p, df.to_csv(index=False, float_format='%.17g').encode('utf-8-sig'))
def log(work, event, **data):
    p = Path(work)/'events.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **data), ensure_ascii=False)+'\n')

@contextmanager
def guarded(work, stage):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    lock = work/'ACTIVE.lock'
    with lock.open('x', encoding='utf-8') as f: f.write('进程运行中；仅确认旧进程已结束后可移除此锁。')
    log(work, 'start', stage=stage)
    try:
        yield
        log(work, 'success', stage=stage)
    except BaseException:
        log(work, 'failure', stage=stage, traceback=traceback.format_exc())
        raise
    finally: lock.unlink()

def discover(root, overrides=None):
    """One directory walk, including nested Drive folders; never guess a match."""
    root = Path(root); overrides = overrides or {}
    check(root.is_dir(), f'云盘未挂载：{root}')
    found = {n: [] for n in NAMES}
    for n, value in overrides.items():
        check(n in found, f'未知输入：{n}')
        if value:
            p = Path(value); check(p.is_file(), f'手动指定的文件不存在：{p}'); found[n] = [p]
    errors = []
    for folder, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        dirs[:] = [d for d in dirs if d not in ['.Trash', '.ipynb_checkpoints']]
        for n in set(files).intersection(NAMES):
            if not overrides.get(n): found[n].append(Path(folder)/n)
    for n, paths in found.items():
        print(n, '\n' + '\n'.join('  '+str(p) for p in paths), flush=True)
        check(paths, f'未找到 {n}。请确认挂载账号或在 OVERRIDES 中填写实际完整路径。扫描错误：{errors[:3]}')
    return found

def zip_member(z, suffix):
    names = [n for n in z.namelist() if n == suffix or n.endswith('/'+suffix)]
    check(len(names)==1, f'压缩包条目不唯一或缺失：{suffix}'); return names[0]

def prepare(found, work):
    work = Path(work).resolve()
    with guarded(work, 'verify_inputs'):
        protocol_paths = []
        for p in found[NAMES[0]]:
            with zipfile.ZipFile(p) as z:
                if sha_bytes(z.read(zip_member(z, 'evaluation_protocol_LOCKED.json'))) == PROTOCOL_SHA:
                    protocol_paths.append(p)
        check(protocol_paths, '没有匹配当前锁定协议的输入包，禁止替换协议继续。')
        check(len({sha(p) for p in protocol_paths})==1, '协议包存在不同内容的副本，请用 OVERRIDES 指定。')
        bundle = work/'inputs'; bundle.mkdir(exist_ok=True)
        with zipfile.ZipFile(protocol_paths[0]) as z:
            anchor = zip_member(z, 'evaluation_protocol_LOCKED.json')
            prefix = anchor[:-len('evaluation_protocol_LOCKED.json')]
            for n in z.namelist():
                if n.endswith('/'): continue
                check(n.startswith(prefix), '协议包目录结构异常')
                rel = n[len(prefix):]; parts = PurePosixPath(rel)
                check(not parts.is_absolute() and '..' not in parts.parts and '\\' not in rel and ':' not in rel, '不安全的ZIP路径')
                dest = bundle/rel; check(dest.resolve().is_relative_to(bundle.resolve()), 'ZIP路径越界')
                b = z.read(n)
                if dest.exists(): check(dest.read_bytes()==b, f'已保存输入发生变化：{rel}')
                else: atomic(dest,b)
        check(sha(bundle/'verify_bundle.py')==VERIFIER_SHA, 'verify_bundle.py 与已审阅版本不符')
        result = subprocess.run([sys.executable, str(bundle/'verify_bundle.py')], capture_output=True, text=True, encoding='utf-8')
        atomic(work/'verify_bundle.log', (result.stdout+'\n'+result.stderr).encode('utf-8'))
        check(result.returncode==0, '协议包核验失败，详见 verify_bundle.log')
        print(result.stdout)
        protocol = read(bundle/'evaluation_protocol_LOCKED.json'); models = read(bundle/'models_LOCKED.json')
        check(len(models)==21 and {m['model_id'] for m in models}=={f'{a}_{s}' for a in METHODS for s in SEEDS}, '模型组或种子错误')
        for m in models:
            text = m['method'].startswith('Text_')
            check(m['label_order']=='E/N/C' and (m['head_input'],m['head_hidden'],m['head_output'],m['input_slice']) == ((512,2048,3,'512:1024') if text else (2048,512,3,'0:2048')), '结构/标签顺序不符')
            t = m['student_temperature']
            check(t is None if text else (isinstance(t,(int,float)) and np.isfinite(t) and t>0), '固定学生温度错误')
        audit = []
        for archive in NAMES[1:]:
            subset = [m for m in models if m['checkpoint_archive']==archive]
            valid = []
            for path in found[archive]:
                try:
                    with zipfile.ZipFile(path) as z:
                        check(len(z.namelist())==len(set(z.namelist())), 'ZIP有重复条目')
                        for m in subset:
                            check(sha_bytes(z.read(m['checkpoint_entry']))==m['checkpoint_sha256'], 'checkpoint SHA256不匹配：'+m['model_id'])
                    valid.append(path)
                except Exception:
                    log(work, 'rejected_archive', path=str(path), traceback=traceback.format_exc())
            check(valid, f'没有符合全部模型哈希的 {archive}，详见 events.jsonl')
            # All accepted copies contain the exact same locked checkpoints.
            with zipfile.ZipFile(valid[0]) as z:
                for m in subset:
                    dest = work/'checkpoints'/f'{m["model_id"]}.pt'
                    atomic(dest, z.read(m['checkpoint_entry']))
                    audit.append(dict(model_id=m['model_id'], archive=str(valid[0]), entry=m['checkpoint_entry'], sha256=sha(dest), expected=m['checkpoint_sha256'], status='PASS'))
        save(work/'checkpoint_audit.json',audit)
        save(work/'input_audit.json',dict(protocol_sha256=PROTOCOL_SHA, protocol_archive=str(protocol_paths[0]),
              protocol_archive_sha256=sha(protocol_paths[0]), files={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()},
              checkpoint_count=len(audit), status='PASS'))
    return dict(work=work, bundle=bundle, models=models, protocol=protocol)

def environment(device):
    import torch
    check(device in ['cuda','cpu'], 'DEVICE 只能为 cuda 或 cpu')
    check(device!='cuda' or torch.cuda.is_available(), '未检测到GPU：选择Colab T4运行时，或明确设置 DEVICE="cpu"。')
    check(str(torch.__version__).split('+')[0]=='2.11.0', '协议要求 PyTorch 2.11.0；请先恢复约定版本。')
    if device=='cuda': check(torch.version.cuda=='13.0', '协议GPU环境为cu130，请先核查环境。')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=str(torch.__version__),
                device=device, gpu=torch.cuda.get_device_name(0) if device=='cuda' else None, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), dtype='float32', tf32=False, deterministic=True, batch_size=256)

def load_data(bundle, split):
    name = 'val_select_replay' if split=='val_select' else split
    with np.load(bundle/f'features/{name}.npz', allow_pickle=False) as a: d = {k:a[k].copy() for k in ['x','y','ids','image_ids']}
    rows = [json.loads(s) for s in (bundle/f'manifests/{split}.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    n = {'val_select':400,'historical_test':1000,'new_dev_holdout':200}[split]
    check(d['x'].shape==(n,2048) and d['x'].dtype==np.float32 and np.isfinite(d['x']).all(), '特征维度/精度/有限性不符')
    check(d['ids'].tolist()==[r['sample_id'] for r in rows] and d['y'].tolist()==[r['label'] for r in rows], '特征与清单ID/标签错位')
    check(d['image_ids'].astype(str).tolist()==[str(r['image_id']) for r in rows] and len(set(d['ids']))==n, '图片ID/样本唯一性错误')
    return d

def infer(ctx, m, d, device):
    import torch
    from torch import nn
    path = ctx['work']/'checkpoints'/f'{m["model_id"]}.pt'
    check(sha(path)==m['checkpoint_sha256'], '推理前checkpoint哈希不符')
    model = nn.Sequential(nn.Linear(m['head_input'],m['head_hidden']),nn.ReLU(),nn.Dropout(m['dropout']),nn.Linear(m['head_hidden'],3))
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    model = model.to(device=device,dtype=torch.float32).eval()
    start,stop = map(int,m['input_slice'].split(':'))
    x = np.ascontiguousarray(d['x'][:,start:stop],dtype=np.float32)
    with torch.inference_mode():
        z = np.concatenate([model(torch.from_numpy(x[i:i+256]).to(device)).cpu().numpy() for i in range(0,len(x),256)])
    check(z.shape==(len(x),3) and np.isfinite(z).all(), '推理输出异常')
    return z

def replay(ctx, device):
    work = ctx['work']
    with guarded(work, 'replay_all_21'):
        env = environment(device)
        config = dict(environment=env, protocol_sha256=PROTOCOL_SHA, model_registry_sha256=sha(ctx['bundle']/'models_LOCKED.json'),
                      code_sha256=sha(__file__), inference_only=True, replay_tolerance=1e-4, label_order=['E','N','C'])
        locked = work/'run_config.json'
        if locked.exists(): check(read(locked)==config, '本轮环境或代码发生变化，请新建独立 WORK，不能混用缓存。')
        else: save(locked,config)
        d = load_data(ctx['bundle'],'val_select'); records=[]
        for m in ctx['models']:
            z = infer(ctx,m,d,device)
            r = pd.read_csv(ctx['bundle']/f'replay_reference/{m["model_id"]}.csv',dtype={'sample_id':str})
            check(not r.sample_id.duplicated().any() and set(r.sample_id)==set(d['ids']), '重放参考样本ID不一致')
            r = r.set_index('sample_id').loc[d['ids']]
            check(np.array_equal(r.gold.to_numpy(),d['y']), '重放参考标签不一致')
            diff = float(np.max(np.abs(z.astype(np.float64)-r[['logit_E','logit_N','logit_C']].to_numpy())))
            mismatches = int(np.sum(z.argmax(1)!=r.prediction.to_numpy()))
            ok = mismatches==0 and diff<=1e-4
            records.append(dict(model_id=m['model_id'], checkpoint_sha256=m['checkpoint_sha256'], max_abs_logit_diff=diff,
                                prediction_mismatches=mismatches, rows=400, status='PASS' if ok else 'FAIL'))
            csv_save(work/'replay_audit.csv',pd.DataFrame(records))
            out = pd.DataFrame(dict(sample_id=d['ids'],gold=d['y'],prediction=z.argmax(1)))
            for i,c in enumerate('ENC'): out['logit_'+c]=z[:,i]
            csv_save(work/'replay_predictions'/f'{m["model_id"]}.csv',out)
            print('重放',m['model_id'],records[-1]['status'],f'最大差={diff:.3g}',flush=True)
            check(ok, '开发集重放失败；评价尚未启动。详见 replay_audit.csv，不能放宽容差。')
        save(work/'replay_PASS.json',dict(count=21,run_config_sha256=sha(locked),audit_sha256=sha(work/'replay_audit.csv')))
    ctx['device']=device; ctx['config']=config
    return pd.DataFrame(records)

def scores(z, y, temperature=1.):
    check(np.isfinite(temperature) and temperature>0, '温度必须为有限正数')
    a = np.asarray(z,dtype=np.float64)/temperature
    a -= a.max(axis=1,keepdims=True)
    logp = a-np.log(np.exp(a).sum(axis=1,keepdims=True)); p = np.exp(logp)
    pred = np.argmax(a,axis=1); y=np.asarray(y,dtype=np.int64)
    f1=[]
    for c in range(3):
        tp=np.sum((pred==c)&(y==c)); den=np.sum(pred==c)+np.sum(y==c)
        f1.append(float(2*tp/den) if den else 0.)
    conf=p.max(axis=1); correct=(pred==y); bins=np.minimum(np.floor(conf*10).astype(int),9)
    ece=0.
    for k in range(10):
        ix=bins==k
        if ix.any(): ece+=float(ix.mean()*abs(correct[ix].mean()-conf[ix].mean()))
    result=dict(macro_f1=float(np.mean(f1)),accuracy=float(correct.mean()),f1_E=f1[0],f1_N=f1[1],f1_C=f1[2],
                nll=float(-logp[np.arange(len(y)),y].mean()),ece10=ece,brier=float(np.square(p-np.eye(3)[y]).sum(axis=1).mean()))
    return p,pred,result

def evaluate(ctx):
    work=ctx['work']; device=ctx['device']
    with guarded(work,'evaluate_frozen_models'):
        check(environment(device)==ctx['config']['environment'], '设备或环境改变')
        check(read(work/'run_config.json')==ctx['config'], '运行配置改变')
        receipt=read(work/'replay_PASS.json')
        check(receipt['count']==21 and receipt['run_config_sha256']==sha(work/'run_config.json') and receipt['audit_sha256']==sha(work/'replay_audit.csv'), '未通过全部21模型重放')
        expected=read(work/'input_audit.json')['files']
        for n,h in expected.items(): check(sha(ctx['bundle']/n)==h, '评价前输入发生变化：'+n)
        metrics=[]; raw_tables=[]; probability_tables=[]
        for split in ['historical_test','new_dev_holdout']:
            d=load_data(ctx['bundle'],split)
            for m in ctx['models']:
                name=m['model_id']; path=work/'raw_cache'/split/f'{name}.npz'; rp=path.with_suffix('.json')
                key=dict(config_sha256=sha(work/'run_config.json'),checkpoint_sha256=m['checkpoint_sha256'],
                         feature_sha256=sha(ctx['bundle']/f'features/{split}.npz'),split=split,model_id=name)
                if path.exists() and rp.exists():
                    r=read(rp); check(r['key']==key and r['sha256']==sha(path), '原始预测缓存发生变化，禁止静默复用')
                    with np.load(path,allow_pickle=False) as a: z=a['logits'].copy()
                    log(work,'reuse_verified_logits',split=split,model_id=name)
                else:
                    if path.exists() or rp.exists(): log(work,'retry_incomplete_cache',split=split,model_id=name)
                    z=infer(ctx,m,d,device); b=io.BytesIO(); np.savez_compressed(b,logits=z)
                    atomic(path,b.getvalue()); save(rp,dict(key=key,sha256=sha(path)))
                check(z.shape==(len(d['y']),3) and z.dtype==np.float32 and np.isfinite(z).all(),'缓存输出异常')
                base=pd.DataFrame(dict(split=split,model_id=name,method=m['method'],seed=m['seed'],sample_id=d['ids'],
                                       image_id=d['image_ids'],gold=d['y'],checkpoint_sha256=m['checkpoint_sha256']))
                for i,c in enumerate('ENC'): base['logit_'+c]=z[:,i]
                variants=[('raw',1.)]
                if m['student_temperature'] is not None: variants.append(('post_temperature',m['student_temperature']))
                raw_metric=None
                for variant,t in variants:
                    p,pred,metric=scores(z,d['y'],t)
                    check(np.array_equal(pred,z.argmax(1)), '温度后处理改变预测类别')
                    if raw_metric is None: raw_metric=metric
                    else:
                        check(all(metric[k]==raw_metric[k] for k in ['macro_f1','accuracy','f1_E','f1_N','f1_C']), '温度后处理改变分类指标')
                    table=base.copy(); table['prediction']=pred; table['variant']=variant; table['student_T']=t
                    for i,c in enumerate('ENC'): table['p_'+c]=p[:,i]
                    table=table[ctx['protocol']['outputs']['required_prediction_fields']]
                    probability_tables.append(table)
                    if variant=='raw': raw_tables.append(table)
                    metrics.append(dict(split=split,display_name='历史1000条' if split=='historical_test' else '补充200条',
                                        model_id=name,method=m['method'],seed=m['seed'],variant=variant,student_T=t,
                                        checkpoint_sha256=m['checkpoint_sha256'],n=len(d['y']),**metric))
                print('完成',split,name,flush=True)
        raw=pd.concat(raw_tables,ignore_index=True); probs=pd.concat(probability_tables,ignore_index=True); per=pd.DataFrame(metrics)
        check(len(raw)==25200 and len(probs)==43200 and len(per)==72,'固定输出数量不符')
        check(not probs.duplicated(['split','model_id','variant','sample_id']).any(),'逐条输出有重复主键')
        summary=[]
        for keys,g in per.groupby(['split','display_name','method','variant'],sort=False):
            check(len(g)==3 and sorted(g.seed)==SEEDS,'三种子不完整，不能生成完整汇总')
            row=dict(zip(['split','display_name','method','variant'],keys)); row['seed_count']=3
            for k in METRICS: row[k+'_mean']=float(g[k].mean()); row[k+'_sd']=float(g[k].std(ddof=1))
            summary.append(row)
        summary=pd.DataFrame(summary)
        for split in ['historical_test','new_dev_holdout']:
            out=work/'report'/split
            csv_save(out/'raw_predictions.csv',raw[raw.split==split])
            csv_save(out/'all_probabilities.csv',probs[probs.split==split])
            csv_save(out/'metrics_per_seed.csv',per[per.split==split])
            csv_save(out/'metrics_mean_sampleSD.csv',summary[summary.split==split])
        save(work/'completion.json',dict(status='COMPLETE',raw_predictions=25200,probability_rows=43200,
             raw_model_split_outputs=42,total_variant_split_outputs=72,training_performed=False,temperature_fitted=False,
             bootstrap_performed=False,protocol_sha256=PROTOCOL_SHA,interpretation=BOUNDARY))
        atomic(work/'解释边界.txt',BOUNDARY.encode('utf-8'))
    return summary

def package(ctx):
    work=ctx['work']; check(read(work/'completion.json')['status']=='COMPLETE','尚未完整完成评价')
    with guarded(work,'package'):
        atomic(work/'code/unified_eval.py',Path(__file__).read_bytes())
    # Models stay in WORK for reproducibility; don't duplicate the original model ZIPs in delivery.
    paths=[p for p in work.rglob('*') if p.is_file() and 'checkpoints' not in p.relative_to(work).parts and p.name not in ['PACKAGE_SHA256.json','ACTIVE.lock'] and not p.name.endswith('.partial')]
    save(work/'PACKAGE_SHA256.json',{p.relative_to(work).as_posix():sha(p) for p in sorted(paths)})
    target=work.parent/(work.name+'_results.zip'); tmp=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[work/'PACKAGE_SHA256.json']: z.write(p,p.relative_to(work).as_posix())
    tmp.replace(target)
    with zipfile.ZipFile(target) as z:
        for n,h in json.loads(z.read('PACKAGE_SHA256.json')).items(): check(sha_bytes(z.read(n))==h,'结果包校验失败：'+n)
    print('结果包：',target,'\nSHA256：',sha(target)); return target
