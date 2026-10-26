"""Independent audit, CPU replay, frozen paired bootstrap and descriptive error transitions.
No uploaded code execution, no fitting, no new model selection.
"""
from pathlib import Path
import zipfile,json,csv,io,hashlib,collections,platform,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--source-zip',type=Path,default=Path('C:/Users/2006/Desktop/Stage2_Hazel_UnifiedEvaluation_20261026-20261026T090809Z-1-001.zip'))
parser.add_argument('--protocol-dir',type=Path,default=ROOT.parent/'stage2_eval_protocol_20261026/handoff')
parser.add_argument('--out',type=Path,default=ROOT/'results')
args=parser.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True);LOCK=args.protocol_dir.resolve();ZIP=args.source_zip.resolve()
z=zipfile.ZipFile(ZIP);prefix='Stage2_Hazel_UnifiedEvaluation_20261026/'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda n:z.read(prefix+n)
j=lambda n:json.loads(read(n))
rows=lambda n:list(csv.DictReader(io.StringIO(read(n).decode('utf-8-sig'))))
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def table(name,rr):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def metrics(logits,y,T=1.):
 a=np.asarray(logits,dtype=np.float64)/T;m=a.max(1);lse=m+np.log(np.exp(a-m[:,None]).sum(1));p=np.exp(a-lse[:,None]);pred=a.argmax(1)
 cm=np.bincount(3*y+pred,minlength=9).reshape(3,3);f=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
 conf=p.max(1);hit=pred==y;bins=np.minimum((conf*10).astype(int),9)
 nll=lse-a[np.arange(len(y)),y];brier=((p-np.eye(3)[y])**2).sum(1)
 vals=dict(macro_f1=float(f.mean()),accuracy=float(hit.mean()),f1_E=float(f[0]),f1_N=float(f[1]),f1_C=float(f[2]),nll=float(nll.mean()),ece10=float(sum(abs(((hit-conf)*(bins==b)).mean()) for b in range(10))),brier=float(brier.mean()))
 return vals,p,pred,nll,brier

assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
hashes=j('PACKAGE_SHA256.json');names={n[len(prefix):] for n in z.namelist() if not n.endswith('/')}
registry=j('inputs/models_LOCKED.json')
expected_unlisted={f'checkpoints/{m["model_id"]}.pt' for m in registry}|{'inputs/PACKAGE_SHA256.json'}
assert not set(hashes)-names and names-set(hashes)-{'PACKAGE_SHA256.json'}==expected_unlisted
for n,h in hashes.items():assert sha(read(n))==h,n
prior=json.loads((LOCK/'PACKAGE_SHA256.json').read_text(encoding='utf8'))
for n,h in prior.items():assert sha(read('inputs/'+n))==h==sha((LOCK/n).read_bytes()),n
assert read('inputs/PACKAGE_SHA256.json')==(LOCK/'PACKAGE_SHA256.json').read_bytes()
protocol=j('inputs/evaluation_protocol_LOCKED.json');models=j('inputs/models_LOCKED.json');config=j('run_config.json');complete=j('completion.json')
assert complete['status']=='COMPLETE' and complete['raw_predictions']==25200 and complete['probability_rows']==43200
assert config['protocol_sha256']==complete['protocol_sha256']==sha(read('inputs/evaluation_protocol_LOCKED.json'))
assert config['model_registry_sha256']==sha(read('inputs/models_LOCKED.json')) and config['code_sha256']==sha(read('code/unified_eval.py'))
assert config['environment']['batch_size']==256 and config['environment']['dtype']=='float32' and config['environment']['tf32']==False
data={};cache={};computed={};metricsrows=[];summaries=[];maxerror=0.;allrowscount=0
for ds in protocol['datasets']:
 split=ds['split']
 with np.load(io.BytesIO(read('inputs/'+ds['features_file'])),allow_pickle=False) as a:d={k:a[k].copy() for k in ['x','y','ids','image_ids']}
 data[split]=d;assert len(d['y'])==ds['rows']
 rawrows=rows(f'report/{split}/raw_predictions.csv');probrows=rows(f'report/{split}/all_probabilities.csv');per=rows(f'report/{split}/metrics_per_seed.csv')
 rawgroups=collections.defaultdict(list);probgroups=collections.defaultdict(list)
 for r in rawrows:rawgroups[r['model_id']].append(r)
 for r in probrows:probgroups[r['model_id'],r['variant']].append(r)
 assert len(rawgroups)==21 and len(probgroups)==36 and len(per)==36
 peridx={(r['method'],int(r['seed']),r['variant']):r for r in per};assert len(peridx)==36
 for model in models:
  mid=model['model_id'];seed=model['seed'];method=model['method'];h=model['checkpoint_sha256']
  assert sha(read(f'checkpoints/{mid}.pt'))==h
  name=f'raw_cache/{split}/{mid}.npz';receipt=j(name.replace('.npz','.json'))
  assert receipt['sha256']==sha(read(name)) and receipt['key']['checkpoint_sha256']==h and receipt['key']['feature_sha256']==ds['npz_sha256']
  assert receipt['key']['config_sha256']==sha(read('run_config.json'))
  with np.load(io.BytesIO(read(name)),allow_pickle=False) as a:logits=a['logits'].copy()
  assert logits.shape==(len(d['y']),3) and logits.dtype==np.float32 and np.isfinite(logits).all();cache[split,mid]=logits
  rr=rawgroups[mid];assert [r['sample_id'] for r in rr]==d['ids'].tolist()
  assert np.max(abs(np.array([[float(r['logit_'+k]) for k in 'ENC'] for r in rr])-logits))<1e-12
  variants=[('raw',1.)]+([] if model['student_temperature'] is None else [('post_temperature',model['student_temperature'])])
  for variant,T in variants:
   m,p,pred,_,_=metrics(logits,d['y'],T);computed[split,method,seed,variant]=m
   rr=probgroups[mid,variant];assert len(rr)==len(d['y']) and [r['sample_id'] for r in rr]==d['ids'].tolist()
   assert [str(r['image_id']) for r in rr]==list(map(str,d['image_ids'])) and [int(r['gold']) for r in rr]==d['y'].tolist()
   assert [int(r['prediction']) for r in rr]==pred.tolist()
   assert all(float(r['student_T'])==T and r['checkpoint_sha256']==h for r in rr)
   assert np.max(abs(np.array([[float(r['p_'+k]) for k in 'ENC'] for r in rr])-p))<1e-12
   assert np.max(abs(np.array([[float(r['logit_'+k]) for k in 'ENC'] for r in rr])-logits))<1e-12
   ref=peridx[method,seed,variant]
   for k,v in m.items():
    err=abs(v-float(ref[k]));maxerror=max(maxerror,err);assert err<1e-10,(mid,k,err)
   metricsrows.append(dict(split=split,method=method,seed=seed,variant=variant,T=T,**m));allrowscount+=len(rr)
 for r in rows(f'report/{split}/metrics_mean_sampleSD.csv'):
  m={}
  for k in next(iter(computed.values())):
   values=[computed[split,r['method'],s,r['variant']][k] for s in [20261010,20261011,20261012]]
   m[k+'_mean']=float(np.mean(values));m[k+'_sd']=float(np.std(values,ddof=1))
   assert abs(m[k+'_mean']-float(r[k+'_mean']))<1e-10 and abs(m[k+'_sd']-float(r[k+'_sd']))<1e-10
  summaries.append(dict(split=split,method=r['method'],variant=r['variant'],**m))
assert allrowscount==43200
table('recomputed_per_seed.csv',metricsrows);table('recomputed_summary.csv',summaries)
print('Package and 43200 probability rows verified; beginning CPU checkpoint replay.',flush=True)

import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
replays=[]
with np.load(io.BytesIO(read('inputs/features/val_select_replay.npz')),allow_pickle=False) as a:devx=a['x'].copy()
for spec in models:
 mid=spec['model_id'];model=torch.nn.Sequential(torch.nn.Linear(spec['head_input'],spec['head_hidden']),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(spec['head_hidden'],3))
 model.load_state_dict(torch.load(io.BytesIO(read(f'checkpoints/{mid}.pt')),map_location='cpu',weights_only=True));model.eval()
 for split,x in [('val_select',devx)]+[(s,d['x']) for s,d in data.items()]:
  xx=np.ascontiguousarray(x[:,512:1024]) if spec['head_input']==512 else x
  with torch.inference_mode():actual=np.concatenate([model(torch.from_numpy(xx[i:i+256])).numpy() for i in range(0,len(xx),256)])
  if split=='val_select':
   rr=rows(f'inputs/replay_reference/{mid}.csv');ref=np.array([[float(r['logit_'+k]) for k in 'ENC'] for r in rr])
  else:ref=cache[split,mid]
  diff=float(np.max(abs(ref-actual)));mismatch=int(np.sum(ref.argmax(1)!=actual.argmax(1)))
  assert diff<=1e-4 and mismatch==0,(split,mid,diff,mismatch)
  replays.append(dict(split=split,model_id=mid,max_abs_logit_difference=diff,prediction_mismatches=mismatch))
 print('Replay PASS',mid,flush=True)
table('independent_CPU_replay.csv',replays)
save('execution_environment.json',dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),device='cpu',purpose='Independent replay and statistics only; reported metrics use original GPU logits'))

# Frozen bootstrap indices are shared by all methods and seeds. Multiplicity counts
# permit fast linear algebra, equivalent to retaining all repeated sampled rows.
intervals=[];errorrows=[];error_details=[];draw_validation=[]
for split,d in data.items():
 with np.load(io.BytesIO(read(f'inputs/bootstrap/{split}_draws.npz')),allow_pickle=False) as a:ordered_ids=a['image_ids'].copy();draws=a['indices'].copy()
 n=len(d['y']);ids={str(v):i for i,v in enumerate(d['image_ids'])};order=np.array([ids[str(v)] for v in ordered_ids]);y=d['y'][order]
 assert draws.shape==(10000,n)
 W=np.zeros((10000,n),dtype=np.float64)
 for start in range(0,10000,500):
  chunk=draws[start:start+500];offset=np.arange(len(chunk))[:,None]*n
  W[start:start+len(chunk)]=np.bincount((chunk+offset).ravel(),minlength=len(chunk)*n).reshape(len(chunk),n)
 assert np.all(W.sum(1)==n)
 bs={}
 for spec in models:
  mid=spec['model_id'];logits=cache[split,mid][order];method=spec['method'];seed=spec['seed']
  variants=[('raw',1.)]+([] if spec['student_temperature'] is None else [('post_temperature',spec['student_temperature'])])
  for variant,T in variants:
   _,p,pred,nll,brier=metrics(logits,y,T);conf=p.max(1);correct=pred==y;bins=np.minimum((conf*10).astype(int),9)
   indicators=np.eye(9)[y*3+pred];eceparts=np.stack([(correct-conf)*(bins==b) for b in range(10)],axis=1)
   aggregate=W@np.column_stack([indicators,nll,brier,eceparts]);cm=aggregate[:,:9].reshape(-1,3,3)
   fs=2*np.diagonal(cm,axis1=1,axis2=2)/np.maximum(cm.sum(1)+cm.sum(2),1)
   scores=dict(macro_f1=fs.mean(1),accuracy=np.trace(cm,axis1=1,axis2=2)/n,f1_E=fs[:,0],f1_N=fs[:,1],f1_C=fs[:,2],nll=aggregate[:,9]/n,brier=aggregate[:,10]/n,ece10=abs(aggregate[:,11:]).sum(1)/n)
   bs[method,seed,variant]=scores
   # Validate several exact draws against a direct recomputation for every variant.
   for ix in [0,123,9999]:
    direct=metrics(logits[draws[ix]],y[draws[ix]],T)[0]
    for k,v in direct.items():assert abs(v-scores[k][ix])<1e-10
 for c in protocol['comparisons']:
  am,av,bm,bv=c['method_a'],c['variant_a'],c['method_b'],c['variant_b']
  for metric in c['metrics']:
   samples=np.mean([bs[am,s,av][metric]-bs[bm,s,bv][metric] for s in [20261010,20261011,20261012]],axis=0)
   point=float(np.mean([computed[split,am,s,av][metric]-computed[split,bm,s,bv][metric] for s in [20261010,20261011,20261012]]))
   lo,hi=np.quantile(samples,[.025,.975],method='linear')
   intervals.append(dict(split=split,comparison=c['id'],method_a=am,variant_a=av,method_b=bm,variant_b=bv,metric=metric,delta=point,ci_low=float(lo),ci_high=float(hi),crosses_zero=bool(lo<=0<=hi),role='primary' if c['id']=='P1_KD_vs_CE' and split=='historical_test' else 'exploratory'))
  if 'macro_f1' in c['metrics']:
   for seed in [20261010,20261011,20261012]:
    pa=cache[split,f'{am}_{seed}'].argmax(1);pb=cache[split,f'{bm}_{seed}'].argmax(1);yy=d['y'];a=pa==yy;b=pb==yy
    fixed=int(sum(a&~b));harmed=int(sum(~a&b))
    errorrows.append(dict(split=split,comparison=c['id'],seed=seed,n=n,b_wrong_a_right=fixed,b_right_a_wrong=harmed,both_right=int(sum(a&b)),both_wrong=int(sum(~a&~b)),net_correct=fixed-harmed,accuracy_delta=(fixed-harmed)/n,changed_class=int(sum(pa!=pb))))
    for i in np.flatnonzero(pa!=pb):error_details.append(dict(split=split,comparison=c['id'],seed=seed,sample_id=str(d['ids'][i]),image_id=str(d['image_ids'][i]),gold=int(yy[i]),prediction_a=int(pa[i]),prediction_b=int(pb[i]),a_correct=bool(a[i]),b_correct=bool(b[i])))
 print('Bootstrap complete:',split,flush=True)
table('paired_intervals.csv',intervals);table('error_transitions_per_seed.csv',errorrows);table('changed_predictions.csv',error_details)
grouped=collections.defaultdict(list)
for r in errorrows:grouped[r['split'],r['comparison']].append(r)
table('error_transitions_summary.csv',[dict(split=s,comparison=c,**{k:float(np.mean([r[k] for r in rr])) for k in ['b_wrong_a_right','b_right_a_wrong','both_right','both_wrong','net_correct','accuracy_delta','changed_class']}) for (s,c),rr in grouped.items()])
audit=dict(status='PASS',archive_sha256=sha(ZIP.read_bytes()),verified_files=len(hashes),protocol_files_matched=len(prior)+1,models=21,metric_rows=72,probability_rows=43200,metrics_max_abs_error=maxerror,cpu_replay_model_split_pairs=len(replays),cpu_replay_max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in replays),cpu_replay_prediction_mismatches=sum(r['prediction_mismatches'] for r in replays),paired_intervals=len(intervals),bootstrap_repeats=10000,bootstrap_shared_frozen_draws=True,direct_draw_validation=True,training=False,temperature_refit=False,scope='Previously opened evaluation sets; supplementary evaluation. Fixed-seed image bootstrap does not remove selection bias.')
save('FINAL_AUDIT.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
print('MAIN INTERVALS')
for r in intervals:
 if r['metric']=='macro_f1':print(json.dumps(r,ensure_ascii=False))
