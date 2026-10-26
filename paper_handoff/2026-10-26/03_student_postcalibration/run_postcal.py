"""Frozen-checkpoint scalar temperature analysis. Never imports supplied experiment code.
Usage: python run_postcal.py prepare /absolute/output; python run_postcal.py run /absolute/output
prepare uses the archived local inputs; run only needs the prepared directory.
"""
from pathlib import Path
import csv,io,json,hashlib,zipfile,sys,math,platform,datetime
import numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def csvrows(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def table(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def softmax(z):
 a=np.exp(z-z.max(1,keepdims=True));return a/a.sum(1,keepdims=True)
def nll(z,y):
 z=np.asarray(z,dtype=np.float64);m=z.max(1)
 return float(np.mean(m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]))
def metrics(z,y):
 z=np.asarray(z,dtype=np.float64);p=softmax(z);pred=z.argmax(1)
 cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,pred),1)
 f=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
 conf=p.max(1);hit=pred==y;bins=np.minimum((conf*10).astype(int),9)
 return dict(macro_f1=float(f.mean()),accuracy=float(hit.mean()),f1_E=float(f[0]),f1_N=float(f[1]),f1_C=float(f[2]),nll=nll(z,y),ece10=float(sum(np.mean(bins==b)*abs(hit[bins==b].mean()-conf[bins==b].mean()) for b in range(10) if (bins==b).any())),brier=float(((p-np.eye(3)[y])**2).sum(1).mean()))

def prepare(out):
 assert not out.exists(),'Use a new output folder; never overwrite a locked run'
 desktop=Path('C:/Users/2006/Desktop')
 source=Path(__file__).resolve().parents[1]/'ablation_20261013/final_delivery/reused_v1'
 kdpath=desktop/'Stage2_Hazel_results_f54a0fef.zip'
 day2path=next(desktop.glob('CLIP_Jev_CE三学习率*20261023T*.zip'))
 kd=zipfile.ZipFile(kdpath);d2=zipfile.ZipFile(day2path);prefix=d2.namelist()[0].split('/')[0]+'/'
 oldcal=json.loads(d2.read(prefix+'student_calibration.json'))
 oldproto=json.loads(kd.read('protocol.json'));inventory={};models=[]
 def put(name,b):
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);inventory[name]=sha(b)
 for split in ['val_fit','val_select']:
  b=d2.read(prefix+'features/val_fit.npz') if split=='val_fit' else (source/'features/val_select.npz').read_bytes()
  expected=oldcal['inputs']['fit_sha256'] if split=='val_fit' else oldproto['cache']['input_sha256']['features/val_select.npz']
  assert sha(b)==expected
  put('inputs/'+split+'.npz',b);put('inputs/'+split+'.jsonl',(source/f'handoff/manifests/{split}.jsonl').read_bytes())
 put('inputs/val_fit_meta.json',d2.read(prefix+'features/val_fit_meta.json'))
 for method in ['CE','KD_raw','KD_calibrated']:
  for seed in [20261010,20261011,20261012]:
   folder=f'Day2_preserved/groups/CE_lr0.0003_eps0/runs/CE_{seed}' if method=='CE' else f'groups/{method}_alpha0.75/runs/{method}_{seed}'
   done=json.loads(kd.read(folder+'/done.json'));b=kd.read(folder+'/best.pt');assert sha(b)==done['checkpoint_sha256']
   mid=f'{method}_{seed}';put(f'inputs/models/{mid}.pt',b);put(f'inputs/models/{mid}_done.json',kd.read(folder+'/done.json'))
   if method=='CE':
    rr=[r for r in csvrows(kd.read('Day2_preserved/report/predictions.csv')) if r['model']==f'CE_eps0_{seed}' and r['variant']=='raw']
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr);b=buf.getvalue().encode('utf8')
   else:b=kd.read(folder+'/predictions.csv')
   put(f'inputs/reference/{mid}.csv',b)
   models.append(dict(id=mid,method=method,seed=seed,checkpoint_sha256=done['checkpoint_sha256'],best_epoch=done['best_epoch']))
 put('inputs/CE_previous_calibration.json',json.dumps([r for r in oldcal['models'] if r['method']=='CE'],ensure_ascii=False).encode('utf8'))
 put('inputs/Day2_reference_per_seed.csv',kd.read('Day2_preserved/report/per_seed.csv'))
 put('run_postcal.py',Path(__file__).read_bytes())
 protocol=dict(kind='Stage2_KD_student_post_temperature_v1',locked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,
  fit='val_fit 400 only; one positive scalar T per checkpoint; minimize NLL; ties prefer abs(log(T)) closest to 0',
  temperature_grid=dict(min=.05,max=20.,geomspace_points=401,include_one=True),
  selection='CE common lr=.0003; KD raw and calibrated alpha=.75 from previous developmental selection; no new selection or training',
  report='val_select 400 reused developmental descriptive comparison; no new independent confirmation; no test evaluation',
  primary_comparison='KD_calibrated raw versus KD_raw post-temperature; NLL and ECE10; also show all families before/after',
  replay='CPU float32 torch 2.11.0; eval mode; chunks256; max abs saved-logit difference <=1e-4 and identical predictions required. Canonical val_select logits retained from original GPU records.',
  statistics='Means and sample SD over fixed 3 seeds, paired seed differences; not confidence intervals',
  archives={kdpath.name:sha(kdpath.read_bytes()),day2path.name:sha(day2path.read_bytes())},inputs_sha256=inventory)
 save(out/'protocol_LOCKED.json',protocol)
 print('PREPARED and LOCKED:',out,flush=True)

def run(out):
 import torch
 from torch import nn
 assert not (out/'COMPLETE.json').exists(),'Run already complete'
 protocol=read(out/'protocol_LOCKED.json')
 assert str(torch.__version__).split('+')[0]=='2.11.0'
 for name,h in protocol['inputs_sha256'].items():assert sha((out/name).read_bytes())==h,name
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 env=dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),device='cpu',dtype='float32',threads=2,platform=platform.platform(),original='torch2.11.0+cu130/numpy2.1.3/python3.13.16 T4; CE originally python3.13.15')
 save(out/'environment.json',env)
 data={}
 for split in ['val_fit','val_select']:
  with np.load(out/f'inputs/{split}.npz',allow_pickle=False) as a:d={k:a[k].copy() for k in ['x','y','ids','image_ids']}
  rr=[json.loads(s) for s in (out/f'inputs/{split}.jsonl').read_text(encoding='utf8').splitlines()]
  assert d['x'].shape==(400,2048) and d['x'].dtype==np.float32 and np.isfinite(d['x']).all()
  assert d['ids'].tolist()==[r['sample_id'] for r in rr] and d['y'].tolist()==[r['label'] for r in rr]
  data[split]=d
 assert not set(data['val_fit']['ids']) & set(data['val_select']['ids'])
 assert not set(map(str,data['val_fit']['image_ids'])) & set(map(str,data['val_select']['image_ids']))
 grid=np.unique(np.r_[1.,np.geomspace(.05,20.,401)])
 records=[];fits=[];audits=[]
 oldce={r['seed']:r for r in read(out/'inputs/CE_previous_calibration.json')}
 for spec in protocol['models']:
  mid=spec['id'];model=nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(.1),nn.Linear(512,3))
  state=torch.load(out/f'inputs/models/{mid}.pt',map_location='cpu',weights_only=True)
  assert set(state)=={'0.weight','0.bias','3.weight','3.bias'} and all(torch.isfinite(v).all() for v in state.values())
  model.load_state_dict(state,strict=True);model.eval();logits={}
  with torch.inference_mode():
   for split,d in data.items():logits[split]=np.concatenate([model(torch.from_numpy(d['x'][i:i+256])).numpy() for i in range(0,400,256)]).astype(np.float64)
  rr=csvrows((out/f'inputs/reference/{mid}.csv').read_bytes());lookup={r['sample_id']:r for r in rr};rr=[lookup[s] for s in data['val_select']['ids']]
  canonical=np.array([[float(r['logit_'+k]) for k in 'ENC'] for r in rr])
  diff=float(np.max(abs(canonical-logits['val_select'])));assert diff<=1e-4,(mid,diff)
  assert np.array_equal(canonical.argmax(1),logits['val_select'].argmax(1))
  for i,r in enumerate(rr):assert int(r.get('label',r.get('gold')))==data['val_select']['y'][i]
  fitz=logits['val_fit'];y=data['val_fit']['y'];values=[nll(fitz/t,y) for t in grid]
  index=min(range(len(grid)),key=lambda i:(values[i],abs(math.log(grid[i]))));t=float(grid[index])
  fit=dict(**spec,T=t,at_boundary=index in [0,len(grid)-1],fit_nll_raw=nll(fitz,y),fit_nll_calibrated=values[index])
  if spec['method']=='CE':
   assert abs(t-oldce[spec['seed']]['T'])<1e-12,(mid,t,oldce[spec['seed']]['T'])
   assert abs(fit['fit_nll_raw']-oldce[spec['seed']]['fit_nll_raw'])<1e-6
  fits.append(fit);audits.append(dict(id=mid,max_abs_gpu_cpu_logit_difference=diff,predictions_equal=True,CE_temperature_reproduced=spec['method']=='CE'))
  folder=out/'outputs'/mid;folder.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(folder/'logits.npz',val_fit=fitz,val_select_canonical=canonical,val_select_cpu=logits['val_select'])
  table(folder/'temperature_grid.csv',[dict(T=float(t0),fit_nll=v) for t0,v in zip(grid,values)])
  for split,z0 in [('val_fit',fitz),('val_select',canonical)]:
   for variant,temperature in [('raw',1.),('post_temperature',t)]:
    z=z0/temperature;prob=softmax(z);yy=data[split]['y'];m=metrics(z,yy)
    assert np.array_equal(z.argmax(1),z0.argmax(1))
    records.append(dict(method=spec['method'],seed=spec['seed'],split=split,variant=variant,T=temperature,**m))
    table(folder/f'{split}_{variant}.csv',[dict(sample_id=str(sid),gold=int(yy[i]),prediction=int(z[i].argmax()),**{'logit_'+k:float(z0[i,j]) for j,k in enumerate('ENC')},**{'p_'+k:float(prob[i,j]) for j,k in enumerate('ENC')}) for i,sid in enumerate(data[split]['ids'])])
  print(mid,'T=',round(t,6),'replay_error=',diff,flush=True)
 # Fit lock exists before development summary is produced.
 save(out/'student_temperature_LOCKED.json',dict(protocol_sha256=sha((out/'protocol_LOCKED.json').read_bytes()),models=fits))
 save(out/'replay_audit.json',audits);table(out/'per_seed.csv',records)
 keys=list(metrics(np.zeros((1,3)),np.array([0])))
 summary=[]
 for method in ['CE','KD_raw','KD_calibrated']:
  for variant in ['raw','post_temperature']:
   rr=[r for r in records if r['method']==method and r['variant']==variant and r['split']=='val_select']
   summary.append(dict(method=method,variant=variant,n_seeds=3,**{k+'_mean':float(np.mean([r[k] for r in rr])) for k in keys},**{k+'_seed_sd':float(np.std([r[k] for r in rr],ddof=1)) for k in keys}))
 table(out/'summary.csv',summary)
 index={(r['method'],r['variant'],r['seed']):r for r in records if r['split']=='val_select'}
 pairs=[('KD_calibrated','raw','KD_raw','post_temperature'),('KD_calibrated','post_temperature','KD_raw','post_temperature'),('KD_raw','post_temperature','CE','post_temperature')]+[(m,'post_temperature',m,'raw') for m in ['CE','KD_raw','KD_calibrated']]
 deltas=[];dsummary=[]
 for a,av,b,bv in pairs:
  name=f'{a} {av} minus {b} {bv}';dr=[]
  for seed in [20261010,20261011,20261012]:
   row=dict(comparison=name,seed=seed,**{k:index[a,av,seed][k]-index[b,bv,seed][k] for k in keys});dr.append(row);deltas.append(row)
  dsummary.append(dict(comparison=name,**{k+'_mean':float(np.mean([r[k] for r in dr])) for k in keys},**{k+'_seed_sd':float(np.std([r[k] for r in dr],ddof=1)) for k in keys}))
 table(out/'paired_deltas.csv',deltas);table(out/'paired_summary.csv',dsummary)
 save(out/'COMPLETE.json',dict(complete=True,models=9,new_KD_calibrations=6,CE_replay_checks=3,test_evaluated=False,retrained=False,all_argmax_unchanged=True,protocol_sha256=sha((out/'protocol_LOCKED.json').read_bytes())))
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
 mode=sys.argv[1];out=Path(sys.argv[2]).resolve()
 if mode=='prepare':prepare(out)
 elif mode=='run':run(out)
 else:raise ValueError(mode)
