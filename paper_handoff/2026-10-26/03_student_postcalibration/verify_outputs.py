"""Independent audit from exported probabilities and saved temperature grid."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(p.read_text(encoding='utf8'))
protocol=read(ROOT/'protocol_LOCKED.json')
for n,h in protocol['inputs_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
fit=read(ROOT/'student_temperature_LOCKED.json');assert fit['protocol_sha256']==hashlib.sha256((ROOT/'protocol_LOCKED.json').read_bytes()).hexdigest()
fitted={r['id']:r for r in fit['models']};count=0;maximum=0.;recomputed={}
for r in rows(ROOT/'per_seed.csv'):
 mid=r['method']+'_'+r['seed'];split=r['split'];variant=r['variant'];rr=rows(ROOT/'outputs'/mid/f'{split}_{variant}.csv')
 assert len(rr)==400 and len({a['sample_id'] for a in rr})==400
 p=np.array([[float(a['p_'+k]) for k in 'ENC'] for a in rr]);y=np.array([int(a['gold']) for a in rr]);pred=p.argmax(1)
 assert np.isfinite(p).all() and (p>0).all() and np.max(abs(p.sum(1)-1))<1e-12
 assert np.array_equal(pred,[int(a['prediction']) for a in rr])
 z=np.array([[float(a['logit_'+k]) for k in 'ENC'] for a in rr]);T=float(r['T']);q=np.exp(z/T-(z/T).max(1,keepdims=True));q/=q.sum(1,keepdims=True)
 assert np.max(abs(p-q))<1e-12 and np.array_equal(z.argmax(1),pred)
 assert T==(1. if variant=='raw' else fitted[mid]['T'])
 cm=np.zeros((3,3),int);np.add.at(cm,(y,pred),1);f=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
 confidence=p.max(1);correct=pred==y;binid=np.minimum((confidence*10).astype(int),9)
 ece=sum(np.mean(binid==b)*abs(correct[binid==b].mean()-confidence[binid==b].mean()) for b in range(10) if (binid==b).any())
 m=dict(macro_f1=f.mean(),accuracy=correct.mean(),f1_E=f[0],f1_N=f[1],f1_C=f[2],nll=-np.log(p[np.arange(400),y]).mean(),ece10=ece,brier=((p-np.eye(3)[y])**2).sum(1).mean())
 for k,v in m.items():maximum=max(maximum,abs(v-float(r[k])));assert abs(v-float(r[k]))<1e-10
 recomputed[r['method'],int(r['seed']),split,variant]=m;count+=400
for mid,r in fitted.items():
 grid=rows(ROOT/'outputs'/mid/'temperature_grid.csv')
 best=min(grid,key=lambda a:(float(a['fit_nll']),abs(math.log(float(a['T'])))))
 assert float(best['T'])==r['T'] and not r['at_boundary']
 with np.load(ROOT/'outputs'/mid/'logits.npz',allow_pickle=False) as a:
  z=a['val_fit'];zs=a['val_select_canonical'];zc=a['val_select_cpu']
 with np.load(ROOT/'inputs/val_fit.npz',allow_pickle=False) as a:y=a['y']
 for g in grid:
  zz=z/float(g['T']);pp=np.exp(zz-zz.max(1,keepdims=True));pp/=pp.sum(1,keepdims=True)
  assert abs(-np.log(pp[np.arange(400),y]).mean()-float(g['fit_nll']))<1e-10
 assert np.array_equal(zs.argmax(1),zc.argmax(1)) and np.max(abs(zs-zc))<1e-4
for r in rows(ROOT/'summary.csv'):
 for k in m:
  vs=[recomputed[r['method'],s,'val_select',r['variant']][k] for s in [20261010,20261011,20261012]]
  assert abs(np.mean(vs)-float(r[k+'_mean']))<1e-10 and abs(np.std(vs,ddof=1)-float(r[k+'_seed_sd']))<1e-10
for r in rows(ROOT/'paired_deltas.csv'):
 a,b=r['comparison'].split(' minus ');am,av=a.split();bm,bv=b.split();s=int(r['seed'])
 for k in m:assert abs(recomputed[am,s,'val_select',av][k]-recomputed[bm,s,'val_select',bv][k]-float(r[k]))<1e-10
# The three known CE post-calibration results must reproduce the prior GPU result table.
for r in rows(ROOT/'inputs/Day2_reference_per_seed.csv'):
 if r['method']!='CE':continue
 variant='raw' if r['variant']=='raw' else 'post_temperature'
 for k in m:assert abs(recomputed['CE',int(r['seed']),'val_select',variant][k]-float(r[k]))<1e-10
result=dict(status='PASS',models=9,exported_prediction_rows= count,unique_examples=800,metrics_max_abs_error=maximum,temperature_grid_minima_verified=9,CE_prior_results_reproduced=True,test_evaluated=False,scope='400 val_fit + 400 val_select, reused developmental data; fixed three seeds; not independent confirmation')
(ROOT/'FINAL_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
