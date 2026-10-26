"""Verify frozen evaluation inputs without inference.
python verify_bundle.py
python verify_bundle.py --archives /folder/containing/the/two/original/ZIPs
"""
from pathlib import Path
import json,hashlib,csv,argparse,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
parser=argparse.ArgumentParser();parser.add_argument('--archives',type=Path);args=parser.parse_args()
hashes=j(ROOT/'PACKAGE_SHA256.json')
for name,h in hashes.items():assert sha((ROOT/name).read_bytes())==h,name
protocol=j(ROOT/'evaluation_protocol_LOCKED.json')
for name,h in protocol['referenced_files_sha256'].items():assert sha((ROOT/name).read_bytes())==h,name
assert sha((ROOT/'models_LOCKED.json').read_bytes())==protocol['model_registry']['sha256']
manifests={p.stem:rows(p) for p in (ROOT/'manifests').glob('*.jsonl')}
image_sets={s:{str(r['image_id']) for r in rr} for s,rr in manifests.items()}
for a,sa in image_sets.items():
 for b,sb in image_sets.items():
  if a!=b:assert not sa&sb,(a,b)
for ds in protocol['datasets']:
 split=ds['split'];b=(ROOT/ds['features_file']).read_bytes();assert sha(b)==ds['npz_sha256']
 meta=j(ROOT/f'features/{split}_meta.json')
 assert sha((ROOT/ds['manifest_file']).read_bytes())==meta['manifest_sha256']==ds['manifest_sha256']
 with np.load(ROOT/ds['features_file'],allow_pickle=False) as a:
  x,y,ids,image_ids=(a[k] for k in ['x','y','ids','image_ids'])
  rr=manifests[split];assert x.shape==(ds['rows'],2048) and x.dtype==np.float32 and np.isfinite(x).all()
  assert ids.tolist()==[r['sample_id'] for r in rr] and y.tolist()==[r['label'] for r in rr]
  assert image_ids.astype(str).tolist()==[str(r['image_id']) for r in rr]
  assert np.max(abs(x[:,1024:1536]-abs(x[:,:512]-x[:,512:1024])))<1e-6
  assert np.max(abs(x[:,1536:]-x[:,:512]*x[:,512:1024]))<1e-6
for d in j(ROOT/'bootstrap/draws_LOCKED.json'):
 assert sha((ROOT/d['draws_file']).read_bytes())==d['sha256']
 with np.load(ROOT/d['draws_file'],allow_pickle=False) as a:
  ids,indices=a['image_ids'],a['indices'];assert ids.tolist()==sorted(image_sets[d['split']])
  assert indices.shape==(10000,len(ids)) and indices.min()>=0 and indices.max()<len(ids)
  assert sha(indices.tobytes(order='C'))==d['index_array_sha256']
models=j(ROOT/'models_LOCKED.json');assert len(models)==21 and len({m['checkpoint_sha256'] for m in models})==21
assert sum(m['student_temperature'] is not None for m in models)==15
for m in models:
 with (ROOT/'replay_reference'/f'{m["model_id"]}.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
 assert len(rr)==400
 assert {r['sample_id']:int(r['gold']) for r in rr}=={r['sample_id']:r['label'] for r in manifests['val_select']}
 assert all(int(r['prediction'])==max(range(3),key=lambda i:float(r['logit_'+'ENC'[i]])) for r in rr)
if args.archives:
 archives={n:zipfile.ZipFile(args.archives/n) for n in {m['checkpoint_archive'] for m in models}}
 try:
  for m in models:assert sha(archives[m['checkpoint_archive']].read(m['checkpoint_entry']))==m['checkpoint_sha256'],m['model_id']
 finally:
  for z in archives.values():z.close()
print(json.dumps(dict(status='PASS',files=len(hashes),checkpoints=21,checkpoint_bytes_checked=bool(args.archives),temperatures=15,evaluation_rows=1200,stage2_evaluation_run=False),ensure_ascii=False,indent=2))
