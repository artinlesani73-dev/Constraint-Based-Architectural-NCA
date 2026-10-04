from pathlib import Path
import json, hashlib, shutil
import numpy as np
ROOT=Path('C:/Users/artin/Documents/Codex/outputs')
SRC=ROOT/'G11-R3-Independent-Review-2026-10-04'
OUT=ROOT/'G11-R3-Preview-2026-10-04'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SRC/'milestone-manifest.json').read_text())['files']
used={}
def read(p):
 rel=p.relative_to(SRC).as_posix();assert sha(p)==manifest[rel];used[rel]=manifest[rel];return p
r=json.loads(read(SRC/'result.json').read_text())
scenes={e['id']:e['scene'] for e in json.loads(read(SRC/'scene-index.json').read_text())['entries']}
cases=[]
for case in dict.fromkeys(o['case'] for o in r['observations']):
 records=[o for o in r['observations'] if o['case']==case]
 item=dict(id=case,cohort=records[0]['cohort'],scene=scenes[records[0]['scene']],request=records[0]['request'],outputs={})
 with np.load(read(SRC/'cases'/case/'hybrid-trajectory.npz')) as a: provenance=a['provenance'].copy()
 with np.load(read(SRC/'cases'/case/'R3-128.npz')) as a: seed=a['field'].astype(bool)&~(provenance>0).any(0)
 assert seed.sum()==1
 for o in records:
  model=o['model'];step=o['steps']
  with np.load(read(SRC/'cases'/case/f'{model}-{step}.npz')) as a:f=a['field'].astype(bool)
  ids=np.flatnonzero(f).tolist()
  labels=np.max(provenance[:step],axis=0).reshape(-1)[ids].tolist() if model=='R3' else [0 if seed.reshape(-1)[i] else 1 for i in ids]
  assert len(ids)==o['score']['occupied_voxels']
  if model=='R3':assert labels.count(2)==o['planner_voxels'] and labels.count(1)==o['learned_voxels']
  item['outputs'][f'{model}-{step}']=dict(voxels=ids,labels=labels,record=o)
 cases.append(item)
cases.sort(key=lambda c:(c['cohort']!='fresh',c['id']))
(OUT/'data.js').write_text('const DATA='+json.dumps(cases,separators=(',',':'))+';',encoding='utf-8')
(OUT/'export-verification.json').write_text(json.dumps(dict(cases=len(cases),outputs=sum(len(c['outputs']) for c in cases),source=str(SRC),source_hashes=used,checks=['source hashes','exact voxel counts','exact planner and learned birth counts']),indent=2),encoding='utf-8')
shutil.copyfile(Path(__file__),OUT/'build_r3_preview.py')
print(OUT)
