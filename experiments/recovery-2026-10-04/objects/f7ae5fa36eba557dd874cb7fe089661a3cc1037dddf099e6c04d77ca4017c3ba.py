from pathlib import Path
from copy import deepcopy
from collections import deque
import sys,json,hashlib,shutil,time,traceback
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G6-Paced-Growth-2026-10-04';REVIEW=BASE/'G6-Reserved-Review-2026-10-04';PREP=BASE/'G1-Preparation-2026-10-03';OUT=BASE/'G7-Vertical-Training-2026-10-04'
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'prepare-data.py')
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
shutil.copytree(REVIEW/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(PREP/'anchored_teacher.py',OUT/'source/anchored_teacher.py')
shutil.copyfile(PREP/'environment.json',OUT/'environment.json')
sys.path.insert(0,str(OUT/'source'));sys.dont_write_bytecode=True
from nca.massing_cases import target_context
from nca.repair_benchmark import condition,context_hash
from nca.generation_data import seed_inputs,teacher_distance
from nca.massing_targets import evaluate_targets
from nca.block_reference import full_origins,cube_union
from nca.budget_reference import budget
from anchored_teacher import generate_anchored_mass,CoverageGeneratorSpec
config=json.loads((OUT/'environment.json').read_text())['config'];prior=json.loads((PREP/'split-manifest.json').read_text())
base=next(e['scene'] for e in prior['entries'] if e['id']=='g1-aligned-y2')
entries=[];seen={e['context_sha256'] for e in prior['entries']}
# Fixed before label construction; no model inference on the new reserved scenes.
designs={'train':[(8,12,18,28,15),(12,8,28,18,15),(10,16,20,28,15),(16,10,28,20,15),(14,14,20,26,15),(16,16,26,22,15)],'reserved':[(9,15,19,27,14),(15,9,27,19,16),(12,18,22,29,14),(18,12,29,22,16)]}
for split,design in designs.items():
 for i,(west,east,wh,eh,y) in enumerate(design):
  scene=deepcopy(base);sid=f'g7-vertical-{split}-{i}';scene['scene_id']=sid;scene['description']='G7 frozen vertical context '+split
  scene['buildings'][0]['z'][1]=wh;scene['buildings'][1]['z'][1]=eh
  for e,z in zip(scene['entrances'],[west,east]):e['z']=z;e['y']=y
  f,d,_=target_context(scene,config);h=context_hash(scene,f,d);assert h not in seen;seen.add(h)
  entries.append(dict(id=sid,split=split,scene=scene,context_sha256=h))
save('split-manifest.json',dict(version='g7_vertical_distribution_v1',requests=[.16,.24,.32],entries=entries,original_train_contexts=9,original_train_rows=27,new_train_contexts=6,new_train_rows=18,total_train_rows=45,reserved_rows=12,teacher_seed=0,frozen_before_labels=True,reserved_model_inference=False,legacy_regression='9 reused G1 development +12 consumed G6 reserved requests;never added to training'))
save('RESUME.json',dict(status='Frozen G7 split;preparing all18newTRAIN teachers',previous=str(REVIEW/'RESUME.json'),next='Inspect data-preparation.json before packaging;never omit failed labels',paid_run_authorized=False,repository_sync_pending=True))
rows=[]
for entry in entries:
 if entry['split']!='train':continue
 scene=entry['scene'];f,d,_=target_context(scene,config)
 for request in [.16,.24,.32]:
  case=entry['id']+f'-v{round(100*request)}';tick=time.perf_counter();row=dict(id=case,split='train',request=request,context_sha256=entry['context_sha256'],admissible=False)
  try:
   c=condition(scene,f,d,request);x=seed_inputs(c);target,route,report=generate_anchored_mass(scene,f,d,0,x['anchor'],CoverageGeneratorSpec(target_fraction=request,max_seconds=15))
   score,_=evaluate_targets(target,scene,f,d);origins=full_origins(target);assert np.array_equal(cube_union(origins),target)
   coords=np.argwhere(origins);anchor=np.array(x['anchor']);roots=coords[((coords<=anchor)&(anchor<coords+3)).all(1)];assert len(roots)
   root=tuple(roots[0]);dist=np.full(origins.shape,-1,np.int16);dist[root]=0;q=deque([root])
   while q:
    p=q.popleft()
    for axis in range(3):
     for sign in [-1,1]:
      n=list(p);n[axis]+=sign;n=tuple(n)
      if all(0<=n[a]<origins.shape[a] for a in range(3)) and origins[n] and dist[n]<0:dist[n]=dist[p]+1;q.append(n)
   assert np.all(dist[origins]>=0) and dist.max()>=2
   D=int(d.sum());B,C=budget(D,request,3)
   # Every origin-BFS stage is connected by construction and must contain the seed.
   for depth in range(int(dist.max())):
    start=cube_union((dist>=0)&(dist<=depth));assert start[tuple(anchor)] and np.all(~start|target) and start.sum()<=C
   path=OUT/f'examples/{case}.npz';path.parent.mkdir(exist_ok=True)
   with path.open('xb') as out:np.savez_compressed(out,condition=c,target=target,seed=x['occupancy'],distance=teacher_distance(target,x['occupancy']),block_distance=dist,target_origins=origins,route=route)
   row.update(arrays=f'examples/{case}.npz',arrays_sha256=sha(path.read_bytes()),score=score,teacher=report,budget=[D,B,C],stage_depth_max=int(dist.max()),admissible=bool(score['contract_pass'] and B<=int(target.sum())<=C))
  except Exception:row['error']=traceback.format_exc()
  row['seconds']=time.perf_counter()-tick;save(f'examples/{case}.json',row);rows.append(row)
  print(case,row['admissible'],row.get('error',''),flush=True)
save('data-preparation.json',dict(rows=rows,count=len(rows),admitted=sum(r['admissible'] for r in rows),ready=len(rows)==18 and all(r['admissible'] for r in rows),reserved_teachers=0,reserved_inference=0))
print('Labels ready:',all(r['admissible'] for r in rows))
