from pathlib import Path
from collections import deque
import sys,json,hashlib,shutil,math,time
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
OUT=BASE/'G11-Allocation-Design-2026-10-04'
OUT.mkdir(exist_ok=False)
sys.dont_write_bytecode=True
SRC=BASE/'G10-Final-Review-2026-10-04/source'
sys.path.insert(0,str(SRC))
from nca.paced_generation import full_origins,connected
from nca.generation_data import seed_inputs
from nca.block_reference import cube_union
from nca.budget_reference import budget
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
 (OUT/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
shutil.copyfile(__file__,OUT/'audit.py')
data=json.loads((BASE/'G9-Training-Diagnosis-2026-10-04/input-data.json').read_text())
save('input-data.json',data)
(OUT/'contexts').mkdir();(OUT/'routes').mkdir()
records=[];started=time.perf_counter()
for row in data['rows']:
 assert row['split']=='train'
 raw=(BASE/'G9-Training-Diagnosis-2026-10-04/inputs'/row['arrays']).read_bytes()
 assert sha(raw)==row['arrays_sha256']
 # Only context is used, never target or teacher distance.
 import io
 with np.load(io.BytesIO(raw),allow_pickle=False) as a:c=a['condition'].copy()
 np.savez_compressed(OUT/'contexts'/f"{row['id']}.npz",condition=c)
 x=seed_inputs(c);legal=x['allowed'];seed=x['occupancy'].astype(bool)
 valid=full_origins(legal);points=np.argwhere(c[5])
 east=c[5].astype(bool)&(np.arange(legal.shape[2])[None,None,:]==points[:,2].max())
 goals=valid & ~full_origins(~east)
 distance=np.full(valid.shape,-1,np.int32);distance[goals]=0
 queue=deque(map(tuple,np.argwhere(goals)))
 while queue:
  p=queue.popleft()
  for axis in range(3):
   for sign in [-1,1]:
    q=list(p);q[axis]+=sign;q=tuple(q)
    if all(0<=q[i]<valid.shape[i] for i in range(3)) and valid[q] and distance[q]<0:
     distance[q]=distance[p]+1;queue.append(q)
 indices=np.indices(valid.shape)
 eligible=valid.copy()
 for axis,s in enumerate(x['anchor']):eligible &= (indices[axis]<=s)&(indices[axis]+3>s)
 starts=np.argwhere(eligible & (distance>=0))
 assert len(starts)
 p=min(map(tuple,starts),key=lambda p:(distance[p],p))
 origins=[p]
 while distance[p]>0:
  choices=[]
  for axis in range(3):
   for sign in [-1,1]:
    q=list(p);q[axis]+=sign;q=tuple(q)
    if all(0<=q[i]<valid.shape[i] for i in range(3)) and distance[q]==distance[p]-1:choices.append(q)
  p=min(choices);origins.append(p)
 route=np.zeros_like(legal)
 for p in origins:route[tuple(slice(v,v+3) for v in p)]=True
 assert (route&seed).any() and (route&east).any() and not (route&~legal).any()
 assert connected(route) and np.array_equal(cube_union(full_origins(route)),route)
 D=int(legal.sum());B,C=budget(D,float(c[6,0,0,0]),3)
 # Full route is an actual access witness; coverage deficits are only optimistic voxel lower bounds.
 xx=np.flatnonzero(legal.any(axis=(0,1)));deficits=[]
 for i in range(3):
  lo=xx[0]+(xx[-1]+1-xx[0])*i/3;hi=xx[0]+(xx[-1]+1-xx[0])*(i+1)/3
  part=legal&((np.arange(legal.shape[2])+.5>=lo)&(np.arange(legal.shape[2])+.5<hi))[None,None,:]
  deficits.append(max(0,math.ceil(.08*int(part.sum())-1e-10)-int((part&route).sum())))
 mass=int(route.sum())
 np.savez_compressed(OUT/'routes'/f"{row['id']}.npz",route=route,origins=np.array(origins),distance=distance)
 records.append(dict(case=row['id'],D=D,B=B,C=C,route_mass=mass,route_cubes=len(origins),route_within_cap=mass<=C,remaining=C-mass,coverage_deficit_lower_bound=sum(deficits),route_plus_coverage_lower_bound=mass+sum(deficits),coverage_budget_not_ruled_out=mass+sum(deficits)<=C))
assert len(records)==45
save('cases.json',records)
save('result.json',dict(cases=45,route_within_cap=sum(r['route_within_cap'] for r in records),coverage_budget_not_ruled_out=sum(r['coverage_budget_not_ruled_out'] for r in records),route_mass_range=[min(r['route_mass'] for r in records),max(r['route_mass'] for r in records)],min_spare_after_route=min(r['remaining'] for r in records),wall_seconds=time.perf_counter()-started,teacher_used=False,model_inference=False,paid_training=False,claim='Legal full-cube route feasibility only; coverage bound is necessary not sufficient; no full nine-family acceptance claim'))
# Freeze source dependencies used by this audit.
shutil.copytree(SRC,OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
print((OUT/'result.json').read_text())

