from pathlib import Path
from collections import deque
import sys,json,hashlib,time
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=Path(__file__).resolve().parent;PREP=BASE/'G1-Preparation-2026-10-03'
sys.path.insert(0,str(BASE/'G3-Final-Review-2026-10-03/source'))
from nca.massing_targets import cube_supported
from nca.evaluation import flood_fill
from nca.budget_reference import budget
from block_reference import full_origins,cube_union,eligible_origins,transition
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2)
def invariant(before,after,allowed,ceiling):
 assert not (before&~after).any() and not (after&~allowed).any() and int(after.sum())<=ceiling
 if after.sum()>1:assert np.array_equal(cube_supported(after,3),after)
 seed=np.zeros_like(after);seed[tuple(np.argwhere(after)[0])]=True
 assert np.array_equal(flood_fill(after,seed),after)

# Exact overlap accounting:36 original cells,then9 and3 new cells,not9+9.
m=np.zeros((5,5,6),bool);m[:3,:3,:4]=True;allowed=np.ones_like(m);shape=full_origins(m).shape;q=np.zeros(shape);q[0,1,0]=.9;q[0,1,1]=.8
after,trace=transition(m,allowed,q,np.ones(shape,bool),48);invariant(m,after,allowed,48)
assert [x['new_cells'] for x in trace['trace']]==[9,3] and after.sum()==48
# Tight remainder produces an explicit stall, not a clipped cube.
stalled,small=transition(m,allowed,q,np.ones(shape,bool),42);assert np.array_equal(stalled,m) and len(small['rejected_budget'])==2
seed=np.zeros((5,5,5),bool);seed[2,2,2]=True;legal=np.ones_like(seed);q=np.ones((3,3,3))
first,first_trace=transition(seed,legal,q,np.ones_like(q,bool),27);assert first.sum()==27 and len(first_trace['accepted'])==1;invariant(seed,first,legal,27)
blocked=legal.copy();blocked[1:4,1:4,1:4]=False;blocked[2,2,2]=True
no_cube,_=transition(seed,blocked,q,np.ones_like(q,bool),27);assert np.array_equal(no_cube,seed)
too_small,_=transition(seed,legal,q,np.ones_like(q,bool),26);assert np.array_equal(too_small,seed)
try:transition(first,legal,q,np.ones_like(q,bool),26)
except ValueError:pass
else:raise AssertionError('Overfull input accepted')
bad=first.copy();bad[4,4,4]=True
try:eligible_origins(bad,legal)
except ValueError:pass
else:raise AssertionError('Nonbulk state accepted')
data=json.loads((PREP/'dataset.json').read_text());rows=[r for r in data['rows'] if r['split']=='train'];assert len(rows)==27
save('protocol.json',dict(purpose='TRAIN teacher representability and target-guided reachability; not a trained benchmark',rows=[r['id'] for r in rows],steps_cap=64,random_seed=2101,width=3,inference_first_cube='highest fired eligible score, no target',oracle='teacher full-cube mask used as perfect scores ONLY in this audit',heldout_access=False))
results=[];t=time.perf_counter()
for row in rows:
 path=PREP/row['arrays'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['arrays_sha256']
 with np.load(path,allow_pickle=False) as a:target=a['target'].astype(bool);seed=a['seed'].astype(bool);allowed=a['context'][0].astype(bool)
 origins=full_origins(target);assert np.array_equal(cube_union(origins),target)
 # A deterministic TRAIN-only root for full-block curriculum construction.
 roots,_=eligible_origins(seed,allowed);root_candidates=np.argwhere(origins&roots);assert len(root_candidates)
 root=tuple(root_candidates[0]);dist=np.full(origins.shape,-1,np.int16);dist[root]=0;queue=deque([root])
 while queue:
  p=queue.popleft()
  for axis in range(3):
   for sign in (-1,1):
    n=list(p);n[axis]+=sign;n=tuple(n)
    if all(0<=n[i]<origins.shape[i] for i in range(3)) and origins[n] and dist[n]<0:dist[n]=dist[p]+1;queue.append(n)
 reached=(dist>=0);represented=cube_union(reached);assert np.array_equal(represented,target)
 for depth in range(int(dist.max())+1):
  stage=cube_union((dist>=0)&(dist<=depth));invariant(seed,stage,allowed,int(target.sum()))
 B,C=budget(int(allowed.sum()),row['request']);field=seed.copy();rng=np.random.default_rng(2101);history=[]
 for step in range(64):
  next_field,report=transition(field,allowed,origins.astype(float),rng.random(origins.shape)<.5,C)
  invariant(field,next_field,allowed,C);field=next_field;history.append(report)
  if np.array_equal(field,target):break
 record=dict(case=row['id'],target_cells=int(target.sum()),represented_cells=int(represented.sum()),teacher_origin_count=int(origins.sum()),connected_origin_count=int(reached.sum()),curriculum_root_zyx=[int(v) for v in root],max_stage_distance=int(dist.max()),
  oracle_steps=len(history),oracle_complete=bool(np.array_equal(field,target)),oracle_missing=int((target&~field).sum()),ceiling=C,history=history)
 results.append(record);save(row['id']+'.json',record)
 with (OUT/(row['id']+'.npz')).open('xb') as f:np.savez_compressed(f,target_origins=origins,stage_distance=dist,oracle_field=field)
 print(row['id'],record['oracle_complete'],len(history),int(dist.max()),flush=True)
result=dict(synthetic_invariants_pass=True,overlap_new_cells=[9,3],representable_targets=len(results),oracle_complete=sum(r['oracle_complete'] for r in results),max_oracle_steps=max(r['oracle_steps'] for r in results),max_stage_distance=max(r['max_stage_distance'] for r in results),wall_seconds=time.perf_counter()-t,trained_model=False,results=results)
save('result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
