from pathlib import Path
import sys,json,hashlib,shutil,zipfile,math,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
OUT=BASE/'G9-Access-Objective-Design-2026-10-04-v2'
OUT.mkdir(exist_ok=False)
PACKAGE=BASE/'G8-Exposure-Training-2026-10-04/package'
REVIEW=BASE/'G8-Final-Review-2026-10-04-v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
manifest=json.loads((PACKAGE/'manifest.json').read_text())
for name,digest in manifest['files'].items():
 assert sha((PACKAGE/name).read_bytes())==digest,name
 if name.endswith('.py') or name=='data.json' or name.startswith('examples/'):
  p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PACKAGE/name,p)
shutil.copyfile(Path(__file__).with_name('access_priority_v1.py'),OUT/'source/access_priority_v1.py')
shutil.copyfile(__file__,OUT/'audit_access_priority.py')
save('parent-manifest.json',manifest)
sys.path.insert(0,str(OUT/'source'))
import numpy as np
import torch
from access_priority_v1 import teacher_graph,priority,temporal_loss
from nca.paced_generation import full_origins,eligibility,block_loss,admit
from nca.block_reference import cube_union
from nca.budget_reference import budget
from nca.generation_data import seed_inputs
torch.set_num_threads(2)
started=time.perf_counter();rows=json.loads((PACKAGE/'data.json').read_text())['rows']
assert len(rows)==45 and all(r['split']=='train' for r in rows)
cases=[];cache={};gradient_checks=0
for row in rows:
 with np.load(OUT/'source'/row['arrays'],allow_pickle=False) as a:
  c=a['condition'].copy();target=a['target'].astype(bool);stage_distance=a['block_distance'].copy()
 x=seed_inputs(c);allowed=x['allowed'];field=x['occupancy'].astype(bool)
 graph=teacher_graph(target,c[5].astype(bool),split=row['split'])
 D=int(allowed.sum());B,C=budget(D,float(c[6,0,0,0]),3);quota=max(9,math.ceil((C-27)/63))
 cache[row['id']]=(c,target,stage_distance,x,graph,(D,B,C))
 path=[];states=[field.copy()]
 # All-fire deterministic teacher oracle: label feasibility only, not model performance.
 for step in range(128):
  positive,phase=priority(field,allowed,graph)
  if phase=='connected':break
  assert phase!='no_teacher_route',row['id']
  e,seed=eligibility(field,full_origins(allowed))
  q=np.where(positive,.9,.1).astype(np.float32)
  out,count=admit(field,e,q,np.ones_like(e),C if seed else min(C,int(field.sum())+quota),seed)
  assert out.sum()>field.sum() and not (out&~target).any(),row['id']
  # Zero-logit gradients explicitly reverse the old reward for delayed teacher cubes.
  pos=e&positive;delay=e&full_origins(target)&~positive
  if pos.any() and delay.any() and not seed:
   tensor=lambda a:torch.from_numpy(a)[None,None]
   z=torch.zeros((1,1,*e.shape),requires_grad=True)
   args=(z,tensor(field),tensor(e),tensor(target.astype(np.float32)),tensor(full_origins(target)),seed,D,B,C)
   loss=temporal_loss(*args,tensor(positive),phase)[0]
   grad=torch.autograd.grad(loss,z)[0].numpy()[0,0]
   assert (grad[pos]<0).all() and (grad[delay]>0).all()
   old=block_loss(*args)[1]
   oldgrad=torch.autograd.grad(old,z)[0].numpy()[0,0]
   assert (oldgrad[delay]<0).all()
   gradient_checks+=1
  path.append(dict(step=step+1,phase=phase,mass_before=int(field.sum()),mass_after=int(out.sum()),positive_origins=int(positive.sum())))
  field=out;states.append(field.copy())
 assert phase=='connected',row['id']
 tensor=lambda a:torch.from_numpy(a)[None,None]
 e,seed=eligibility(field,full_origins(allowed));z=torch.zeros((1,1,*e.shape),requires_grad=True)
 args=(z,tensor(field),tensor(e),tensor(target.astype(np.float32)),tensor(full_origins(target)),seed,D,B,C)
 original=block_loss(*args);new=temporal_loss(*args,tensor(np.zeros_like(e)),'connected')
 assert all(torch.equal(a,b) for a,b in zip(original,new))
 p=OUT/'oracle'/row['id'];p.parent.mkdir(exist_ok=True)
 np.savez_compressed(str(p)+'.npz',states=np.stack(states),distance=graph[0])
 rec=dict(case=row['id'],connection_steps=len(path),connection_mass=int(field.sum()),ceiling=C,remaining=C-int(field.sum()),steps=path)
 cases.append(rec)
 print(row['id'],len(path),int(field.sum()),'/',C,flush=True)
save('oracle-cases.json',cases)
starts=[]
for i in range(1,428):
 rec=json.loads((REVIEW/f'import/worker/update-{i:04d}.json').read_text())
 row=rows[rec['row_index']];c,t,dist,x,graph,budget_values=cache[row['id']]
 start=x['occupancy'].astype(bool) if rec['start']['kind']=='seed' else cube_union((dist>=0)&(dist<=rec['start']['depth']))
 assert sha(start.astype(np.uint8).tobytes())==rec['start']['sha256']
 pos,phase=priority(start,x['allowed'],graph)
 starts.append(dict(update=i,case=row['id'],start_sha256=rec['start']['sha256'],phase=phase,positive_origins=int(pos.sum())))
save('frozen-start-audit.json',starts)
# Explicit boundary checks: no label computation on heldout, off-teacher fallback,
# no positive fired => suppress delayed offers; connected delegates identically.
try:teacher_graph(t,c[5].astype(bool),split='reserved')
except ValueError:pass
else:raise AssertionError('Heldout accepted')
z=torch.zeros((1,1,2,2,2),requires_grad=True);m=torch.zeros((1,1,4,4,4),dtype=torch.bool)
e=torch.ones_like(z,dtype=torch.bool);pos=torch.zeros_like(e);pos.flatten()[0]=True
fire=e.clone();fire.flatten()[0]=False
loss=temporal_loss(z,m,fire,m.float(),e,False,64,20,28,pos,'advance_access')[0]
grad=torch.autograd.grad(loss,z)[0];assert grad.flatten()[0]==0 and (grad.flatten()[1:]>0).all()
fallback=temporal_loss(z,m,e,m.float(),e,False,64,20,28,pos,'no_teacher_route')
assert all(torch.equal(a,b) for a,b in zip(fallback,block_loss(z,m,e,m.float(),e,False,64,20,28)))
summary=dict(train_cases=45,oracle_connections=sum(c['remaining']>=0 for c in cases),oracle_max_steps=max(c['connection_steps'] for c in cases),minimum_remaining_mass=min(c['remaining'] for c in cases),gradient_checks=gradient_checks,post_connection_loss_exact_cases=45,frozen_starts=427,start_phases={p:sum(r['phase']==p for r in starts) for p in sorted({r['phase'] for r in starts})},heldout_rejected=True,no_fired_progress_check=True,fallback_loss_exact=True,optimizer_updates=0,model_rollouts=0,heldout_inference=0,elapsed_seconds=time.perf_counter()-started)
save('result.json',summary);print(json.dumps(summary,indent=2))
