from pathlib import Path
import sys,json,time
import numpy as np
import torch
from torch.nn import functional as F
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=Path(__file__).resolve().parent;PRIOR=BASE/'G2-Final-Review-2026-10-03';PREP=BASE/'G1-Preparation-2026-10-03'
sys.path.insert(0,str(PRIOR/'source'))
from budget_reference import budget,admit,feedback,band_loss
from nca.connected_repair import ConnectedRepair,neighbors6
from nca.repair_training import perceive
from nca.generation_data import seed_inputs
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2)

# Boundary, tie and gradient direction invariants, no optimization.
m=np.zeros((3,3,3),bool);m[1,1,1]=True;e=np.ones_like(m)&~m;q=np.full(m.shape,.7)
b,report=admit(m,e,q,4);assert int(b.sum())==3 and np.array_equal(np.flatnonzero(b),[0,1,2])
b,_=admit(m,e,q,1);assert not b.any()
try:admit(m,e,q,0)
except ValueError:pass
else:raise AssertionError('Overfull start accepted')
for label,count,target,cap,sign in [('under',0,3,4,-1),('over',4,3,4,1),('within',2,2,4,0)]:
 x=torch.zeros(4,requires_grad=True);state=torch.zeros(4);state[:count]=1;eligible=torch.ones(4,dtype=torch.bool)
 # Pure scalar loss probe; eligibility is independent here to exercise band signs.
 loss=band_loss(x,state,eligible,100,target,cap);g=torch.autograd.grad(loss,x)[0]
 assert (bool((g<0).all()) if sign<0 else bool((g>0).all()) if sign>0 else bool((g==0).all()))
for seed in range(20):
 rng=np.random.default_rng(seed);m=rng.random((5,5,5))<.2;e=rng.random(m.shape)<.5;q=rng.random(m.shape);cap=int(m.sum())+seed%8
 born,_=admit(m,e,q,cap);assert not (born&m).any() and not (born&~e).any() and int((m|born).sum())<=cap
data=json.loads((PREP/'dataset.json').read_text());train=[r for r in data['rows'] if r['split']=='train'];admission=[]
for row in train:
 B,C=budget(row['score']['domain_voxels'],row['request']);count=row['score']['occupied_voxels'];assert B<=count<=C
 admission.append(dict(case=row['id'],target=B,ceiling=C,teacher_cells=count))
rows=[r for r in train if '-y2-v24' in r['id']];assert len(rows)==3
save('protocol.json',dict(status='design reference; guard-only G2 diagnostic, not trained G3',cases=[r['id'] for r in rows],steps=64,firing_seed=2101,requests=[.16,.24,.32],pilot_width=3,feedback_used_in_model=False,optimizer_updates=0))
model=ConnectedRepair().float();p=torch.load(PRIOR/'import/worker/checkpoint-0256.pt',map_location='cpu',weights_only=True);model.load_state_dict(p['model']);model.eval()
scenes={r['id']:r['scene'] for r in json.loads((PREP/'split-manifest.json').read_text())['entries']};config=json.loads((PREP/'environment.json').read_text())['config'];records=[]
for row in rows:
 with np.load(PREP/row['arrays'],allow_pickle=False) as a:context=a['context'].copy()
 x=seed_inputs(context);m=torch.from_numpy(x['occupancy'])[None,None].bool();initial=m.clone();hidden=torch.zeros(1,7,*m.shape[2:]);allowed=torch.from_numpy(x['allowed'])[None,None];c=torch.from_numpy(context)[None];features=perceive(c);g=torch.Generator().manual_seed(2101)
 B,C=budget(int(context[0].sum()),row['request']);trace=[]
 with torch.no_grad():
  raw=model.rollout(initial.float(),features,allowed,torch.Generator().manual_seed(2101),64)['field'].numpy()[0,0]
  for step in range(64):
   output=model.last(F.relu(model.first(torch.cat((perceive(torch.cat((m.float(),hidden),1)),features),1))));q=torch.sigmoid(output[:,:1]);fire=torch.rand(m.shape,generator=g)<.5
   eligible=allowed & ~m & neighbors6(m) & fire
   born,report=admit(m.numpy()[0,0],eligible.numpy()[0,0],q.numpy()[0,0],C)
   m=m|torch.from_numpy(born)[None,None];hidden=(hidden+output[:,1:]*fire)*allowed
   trace.append(dict(step=step+1,mass=int(m.sum()),remaining_fraction=feedback(m.numpy(),int(context[0].sum()),B),**report))
   assert int(m.sum())<=C and not (m&~allowed).any() and not (initial&~m).any()
 scene=scenes[row['id'].rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config)
 raw_score,_=evaluate_targets(raw,scene,fields,domain);score,_=evaluate_targets(m.numpy()[0,0],scene,fields,domain)
 r=dict(case=row['id'],target=B,ceiling=C,guard_score=score,raw_score=raw_score,trace=trace);records.append(r);save(row['id']+'.json',r)
 with (OUT/(row['id']+'.npz')).open('xb') as f:np.savez_compressed(f,raw=raw,guarded=m.numpy()[0,0])
 print(row['id'],'mass',int(m.sum()),'band',B,C,'failed',[k for k,v in score['family_pass'].items() if not v],flush=True)
save('result.json',dict(reference_checks_pass=True,random_cases=20,teachers_inside_band=len(admission),teacher_admission=admission,diagnostics=records,trained_G3=False,paid_run=False))
