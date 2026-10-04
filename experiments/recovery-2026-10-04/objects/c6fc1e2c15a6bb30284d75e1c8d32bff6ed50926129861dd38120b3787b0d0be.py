from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G4-Block-Training-2026-10-03-v2');ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
from nca.block_generation import BlockNCA,full_origins,eligibility,admit,soft_union,block_loss,device_probe,connected
from nca.block_reference import transition,cube_union
from nca.generation_package import verify
from nca.generation_data import seed_inputs,teacher_distance
from nca.generation_training import GenerationSession,SETTINGS
from nca.repair_training import perceive
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
started=time.perf_counter();checks={};_,data=verify(ROOT)
checks['device_probe']=device_probe('cpu')

# Two half-probability cubes overlap on18voxels: union=.75,edges=.5.
logits=torch.zeros((1,1,1,1,2),requires_grad=True);e=torch.ones_like(logits,dtype=torch.bool);m=torch.zeros((1,1,3,3,4),dtype=torch.bool)
u=soft_union(logits,e,m)
assert torch.all(u[:,:,:,:,0]==.5) and torch.all(u[:,:,:,:,3]==.5)
assert torch.allclose(u[:,:,:,:,1:3],torch.full_like(u[:,:,:,:,1:3],.75))
assert abs(float(u.sum().detach())-22.5)<1e-6
u.sum().backward();assert torch.isfinite(logits.grad).all() and (logits.grad>0).all()
for value in [-100.,100.]:
    logits=torch.full((1,1,1,1,2),value,requires_grad=True);v=soft_union(logits,e,m);v.sum().backward()
    assert torch.isfinite(v).all() and torch.isfinite(logits.grad).all() and (v>=0).all() and (v<=1).all()
checks['analytic_overlap_and_extreme_logits']=True
for bounds,sign in [((30,35),-1),((1,5),1)]:
    logits=torch.zeros((1,1,1,1,2),requires_grad=True)
    _,_,_,band=block_loss(logits,m,e,torch.zeros_like(m).float(),torch.zeros_like(e),False,100,*bounds)
    band.backward();assert (logits.grad*sign>0).all()
logits=torch.zeros((1,1,1,1,2),requires_grad=True);labels=torch.tensor([True,False]).reshape_as(e)
loss,front,volume,band=block_loss(logits,m,e,torch.zeros_like(m).float(),labels,True,100,30,35)
assert float(volume.detach())==float(band.detach())==0 and torch.equal(loss,front)
loss.backward();assert logits.grad.flatten()[0]<0 and logits.grad.flatten()[1]>0
checks['band_gradient_direction_and_seed_bce_only']=True

# Full teacher cube graphs independently recomputed from target and context seed.
for row in data['rows']:
    with np.load(ROOT/row['arrays'],allow_pickle=False) as a:
        c=a['condition'];target=a['target'].astype(bool);d=a['block_distance'];origins=a['target_origins'];seed=seed_inputs(c)['occupancy'].astype(bool)
        assert np.array_equal(origins,full_origins(target)) and np.array_equal(cube_union(origins),target)
        valid,_=eligibility(seed,origins);root=tuple(np.argwhere(valid)[0]);origin_seed=np.zeros_like(origins);origin_seed[root]=True
        assert np.array_equal(teacher_distance(origins,origin_seed),d)
        for depth in [0,int(d.max())//2,int(d.max())]:
            stage=cube_union((d>=0)&(d<=depth));assert connected(stage) and stage[seed].all() and not (stage&~target).any()
checks['train_graphs_recomputed']=27

# Realistic crowded offers and tight capacity; compare optimized adapter to independent reference.
admission_seconds=[]
for i in range(20):
    rng=np.random.default_rng(i);f=np.zeros((32,32,32),bool);f[4:13,4:13,4:13]=True;legal=np.ones_like(f)
    q=np.round(rng.random((30,30,30)),2).astype(np.float32);fire=rng.random(q.shape)<.5;cap=int(f.sum())+(i*7)%140
    expected,report=transition(f,legal,q,fire,cap)
    t=time.perf_counter();eligible,seed_phase=eligibility(f,full_origins(legal));actual,count=admit(f,eligible,q,fire,cap,seed_phase);admission_seconds.append(time.perf_counter()-t)
    assert np.array_equal(actual,expected) and int(actual.sum())<=cap and connected(actual) and np.array_equal(cube_union(full_origins(actual)),actual)
    assert count[1]==sum(count[2:6]) and count[0]+count[6]==int(actual.sum())
checks['realistic_reference_cases']=20
checks['cpu_admission_seconds']={'median':float(np.median(admission_seconds)),'max':max(admission_seconds),'includes_eligibility':True,'gpu_transfer_included':False}

identity={'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'settings':SETTINGS}
s=GenerationSession(ROOT,data['rows'],identity,device='cpu',seed=1201)
occ,features,allowed,target=s.tensors(0)
g=torch.Generator().manual_seed(2101)
t=time.perf_counter();result=s.model.rollout(occ,features,allowed,g,12,target=target,capture=True);result['loss'].backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in s.model.parameters())
field=occ[0,0].numpy().astype(bool)
for born in result['births']:
    field |= born[0,0].numpy()
    assert connected(field) and (field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field))
checks['rollout12_forward_backward_seconds']=time.perf_counter()-t
checks['rollout12_mass']=int(field.sum());checks['finite_model_gradients']=True

def rejected(start):
    try:s.model.rollout(start,features,allowed,torch.Generator().manual_seed(1),1)
    except ValueError:return True
    return False
thin=occ.clone();point=np.argwhere(allowed[0,0].numpy()&~occ[0,0].numpy().astype(bool))[0];thin[(0,0,*point)]=1
assert rejected(thin)
assert rejected(allowed.float())
checks['invalid_starts_rejected']=True
# Distinguish connectivity rejection from connected-but-thin rejection.
small_context=torch.zeros((1,7,10,10,10));small_context[:,:2]=1;small_context[:,6]=.16
small_features=perceive(small_context);small_allowed=torch.ones((1,1,10,10,10),dtype=torch.bool)
for kind in ['thin','disconnected']:
    start=torch.zeros_like(small_allowed,dtype=torch.float32)
    if kind=='thin':start[0,0,4,4,4:6]=1
    else:start[0,0,1:4,1:4,1:4]=1;start[0,0,6:9,6:9,6:9]=1
    try:s.model.rollout(start,small_features,small_allowed,torch.Generator().manual_seed(1),1)
    except ValueError as exc:assert kind.lower() in str(exc).lower()
    else:raise AssertionError('Invalid start accepted')
checks['connected_thin_and_disconnected_cube_union_rejected']=True
old=Path('C:/Users/artin/Documents/Codex/outputs/G3-Final-Review-2026-10-03/import/worker/checkpoint-0256.pt')
try:s.restore(old)
except ValueError:checks['g3_checkpoint_rejected']=True
else:raise AssertionError('Old semantic identity accepted')
checks['elapsed_seconds']=time.perf_counter()-started
(OUT/'implementation-checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
