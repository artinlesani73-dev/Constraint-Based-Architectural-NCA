from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G5-Destination-Guidance-2026-10-04';ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
from nca.destination_cue import destination_cue,_cached,device_probe
from nca.block_generation import full_origins,eligibility,connected
from nca.block_reference import cube_union,adjacent_origins
from nca.generation_data import seed_inputs
from nca.generation_training import GenerationSession,SETTINGS
from nca.generation_package import verify
from nca.budget_reference import budget
sha=lambda b:hashlib.sha256(b).hexdigest()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.perf_counter();results={};_,data=verify(ROOT)

def reference_distance(allowed,interfaces):
    valid=full_origins(allowed);coords=np.argwhere(allowed&interfaces)
    goals=valid & ~full_origins(~(interfaces&allowed&(np.arange(allowed.shape[2])[None,None,:]==coords[:,2].max())))
    d=np.full(valid.shape,-1,np.int32);d[goals]=0;front=goals.copy();depth=0
    while front.any():
        front=adjacent_origins(front)&valid&(d<0);depth+=1;d[front]=depth
    return d

allowed=np.ones((9,9,9),bool);interfaces=np.zeros_like(allowed);interfaces[4,4,0]=True;interfaces[4,4,8]=True
cue,d=destination_cue(allowed,interfaces);z,y,x=np.indices(d.shape)
analytic=6-x+np.maximum(2-z,0)+np.maximum(z-4,0)+np.maximum(2-y,0)+np.maximum(y-4,0)
assert np.array_equal(d,analytic)
blocked=allowed.copy();blocked[:,:,4]=False
bc,bd=destination_cue(blocked,interfaces);assert (bd[:,:,0:2]==-1).all() and (bc[1,1:-1,1:-1,1:3]==0).all()
assert np.array_equal(bd,reference_distance(blocked,interfaces))
try:cue[0,0,0,0]=99
except ValueError:results['cache_values_immutable']=True
else:raise AssertionError('Cache writable')
cached=cue.copy();_cached.cache_clear();assert np.array_equal(destination_cue(allowed,interfaces)[0],cached)
for invalid in [np.zeros_like(interfaces),interfaces& (np.arange(9)[None,None,:]==0)]:
    try:destination_cue(allowed,invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid interfaces accepted')
results.update(analytic_graph_distance=True,unreachable_distinct_from_goal=True,cache_clear_reproducible=True,invalid_interfaces_rejected=True,device_probe=device_probe('cpu'))

cases=[]
for row in data['rows']:
    with np.load(ROOT/row['arrays'],allow_pickle=False) as a:c=a['condition'].copy()
    seed=seed_inputs(c);legal=seed['allowed'];interfaces=c[5].astype(bool)
    _cached.cache_clear();t=time.perf_counter();cue,d=destination_cue(legal,interfaces);seconds=time.perf_counter()-t
    assert np.array_equal(d,reference_distance(legal,interfaces)) and np.isfinite(cue).all() and ((cue>=0)&(cue<=1)).all()
    valid=full_origins(legal);e,_=eligibility(seed['occupancy'].astype(bool),valid)
    options=np.argwhere(e&(d>=0));assert len(options)
    root=tuple(min(map(tuple,options),key=lambda p:(d[p],p)));current=root;path=[current]
    while d[current]>0:
        neighbors=[]
        for axis in range(3):
            for sign in [-1,1]:
                n=list(current);n[axis]+=sign;n=tuple(n)
                if all(0<=n[i]<d.shape[i] for i in range(3)) and d[n]==d[current]-1:neighbors.append(n)
        assert neighbors;current=min(neighbors);path.append(current)
    route=np.zeros_like(d,dtype=bool)
    for p in path:route[p]=True
    witness=cube_union(route);D=int(legal.sum());B,C=budget(D,float(c[6,0,0,0]),3)
    coords=np.argwhere(legal&interfaces);west=interfaces&(np.arange(legal.shape[2])[None,None,:]==coords[:,2].min());east=interfaces&(np.arange(legal.shape[2])[None,None,:]==coords[:,2].max())
    assert connected(witness) and not (witness&~legal).any() and witness[seed['occupancy'].astype(bool)].all()
    assert (witness&west).any() and (witness&east).any() and witness.sum()<=C
    with (OUT/(row['id']+'-cue.npz')).open('xb') as f:np.savez_compressed(f,cue=cue,distance=d,witness=witness,route_origins=route)
    cases.append(dict(id=row['id'],cold_compute_seconds=seconds,maximum_graph_distance=int(d.max()),root_distance=int(d[root]),reachable_origins=int((d>=0).sum()),legal_origins=int(valid.sum()),witness_voxels=int(witness.sum()),ceiling=C,witness_is_context_only=True,witness_is_not_model_output=True))
results['train_cases']=cases;results['all27_contexts_have_cube_route_under_cap']=True
results['median_cold_cue_seconds']=float(np.median([c['cold_compute_seconds'] for c in cases]))

identity={'manifest_sha256':sha((ROOT/'manifest.json').read_bytes()),'settings':SETTINGS}
session=GenerationSession(ROOT,data['rows'],identity,device='cpu',seed=1201)
old_files=list((BASE/'G4-Block-Training-2026-10-03-v2/package/generation-runs').glob('*/worker/checkpoint-0000.pt'));assert len(old_files)==1
old=torch.load(old_files[0],map_location='cpu',weights_only=False)['model'];new=session.model.state_dict()
assert torch.equal(new['first.weight'][:,:61],old['first.weight']) and torch.count_nonzero(new['first.weight'][:,61:])==0
assert all(torch.equal(new[k],old[k]) for k in old if k!='first.weight')
results['fresh_core_parameters_equal_g4']=True;results['cue_weights_initially_zero']=True
occ,features,allowed,target=session.tensors(0)
# Deliberately nonzero output weights solely for a synthetic gradient probe,not a retained update.
with torch.no_grad():session.model.last.weight.fill_(.02)
r=session.model.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),12,target=target)
r['loss'].backward();grad=session.model.first.weight.grad[:,61:]
assert torch.isfinite(grad).all() and (grad.abs().sum((0,2,3,4))>0).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in session.model.parameters())
results['both_cue_channels_receive_finite_nonzero_gradients']=True
results['synthetic_gradient_probe_optimizer_updates']=0
try:session.restore(old_files[0])
except ValueError:results['g4_checkpoint_identity_rejected']=True
else:raise AssertionError('Old checkpoint accepted')
results['elapsed_seconds']=time.perf_counter()-started
(OUT/'implementation-checks.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v for k,v in results.items() if k!='train_cases'},indent=2))
