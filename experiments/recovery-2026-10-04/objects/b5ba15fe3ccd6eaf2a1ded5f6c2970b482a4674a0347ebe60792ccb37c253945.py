from pathlib import Path
import sys,os,json
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
OUT=Path(__file__).resolve().parent;ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from nca.budget_generation import BudgetNCA,device_admit,device_probe
from nca.budget_reference import budget,band_loss
from nca.connected_repair import ConnectedRepair,neighbors6
from nca.generation_data import seed_inputs,generate
from nca.repair_training import perceive
from nca.generation_package import verify
from nca.generation_training import GenerationSession
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
probe=device_probe(torch.device('cpu'));_,data=verify(ROOT)
torch.manual_seed(1201);base=ConnectedRepair();torch.manual_seed(1201);model=BudgetNCA()
assert torch.equal(model.first.weight[:,:60],base.first.weight) and torch.equal(model.first.bias,base.first.bias)
assert torch.equal(model.last.weight,base.last.weight) and torch.equal(model.last.bias,base.last.bias)
assert torch.count_nonzero(model.first.weight[:,60:])==0
assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())==64
# Forced proposals exercise several capped steps; preserve connectivity to the seed.
c=torch.zeros(1,7,5,5,5);c[:,:2]=1;c[:,6]=.16;occ=torch.zeros(1,1,5,5,5);occ[:,:,2,2,2]=1;allowed=torch.ones_like(occ,dtype=torch.bool)
with torch.no_grad():model.last.bias[0]=10
with torch.no_grad():r=model.rollout(occ,perceive(c),allowed,torch.Generator().manual_seed(2101),16,capture=True)
D,B,C=r['budget'].tolist();assert (D,B,C)==(125,20,28) and int(r['field'].sum())==C
m=occ.bool()
for born,counts in zip(r['births'],r['admission_counts']):
 assert not (born&m).any() and not (born&~neighbors6(m)).any();m|=born;assert int(m.sum())<=C
assert int(r['admission_counts'][:,3].sum())>0 and int(r['pre_admission_candidates'].sum(dim=(1,2,3,4,5)).max())>C
overfull=allowed.float()
try:model.rollout(overfull,perceive(c),allowed,torch.Generator(),1)
except ValueError:pass
else:raise AssertionError('Overfull start accepted')
# Correct signed loss derivatives, before the detached cap; inactive entries stay zero.
for mass,B,C,sign in [(0,3,4,-1),(4,3,4,1),(2,2,4,0)]:
 logits=torch.zeros(5,requires_grad=True);m=torch.zeros(5);m[:mass]=1;eligible=torch.tensor([1,1,1,1,0],dtype=torch.bool)
 g=torch.autograd.grad(band_loss(logits,m,eligible,100,B,C),logits)[0]
 assert g[-1]==0
 assert bool((g[:4]<0).all()) if sign<0 else bool((g[:4]>0).all()) if sign>0 else bool((g[:4]==0).all())
# Full integrated backward on one genuine TRAIN example, without optimizer update.
with np.load(ROOT/data['rows'][0]['arrays'],allow_pickle=False) as a:context=a['condition'].copy();target=a['target'].astype(np.float32)
x=seed_inputs(context);torch.manual_seed(1201);model=BudgetNCA()
with torch.no_grad():model.last.weight.normal_(0,.01);model.last.bias[0]=.1
inp=torch.from_numpy(x['occupancy'])[None,None];static=perceive(torch.from_numpy(context)[None]);allowed=torch.from_numpy(x['allowed'])[None,None]
r=model.rollout(inp,static,allowed,torch.Generator().manual_seed(2101),4,target=torch.from_numpy(target)[None,None]);r['loss'].backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
extra_grad=float(model.first.weight.grad[:,60:].abs().sum());assert extra_grad>0
assert torch.allclose(r['loss'],r['frontier_loss']+.25*r['volume_loss']+r['band_loss'])
first=generate(model,context,2101,4)['field'];target[:]=0;second=generate(model,context,2101,4)['field'];assert torch.equal(first,second)
session=GenerationSession(ROOT,data['rows'],{'test':'G3'},device='cpu');assert session.model.first.in_channels==61
assert session.identity['objective']['global_band']==1 and session.identity['global_budget_operations']
prior=OUT.parent/'G2-Balanced-Growth-2026-10-03/package/generation-runs/20261003T190403Z_08d5aa94df57/worker/checkpoint-0003.pt'
try:session.restore(prior)
except ValueError:pass
else:raise AssertionError('G2 checkpoint accepted')
result=dict(reference_probe=probe,base_parameters_copied_exactly=True,new_weights_initially_zero=True,added_parameters=64,cap_and_connectivity_pass=True,overfull_start_rejected=True,
 raw_overflow_retained=True,band_gradient_signs_pass=True,integrated_gradients_finite=True,extra_channel_gradient_abs_sum=extra_grad,target_independent_inference=True,g2_checkpoint_rejected=True,optimizer_updates=0)
with (OUT/'INTEGRATION-VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
