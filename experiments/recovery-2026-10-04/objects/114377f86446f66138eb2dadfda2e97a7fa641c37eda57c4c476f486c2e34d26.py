from pathlib import Path
import os,sys,json,importlib.util
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
OUT=Path(__file__).resolve().parent;ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from nca.connected_repair import step_loss,ConnectedRepair
from nca.generation_package import verify
from nca.generation_training import GenerationSession
from nca.repair_training import perceive
torch.set_num_threads(2)
oldpath=OUT.parent/'G1-Portable-Paths-2026-10-03/verification-package/nca/connected_repair.py'
spec=importlib.util.spec_from_file_location('g1_reference',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
checks=[]
for kind in ('both_classes','positive_only','negative_only','empty'):
 logits=torch.linspace(-2,2,125).reshape(1,1,5,5,5).requires_grad_();m=torch.zeros_like(logits,dtype=torch.bool);target=torch.zeros_like(logits);target[:,:,1:4,1:4,1:4]=1
 eligible=torch.ones_like(m) if kind=='both_classes' else (target.bool() if kind=='positive_only' else (~target.bool() if kind=='negative_only' else torch.zeros_like(m)))
 positive=eligible&target.bool()
 g1,_,v1=old.step_loss(logits,m,eligible,target,False);g2,_,v2=step_loss(logits,m,eligible,target,False)
 expected=.5*F.softplus(-logits[positive]).mean() if positive.any() else logits.sum()*0
 assert torch.allclose(g2-g1,expected,atol=1e-7) and torch.equal(v1,v2)
 a=torch.autograd.grad(g1,logits,retain_graph=True)[0];b=torch.autograd.grad(g2,logits)[0]
 delta=torch.zeros_like(a)
 if positive.any():delta[positive]=.5*(torch.sigmoid(logits.detach()[positive])-1)/int(positive.sum())
 assert torch.allclose(b-a,delta,atol=1e-7) and torch.equal(a[~positive],b[~positive])
 checks.append(kind)
# Identical initialization and forward dynamics: only supervised objective changed.
torch.manual_seed(1201);a=old.ConnectedRepair();torch.manual_seed(1201);b=ConnectedRepair()
assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
a.last.bias.data[0]=.1;b.load_state_dict(a.state_dict())
occ=torch.zeros(1,1,5,5,5);occ[:,:,2,2,2]=1;allowed=torch.ones_like(occ,dtype=torch.bool);features=perceive(torch.zeros(1,7,5,5,5))
with torch.no_grad():
 x=a.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),4)
 y=b.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),4)
assert all(torch.equal(x[k],y[k]) for k in x)
_,data=verify(ROOT);session=GenerationSession(ROOT,data['rows'],{'check':'G2'},device='cpu')
assert session.identity['objective']['frontier_positive']==1.0 and session.identity['generation_settings']['objective']['frontier_positive']==1.0
# A G1 checkpoint must not be admitted as a G2 continuation.
prior=OUT.parent/'G1-Training-2026-10-03/package/generation-runs/20261003T162226Z_0089b332210b/worker/checkpoint-0003.pt'
try:session.restore(prior)
except ValueError:pass
else:raise AssertionError('G1 checkpoint accepted into G2')
r=dict(loss_gradient_cases=checks,positive_gradient_increment_exact=True,negative_and_inactive_gradients_unchanged=True,volume_term_unchanged=True,initial_weights_equal=True,inference_dynamics_equal=True,identity_records_new_weight=True,g1_checkpoint_rejected=True,package_verified=True)
with (OUT/'CHANGE-VERIFICATION.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
