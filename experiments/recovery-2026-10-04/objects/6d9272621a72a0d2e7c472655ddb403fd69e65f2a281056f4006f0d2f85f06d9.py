from pathlib import Path
import sys,json,torch,numpy as np,hashlib,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G10-One-Sided-Training-2026-10-04';ROOT=OUT/'package'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from nca.paced_generation import PacedNCA
from nca.ranked_generation import RankedNCA
from nca.generation_data import generate,seed_inputs
from nca.repair_training import perceive
from nca.generation_training import equal_tree
from nca.access_labels import cached_teacher_graph
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
checkpoint=BASE/'G9-Final-Review-2026-10-04/import/worker/checkpoint-0427.pt'
weights=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
old=PacedNCA();new=RankedNCA();old.load_state_dict(weights);new.load_state_dict(weights)
rows=json.loads((ROOT/'data.json').read_text())['rows'];checks=[]
for index in [0,27,36]:
 with np.load(ROOT/rows[index]['arrays'],allow_pickle=False) as a:c=a['condition'].copy();target=a['target'].copy()
 for horizon in [64,128]:
  a=generate(old,c,2101,horizon);b=generate(new,c,2101,horizon)
  assert equal_tree(a,b)
  checks.append(dict(case=rows[index]['id'],horizon=horizon,every_result_tensor_equal=True))
 x=seed_inputs(c);occ=torch.from_numpy(x['occupancy'])[None,None];features=perceive(torch.from_numpy(c)[None]);allowed=torch.from_numpy(x['allowed'])[None,None];t=torch.from_numpy(target.astype(np.float32))[None,None]
 with torch.no_grad():
  a=old.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),64,target=t)
  cached_teacher_graph.cache_clear()
  b=new.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),64,target=t,training_split='train')
 for key in a:
  if key!='loss':assert equal_tree(a[key],b[key]),key
 assert torch.allclose(b['loss'],a['loss']+b['ranking_loss'],rtol=1e-6,atol=1e-6)
 assert len(b['access_phase_trace'])==64
 # Training-only labels do not alter forward transition or firing consumption.
 checks.append(dict(case=rows[index]['id'],base_terms_and_forward_equal=True,loss_is_base_plus_ranking=True,ranking_loss=float(b['ranking_loss'])))
try:new.rollout(occ,features,allowed,torch.Generator().manual_seed(2101),1,target=t,training_split='reserved')
except ValueError as e:assert 'TRAIN' in str(e)
else:raise AssertionError('Reserved supervision accepted')
from nca.access_ranking import access_ranking
from torch.nn import functional as F
import math
for values in [[-1.,.2,.8,2.],[1000.,-1000.,3.,-2.]]:
 z=torch.tensor(values).reshape(1,1,1,1,4).requires_grad_()
 e=torch.ones_like(z,dtype=torch.bool);positive=e.clone();positive[...,2:]=False
 newloss=access_ranking(z,e,e,positive,'advance_access')
 oldloss=F.softplus(1+torch.logsumexp(z[~positive],0)-math.log(2)+torch.logsumexp(-z[positive],0)-math.log(2))
 ng=torch.autograd.grad(newloss,z,retain_graph=True)[0];og=torch.autograd.grad(oldloss,z)[0]
 assert torch.equal(newloss,oldloss) and torch.equal(ng[positive],og[positive]) and (ng[~positive]==0).all()
 assert torch.isfinite(ng).all()
assert access_ranking(z,~e,e,positive,'advance_access')==0
result=dict(semi_gradient_matches_G9_value_and_advancing_gradient=True,other_auxiliary_gradient_zero=True,passed=True,checks=checks,heldout_supervision_rejected=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),optimizer_updates=0,heldout_inference=0,interpretation='TRAIN-only engineering parity,not a G10 quality benchmark')
with (OUT/'integration-check.json').open('x') as f:json.dump(result,f,indent=2)
shutil.copyfile(__file__,OUT/'check-integration.py');print(json.dumps(result,indent=2))

