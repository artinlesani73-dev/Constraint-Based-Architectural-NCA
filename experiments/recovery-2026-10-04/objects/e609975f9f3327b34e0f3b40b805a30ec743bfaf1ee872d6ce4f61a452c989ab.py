"""G9 access-ranking paced cube-proposal generation, using portable completed-update checkpoints."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from nca.connected_repair import ConnectedSession,LOSS as BASE_LOSS
from nca.ranked_generation import RankedNCA,COUNT_COLUMNS
from nca.block_reference import cube_union
LOSS={"frontier_positive":1.0,"frontier_negative":1.0,"volume":.25,"cube":3,"global_band":1.0,"access_ranking":1.0,"ranking_margin":1.0}
from nca.repair_training import perceive
from nca.generation_data import seed_inputs,training_start,generate
VERSION='seed_generation_training_v9_access_ranking'
SEEDS=(1201,)
SETTINGS=dict(version=VERSION,updates=427,train_steps=64,review_steps=64,stability_steps=128,
 seed=1201,seconds=600,seed_start_fraction=.5,teacher_stages='alternating seed / cube union; sha256(update:row) modulo maximum origin distance',
 objective=LOSS,review_firing_seed=2101,automatic_retry=False,architecture="61-64-8",admission="cpu_paced_cube_overlap_v1",quota_rule="max(9,ceil((C-27)/63))",pacing_horizon_constant=64,budget_width=3,proposal_shape=[30,30,30],seed_loss="frontier_only",union_surrogate="independent_before_hard_selection",admission_count_columns=COUNT_COLUMNS)

class GenerationSession(ConnectedSession):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  model=RankedNCA().float().to(self.device);model.import_base(self.model);self.model=model
  self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
  self.identity.update(model_semantics=VERSION,train_steps=64,generation_settings=SETTINGS,objective=LOSS,global_budget_operations=True,admission_is_detached=True)
 def tensors(self,index):
  row=self.rows[index];path=(self.root/row['arrays']).resolve()
  if not path.is_relative_to(self.root.resolve()):raise ValueError('Outside dataset')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['arrays_sha256']:raise ValueError('Changed data')
  with np.load(path,allow_pickle=False) as a:
   c=a['condition'].copy();target=a['target'].copy();distance=a['block_distance'].copy()
  x=seed_inputs(c);depth=0
  if self.completed%2:
   maximum=int(distance.max())
   if maximum<2:raise ValueError('Teacher too small')
   key=hashlib.sha256(f'{self.completed}:{index}'.encode()).digest()
   depth=int.from_bytes(key[:8],'little')%maximum
  seed_start=self.completed%2==0
  start=x['occupancy'].copy() if seed_start else cube_union((distance>=0)&(distance<=depth)).astype(np.float32)
  if not np.all(start[x['occupancy'].astype(bool)]==1):raise ValueError('Seed lost')
  self.last_start=dict(kind='seed' if seed_start else 'cube_teacher_stage',depth=None if seed_start else depth,occupied=int(start.sum()),sha256=hashlib.sha256(start.astype(np.uint8).tobytes(order='C')).hexdigest())
  self.last_start_field=start.astype(np.uint8).copy()
  tensor=lambda a:torch.as_tensor(a,device=self.device,dtype=torch.float32)
  return tensor(start)[None,None],perceive(tensor(c)[None]).detach(),torch.as_tensor(x['allowed'],device=self.device)[None,None],tensor(target)[None,None]
 def step(self):
  index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index);self.optimizer.zero_grad(set_to_none=True)
  r=self.model.rollout(occupancy,features,allowed,self.firing,64,target=target,training_split="train");loss=r['loss']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.synchronize()
  if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Invalid weights')
  self.completed+=1
  row=dict(update=self.completed,row_index=index,loss=float(loss.detach()),frontier_loss=float(r['frontier_loss'].detach()),volume_loss=float(r['volume_loss'].detach()),pre_clip_gradient_norm=float(norm),start=self.last_start,band_loss=float(r["band_loss"].detach()),ranking_loss=float(r["ranking_loss"].detach()),access_phase_trace=r["access_phase_trace"],admission_counts=r["admission_counts"].detach().cpu().tolist(),budget=r["budget"].detach().cpu().tolist(),quota=int(r["quota"].detach().cpu()),step_ceilings=r["step_ceilings"].detach().cpu().tolist())
  self.trace.append(row);return row,r['state'].detach().cpu().numpy()[0].copy()
 def evaluate(self,index=0):
  with np.load(self.root/self.rows[index]['arrays'],allow_pickle=False) as a:c=a['condition'].copy()
  r=generate(self.model,c,2101,64)
  return {k:v.detach().cpu().numpy() for k,v in r.items()}

def equal_tree(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b
