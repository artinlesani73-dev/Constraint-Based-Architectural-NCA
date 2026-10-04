"""G2 positive-balanced seed generation, using portable completed-update checkpoints."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from nca.connected_repair import ConnectedSession,LOSS
from nca.repair_training import perceive
from nca.generation_data import seed_inputs,training_start,generate
VERSION='seed_generation_training_v2_positive_balance'
SEEDS=(1201,)
SETTINGS=dict(version=VERSION,updates=256,train_steps=64,review_steps=64,stability_steps=128,
 seed=1201,seconds=600,seed_start_fraction=.5,teacher_stages='alternating; sha256(update:row) modulo max teacher distance',
 objective=LOSS,review_firing_seed=2101,automatic_retry=False)

class GenerationSession(ConnectedSession):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.identity.update(model_semantics=VERSION,train_steps=64,generation_settings=SETTINGS)
 def tensors(self,index):
  row=self.rows[index];path=(self.root/row['arrays']).resolve()
  if not path.is_relative_to(self.root.resolve()):raise ValueError('Outside dataset')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['arrays_sha256']:raise ValueError('Changed data')
  with np.load(path,allow_pickle=False) as a:
   c=a['condition'].copy();target=a['target'].copy();distance=a['distance'].copy()
  x=seed_inputs(c);depth=0
  if self.completed%2:
   maximum=int(distance.max())
   if maximum<2:raise ValueError('Teacher too small')
   key=hashlib.sha256(f'{self.completed}:{index}'.encode()).digest()
   depth=1+int.from_bytes(key[:8],'little')%(maximum-1)
  start=training_start(distance,depth,'train')
  if depth==0 and not np.array_equal(start,x['occupancy']):raise ValueError('Seed differs')
  self.last_start=dict(kind='seed' if depth==0 else 'teacher_stage',depth=depth,occupied=int(start.sum()))
  tensor=lambda a:torch.as_tensor(a,device=self.device,dtype=torch.float32)
  return tensor(start)[None,None],perceive(tensor(c)[None]).detach(),torch.as_tensor(x['allowed'],device=self.device)[None,None],tensor(target)[None,None]
 def step(self):
  index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index);self.optimizer.zero_grad(set_to_none=True)
  r=self.model.rollout(occupancy,features,allowed,self.firing,64,target=target);loss=r['loss']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.synchronize()
  if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Invalid weights')
  self.completed+=1
  row=dict(update=self.completed,row_index=index,loss=float(loss.detach()),frontier_loss=float(r['frontier_loss'].detach()),volume_loss=float(r['volume_loss'].detach()),pre_clip_gradient_norm=float(norm),start=self.last_start)
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
