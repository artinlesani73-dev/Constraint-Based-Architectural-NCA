"""CGR3: CGR1 math with TRAIN-only intermediate start states."""
from copy import deepcopy
import hashlib
import numpy as np
import torch
from nca.connected_repair import ConnectedSession as BaseSession,ConnectedRepair,LOSS,SETTINGS as BASE_SETTINGS
from nca.repair_curriculum import training_start,VERSION as START_VERSION
from nca.repair_portable import read_portable
VERSION='curriculum_constructive_repair_v1'
SETTINGS=deepcopy(BASE_SETTINGS)
SETTINGS.update(version=VERSION,start_sampler=START_VERSION)
SEEDS=(1201,)

class ConnectedSession(BaseSession):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.identity.update(model_semantics=VERSION,start_sampler=START_VERSION)
  self.visits=[0]*len(self.rows);self.last_start=None
 def step(self):
  index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index)
  raw,meta=training_start(occupancy.cpu().numpy()[0,0],target.cpu().numpy()[0,0],allowed.cpu().numpy()[0,0],split=self.rows[index]['split'],row_hash=self.rows[index]['arrays_sha256'],visit=self.visits[index],seed=self.identity['seed'])
  occupancy=torch.from_numpy(raw)[None,None].to(self.device);self.last_start=raw.copy()
  meta=dict(meta,rng_seed=meta.get('rng_seed'),occupancy_sha256=hashlib.sha256(raw.astype('<f4').tobytes(order='C')).hexdigest())
  self.optimizer.zero_grad(set_to_none=True)
  r=self.model.rollout(occupancy,features,allowed,self.firing,32,target=target);loss=r['loss']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.synchronize()
  if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite parameters')
  self.visits[index]+=1;self.completed+=1
  row=dict(update=self.completed,row_index=index,loss=float(loss.detach()),frontier_loss=float(r['frontier_loss'].detach()),volume_loss=float(r['volume_loss'].detach()),pre_clip_gradient_norm=float(norm),start=meta)
  self.trace.append(row);return row,r['state'].detach().cpu().numpy()[0].copy()
 @staticmethod
 def validate_visits(visits,trace,completed,size):
  if len(visits)!=size or any(type(v) is not int or v<0 for v in visits):raise ValueError('Invalid visit counters')
  counts=[0]*size
  for row in trace:
   i=row['row_index']
   if type(i) is not int or not 0<=i<size or row['start']['visit']!=counts[i]:raise ValueError('Start visit history differs')
   counts[i]+=1
  if counts!=visits or sum(visits)!=completed:raise ValueError('Visit cursor differs')
 def payload(self):
  self.validate_visits(self.visits,self.trace,self.completed,len(self.rows))
  p=super().payload();p['start_visits']=self.visits.copy();return p
 def restore(self,path):
  p=read_portable(path,self.identity)
  self.validate_visits(p['start_visits'],p['trace'],p['completed'],len(self.rows))
  super().restore(path);self.visits=p['start_visits'].copy();self.last_start=None
 # Inherit original tensors/evaluate unchanged: curriculum is exclusively in step.
