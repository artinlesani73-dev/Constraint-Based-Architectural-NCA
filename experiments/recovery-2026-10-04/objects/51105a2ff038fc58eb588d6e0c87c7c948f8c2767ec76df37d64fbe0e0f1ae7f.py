"""CGR1 constructive repair. Hard births detached; no target in inference."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from nca.repair_training import RepairNCA,perceive
from nca.repair_portable import PortableSession
VERSION='connected_generation_positive_balance_v1'
LOSS={'frontier_positive':1.,'frontier_negative':1.,'intact_negative':1.5,'volume':.25,'cube':3}
SETTINGS={'version':VERSION,'seeds':[1201],'updates':256,'train_steps':32,'seconds_per_job':600,'jobs':1,
 'dataset_run':'20260925T094341Z_316cff241020','loss':LOSS,'threshold':.5,'review_steps':32,
 'review_firing_seed':2101,'review_split':'validation','review_rows':27,'evaluation_device':'cpu',
 'evaluation_dtype':'float32','initialization':'fresh','automatic_retry':False,'test_evaluation':False,
 'gpu_job_requires_approval':True,'drive_operations_require_separate_approval':True}
SEEDS=(1201,)

def neighbors6(m):
 p=F.pad(m.float(),(1,1,1,1,1,1))
 return (p[:,:,:-2,1:-1,1:-1]+p[:,:,2:,1:-1,1:-1]+p[:,:,1:-1,:-2,1:-1]+p[:,:,1:-1,2:,1:-1]+p[:,:,1:-1,1:-1,:-2]+p[:,:,1:-1,1:-1,2:])>0

def cube_mean(x):
 # Same valid 3-cube mean; conv3d uses the deterministic convolution path.
 kernel=x.new_full((1,1,3,3,3),1/27)
 return F.conv3d(x,kernel)

def step_loss(logits,m,eligible,target,intact):
 zero=logits.sum()*0
 positive=eligible & target.bool();negative=eligible & ~target.bool()
 front=(1.*F.softplus(-logits[positive]).mean() if positive.any() else zero)
 front=front+((1.5 if intact else 1.)*F.softplus(logits[negative]).mean() if negative.any() else zero)
 soft=m.float()+eligible.float()*torch.sigmoid(logits)
 windows=F.max_pool3d(eligible.float(),3,1)>0
 delta=(cube_mean(soft)-cube_mean(target)).abs()
 volume=delta[windows].mean() if windows.any() else zero
 return front+.25*volume,front,volume

class ConnectedRepair(RepairNCA):
 def rollout(self,occupancy,static_features,allowed,generator,steps=32,*,target=None,capture=False):
  if type(steps) is not int or steps<1:raise ValueError('Positive integer steps required')
  if occupancy.ndim!=5 or occupancy.shape[1]!=1 or min(occupancy.shape[2:])<3:raise ValueError('One occupancy channel, size>=3')
  if allowed.shape!=occupancy.shape or allowed.dtype!=torch.bool:raise ValueError('Boolean matching allowed mask required')
  if not torch.isfinite(occupancy).all() or not ((occupancy==0)|(occupancy==1)).all():raise ValueError('Binary input required')
  if (occupancy.bool()&~allowed).any():raise ValueError('Input outside legal domain')
  if not occupancy.flatten(1).any(1).all():raise ValueError('Empty input unsupported')
  if occupancy.shape[0]!=1:raise ValueError('Frozen batch size1')
  if static_features.shape!=(1,28,*occupancy.shape[2:]) or not torch.isfinite(static_features).all():raise ValueError('Context differs')
  if target is not None:
   if target.shape!=occupancy.shape or not ((target==0)|(target==1)).all() or (target.bool()&~allowed).any() or (occupancy.bool()&~target.bool()).any():raise ValueError('Removal-only binary target required')
  m=occupancy.bool().detach();hidden=torch.zeros_like(occupancy).expand(-1,7,-1,-1,-1).clone()
  intact=target is not None and torch.equal(occupancy,target)
  losses=[];fronts=[];volumes=[];births=[];proposals=[]
  for _ in range(steps):
   output=self.last(F.relu(self.first(torch.cat((perceive(torch.cat((m.float(),hidden),1)),static_features),1))))
   logits=output[:,:1];q=torch.sigmoid(logits)
   fire=torch.rand(m.shape,generator=generator,device=m.device)<.5
   eligible=allowed & ~m & neighbors6(m) & fire
   if target is not None:
    loss,front,volume=step_loss(logits,m,eligible,target,intact);losses.append(loss);fronts.append(front);volumes.append(volume)
   born=(eligible & (q>.5)).detach()
   if capture:births.append(born.detach().cpu());proposals.append(q.detach().cpu())
   m=m|born;hidden=(hidden+output[:,1:]*fire)*allowed
  result={'state':torch.cat((m.float(),hidden),1),'field':m,'proposal':q}
  if target is not None:result.update(loss=torch.stack(losses).mean(),frontier_loss=torch.stack(fronts).mean(),volume_loss=torch.stack(volumes).mean())
  if capture:result.update(births=torch.stack(births),proposals=torch.stack(proposals),initial=occupancy.detach().cpu())
  return result

class ConnectedSession(PortableSession):
 def __init__(self,root,rows,identity,**kwargs):
  super().__init__(root,rows,identity,**kwargs)
  # Reuse identically initialized parameter tensors, never an old checkpoint.
  model=ConnectedRepair().float().to(self.device);model.load_state_dict(self.model.state_dict());self.model=model
  self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
  self.identity.update(model_semantics=VERSION,objective=deepcopy(LOSS),train_steps=32,hard_birth_gradients=False)
 def step(self):
  index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index);self.optimizer.zero_grad(set_to_none=True)
  r=self.model.rollout(occupancy,features,allowed,self.firing,32,target=target);loss=r['loss']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.synchronize()
  if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite parameters')
  self.completed+=1
  row={'update':self.completed,'row_index':index,'loss':float(loss.detach()),'frontier_loss':float(r['frontier_loss'].detach()),'volume_loss':float(r['volume_loss'].detach()),'pre_clip_gradient_norm':float(norm)}
  self.trace.append(row);return row,r['state'].detach().cpu().numpy()[0].copy()
 def evaluate(self,index=0):
  occupancy,features,allowed,_=self.tensors(index);g=torch.Generator(device=self.device).manual_seed(2101)
  with torch.no_grad():r=self.model.rollout(occupancy,features,allowed,g,32,capture=True)
  return {k:v.detach().cpu().numpy() for k,v in r.items()}
