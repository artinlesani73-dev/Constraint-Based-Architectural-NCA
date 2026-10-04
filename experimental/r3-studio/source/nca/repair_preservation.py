"""NR4 preservation objective; NR1/NR2/NR3 implementations remain unchanged."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from nca.repair_training import balanced_loss
from nca.repair_portable import PortableSession
LOSS={'version':'preservation_bce_v1','extra_negative_weight':0.5,'extra_intact_base_weight':1.0}
SEEDS=(1201,)
SETTINGS={'version':'NR4_preservation_v1','seeds':[1201],'updates':256,'train_steps':16,
 'seconds_per_job':600,'jobs':1,'dataset_run':'20260925T094341Z_316cff241020',
 'loss':LOSS,'threshold':0.5,'review_steps':32,'review_firing_seed':2101,
 'review_split':'validation','review_rows':27,'evaluation_device':'cpu','evaluation_dtype':'float32',
 'initialization':'fresh_same_seed_as_NR3','automatic_retry':False,'test_evaluation':False,
 'gpu_job_requires_approval':True,'drive_operations_require_separate_approval':True}

def preservation_loss(logits,target,allowed,occupancy):
    if occupancy.shape!=target.shape:raise ValueError('Occupancy shape mismatch')
    if not ((occupancy==0)|(occupancy==1)).all() or not ((target==0)|(target==1)).all():raise ValueError('Binary input and target required')
    base=balanced_loss(logits,target,allowed)
    negative=F.softplus(logits[allowed & ~target.bool()]).mean()
    intact=torch.equal(occupancy,target)
    loss=base+LOSS['extra_negative_weight']*negative
    if intact:loss=loss+LOSS['extra_intact_base_weight']*base
    return loss,{'base_bce':float(base.detach()),'negative_bce':float(negative.detach()),'intact':intact}

class PreservationSession(PortableSession):
    def __init__(self,root,rows,identity,**kwargs):
        super().__init__(root,rows,identity,**kwargs)
        self.identity['objective']=deepcopy(LOSS)

    def step(self):
        index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index)
        self.optimizer.zero_grad(set_to_none=True)
        state=self.model.rollout(occupancy,features,allowed,self.firing,16)
        loss,components=preservation_loss(state[:,:1],target,allowed,occupancy)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.synchronize()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite weights')
        self.completed+=1
        row={'update':self.completed,'row_index':index,'loss':float(loss.detach()),'pre_clip_gradient_norm':float(norm)}
        row.update(components)
        self.trace.append(row)
        return row,state.detach().cpu().numpy()[0].copy()

