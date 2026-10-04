"""NR5 fixed32-step training: identical NR4 loss, architecture and optimizer."""
from copy import deepcopy
import torch
from nca.repair_portable import PortableSession
from nca.repair_preservation import LOSS,preservation_loss,SETTINGS as NR4_SETTINGS
SETTINGS=deepcopy(NR4_SETTINGS)
SETTINGS.update(version='NR5_horizon32_v1',train_steps=32,initialization='fresh_same_seed_as_NR4')
SEEDS=(1201,)

class HorizonSession(PortableSession):
    def __init__(self,root,rows,identity,**kwargs):
        super().__init__(root,rows,identity,**kwargs)
        self.identity['objective']=deepcopy(LOSS)
        self.identity['train_steps']=32

    def step(self):
        index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index)
        self.optimizer.zero_grad(set_to_none=True)
        state=self.model.rollout(occupancy,features,allowed,self.firing,32)
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


    def evaluate(self,index=0):
        occupancy,features,allowed,target=self.tensors(index)
        generator=torch.Generator(device=self.device).manual_seed(2101)
        with torch.no_grad():
            state=self.model.rollout(occupancy,features,allowed,generator,32)
            probability=torch.sigmoid(state[:,:1])*allowed
        self.synchronize()
        return {'state':state.cpu().numpy()[0].copy(),'probability':probability.cpu().numpy()[0,0].copy()}

