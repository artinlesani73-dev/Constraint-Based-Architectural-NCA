"""G3 hybrid global-budget/local-growth model; no teacher data at inference."""
import torch
from torch import nn
from torch.nn import functional as F
from nca.connected_repair import ConnectedRepair,neighbors6,step_loss
from nca.repair_training import perceive
from nca.budget_reference import budget,band_loss

VERSION='budget_feedback_generation_v1'

def device_admit(m,eligible,q,ceiling):
    """Stable descending probability; ties keep ascending flat ZYX index."""
    offered=eligible & ~m & (q>.5)
    scores=torch.where(offered,q,torch.full_like(q,-torch.inf)).flatten()
    order=torch.argsort(scores,descending=True,stable=True)
    room=torch.clamp(ceiling-m.sum(),min=0)
    selected=(torch.arange(order.numel(),device=q.device)<room)&offered.flatten()[order]
    born=torch.zeros_like(offered).flatten().scatter(0,order,selected).reshape_as(m)
    return born

def device_probe(device):
    """Device reference check used inside the bounded GPU job before updates."""
    import numpy as np
    from nca.budget_reference import admit
    checks=0
    for seed in range(8):
        rng=np.random.default_rng(seed);m=rng.random((5,5,5))<.2;e=rng.random(m.shape)<.6
        q=np.round(rng.random(m.shape),1).astype(np.float32);cap=int(m.sum())+seed
        expected,_=admit(m,e,q,cap)
        actual=device_admit(torch.from_numpy(m).to(device),torch.from_numpy(e).to(device),torch.from_numpy(q).to(device),cap)
        if not np.array_equal(actual.cpu().numpy(),expected):raise AssertionError('Device admission differs from reference')
        checks+=1
    return {'device':str(device),'reference_cases':checks,'passed':True}

class BudgetNCA(ConnectedRepair):
    def __init__(self):
        super().__init__()
        old=self.first
        self.first=nn.Conv3d(61,64,1)
        with torch.no_grad():
            self.first.weight[:,:60].copy_(old.weight);self.first.weight[:,60:].zero_();self.first.bias.copy_(old.bias)

    def import_base(self,base):
        with torch.no_grad():
            self.first.weight[:,:60].copy_(base.first.weight);self.first.weight[:,60:].zero_();self.first.bias.copy_(base.first.bias)
            self.last.load_state_dict(base.last.state_dict())

    def rollout(self,occupancy,static_features,allowed,generator,steps=64,*,target=None,capture=False):
        if type(steps) is not int or steps<1:raise ValueError('Positive steps required')
        if occupancy.ndim!=5 or occupancy.shape[:2]!=(1,1) or min(occupancy.shape[2:])<3:raise ValueError('Batch1 occupancy required')
        if allowed.shape!=occupancy.shape or allowed.dtype!=torch.bool:raise ValueError('Boolean allowed mask required')
        if not torch.isfinite(occupancy).all() or not ((occupancy==0)|(occupancy==1)).all() or not occupancy.any():raise ValueError('Nonempty binary start required')
        if static_features.shape!=(1,28,*occupancy.shape[2:]) or not torch.isfinite(static_features).all():raise ValueError('Seven perceived static channels required')
        domain=static_features[:,:1]
        if not ((domain==0)|(domain==1)).all() or not torch.equal(domain.bool(),allowed):raise ValueError('Domain must match allowed mask')
        if (occupancy.bool()&~allowed).any():raise ValueError('Illegal start')
        request=static_features[:,6:7]
        if not (request==request.flatten()[0]).all():raise ValueError('Uniform request required')
        D=int(domain.sum());B,C=budget(D,float(request.flatten()[0]),3)
        if int(occupancy.sum())>C:raise ValueError('Start exceeds budget')
        if target is not None:
            if target.shape!=occupancy.shape or not ((target==0)|(target==1)).all() or (target.bool()&~allowed).any() or (occupancy.bool()&~target.bool()).any():raise ValueError('Invalid training label')
            if not B<=int(target.sum())<=C:raise ValueError('Teacher outside budget band')
        m=occupancy.bool().detach();hidden=torch.zeros_like(occupancy).expand(-1,7,-1,-1,-1).clone()
        losses=[];fronts=[];volumes=[];bands=[];counts=[];births=[];proposals=[];candidates=[]
        intact=target is not None and torch.equal(occupancy,target)
        for _ in range(steps):
            remaining=(B-m.float().sum())/D
            inputs=torch.cat((perceive(torch.cat((m.float(),hidden),1)),static_features,remaining.expand_as(occupancy)),1)
            output=self.last(F.relu(self.first(inputs)));logits=output[:,:1];q=torch.sigmoid(logits)
            fire=torch.rand(m.shape,generator=generator,device=m.device)<.5
            eligible=allowed & ~m & neighbors6(m) & fire
            if target is not None:
                local,front,volume=step_loss(logits,m,eligible,target,intact)
                band=band_loss(logits,m,eligible,D,B,C)
                losses.append(local+band);fronts.append(front);volumes.append(volume);bands.append(band)
            offered=(eligible&(q>.5)).detach();born=device_admit(m,eligible,q.detach(),C)
            counts.append(torch.stack((m.sum(),offered.sum(),born.sum(),offered.sum()-born.sum())))
            if capture:
                births.append(born.cpu());proposals.append(q.detach().cpu());candidates.append((m|offered).cpu())
            m=m|born;hidden=(hidden+output[:,1:]*fire)*allowed
        result={'state':torch.cat((m.float(),hidden),1),'field':m,'proposal':q,'admission_counts':torch.stack(counts),'budget':torch.tensor([D,B,C],device=m.device)}
        if target is not None:result.update(loss=torch.stack(losses).mean(),frontier_loss=torch.stack(fronts).mean(),volume_loss=torch.stack(volumes).mean(),band_loss=torch.stack(bands).mean())
        if capture:result.update(births=torch.stack(births),proposals=torch.stack(proposals),pre_admission_candidates=torch.stack(candidates),initial=occupancy.detach().cpu())
        return result
