"""G5: destination-conditioned cube proposals with detached, deterministic CPU admission."""
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from nca.destination_cue import destination_cue
from nca.budget_generation import BudgetNCA
from nca.budget_reference import budget
from nca.block_reference import adjacent_origins,cube_union
from nca.connected_repair import cube_mean
from nca.repair_training import perceive

VERSION='destination_conditioned_block_generation_v1'
COUNT_COLUMNS=['initial_mass','offered_blocks','accepted_blocks','budget_rejected_blocks','redundant_blocks','deferred_seed_blocks','added_voxels']

def full_origins(field):
    shape=tuple(n-2 for n in field.shape)
    result=np.ones(shape,bool)
    for z in range(3):
        for y in range(3):
            for x in range(3):result &= field[z:z+shape[0],y:y+shape[1],x:x+shape[2]]
    return result

def connected(field):
    first=tuple(np.argwhere(field)[0]);seen={first};queue=deque([first])
    while queue:
        p=queue.popleft()
        for a in range(3):
            for sign in (-1,1):
                n=list(p);n[a]+=sign;n=tuple(n)
                if all(0<=n[i]<field.shape[i] for i in range(3)) and field[n] and n not in seen:seen.add(n);queue.append(n)
    return len(seen)==int(field.sum())

def eligibility(field,valid):
    if int(field.sum())==1:
        seed=np.argwhere(field)[0];e=valid.copy()
        for axis,c in enumerate(np.indices(valid.shape)):e &= (c<=seed[axis])&(seed[axis]<c+3)
        return e,True
    full=full_origins(field)
    return valid & ~full & adjacent_origins(full),False

def admit(field,eligible,q,fire,ceiling,seed_phase):
    """Eligibility frozen at step start; overlap counted after each acceptance."""
    if q.shape!=eligible.shape or fire.shape!=eligible.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid proposal')
    offered=np.flatnonzero(eligible & fire & (q>.5))
    order=offered[np.lexsort((offered,-q.ravel()[offered]))]
    out=field.copy();mass=int(field.sum());initial=mass;accepted=rejected=redundant=deferred=0
    for index in order:
        origin=np.unravel_index(index,q.shape);region=tuple(slice(int(v),int(v)+3) for v in origin)
        delta=27-int(out[region].sum())
        if not delta:redundant+=1;continue
        if mass+delta>ceiling:rejected+=1;continue
        out[region]=True;mass+=delta;accepted+=1
        if seed_phase:
            deferred=len(order)-accepted-rejected-redundant;break
    return out,[initial,len(offered),accepted,rejected,redundant,deferred,mass-initial]

def soft_union(logits,eligible,m):
    # Independent-proposal surrogate BEFORE ranking/cap. Stable even at large logits.
    log_empty=-F.softplus(logits)*eligible
    log_empty=F.conv_transpose3d(log_empty,logits.new_ones((1,1,3,3,3)))
    return m.float()+(~m).float()*(-torch.expm1(log_empty))

def block_loss(logits,m,eligible,target,origins,seed_phase,D,B,C):
    zero=logits.sum()*0
    positive=eligible & origins;negative=eligible & ~origins
    front=(F.softplus(-logits[positive]).mean() if positive.any() else zero)
    front=front+(F.softplus(logits[negative]).mean() if negative.any() else zero)
    if seed_phase:return front,front,zero,zero
    soft=soft_union(logits,eligible,m)
    covered=F.conv_transpose3d(eligible.float(),logits.new_ones((1,1,3,3,3)))>0
    windows=F.max_pool3d((covered & ~m).float(),3,1)>0
    delta=(cube_mean(soft)-cube_mean(target)).abs()
    volume=delta[windows].mean() if windows.any() else zero
    count=soft.sum();band=(F.relu(B-count)+F.relu(count-C))/D
    return front+.25*volume+band,front,volume,band

class GuidedNCA(BudgetNCA):
    def __init__(self):
        super().__init__()
        old=self.first;self.first=nn.Conv3d(63,64,1)
        with torch.no_grad():
            self.first.weight[:,:61].copy_(old.weight);self.first.weight[:,61:].zero_();self.first.bias.copy_(old.bias)

    def import_base(self,base):
        # Parent copies identically initialized60core channels; all3added inputs zero.
        super().import_base(base)

    def rollout(self,occupancy,static_features,allowed,generator,steps=64,*,target=None,capture=False):
        if type(steps) is not int or steps<1:raise ValueError('Positive steps required')
        if occupancy.ndim!=5 or occupancy.shape[:2]!=(1,1) or min(occupancy.shape[2:])<3:raise ValueError('Batch1 occupancy required')
        if allowed.shape!=occupancy.shape or allowed.dtype!=torch.bool:raise ValueError('Boolean allowed mask required')
        if not torch.isfinite(occupancy).all() or not ((occupancy==0)|(occupancy==1)).all() or not occupancy.any():raise ValueError('Nonempty binary start required')
        if static_features.shape!=(1,28,*occupancy.shape[2:]) or not torch.isfinite(static_features).all():raise ValueError('Seven perceived static channels required')
        domain=static_features[:,:1];request=static_features[:,6:7]
        if not ((domain==0)|(domain==1)).all() or not torch.equal(domain.bool(),allowed):raise ValueError('Domain differs from allowed')
        if not (request==request.flatten()[0]).all():raise ValueError('Uniform request required')
        field=occupancy[0,0].bool().detach().cpu().numpy().copy();legal=allowed[0,0].cpu().numpy()
        D=int(legal.sum());B,C=budget(D,float(request.flatten()[0]),3)
        if (field&~legal).any() or int(field.sum())>C:raise ValueError('Illegal or over-budget start')
        if not connected(field):raise ValueError('Disconnected start')
        if field.sum()!=1 and not np.array_equal(cube_union(full_origins(field)),field):raise ValueError('Thin start')
        valid=full_origins(legal)
        interfaces=static_features[0,5].detach().cpu().numpy()
        if not np.isin(interfaces,[0,1]).all():raise ValueError('Binary interface channel required')
        cue,_=destination_cue(legal,interfaces.astype(bool))
        cue_tensor=torch.tensor(cue,device=occupancy.device)[None]

        if target is not None:
            if target.shape!=occupancy.shape or not ((target==0)|(target==1)).all() or (target.bool()&~allowed).any() or (occupancy.bool()&~target.bool()).any():raise ValueError('Invalid target')
            if not B<=int(target.sum())<=C:raise ValueError('Teacher outside band')
            label=target[0,0].bool().cpu().numpy()
            origins_np=full_origins(label)
            if not np.array_equal(cube_union(origins_np),label):raise ValueError('Thin teacher')
            origins=torch.as_tensor(origins_np,device=occupancy.device)[None,None]
        tensor=lambda a:torch.as_tensor(a,device=occupancy.device)[None,None]
        m=tensor(field.copy());hidden=torch.zeros_like(occupancy).expand(-1,7,-1,-1,-1).clone()
        losses=[];fronts=[];volumes=[];bands=[];counts=[];births=[];proposals=[];candidates=[]
        for _ in range(steps):
            eligible_np,seed_phase=eligibility(field,valid)
            remaining=occupancy.new_tensor((B-int(field.sum()))/D)
            inputs=torch.cat((perceive(torch.cat((m.float(),hidden),1)),static_features,remaining.expand_as(occupancy),cue_tensor),1)
            output=self.last(F.relu(self.first(inputs)));logits=output[:,:1,1:-1,1:-1,1:-1];q=torch.sigmoid(logits)
            fire=torch.rand(q.shape,generator=generator,device=q.device)<.5
            eligible=tensor(eligible_np)&fire
            if target is not None:
                loss,front,volume,band=block_loss(logits,m,eligible,target,origins,seed_phase,D,B,C)
                losses.append(loss);fronts.append(front);volumes.append(volume);bands.append(band)
            # One synchronized copy of probability and firing; CPU sort/overlap selection is explicit.
            copied=torch.cat((q.detach(),fire.float()),1)[0].cpu().numpy()
            new_field,count=admit(field,eligible_np,copied[0],copied[1].astype(bool),C,seed_phase);counts.append(count)
            if capture:
                births.append(tensor(new_field&~field).cpu());proposals.append(q.detach().cpu())
                candidates.append(tensor(field|cube_union(eligible_np & copied[1].astype(bool) & (copied[0]>.5))).cpu())
            field=new_field;m=tensor(field.copy());hidden=(hidden+output[:,1:]*F.pad(fire.float(),(1,1,1,1,1,1)))*allowed
        result=dict(state=torch.cat((m.float(),hidden),1),field=m,proposal=q,admission_counts=torch.tensor(counts,device=m.device),budget=torch.tensor([D,B,C],device=m.device))
        if target is not None:result.update(loss=torch.stack(losses).mean(),frontier_loss=torch.stack(fronts).mean(),volume_loss=torch.stack(volumes).mean(),band_loss=torch.stack(bands).mean())
        if capture:result.update(births=torch.stack(births),proposals=torch.stack(proposals),pre_admission_candidates=torch.stack(candidates),initial=occupancy.detach().cpu())
        return result

def device_probe(device):
    from nca.block_reference import transition,eligible_origins
    checks=0
    for seed in range(12):
        rng=np.random.default_rng(seed);field=np.zeros((10,10,10),bool);legal=np.ones_like(field)
        field[4,4,4]=True
        if seed%2:field[3:6,3:6,3:6]=True
        q=np.round(rng.random((8,8,8)),1).astype(np.float32);fire=rng.random(q.shape)<.5
        cap=int(field.sum())+[0,26,70][seed%3]
        expected,report=transition(field,legal,q,fire,cap)
        # Exercise the same copy boundary as rollout on the selected device.
        copied=torch.as_tensor(np.stack((q,fire)),device=device).float().cpu().numpy()
        e,s=eligibility(field,full_origins(legal));reference_e,_=eligible_origins(field,legal)
        assert np.array_equal(e,reference_e)
        actual,count=admit(field,e,copied[0],copied[1].astype(bool),cap,s)
        assert np.array_equal(actual,expected)
        assert count==[report['initial_mass'],report['offered_count'],len(report['accepted']),len(report['rejected_budget']),len(report['redundant']),report['deferred_after_first_cube'],report['final_mass']-report['initial_mass']]
        checks+=1
    # CUDA transposed convolution/backward must support the deterministic runtime.
    logits=torch.zeros((1,1,2,2,2),device=device,requires_grad=True);e=torch.ones_like(logits,dtype=torch.bool);m=torch.zeros((1,1,4,4,4),device=device,dtype=torch.bool)
    soft_union(logits,e,m).sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad>0).all()
    return dict(device=str(device),reference_cases=checks,union_backward_finite=True,admission_backend='numpy_cpu',passed=True)
