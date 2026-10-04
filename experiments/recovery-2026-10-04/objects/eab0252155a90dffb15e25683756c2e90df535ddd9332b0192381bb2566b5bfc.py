"""G1 seed-only inference contract and TRAIN-only teacher trajectories."""
from collections import deque
import numpy as np

VERSION = 'seed_generation_data_v1'

def seed_inputs(context):
    c=np.asarray(context,dtype=np.float32)
    if c.ndim!=4 or c.shape[0]!=7 or min(c.shape[1:])<3 or not np.isfinite(c).all():
        raise ValueError('Finite seven-channel 3D context required')
    if not np.isin(c[:6],[0,1]).all() or not np.all(c[6]==c[6,0,0,0]) or not 0<c[6,0,0,0]<=1:
        raise ValueError('Binary context and constant request required')
    allowed=c[0].astype(bool)&c[1].astype(bool)&~c[2].astype(bool)&~c[3].astype(bool)
    coords=np.argwhere(allowed & c[5].astype(bool))
    if not len(coords): raise ValueError('No legal interface seed')
    side=coords[coords[:,2]==coords[:,2].min()]
    anchor=tuple(side[np.argmin(((side-side.mean(axis=0))**2).sum(axis=1))])
    occupancy=np.zeros(allowed.shape,np.float32);occupancy[anchor]=1
    return {'occupancy':occupancy,'context':c.copy(),'allowed':allowed,'anchor':anchor}

def teacher_distance(target, seed):
    target=np.asarray(target)
    if not np.isin(target,[0,1]).all(): raise ValueError('Binary teacher required')
    target=target.astype(bool)
    if target.shape!=seed.shape or np.count_nonzero(seed)!=1 or not np.isin(seed,[0,1]).all():
        raise ValueError('Single binary seed required')
    origin=tuple(np.argwhere(seed)[0])
    if not target[origin]: raise ValueError('Teacher excludes seed')
    d=np.full(target.shape,-1,np.int16);d[origin]=0;q=deque([origin])
    while q:
        p=q.popleft()
        for axis in range(3):
            for sign in (-1,1):
                n=list(p);n[axis]+=sign;n=tuple(n)
                if all(0<=n[i]<target.shape[i] for i in range(3)) and target[n] and d[n]<0:
                    d[n]=d[p]+1;q.append(n)
    if (target & (d<0)).any(): raise ValueError('Teacher disconnected from seed')
    return d

def training_start(distance, depth, split):
    if split!='train': raise ValueError('Teacher-derived starts are TRAIN-only')
    if type(depth) is not int or depth<0: raise ValueError('Nonnegative depth required')
    return ((distance>=0)&(distance<=depth)).astype(np.float32)

def generate(model, context, firing_seed=2101, steps=32):
    # No label argument, no teacher trajectory, no procedural route at inference.
    import torch
    from nca.repair_training import perceive
    x=seed_inputs(context);device=next(model.parameters()).device
    occ=torch.as_tensor(x['occupancy'],device=device)[None,None]
    c=torch.as_tensor(x['context'],device=device)[None]
    allowed=torch.as_tensor(x['allowed'],device=device)[None,None]
    with torch.no_grad():
        return model.rollout(occ,perceive(c),allowed,torch.Generator(device=device).manual_seed(firing_seed),steps)
