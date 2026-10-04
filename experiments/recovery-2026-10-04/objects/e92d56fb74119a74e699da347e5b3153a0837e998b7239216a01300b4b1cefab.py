"""G5 static context-only distances on legal overlapping3cube origins."""
from collections import deque
from functools import lru_cache
import numpy as np
from nca.block_generation import full_origins

VERSION='opposite_x_interface_cube_distance_v1'

@lru_cache(maxsize=32)
def _cached(shape,allowed_bytes,interface_bytes):
    allowed=np.frombuffer(allowed_bytes,dtype=bool).reshape(shape)
    interfaces=np.frombuffer(interface_bytes,dtype=bool).reshape(shape)
    points=np.argwhere(interfaces&allowed)
    if len(points)==0 or points[:,2].min()==points[:,2].max():raise ValueError('Two separated X interface planes required')
    # Same orientation as G1 context seed: minimumX is origin,maximumX destination.
    goal=(interfaces&allowed)&(np.arange(shape[2])[None,None,:]==points[:,2].max())
    valid=full_origins(allowed);goals=valid & ~full_origins(~goal)
    if not goals.any():raise ValueError('Destination has no legal full-cube origin')
    distance=np.full(valid.shape,-1,np.int32);distance[goals]=0
    queue=deque(map(tuple,np.argwhere(goals)))
    while queue:
        p=queue.popleft()
        for axis in range(3):
            for sign in (-1,1):
                n=list(p);n[axis]+=sign;n=tuple(n)
                if all(0<=n[i]<valid.shape[i] for i in range(3)) and valid[n] and distance[n]<0:
                    distance[n]=distance[p]+1;queue.append(n)
    reachable=distance>=0;scale=max(1,int(distance.max()))
    cue=np.zeros((2,*shape),np.float32)
    cue[0,1:-1,1:-1,1:-1]=np.where(reachable,distance/scale,0).astype(np.float32)
    cue[1,1:-1,1:-1,1:-1]=reachable.astype(np.float32)
    distance.setflags(write=False);cue.setflags(write=False)
    return cue,distance

def destination_cue(allowed,interfaces):
    allowed=np.asarray(allowed);interfaces=np.asarray(interfaces)
    if allowed.ndim!=3 or allowed.shape!=interfaces.shape or min(allowed.shape)<3 or allowed.dtype!=bool or interfaces.dtype!=bool:raise ValueError('Matching Boolean3D masks required')
    # No labels,requested volume,current occupancy or training history in this key.
    return _cached(allowed.shape,allowed.tobytes(order='C'),interfaces.tobytes(order='C'))

def device_probe(device):
    import torch
    from nca.block_generation import device_probe as block_probe
    result=block_probe(device)
    allowed=np.ones((9,9,9),bool);interfaces=np.zeros_like(allowed)
    interfaces[4,4,0]=True;interfaces[4,4,8]=True
    cue,distance=destination_cue(allowed,interfaces)
    assert np.isfinite(cue).all() and np.array_equal(torch.tensor(cue,device=device).cpu().numpy(),cue)
    assert distance[2,2,6]==0 and distance[2,2,0]==6
    result.update(destination_cue_version=VERSION,cue_device_copy_exact=True)
    return result
