"""CGR3 proposed TRAIN start-state sampler. Not integrated into training yet."""
import hashlib
import numpy as np
import torch
from nca.connected_repair import neighbors6
VERSION='intermediate_repair_starts_v1'

def training_start(occupancy,target,allowed,*,split,row_hash,visit,seed=1201):
 if split!='train':raise ValueError('TRAIN only')
 if type(visit) is not int or visit<0:raise ValueError('Zero-based row visit required')
 o=np.asarray(occupancy);t=np.asarray(target);a=np.asarray(allowed)
 if o.ndim!=3 or o.shape!=t.shape or o.shape!=a.shape:raise ValueError('Matching 3D arrays required')
 if not all(np.isin(x,[0,1]).all() for x in [o,t,a]):raise ValueError('Binary arrays required')
 if not o.any() or (o.astype(bool)&~t.astype(bool)).any() or (t.astype(bool)&~a.astype(bool)).any():raise ValueError('Legal removal-only pair required')
 m=torch.from_numpy(o.astype(bool))[None,None];truth=torch.from_numpy(t.astype(bool))[None,None]
 record=dict(version=VERSION,visit=visit,mode='original',added=0,stages=0)
 if visit%2==0 or torch.equal(m,truth):return o.astype(np.float32).copy(),record
 token=f'{VERSION}|{seed}|{row_hash}|{visit}'.encode();rng_seed=int.from_bytes(hashlib.sha256(token).digest()[:8],'big')%(2**63-1)
 g=torch.Generator().manual_seed(rng_seed);stages=1+(visit//2)%3
 for _ in range(stages):
  frontier=neighbors6(m)&truth&~m;born=frontier&(torch.rand(m.shape,generator=g)<.5);candidate=m|born
  # Keep damaged augmentation genuinely incomplete, never teach only intact starts.
  if torch.equal(candidate,truth):break
  m=candidate
 result=m[0,0].numpy().astype(np.float32)
 record.update(mode='intermediate' if not np.array_equal(result,o) else 'original_fallback',added=int(result.sum()-o.sum()),stages=stages,rng_seed=rng_seed)
 return result,record
