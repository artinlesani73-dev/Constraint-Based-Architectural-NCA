from collections import deque
import numpy as np
from nca.generation_data import seed_inputs
from nca.paced_generation import full_origins,connected
from nca.block_reference import cube_union

def route_from_context(c):
 x=seed_inputs(c);legal=x['allowed'];seed=x['occupancy'].astype(bool)
 valid=full_origins(legal);points=np.argwhere(c[5])
 east=c[5].astype(bool)&(np.arange(legal.shape[2])[None,None,:]==points[:,2].max())
 goals=valid & ~full_origins(~east)
 distance=np.full(valid.shape,-1,np.int32);distance[goals]=0
 queue=deque(map(tuple,np.argwhere(goals)))
 while queue:
  p=queue.popleft()
  for axis in range(3):
   for sign in [-1,1]:
    q=list(p);q[axis]+=sign;q=tuple(q)
    if all(0<=q[i]<valid.shape[i] for i in range(3)) and valid[q] and distance[q]<0:
     distance[q]=distance[p]+1;queue.append(q)
 indices=np.indices(valid.shape)
 eligible=valid.copy()
 for axis,s in enumerate(x['anchor']):eligible &= (indices[axis]<=s)&(indices[axis]+3>s)
 starts=np.argwhere(eligible & (distance>=0))
 if not len(starts):return None
 p=min(map(tuple,starts),key=lambda p:(distance[p],p))
 origins=[p]
 while distance[p]>0:
  choices=[]
  for axis in range(3):
   for sign in [-1,1]:
    q=list(p);q[axis]+=sign;q=tuple(q)
    if all(0<=q[i]<valid.shape[i] for i in range(3)) and distance[q]==distance[p]-1:choices.append(q)
  p=min(choices);origins.append(p)
 route=np.zeros_like(legal)
 for p in origins:route[tuple(slice(v,v+3) for v in p)]=True
 assert (route&seed).any() and (route&east).any() and not (route&~legal).any()
 assert connected(route) and np.array_equal(cube_union(full_origins(route)),route)

 return route
