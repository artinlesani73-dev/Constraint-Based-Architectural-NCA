"""Experimental waypoint routes, not a replacement for frozen R3."""
from collections import deque
import numpy as np
from nca.generation_data import seed_inputs
from nca.paced_generation import full_origins,connected
from nca.block_reference import cube_union

def route_via(c,side):
 x=seed_inputs(c);legal=x['allowed'];valid=full_origins(legal);pts=np.argwhere(c[5])
 east=c[5].astype(bool)&(np.arange(legal.shape[2])[None,None,:]==pts[:,2].max())
 goals=valid&~full_origins(~east)
 def adjacent(p):
  for a in range(3):
   for sign in [-1,1]:
    q=list(p);q[a]+=sign;q=tuple(q)
    if all(0<=q[i]<valid.shape[i] for i in range(3)) and valid[q]:yield q
 def distance(targets):
  d=np.full(valid.shape,-1,np.int32);d[targets]=0;queue=deque(map(tuple,np.argwhere(targets)))
  while queue:
   p=queue.popleft()
   for q in adjacent(p):
    if d[q]<0:d[q]=d[p]+1;queue.append(q)
  return d
 dg=distance(goals);indices=np.indices(valid.shape);start=valid.copy()
 for a,s in enumerate(x['anchor']):start&=(indices[a]<=s)&(indices[a]+3>s)
 reachable=np.argwhere(valid&(dg>=0));assert len(reachable)
 # Place the waypoint toward opposite Y edges at the mid-gap X plane.
 midx=int((pts[:,2].min()+pts[:,2].max())//2)-1
 candidates=reachable[reachable[:,2]==midx]
 if not len(candidates):return None,dict(reason='no middle-plane waypoint')
 targety=int(np.quantile(candidates[:,1],.2 if side=='low_y' else .8));targetz=int(np.median(pts[:,0]))-1
 point=min(map(tuple,candidates),key=lambda p:(abs(p[1]-targety)+abs(p[0]-targetz),p))
 target=np.zeros_like(valid);target[point]=True;dw=distance(target);starts=np.argwhere(start&(dw>=0))
 if not len(starts):return None,dict(reason='waypoint unreachable from seed',waypoint=point)
 p=min(map(tuple,starts),key=lambda p:(dw[p],p));origins=[p]
 for d in [dw,dg]:
  while d[p]>0:p=min(q for q in adjacent(p) if d[q]==d[p]-1);origins.append(p)
 route=np.zeros_like(legal)
 for p in origins:route[tuple(slice(int(v),int(v)+3) for v in p)]=True
 assert connected(route) and not(route&~legal).any() and (route&x['occupancy'].astype(bool)).any() and (route&east).any()
 assert np.array_equal(route,cube_union(full_origins(route)))
 return route,dict(waypoint=[int(v) for v in point],target_y=targety,origin_count=len(origins),route_voxels=int(route.sum()))
