"""Reference binary transition only. No learned network or GPU implementation."""
from collections import deque
import numpy as np

def neighbors(m):
 n=np.zeros_like(m)
 for axis in range(3):
  a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
  n[tuple(a)]|=m[tuple(b)];n[tuple(b)]|=m[tuple(a)]
 return n

def transition(original,current,allowed,fire,probability):
 o=np.asarray(original,dtype=bool);m=np.asarray(current,dtype=bool);legal=np.asarray(allowed,dtype=bool);f=np.asarray(fire,dtype=bool);q=np.asarray(probability)
 if o.ndim!=3 or any(x.shape!=o.shape for x in [m,legal,f,q]):raise ValueError('Matching3D arrays required')
 if not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Probability range')
 if not o.any() or (o&~m).any() or (m&~legal).any():raise ValueError('Invalid initial state')
 active=legal&~o&f&(m|neighbors(m))
 candidate=o|(m&~active)|(active&(q>.5))
 reached=o.copy();queue=deque(map(tuple,np.argwhere(o)))
 while queue:
  x=queue.popleft()
  for axis in range(3):
   for delta in [-1,1]:
    y=list(x);y[axis]+=delta;y=tuple(y)
    if 0<=y[axis]<m.shape[axis] and candidate[y] and not reached[y]:reached[y]=True;queue.append(y)
 return dict(field=reached,active=active,candidate=candidate,direct_removed=m&~candidate,projection_removed=candidate&~reached,births=reached&~m)
