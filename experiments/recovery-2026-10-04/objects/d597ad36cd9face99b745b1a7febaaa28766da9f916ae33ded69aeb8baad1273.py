from pathlib import Path
import json,hashlib
from collections import deque
import numpy as np
out=Path('C:/Users/artin/Documents/Codex/outputs/Reversible-Repair-Design-2026-10-03');out.mkdir(exist_ok=True)
code='''"""Reference binary transition only. No learned network or GPU implementation."""
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
'''
(out/'reference_transition.py').write_text(code,encoding='utf-8');ns={};exec(code,ns);step=ns['transition']
shape=(7,7,7);o=np.zeros(shape,bool);o[2,3,3]=True;m=o.copy();m[3,3,3]=True;m[4,3,3]=True;legal=np.ones(shape,bool);fire=np.ones(shape,bool);q=np.zeros(shape);q[4,3,3]=.9
r=step(o,m,legal,fire,q);assert r['field'][2,3,3] and not r['field'][3,3,3] and r['projection_removed'][4,3,3]
# No births selected in this second probe; original must survive even q=0.
z=step(o,m,legal,fire,np.zeros(shape));assert np.array_equal(z['field'],o)
none=step(o,m,legal,np.zeros(shape,bool),np.zeros(shape));assert np.array_equal(none['field'],m)
legal[1,3,3]=False;x=step(o,m,legal,fire,np.ones(shape));assert not x['field'][1,3,3] and not (x['births']&~ns['neighbors'](m)).any()
o2=o.copy();o2[6,6,6]=True;y=step(o2,o2,legal,np.zeros(shape,bool),np.zeros(shape));assert np.array_equal(y['field'],o2)
checks=dict(original_preserved=True,wrong_added_cell_can_be_removed=True,detached_descendants_pruned=True,no_firing_preserves_valid_anchored_state=True,illegal_births_blocked=True,births_use_previous_six_face_frontier=True,disconnected_originals_preserved_not_forced_to_merge=True)
(out/'synthetic-checks.json').write_text(json.dumps(dict(checks=checks,scope='Reference binary transition only; no learning, recovery or GPU evidence.',source_sha256=hashlib.sha256(code.encode()).hexdigest()),indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
