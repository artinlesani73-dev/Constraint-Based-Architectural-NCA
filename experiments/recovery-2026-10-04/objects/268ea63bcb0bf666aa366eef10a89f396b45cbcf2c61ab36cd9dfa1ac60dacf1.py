"""TRAIN-only, context-defined seed audit; never loads held-out arrays."""
import json, hashlib
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

ROOT = Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
OUT = Path('C:/Users/artin/Documents/Codex/outputs/Generation-Milestone-2026-10-03')
OUT.mkdir(exist_ok=True)
source = ROOT / '.local-artifacts/runs/20260925T094341Z_316cff241020'
study = json.loads((source/'study.json').read_text())
records=[]; groups=defaultdict(list)
for row in study['examples']:
    if row['split']!='train' or row['damage']!='intact': continue
    path=source/row['arrays']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['arrays_sha256']
    with np.load(path,allow_pickle=False) as a:
        c=a['condition'].copy(); target=a['target'].astype(bool)
    legal=(c[0]>0)&(c[1]>0)&~(c[2]>0)&~(c[3]>0)
    candidates=np.argwhere(legal & (c[5]>0))
    assert len(candidates)
    # Lowest x interface, then nearest its centroid, ties in z,y,x order.
    side=candidates[candidates[:,2]==candidates[:,2].min()]
    seed=tuple(side[np.argmin(((side-side.mean(axis=0))**2).sum(axis=1))])
    distance=np.full(legal.shape,-1,dtype=np.int32); distance[seed]=0
    queue=deque([seed])
    while queue:
        pos=queue.popleft()
        for axis in range(3):
            for sign in (-1,1):
                nxt=list(pos); nxt[axis]+=sign; nxt=tuple(nxt)
                if all(0<=nxt[i]<legal.shape[i] for i in range(3)) and legal[nxt] and distance[nxt]<0:
                    distance[nxt]=distance[pos]+1; queue.append(nxt)
    key=hashlib.sha256(c.tobytes()).hexdigest()
    groups[key].append(hashlib.sha256(target.tobytes()).hexdigest())
    records.append(dict(case=row['case'],site=row['site'],input_sha256=row['arrays_sha256'],condition_sha256=key,
        seed_zyx=[int(v) for v in seed],seed_in_teacher=bool(target[seed]),target_cells=int(target.sum()),
        unreachable_teacher_cells=int((target & (distance<0)).sum()),
        max_legal_path_distance=int(distance[target].max()),teacher_cells_beyond_32=int((target & (distance>32)).sum())))
result=dict(version='generation_seed_audit_v1',scope='27 TRAIN intact examples only; no held-out arrays loaded',
    caveat='Legal-domain path lengths are optimistic reachability bounds, not learned or teacher-only growth evidence.',
    study_sha256=hashlib.sha256((source/'study.json').read_bytes()).hexdigest(),records=records,
    summary=dict(targets=len(records),contexts=len(set(r['site'] for r in records)),
        seeds_inside_teacher=sum(r['seed_in_teacher'] for r in records),
        max_legal_path_distance=max(r['max_legal_path_distance'] for r in records),
        targets_beyond_32=sum(r['teacher_cells_beyond_32']>0 for r in records),
        conditioning_groups=len(groups),groups_with_multiple_distinct_teachers=sum(len(set(v))>1 for v in groups.values())))
(OUT/'seed-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary'],indent=2))
