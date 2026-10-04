from pathlib import Path
import json,hashlib,shutil
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Independent-Review-2026-10-04')
r=json.loads((OUT/'result.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
frozen=json.loads((OUT/'frozen-execution-hashes.json').read_text())
assert all(sha((OUT/n).read_bytes())==v for n,v in frozen.items())
comparisons=[];stats={}
for cohort in ['regression','fresh']:
 for step in [64,128]:
  rows=[o for o in r['observations'] if o['cohort']==cohort and o['model']=='R3' and o['steps']==step]
  gains=[];losses=[]
  for o in rows:
   before=next(b for b in r['observations'] if b['case']==o['case'] and b['model']=='G10' and b['steps']==step)
   bp=before['status']=='evaluated' and before['score']['contract_pass']
   ap=o['status']=='evaluated' and o['score']['contract_pass']
   if ap and not bp:gains.append(o['case'])
   if bp and not ap:losses.append(o['case'])
  comparisons.append(dict(cohort=cohort,steps=step,gains=gains,losses=losses))
 for label in ['G10','R3']:
  rows=[o for o in r['observations'] if o['cohort']==cohort and o['model']==label and o['steps']==128 and o['status']=='evaluated']
  sts=[s for s in r['stability'] if s['cohort']==cohort and s['model']==label]
  stats[cohort+'_'+label]=dict(stable=sum(s['passed'] for s in sts),count=len(sts),max_growth=max((s['growth'] for s in sts if s['growth'] is not None),default=None))
  if label=='R3':
   shares=[o['planner_voxels']/(o['score']['occupied_voxels']-1) for o in rows]
   stats[cohort+'_'+label].update(planner_share_median=float(np.median(shares)) if shares else None)
# Replay union reservation & saved outputs; evaluator already checked ceilings and finite states.
checks=0
for p in sorted((OUT/'cases').iterdir()):
 if not (p/'hybrid-trajectory.npz').exists():continue
 with np.load(p/'hybrid-trajectory.npz') as a,np.load(p/'witness.npz') as w:
  assert np.isin(a['provenance'],[0,1,2]).all()
  for step in [64,128]:
   with np.load(p/f'R3-{step}.npz') as f:assert np.array_equal(f['field'],a[f'state{step}'][0,0].astype(bool))
   assert np.isfinite(a[f'state{step}']).all();checks+=1
summary=dict(paired=comparisons,stats=stats,frozen_inputs_sources_unchanged=True,finite_horizon_states=checks,certificate_failures=[c['case'] for c in r['certificates'] if not c['certified']],unstable=[s for s in r['stability'] if not s['passed']])
(OUT/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
shutil.copyfile(__file__,OUT/'audit-results.py')
print(json.dumps(summary,indent=2))

