from pathlib import Path
import json,shutil
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G9-Final-Review-2026-10-04');BASE=OUT.parent
r=json.loads((OUT/'result.json').read_text());checks=[]
for s in r['stability']:
 case=s['case']
 with np.load(OUT/f'observations/{case}-64.npz') as a,np.load(OUT/f'observations/{case}-128.npz') as b:
  assert np.array_equal(a['births'],b['births'][:64]);assert np.isfinite(a['state']).all() and np.isfinite(b['state']).all()
  checks.append(dict(case=case,prefix64_equal=True,finite_states=True,changed_cells=int((a['field']!=b['field']).sum())))
old=json.loads((BASE/'G8-Final-Review-2026-10-04-v2/result.json').read_text())['observations'];comparison=[]
for o in r['observations']:
 if o['cohort']=='baseline_fresh':continue
 if o['cohort']=='regression':before=next(x for x in old if x['case']==o['case'] and x['steps']==o['steps'])
 else:before=next(x for x in r['observations'] if x['case']=='g8-baseline-'+o['case'] and x['steps']==o['steps'])
 comparison.append(dict(case=o['case'],cohort=o['cohort'],steps=o['steps'],g8_pass=before['score']['contract_pass'],g9_pass=o['score']['contract_pass'],g8_failed=[k for k,v in before['score']['family_pass'].items() if not v],g9_failed=[k for k,v in o['score']['family_pass'].items() if not v]))
paired={}
for cohort in ['regression','fresh_reserved']:
 paired[cohort]={}
 for step in [64,128]:
  group=[x for x in comparison if x['cohort']==cohort and x['steps']==step]
  paired[cohort][str(step)]=dict(g8=sum(x['g8_pass'] for x in group),g9=sum(x['g9_pass'] for x in group),improved=[x['case'] for x in group if x['g9_pass'] and not x['g8_pass']],regressed=[x['case'] for x in group if x['g8_pass'] and not x['g9_pass']])
result=dict(prefix_and_finite_checks=checks,paired_comparison=comparison,paired_summary=paired)
with (OUT/'result-audit.json').open('x') as f:json.dump(result,f,indent=2)
shutil.copyfile(__file__,OUT/'audit-results.py')
print(json.dumps(dict(paired=paired,failures=[dict(case=o['case'],cohort=o['cohort'],steps=o['steps'],failed=[k for k,v in o['score']['family_pass'].items() if not v],contacts=o['direct_interface_contact_voxels'],unused=o['unused_capacity']) for o in r['observations'] if not o['score']['contract_pass']],stability_failures=[s for s in r['stability'] if s['relative_mass_change']>.05]),indent=2))

