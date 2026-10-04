from pathlib import Path
import json,hashlib,shutil
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G7-Final-Review-2026-10-04');BASE=OUT.parent
r=json.loads((OUT/'result.json').read_text());checks=[]
for s in r['stability']:
 case=s['case']
 with np.load(OUT/f'observations/{case}-64.npz') as a,np.load(OUT/f'observations/{case}-128.npz') as b:
  assert np.array_equal(a['births'],b['births'][:64]);assert np.isfinite(a['state']).all() and np.isfinite(b['state']).all()
  checks.append(dict(case=case,prefix64_equal=True,finite_states=True,changed_cells=int((a['field']!=b['field']).sum())))
old=[]
for folder in ['G6-Final-Review-2026-10-04','G6-Reserved-Review-2026-10-04']:
 old+=json.loads((BASE/folder/'result.json').read_text())['observations']
comparison=[]
for o in r['observations']:
 if o['cohort']!='regression':continue
 before=next(x for x in old if x['case']==o['case'] and x['steps']==o['steps'])
 comparison.append(dict(case=o['case'],steps=o['steps'],g6_pass=before['score']['contract_pass'],g7_pass=o['score']['contract_pass'],g6_failed_families=[k for k,v in before['score']['family_pass'].items() if not v],g7_failed_families=[k for k,v in o['score']['family_pass'].items() if not v]))
result=dict(prefix_and_finite_checks=checks,paired_regression_comparison=comparison,cohort_claim='G6 has not been evaluated on the fresh G7 reserved cohort;no fresh-cohort causal comparison claimed')
with (OUT/'result-audit.json').open('x') as f:json.dump(result,f,indent=2)
shutil.copyfile(__file__,OUT/'audit-results.py')
print(json.dumps(dict(failures=[dict(case=o['case'],cohort=o['cohort'],steps=o['steps'],families=[k for k,v in o['score']['family_pass'].items() if not v],error=o['absolute_fraction_error'],contacts=o['direct_interface_contact_voxels'],unused=o['unused_capacity']) for o in r['observations'] if not o['score']['contract_pass']],stability_failures=[s for s in r['stability'] if s['relative_mass_change']>.05],changes=[c for c in comparison if c['g6_pass']!=c['g7_pass']]),indent=2))
