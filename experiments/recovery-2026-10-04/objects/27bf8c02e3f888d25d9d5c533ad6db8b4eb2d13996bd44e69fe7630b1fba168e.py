from pathlib import Path
import json,shutil
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G10-Final-Review-2026-10-04');BASE=OUT.parent
r=json.loads((OUT/'result.json').read_text());checks=[]
for s in r['stability']:
 case=s['case']
 with np.load(OUT/f'observations/{case}-64.npz') as a,np.load(OUT/f'observations/{case}-128.npz') as b:
  assert np.array_equal(a['births'],b['births'][:64]);assert np.isfinite(a['state']).all() and np.isfinite(b['state']).all()
  checks.append(dict(case=case,prefix64_equal=True,finite_states=True,changed_cells=int((a['field']!=b['field']).sum())))
old=json.loads((BASE/'G9-Final-Review-2026-10-04/result.json').read_text())['observations'];comparison=[]
for o in r['observations']:
 if o['cohort'].startswith('baseline_'):continue
 labels=['G9'] if o['cohort']=='regression' else ['G8','G9']
 for label in labels:
  before=next(x for x in (old if o['cohort']=='regression' else r['observations']) if x['case']==(o['case'] if o['cohort']=='regression' else label.lower()+'-baseline-'+o['case']) and x['steps']==o['steps'])
  comparison.append(dict(case=o['case'],cohort=o['cohort'],baseline=label,steps=o['steps'],baseline_pass=before['score']['contract_pass'],g10_pass=o['score']['contract_pass'],baseline_failed=[k for k,v in before['score']['family_pass'].items() if not v],g10_failed=[k for k,v in o['score']['family_pass'].items() if not v]))
paired={}
for cohort,label in [('regression','G9'),('fresh_reserved','G9'),('fresh_reserved','G8')]:
 key=cohort+'_'+label;paired[key]={}
 for step in [64,128]:
  group=[x for x in comparison if x['cohort']==cohort and x['baseline']==label and x['steps']==step]
  paired[key][str(step)]=dict(baseline=sum(x['baseline_pass'] for x in group),g10=sum(x['g10_pass'] for x in group),improved=[x['case'] for x in group if x['g10_pass'] and not x['baseline_pass']],regressed=[x['case'] for x in group if x['baseline_pass'] and not x['g10_pass']])
result=dict(prefix_and_finite_checks=checks,paired_comparison=comparison,paired_summary=paired)
with (OUT/'result-audit.json').open('x') as f:json.dump(result,f,indent=2)
shutil.copyfile(__file__,OUT/'audit-results.py')
print(json.dumps(dict(paired=paired,failures=[dict(case=o['case'],steps=o['steps'],failed=[k for k,v in o['score']['family_pass'].items() if not v],unused=o['unused_capacity']) for o in r['observations'] if not o['cohort'].startswith('baseline_') and not o['score']['contract_pass']],stability_failures=[s for s in r['stability'] if not s['cohort'].startswith('baseline_') and s['relative_mass_change']>.05]),indent=2))

