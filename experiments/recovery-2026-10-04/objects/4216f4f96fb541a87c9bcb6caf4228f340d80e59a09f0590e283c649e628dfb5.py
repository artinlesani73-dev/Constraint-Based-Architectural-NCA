from pathlib import Path
import json,hashlib,shutil
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R2-Packing-2026-10-04')
r=json.loads((OUT/'result.json').read_text());records=[]
cases=list(dict.fromkeys(x['case'] for x in r['observations']))
for case in cases:
 p=OUT/'cases'/case;wr=json.loads((p/'witness.json').read_text())
 with np.load(p/'context.npz') as a:c=a['condition'];allowed=c[0].astype(bool)
 with np.load(p/'witness.npz') as a:W=a['field']
 with np.load(p/'G11-R2-trajectory.npz') as a:births=a['births'];provenance=a['provenance']
 assert np.array_equal(provenance>0,births)
 import sys
 sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
 from nca.generation_data import seed_inputs
 f=seed_inputs(c)['occupancy'].astype(bool)
 trace=json.loads((p/'hybrid-trace.json').read_text());K=max(9,int(np.ceil((wr['C']-27)/63)))
 for i,b in enumerate(births):
  before=int(f.sum());assert not (f&b).any()
  f |= b;cap=wr['C'] if before==1 else min(wr['C'],before+K)
  assert f.sum()<=cap and not(f&~allowed).any() and (f|W).sum()<=wr['C']
  assert trace[i]['mass']==f.sum() and trace[i]['cap']==cap
  assert trace[i]['procedural_voxels']==(provenance[i]==2).sum()
  assert trace[i]['learned_voxels']==(provenance[i]==1).sum()
  if i+1 in [64,128]:
   with np.load(p/f'G11-R2-{i+1}.npz') as a:assert np.array_equal(a['field'],f)
 rec=dict(case=case)
 for label in ['G10','G11-R2']:
  s=json.loads((p/f'{label}-stability.json').read_text());rec[label]=s
 records.append(rec)
summary={}
for label in ['G10','G11-R2']:
 selected=[o for o in r['observations'] if o['model']==label and o['steps']==128]
 h64=[o for o in r['observations'] if o['model']==label and o['steps']==64]
 summary[label]=dict(stable=sum(x[label]['passed'] for x in records),max_growth=max(x[label]['growth'] for x in records),median_seconds128=float(np.median([o['seconds128'] for o in selected])),failures64=[dict(case=o['case'],families=[k for k,v in o['score']['family_pass'].items() if not v]) for o in h64 if not o['score']['contract_pass']],failures128=[dict(case=o['case'],families=[k for k,v in o['score']['family_pass'].items() if not v]) for o in selected if not o['score']['contract_pass']])
hy=[o for o in r['observations'] if o['model']=='G11-R2' and o['steps']==128]
shares=[o['procedural_voxels']/(o['score']['occupied_voxels']-1) for o in hy]
summary['procedural_share128']=dict(min=min(shares),median=float(np.median(shares)),max=max(shares))
summary['witness_complete64']=sum(o['witness_missing']==0 for o in r['observations'] if o['model']=='G11-R2' and o['steps']==64)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((OUT/'model/checkpoint-0427.pt').read_bytes())==json.loads((OUT/'runtime.json').read_text())['checkpoint_sha256']
(OUT/'audit.json').write_text(json.dumps(dict(summary=summary,step_account_checks=len(cases)*128,checkpoint_unchanged=True,stability=records),indent=2),encoding='utf-8')
shutil.copyfile(__file__,OUT/'audit-results.py')
print(json.dumps(summary,indent=2))

