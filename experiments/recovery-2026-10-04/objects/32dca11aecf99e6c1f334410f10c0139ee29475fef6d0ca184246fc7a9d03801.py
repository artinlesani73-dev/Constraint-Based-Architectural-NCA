from pathlib import Path
import json,hashlib,numpy as np
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR2-Final-Review-2026-09-29')
prior={'CGR1':Path('C:/Users/artin/Documents/Codex/outputs/CGR1-Final-Review-2026-09-28'),'NR5':Path('C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/NR5-Single-Trial-Review')}
def read(p):return json.loads(p.read_bytes())
def write(p,v):
 with p.open('x',encoding='utf-8') as f:f.write(v if isinstance(v,str) else json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[read(p) for p in sorted((out/'observations').glob('*.json'))]
comparison={}
for name,root in prior.items():
 old=[]
 for p in sorted((out/'observations').glob('*.json')):
  a=read(p);b=read(root/'observations'/p.name)
  assert sha(p.with_suffix('.npz'))==a['arrays_sha256'] and sha((root/'observations'/p.name).with_suffix('.npz'))==b['arrays_sha256']
  assert (a['case'],a['damage'],a['baselines'])==(b['case'],b['damage'],b['baselines']);old.append(b)
 comparison[name]={}
 for group in ['all','damaged','intact']:
  ms=[x['metrics'] for x in old if group=='all' or (x['damage']=='intact')==(group=='intact')]
  comparison[name][group]=dict(all_nine_pass=sum(x['targets']['contract_pass'] for x in ms),median_iou=float(np.median([x['iou'] for x in ms])),median_absolute_request_error_cells=float(np.median([abs(x['request_error_cells']) for x in ms])),**{k:sum(x[k] for x in ms) for k in ['recovered_cells','false_positive_cells','surviving_cells_removed']})
failures=[dict(case=x['case'],damage=x['damage'],families=[k for k,v in x['metrics']['targets']['family_pass'].items() if not v],bulk_fraction=x['metrics']['targets']['bulk_fraction']) for x in rows if not x['metrics']['targets']['contract_pass']]
result=read(out/'result.json');imports=read(out/'imports.json')
extra=dict(priors=comparison,verified_prior_arrays=54,failures=failures,intact_below_099=sum(x['metrics']['iou']<.99 for x in rows if x['damage']=='intact'),detached_voxels=sum(x['metrics']['detached_correct']+x['metrics']['detached_excess'] for x in rows))
write(out/'comparison.json',extra)
write(r/'experiments/reports/CGR2-final-review.json',dict(run_id='20260929T062933Z_465972e5c250',receipt=imports['receipt'],review=result,comparison=extra,artifact_location=str(out),accepted=False))
write(r/'experiments/records/20260929T062933Z_465972e5c250.json',dict(run_id='20260929T062933Z_465972e5c250',status='completed',completed_updates=256,receipt=imports['receipt'],model_version='bulk_constructive_repair_v1',controlled_seconds=imports['result']['wall_seconds'],artifact_location=str(out),execution_provenance=imports['execution_provenance'],quality_accepted=False,parent_comparison='20260928T153419Z_44ad70fb1464'))
s=Path('C:/Users/artin/Documents/Codex/2026-09-06/cre/review_bulk_run.py').read_text();s=s.replace("ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')",'ROOT=Path(__file__).resolve().parents[1]')
write(r/'scripts/review_bulk_run.py',s)
doc='''# CGR2 final development review — 2026-09-29

CGR2 completes successfully but does not meet frozen acceptance. Reduced excess
comes with reduced repair recovery; retain CGR1 as the experimental reference.
MG7 remains live. No model promotion or additional training.

## Verified evidence

Run20260929T062933Z_465972e5c250:256 CUDA updates,seed1201,32steps;
78.467s controlled,75.666s worker,714MiB peak reserved memory,cleanup exit0.
Verified outer receipt, all1,036 payload hashes, exact unique ZIP membership,
final checkpoint digest/bytes/semantic identity and optimizer/sampler/trace cursor.
User-supplied execution evidence is not standing permission for another paid run.
GPU execution succeeds; exact GPU interruption/recovery was not tested here.

Evaluation: final256 only, CPUfloat32,32steps,firing2101,27 existing development
cases. Accepted binary occupancy, no cleanup or threshold tuning. No TEST.
All histories, raw proposals, birth masks and state arrays preserved. Verified
54 prior CGR1/NR5 arrays and matching case,damage and baseline records.

| Metric | NR5 | CGR1 | CGR2 | Closing3 |
|---|---:|---:|---:|---:|
| All-nine pass, all27 |17|24|24|24|
| All-nine pass, damaged18 |11|15|15|15|
| Damaged median IoU |.97058|.97504|.97096|.97268|
| Damaged recovered cells |1945|1959|1828|1464|
| Damaged excess cells |325|245|165|13|
| Damaged median absolute volume error, cells |19|15.5|23.5|38|
| Intact excess cells |172|117|83|13|
| Intact median IoU |.98633|.99426|.99316|1.00000|

CGR2 improves excess counts but does not dominate CGR1 or the simple baseline.
Despite fewer total intact excess cells, median intact overlap slightly worsens:
aggregate counts and medians describe different aspects of the distribution.

## Frozen decision

Four of eight gates fail: all-intact overlap (3/9 below.99), damaged validity
(15/18 versus17required), damaged recovery (1828 versus1945required), and median
absolute volume error (23.5 versus19maximum). Intact validity9/9, damaged overlap,
damaged excess and preservation pass. No surviving input cells are removed.

The same three cube5 examples fail access and thickness. Support passes27/27;
zero detached occupied voxels. Enforced attachment does not ensure a thick bulk
connection. The new supervision did not resolve those failures in this one run.
This does not establish that either new loss term individually caused the result:
both changed together, with one training seed and repeatedly used development data.

## Recommended next step

Pause further coefficient trials. Use saved trajectories to inspect where repair
stalls on TRAIN examples and whether missing bridge cells ever become eligible,
remain below threshold, or are blocked by earlier irreversible growth. Compare
CGR1 and CGR2 on identical inputs/firing without optimizing against TEST. This
local diagnosis should distinguish an objective imbalance from the limitations
of detached hard births before proposing another bounded experiment. No claim
that increasing grid size or training duration alone will fix it. Maintain the
nine families and overall-volume concept; this finding does not call for rooms.

## Artifacts and resume

Full evidence: C:/Users/artin/Documents/Codex/outputs/CGR2-Final-Review-2026-09-29.
Review script: scripts/review_bulk_run.py; requires fresh output directory.
Small record: experiments/reports/CGR2-final-review.json. The supplied ZIP and
receipt and every case remain local; an adjacent verified milestone ZIP includes
these and final project records. Same-disk copies are not off-device backup.
No Drive access, push, live replacement, extra seed or paid run performed.
'''
write(r/'docs/next-phase/CGR2_FINAL_REVIEW.md',doc);write(out/'FINDINGS.md',doc)
entry='''## D094 — CGR2 completed; retain CGR1 reference — 2026-09-29

Run20260929T062933Z_465972e5c250 completed256 GPU updates in78.467s;
1036payload hashes and checkpoint identity/cursors verified. Final frozen CPU
review24/27valid; damaged15/18,IoU.970964,recovered1828,excess165,error23.5.
Four gates fail: intact overlap,damaged validity,recovery,volume error. CGR1 had
two failures; its1959recovered versus CGR2 1828. Reduced excess is a tradeoff,
not overall acceptance. Same3cube5 cases fail access+thickness; zero detached
voxels. Keep CGR1 reference,MG7 live. No TEST,paid retry,Drive or push.
See CGR2_FINAL_REVIEW.md and experiments/reports/CGR2-final-review.json.
Evidence: C:/Users/artin/Documents/Codex/outputs/CGR2-Final-Review-2026-09-29.
Next: local TRAIN trajectory diagnosis (eligibility versus rejected births versus
irreversible-growth effects), then specify a justified revision before any new
compute request. Do not repeat the historical CGR2 launch instructions below.

'''
for name in ['RESUME.md','PLAN.md']:
 p=r/'docs/next-phase'/name;a,b=p.read_text(encoding='utf-8').split('\n',1);p.write_text(a+'\n\n'+entry+b,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 with (r/'docs/next-phase'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry.rstrip()+'\n')
print(json.dumps(extra,indent=2))
