from pathlib import Path
import json,hashlib,zipfile
repo=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
out=Path('C:/Users/artin/Documents/Codex/outputs/CGR1-Final-Review-2026-09-28')
old=Path('C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/NR5-Single-Trial-Review')
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f: f.write(json.dumps(x,indent=2)+'\n' if not isinstance(x,str) else x)
rows=[];prev=[]
for p in sorted((out/'observations').glob('*.json')):
 r=read(p); q=read(old/'observations'/p.name)
 assert sha(p.with_suffix('.npz'))==r['arrays_sha256']
 assert sha((old/'observations'/p.name).with_suffix('.npz'))==q['arrays_sha256']
 assert (r['case'],r['damage'],r['baselines'])==(q['case'],q['damage'],q['baselines'])
 rows.append(r);prev.append(q)
fails=[dict(case=r['case'],damage=r['damage'],bulk_fraction=r['metrics']['targets']['bulk_fraction'],unreached_bulk=r['metrics']['targets']['unreached_bulk_voxels'],families=[k for k,v in r['metrics']['targets']['family_pass'].items() if not v]) for r in rows if not r['metrics']['targets']['contract_pass']]
result=read(out/'result.json');imports=read(out/'imports.json')
comparison=dict(nr5_verified_arrays=27,matched_case_damage_baselines=True,nr5_all_pass=sum(r['metrics']['targets']['contract_pass'] for r in prev),intact_below_099=[dict(case=r['case'],iou=r['metrics']['iou']) for r in rows if r['damage']=='intact' and r['metrics']['iou']<.99],detached_correct=sum(r['metrics']['detached_correct'] for r in rows),detached_excess=sum(r['metrics']['detached_excess'] for r in rows),failures=fails)
write(out/'comparison.json',comparison)
summary=dict(run_id='20260928T153419Z_44ad70fb1464',receipt=imports['receipt'],review=result,comparison=comparison,artifacts=str(out),decision='Experimental only; frozen acceptance not met.')
write(repo/'experiments/reports/CGR1-final-review.json',summary)
write(repo/'experiments/records/20260928T153419Z_44ad70fb1464.json',dict(run_id=summary['run_id'],status='completed',completed_updates=256,verified_payloads=1036,receipt=imports['receipt'],artifact_location=str(out),previous_attempt='20260928T113537Z_6f25fe81a893',model_version='connected_constructive_repair_v2',execution_provenance=imports['execution_provenance'],quality_accepted=False))
doc='''# CGR1 final development review — 2026-09-28

The connected constructive model improves the NR5 development results, but fails two frozen acceptance conditions. Keep it experimental; MG7 remains live.

## Evidence and evaluation

User supplied run `20260928T153419Z_44ad70fb1464`: 256 CUDA updates completed, seed 1201, 32 growth steps. Controlled duration 61.469 seconds, peak reserved GPU memory 710 MiB, cleanup exit 0. Verified outer receipt SHA256, all 1,036 payload hashes and exact unique ZIP membership; verified final checkpoint bytes/hash, semantic identity, optimizer/sampler cursor and trace. This establishes successful GPU execution of v2, not exact GPU interruption/recovery equivalence. The original failed v1 attempt remains preserved.

Final checkpoint 256 only; CPU float32, 32 steps, firing seed 2101, all 27 existing validation/development cases. Scored accepted binary occupancy, without cleanup or threshold changes. Saved all proposals, birth masks, hidden state, inputs, outputs and per-case metrics. Verified the 27 prior NR5 array hashes and matching case/damage/baseline identities. No TEST cases inspected. Repeated development cases and one training seed cannot establish generalization.

## Comparison

| Metric | NR5 | CGR1 |
|---|---:|---:|
| All nine checks, all cases | 17/27 | 24/27 |
| All nine checks, damaged cases | 11/18 | 15/18 |
| Damaged median IoU | 0.97058 | 0.97504 |
| Damaged excess voxels | 325 | 245 |
| Damaged correctly recovered voxels | 1,945 | 1,959 |
| Damaged median absolute requested-volume error, cells | 19 | 15.5 |
| Intact all-nine validity | 6/9 | 9/9 |
| Intact median IoU | 0.98633 | 0.99426 |
| Intact excess voxels | 172 | 117 |

The simple closing3 baseline also passes 24/27 overall and 15/18 damaged cases. It recovers fewer damaged cells (1,464) but adds only 13 excess damaged cells. CGR1 is therefore not a universal winner against the simple baseline.

## What remains wrong

All three failed cases are cube5 damage. Each fails both access and thickness: the occupied field is attached, but its thick bulk does not fully connect to the interfaces. All 27 pass support; detached correct and detached excess voxel counts are both zero. Attachment and no deletion are enforced properties, not learned guarantees. Single-cell connectivity is insufficient for volumetric connectivity.

Six of eight frozen conditions pass. Damaged validity is 15/18, below the required 17/18. Three intact examples have IoU below the required 0.99 for every intact example, despite their median exceeding 0.99. Intact inputs still accumulate 117 unwanted cells. Wrong births cannot be undone by this monotonic model.

## Next step

Keep this checkpoint as the connected baseline. Before another paid run, prepare one focused revision of training supervision for connected thick bulk and stopping growth on intact inputs. Retain the nine families and overall-building-volume semantics; do not prescribe rooms or fill exterior gaps. First use existing TRAIN cases to choose a differentiable bulk-aware objective and quantify its gradients and interaction with the existing frontier loss. Freeze the resulting specification, budget and comparison before requesting a single new run. Do not silently relax criteria, select a better horizon, tune on TEST, or launch a seed sweep. This recommendation is a design direction, not evidence that the revised loss will work.

## Preservation and resumption

Full local evidence: `C:/Users/artin/Documents/Codex/outputs/CGR1-Final-Review-2026-09-28`.
Reproducible review: `scripts/review_connected_run.py` (requires a fresh output directory). Small summary: `experiments/reports/CGR1-final-review.json`. Original supplied ZIP/receipt, review source snapshot and all 27 observation arrays retained. Milestone archive is adjacent to the evidence directory; its receipt records hashes. Same-disk copies are not off-device backup. No Drive operations, new training, push or live-model replacement performed.
'''
write(repo/'docs/next-phase/CGR1_FINAL_REVIEW.md',doc);write(out/'FINDINGS.md',doc)
entry='''## CGR1 v2 completed and reviewed — 2026-09-28

D092: run20260928T153419Z_44ad70fb1464 completed256 GPU updates in61.469s;
1036 payload hashes and final checkpoint identity/cursors verified. Frozen final
CPU review:24/27 all-nine valid (NR5 17/27); damaged15/18, medianIoU0.975039,
excess245, recovered1959, volume-error15.5. Intact9/9 valid but3/9 IoU<0.99.
Two frozen conditions fail: damaged validity and per-intact overlap. All3failed
cube5 cases fail access+thickness; zero detached voxels. Keep experimental; MG7
stays live. No TEST/new training/Drive/push. Successful GPU execution does not
prove exact GPU recovery. See CGR1_FINAL_REVIEW.md and CGR1-final-review.json.
Full evidence C:/Users/artin/Documents/Codex/outputs/CGR1-Final-Review-2026-09-28.
Next: specify one TRAIN-grounded bulk-aware/stopping supervision revision,
freeze it and its compute budget, then request one concrete GPU run approval.
Do not repeat the completed CGR1 launch instructions below; they are history.

'''
for name in ['RESUME.md','PLAN.md']:
 p=repo/'docs/next-phase'/name;s=p.read_text(encoding='utf-8');a,b=s.split('\n',1);p.write_text(a+'\n\n'+entry+b,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 p=repo/'docs/next-phase'/name
 with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry.replace('## CGR1 v2 completed and reviewed','## D092 — CGR1 v2 completed and reviewed'))
print(json.dumps(comparison,indent=2))
