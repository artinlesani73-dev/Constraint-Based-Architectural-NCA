from pathlib import Path
import json,hashlib,numpy as np,shutil
out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Final-Review-2026-10-03');base=out.parent
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roots={'CGR1':base/'CGR1-Final-Review-2026-09-28','CGR2':base/'CGR2-Final-Review-2026-09-29','NR5':Path('C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/NR5-Single-Trial-Review')}
rows=[read(p) for p in sorted((out/'observations').glob('*.json'))];comparison={}
for label,root in roots.items():
 old=[]
 for p in sorted((out/'observations').glob('*.json')):
  a=read(p);b=read(root/'observations'/p.name)
  assert sha(p.with_suffix('.npz'))==a['arrays_sha256'] and sha((root/'observations'/p.name).with_suffix('.npz'))==b['arrays_sha256']
  assert (a['case'],a['damage'],a['baselines'])==(b['case'],b['damage'],b['baselines']);old.append(b)
 comparison[label]={}
 for group in ['all','damaged','intact']:
  ms=[r['metrics'] for r in old if group=='all' or (r['damage']=='intact')==(group=='intact')]
  comparison[label][group]=dict(all_nine_pass=sum(m['targets']['contract_pass'] for m in ms),median_iou=float(np.median([m['iou'] for m in ms])),median_absolute_request_error_cells=float(np.median([abs(m['request_error_cells']) for m in ms])),**{k:sum(m[k] for m in ms) for k in ['recovered_cells','false_positive_cells','surviving_cells_removed']})
fails=[dict(case=r['case'],damage=r['damage'],failed_families=[k for k,v in r['metrics']['targets']['family_pass'].items() if not v]) for r in rows if not r['metrics']['targets']['contract_pass']]
extra=dict(priors=comparison,verified_prior_arrays=81,failures=fails,intact_below_099=sum(r['metrics']['iou']<.99 for r in rows if r['damage']=='intact'),detached_voxels=sum(r['metrics']['detached_correct']+r['metrics']['detached_excess'] for r in rows))
(out/'comparison.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
imports=read(out/'imports.json');result=read(out/'result.json')
record=dict(run_id='20261003T102352Z_a40c9f968e66',status='completed',completed_updates=256,verified_payloads=1036,receipt=imports['receipt'],controlled_seconds=imports['result']['wall_seconds'],quality_accepted=False,review=result,comparison=extra,curriculum=read(out/'curriculum-verification.json'),artifact_location=str(out),repository_sync_pending=True,download_policy='Keep full ZIP; user withdrew small review ZIP proposal.',execution_provenance='User supplied completed evidence; no new paid launch approval inferred.')
(out/'project-record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
doc='''# CGR3 final review — 2026-10-03

CGR3 completed, but failed two frozen acceptance conditions. Keep CGR1 as the
experimental reference and MG7 live. Do not promote this checkpoint or launch
another training trial automatically.

## Verified evidence

Run20261003T102352Z_a40c9f968e66 completed256updates in59.970controlled seconds,
seed1201,T4,peak reserved710MiB,exit0. Verified original receipt hash and all1036
payloads,exact unique membership,final checkpoint digest/identity/cursors and Adam
step counts. Independently matched all256 saved start arrays to the frozen TRAIN
schedule and verified their hashes,per-update traces,source row hashes and visit
history:194original,62intermediate. No incomplete-download or curriculum mismatch.

Runtime is Torch2.11.0+cu130/CUDA13.0/cuDNN92700. The prior17-payload compatibility
run20261003T094720Z_e18428c17993 independently passed full-payload augmented-step
recovery. That is same-runtime recovery evidence, not CUDA12.8/13.0 equivalence.
CGR1/CGR2 were trained on cu128, so runtime is a confound in curriculum attribution.

Frozen evaluation: final256,CPUfloat32,32steps,firing2101,27existing development
examples using original inputs only. No teacher augmentation at evaluation,
threshold tuning,cleanup,TEST or checkpoint/horizon selection. Preserved proposals,
births,states and all case metrics. Verified81prior NR5/CGR1/CGR2 arrays and matched
case/damage/baseline identities. One seed,reused development scenes; not generalization.

## Comparison

| Metric | CGR1 | CGR2 | CGR3 |
|---|---:|---:|---:|
| All nine pass, all27 |24|24|24|
| All nine pass, damaged18 |15|15|15|
| Damaged median IoU |.97504|.97096|.97214|
| Damaged recovered cells |1959|1828|1949|
| Damaged excess cells |245|165|268|
| Damaged median absolute volume error |15.5|23.5|10.5|
| Intact excess cells |117|83|139|
| Intact median IoU |.99426|.99316|.98790|

CGR3 recovers most of CGR1's repaired volume and improves requested-volume error,
but increases excess and worsens intact overlap. Lower volume error can coexist
with worse shape agreement, because missing and excess cells can offset in a
volume total. Closing3 also passes24/27,with damaged IoU.97268,recovered1464,
excess13 and volume error38; CGR3 is not a universal winner over this simple baseline.

## Frozen decision and actual failure changes

Six of eight gates pass. Damaged validity remains15/18 versus17required. Five
intact cases are below.99IoU (every intact case must pass). All9intact cases still
satisfy the nine geometric checks. No surviving input cells removed; no detached
voxels. Input preservation and attached growth are enforced, not learned.

Same three cube5 cases fail, but two individual family failures are resolved:
- v16,s0 now passes access and still fails thickness.
- v16,s2 still fails access and thickness.
- v24,s2 now passes thickness and still fails access.
Access and thickness each improve24/27 to25/27; all remaining families pass27/27.
The curriculum therefore shows a partial change, not full failure resolution.

## Recommendation

Close this bounded curriculum trial as mixed/not accepted. Do not infer that
more voxels,more rollout steps,or another small loss adjustment will solve it.
Before another paid job, consolidate the experiments into a model-design decision:
compare the current irreversible,detached birth rule with a trainable proposal
that can revise its own additions while preserving original input. This is a
candidate for analysis, not an approved architecture or proven solution. Examine
objective versus deployment-rule alignment and keep the nine families fixed.
These trials assess repair of known volumes; they do not demonstrate generation
of new architectural volumes from scene constraints. Keep that distinction in
the larger project roadmap. Retain CGR1 until a replacement meets the frozen gates.

## Records and downloads

User explicitly withdrew the small-review-ZIP proposal. Continue full ZIP exports;
no new split/review export workflow introduced. Original files and all evaluation
evidence are preserved here. Repository writes were not granted, so this record,
review script and pending resume instructions are local and not Git-committed.
No Drive access,new paid training,push or live-model change. Local copies are
same-disk archives,not an off-device backup.
'''
(out/'FINDINGS.md').write_text(doc,encoding='utf-8')
(out/'RESUME-PENDING-SYNC.json').write_text(json.dumps(dict(status='CGR3_review_complete_not_accepted',repository_write_access=False,record=str(out/'project-record.json'),findings=str(out/'FINDINGS.md'),pending_prior_records=[str(base/'CGR3-cu130-Preflight/READINESS.json'),str(base/'CGR3-cu130-Verified-20261003T094720Z/verified-record.json'),str(base/'CGR3-Curriculum-cu130/RESUME-PENDING-SYNC.json')],next='Consolidate repair trial findings into architecture/training decision before another paid experiment.',user_export_preference='Full ZIP; small review ZIP withdrawn.'),indent=2),encoding='utf-8')
transfer=base/'CGR3-Transfer-20261003T102352Z'
(transfer/'completed.json').write_text(json.dumps(dict(status='full_archive_received_and_verified',sha256=imports['receipt']['sha256'],small_review_zip_requested=False,review=str(out)),indent=2),encoding='utf-8')
print('Verified comparisons; intact below .99:',extra['intact_below_099'],'detached:',extra['detached_voxels'])
