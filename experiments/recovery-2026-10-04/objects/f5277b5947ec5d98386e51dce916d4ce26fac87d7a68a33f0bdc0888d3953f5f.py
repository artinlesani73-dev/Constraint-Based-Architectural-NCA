from pathlib import Path
import sys,json,zipfile,hashlib,shutil
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');AUDIT=BASE/'G6-Objective-Audit-2026-10-04';OUT=BASE/'G6-Paced-Growth-2026-10-04';ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
from nca.generation_package import verify
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(folder,name,value):
    with (folder/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
manifest,data=verify(ROOT);runs=list((ROOT/'generation-runs').glob('*/result.json'));assert len(runs)==1
run=runs[0].parent;result=json.loads(runs[0].read_text());assert result['status']=='completed' and result['worker']['completed']==3 and result['cleanup']['active_processes_after_stop']==0
assert result['request']['manifest_sha256']==sha((ROOT/'manifest.json').read_bytes())
archive=run.with_suffix('.zip');receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(archive.read_bytes())==receipt['sha256']
with zipfile.ZipFile(archive) as z:
    em=json.loads(z.read('evidence-manifest.json'));assert len(em)==receipt['files']
    assert len(z.namelist())==len(set(z.namelist()))==len(em)+1 and set(z.namelist())==set(em)|{'evidence-manifest.json'}
    assert all(sha(z.read(k))==v for k,v in em.items())
recoveries=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]];assert all(r['full_payload_equal'] and r['state_equal'] for r in recoveries)
traces=[]
for i in range(1,4):
    trace=json.loads((run/f'worker/update-{i:04d}.json').read_text());cs=np.asarray(trace['admission_counts']);caps=np.asarray(trace['step_ceilings']);D,B,C=trace['budget'];K=max(9,int(np.ceil((C-27)/63)))
    assert trace['quota']==K and cs.shape==(64,7) and caps.shape==(64,)
    assert np.array_equal(caps,np.where(cs[:,0]==1,C,np.minimum(C,cs[:,0]+K)))
    assert (cs[:,1]==cs[:,2:6].sum(1)).all() and (cs[1:,0]==cs[:-1,0]+cs[:-1,6]).all() and (cs[:,0]+cs[:,6]<=caps).all()
    with np.load(run/f'worker/training-{i:04d}.npz',allow_pickle=False) as a:
        assert sha(a['start'].tobytes(order='C'))==trace['start']['sha256'] and int(a['start'].sum())==cs[0,0]
        assert int(a['state'][0].sum())==cs[-1,0]+cs[-1,6]
    traces.append(trace)
g4files=list((BASE/'G4-Block-Training-2026-10-03-v2/package/generation-runs').glob('*/worker/checkpoint-0000.pt'));assert len(g4files)==1
g4=torch.load(g4files[0],map_location='cpu',weights_only=False);g6=torch.load(run/'worker/checkpoint-0000.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(g4['model'][k],g6['model'][k]) for k in g4['model'])
old_final=torch.load(g4files[0].with_name('checkpoint-0003.pt'),map_location='cpu',weights_only=False);new_final=torch.load(run/'worker/checkpoint-0003.pt',map_location='cpu',weights_only=False)
assert [(x['row_index'],x['start']) for x in traces]==[(x['row_index'],x['start']) for x in old_final['trace']]
assert torch.equal(old_final['rng']['firing'],new_final['rng']['firing'])
record=dict(run=run.name,controlled_seconds=result['wall_seconds'],worker_seconds=result['worker']['wall_seconds'],retained_updates=3,exact_recoveries=recoveries,evidence_payloads=len(em),evidence_sha256=receipt['sha256'],verified_step_accounts_and_caps=192,all_start_hashes_verified=True,initial_parameters_equal_g4=True,row_start_schedule_equal_g4=True,firing_rng_equal_g4=True,quality_evaluated=False)
save(OUT,'rehearsal-verification.json',record)
package=json.loads((OUT/'package-receipt.json').read_text());nb=json.loads((OUT/'NCA-G6-Paced.ipynb').read_text());s='\n'.join(''.join(c['source']) for c in nb['cells'])
assert package['sha256'] in s and 'APPROVED_G6_JOB=False' in s
for c in nb['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
audit=json.loads((AUDIT/'result.json').read_text());probe=json.loads((AUDIT/'fixed-state-pacing-probe.json').read_text());paced=json.loads((AUDIT/'paced-rollout-result.json').read_text())
report=f'''# Objective and growth-timing audit — 2026-10-04

The evidence shifts the next experiment toward **growth timing**, rather than
another input channel or an immediate loss-weight change. G4 usually already
scores advancing teacher cubes higher than other teacher cubes, but its
admission rule accepts many above-threshold proposals in the same step.

## What was audited

Used the existing final G4 checkpoint on all 27 TRAIN cases, seed-only starts,
firing seed 2101 and a fixed 32-step observation window. Saved 864 step records
and 189 gradient snapshots at steps 1, 4, 8, 12, 16, 24 and 32. Gradients were
decomposed into teacher BCE, weighted local volume and global band terms at
both actual logits and neutral zero logits. State, eligibility and firing were
fixed during each gradient probe. These are instantaneous logit gradients,
not full-rollout parameter updates or causal estimates of training outcomes.
One complete 32-step replay matched the original model's field, hidden state
and count trace exactly. No optimizer updates or held-out evaluation occurred.

“Progress” here means lowering the minimum context cube-graph distance to the
opposite interface. “Other” can include useful target volume; it does not mean
wrong geometry or exclusively sideways movement. Only comparisons with both
teacher-positive groups available, before connection and before the cap, are
used for the matched statistics below.

## Findings

- In 278 matched decision steps, other teacher-positive proposals had a higher
  mean score than progress proposals only 10 times. In 131 steps, both groups'
  mean probabilities exceeded the hard 0.5 acceptance threshold.
- At all 61 matched gradient snapshots, neutral BCE treated both positive
  groups equally. The complete current loss, at actual logits, still encouraged
  the other positive group on average in all 61 snapshots. It teaches eventual
  target membership, not a preferred time to add each target cube.
- Median instantaneous gradient L1 was 0.9723 for BCE, 0.00764 for weighted local
  volume and 0.00653 for global band error. The inspected local gradients do not
  support blaming the global band term alone. These magnitudes are not parameter
  gradients and do not isolate every effect of each term through training.
- Only 2,905 of 17,212 newly added voxels in the matched decision steps belonged
  to strictly advancing cube admissions. This excludes other useful additions
  from the progress category; it is not a fraction of “correct” voxels.
- Across G4's retained training traces, 13,549 of 16,384 steps (82.7%) began at
  the global cap; G5 had 12,301 (75.1%). Growth is impossible in those states,
  although hidden-state updates and gradients still occur. These are not all
  computationally meaningless steps, but training is dominated by capped states.

## One predetermined pacing probe

Set a per-step allowance K=max(9,ceil((C-27)/63)), where C is the unchanged global
ceiling. The first cube still contains 27 cells and follows the original seed
rule. Later steps admit at most K new voxels through the original whole-cube,
overlap-aware score ordering. The floor 9 allows one maximally costly adjacent
cube. Unused allowance does not carry over. The same K applies at 64 and 128
steps; it is never recomputed from the requested evaluation horizon.

In 67 saved states with available progress proposals, pacing increased the
share of added volume assigned to progress in 39 states, tied in 3 and reduced
it in 25. Aggregated progress share changed from681/3994 (17.1%) to386/937
(41.2%), while absolute progress additions fell. A fixed-state replay cannot
establish whole-run success, so the same single policy was also rolled out.
There was no quota search or coefficient tuning.

## Complete TRAIN-only counterfactual

Existing G4 weights were held fixed. Only the admission allowance changed.
Both 64-step and 128-step outputs were retained for every TRAIN case. Original
G4 baseline fields already reached their immutable global cap by step32, so
their saved occupancy is exactly the later baseline occupancy under the
monotone cap. No hidden-state equivalence at later steps is claimed.

| Check | Original G4 | Paced at 64 | Paced at 128 |
|---|---:|---:|---:|
| All nine families pass | 2/27 | 12/27 | 11/27 |
| Access | 2/27 | 22/27 | 22/27 |
| Coverage | 3/27 | 24/27 | 24/27 |
| Facade | 26/27 | 14/27 | 13/27 |
| Each of the other six families | 27/27 | 27/27 | 27/27 |

Only 18/27 paced cases satisfy the 5% mass-stability limit. Maximum mass change
is18.30%. Median volume-fraction error at64 is0.245 percentage points and maximum
3.322 points. All individual scores and geometries, including regressions, are
saved under paced-rollouts. This is not a trained G6 result or generalization
evidence. Neither policy is accepted for deployment.

## Decision and ready next experiment

Prepare G6 as a **pacing-only training change from G4**: same 61-input model,
fresh paired initialization, TRAIN data, teacher stages, losses and firing.
Do not carry G5's extra inputs into this experiment. This isolates the admission
schedule and its induced training-state distribution before adding explicit
task losses. It is a test of a plausible mechanism, not a promised solution.
Facade and stability regressions remain explicit risks.

The new code matches the independent paced rollout exactly for both checked
horizons and all128 step counts. Its loss function is byte-for-byte equal to
G4's function. A packaged CPU rehearsal completed3retained updates and two exact
full-payload/state recovery replays in{result['wall_seconds']:.3f}s. All192step ceilings
and accounting records were verified. Initial weights, row/start schedule and
firing consumption match G4. No paid job was launched during this work.

Next package: {OUT.name}. One proposed Tesla T4 job, seed1201,256updates64steps,
maximum600controlledseconds; setup/export/download/idle extra. Explicit approval
is still required. No automatic retry. Frozen development gates remain unchanged.
No development or reserved evaluation occurred in this audit. Repeated prior
development use remains a limitation to disclose in the next review.

All work is documented and locally archived. The repository checkout has not
been synchronized and its older RESUME is stale. Use the latest package's
RESUME.json. Archives are on the same disk, not an off-device backup. No Drive,
push, publication or live-model replacement occurred; MG7 remains live.
'''
with (AUDIT/'FINDINGS.md').open('x',encoding='utf-8') as f:f.write(report)
save(AUDIT,'RESUME.json',dict(status='Objective/timing audit and one TRAIN-only pacing counterfactual complete',next=str(OUT/'RESUME.json'),previous=str(BASE/'G5-Final-Review-2026-10-04/RESUME.json'),findings=str(AUDIT/'FINDINGS.md'),paid_run_authorized=False,development_or_reserved_evaluation=False,optimizer_updates_in_audit=0,repository_sync_pending=True,live_model='MG7 unchanged'))
save(AUDIT,'project-record.json',dict(event='G6 decision based on objective/timing audit',evidence={'gradient_snapshot_count':189,'matched_steps':278,'matched_gradients':61,'fixed_state_probe':probe['summary'],'train_paced_summary':paced['summary'],'stable_cases':paced['stable_within5percent']},decision='One focused pacing-only training pilot;do not attribute failure solely to missing destination information or global band weight.',limitations=['TRAIN-only fixed G4weights in pacing diagnostic','Instantaneous gradients,not network-update causality','Facade/stability regressions','No trained G6quality result','No held-out evaluation'],next=str(OUT/'RESUME.json')))
dependencies={}
for folder in [BASE/'G4-Final-Review-2026-10-04',BASE/'G5-Final-Review-2026-10-04']:
    for p in (folder/'import/worker').glob('update-*.json'):dependencies[str(p)]=sha(p.read_bytes())
save(AUDIT,'training-trace-dependencies.json',dependencies)
save(AUDIT,'source-fingerprints.json',{p.relative_to(AUDIT).as_posix():sha(p.read_bytes()) for p in sorted((AUDIT/'source').rglob('*.py'))})
save(OUT,'readiness.json',dict(status='ready_for_explicit_one_job_approval',package=package,gpu='Tesla T4',seed=1201,updates=256,steps=64,max_controlled_seconds=600,paid_run_authorized=False,setup_export_idle_extra=True,rehearsal=record,evidence_basis=str(AUDIT/'FINDINGS.md'),development_or_reserved_evaluation=False,repository_sync_pending=True,live_model='MG7 unchanged'))
save(OUT,'RESUME.json',dict(status='G6 pacing-only package verified;one paid job pending approval',previous=str(AUDIT/'RESUME.json'),folder=str(OUT),next='Ask approval for ONE seed1201 Tesla T4 run,256updates64steps,600controlledseconds,setup/export/idle extra,no retry. Open notebook and upload this ZIP;approval flag remainsFalse. Receive FULL ZIP+receipt. Verify exact package identity,all256start hashes,quota and every effective step ceiling,7column accounts and exact recovery. Column3 is now allowance rejection,not solely global-cap rejection. Evaluate final256 with PacedNCA on unchanged9development requests at64/128 using same frozenquota,CPUfloat32,firing2101. Compare G4;report facade and stability regressions. Keep reserved labels unopened.',model_class='nca.paced_generation.PacedNCA',review_parent=str(BASE/'G4-Final-Review-2026-10-04/review-script.py'),paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save(OUT,'project-record.json',dict(event='G6 pacing-only implementation and package complete',change='Effective admission ceiling min(global ceiling,current mass+K) after first cube;K=max(9,ceil((C-27)/63));unchanged loss/model/data/firing.',trace_semantics='Column3 allowance_rejected_blocks;quota and per-step ceilings saved separately.',checks=record,local_prior='Existing G4weights paced on27TRAIN;12/27valid64,11/27valid128;facade/stability regressions retained.',paid_run_authorized=False,quality_evaluated=False,repository_sync_pending=True))
notes=f'''# G6 handoff

G6 changes how quickly cubes are admitted. It uses G4's model and losses with
the same fresh initialization; no G5 distance inputs or trained warm start.
Read {AUDIT/'FINDINGS.md'} for the evidence and limitations.

The fixed allowance is K=max(9,ceil((C-27)/63)) new voxels per step after the
first seed-containing cube. The global volume ceiling and nine families remain
unchanged. Quota stays fixed in128step evaluation. This does not guarantee
connection,facade compliance,requested size or stability.

The independent TRAIN diagnostic improved all-nine validity from2/27to12/27
at64steps,using unchanged G4weights. Facade compliance worsened and only18/27
cases were stable. This is preliminary TRAIN evidence,not trained G6performance.

Local package/trajectory comparison and exact recovery passed. Rehearsal:
{run.name},3updates plus two replays,{result['wall_seconds']:.3f}controlledseconds.
Initial parameters and row/start/firing schedules match G4. See PROTOCOL.md.

After explicit approval for one T4job,256updates64steps,maximum600controlled
seconds,open NCA-G6-Paced.ipynb in Colab and upload NCA-G6-Paced-Package.zip.
Set APPROVED_G6_JOB=True only after that approval;distributed flag isFalse.
Setup/export/download/idle extra. Run once. Return full evidenceZIP+receipt,
including failures. No automatic retry,Drive operation or model promotion.

Results and resume records are locally saved. Repository synchronization and
off-device backup remain pending. MG7 stays live.
'''
with (OUT/'IMPLEMENTATION-NOTES.md').open('x',encoding='utf-8') as f:f.write(notes)
for file,target in [('check_g6_package.py','check-package.py'),('finalize_g6.py','verify-and-record.py')]:shutil.copyfile(Path(__file__).with_name(file),OUT/target)
archives=[]
for folder in [AUDIT,OUT]:
    files={p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    save(folder,'milestone-manifest.json',{'files':files});files['milestone-manifest.json']=sha((folder/'milestone-manifest.json').read_bytes())
    archive=folder.with_suffix('.verified.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(folder/name,name)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
    ar=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
    archive.with_suffix('.receipt.json').write_text(json.dumps(ar,indent=2));archives.append(ar)
print(json.dumps(dict(rehearsal=record,archives=archives,package=package),indent=2))
