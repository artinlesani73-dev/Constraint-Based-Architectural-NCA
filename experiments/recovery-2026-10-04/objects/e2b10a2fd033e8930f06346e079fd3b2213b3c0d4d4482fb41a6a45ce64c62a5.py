from pathlib import Path
import sys,json,hashlib,zipfile,shutil
import torch
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G8-Exposure-Training-2026-10-04';AUDIT=BASE/'G7-Training-Diagnosis-2026-10-04';OLD=BASE/'G7-Vertical-Training-2026-10-04-v2';ROOT=OUT/'package'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from nca.generation_training import equal_tree
from nca.generation_package import verify
from nca.repair_portable import TrainingOrder
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(root,name,v):
 with (root/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
_,data=verify(ROOT);runs=list((ROOT/'generation-runs').glob('*/result.json'));assert len(runs)==1
run=runs[0].parent;res=json.loads(runs[0].read_text());assert res['status']=='completed' and res['worker']['completed']==3 and res['cleanup']['active_processes_after_stop']==0
receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(run.with_suffix('.zip').read_bytes())==receipt['sha256']
with zipfile.ZipFile(run.with_suffix('.zip')) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'] and len(z.namelist())==len(set(z.namelist()))==len(m)+1 and set(z.namelist())==set(m)|{'evidence-manifest.json'}
 assert all(sha(z.read(k))==v for k,v in m.items())
recoveries=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]];assert all(c['full_payload_equal'] and c['state_equal'] for c in recoveries)
oldrun=next((OLD/'package/generation-runs').glob('*/worker/checkpoint-0000.pt')).parent
paired=[]
for i in [0,3]:
 a=torch.load(oldrun/f'checkpoint-{i:04d}.pt',map_location='cpu',weights_only=False);b=torch.load(run/f'worker/checkpoint-{i:04d}.pt',map_location='cpu',weights_only=False)
 assert a.keys()==b.keys()
 keys=[k for k in a if k!='identity'];assert all(equal_tree(a[k],b[k]) for k in keys)
 paired.append(dict(update=i,equal_keys=keys,identity_separate=True))
for i in [1,2,3]:
 t=json.loads((run/f'worker/update-{i:04d}.json').read_text());cs=np.asarray(t['admission_counts']);caps=np.asarray(t['step_ceilings']);C=t['budget'][2];K=max(9,int(np.ceil((C-27)/63)))
 assert t['quota']==K and np.array_equal(caps,np.where(cs[:,0]==1,C,np.minimum(C,cs[:,0]+K))) and (cs[:,0]+cs[:,6]<=caps).all()
order=TrainingOrder(45,1203);visits=np.zeros(45,int)
for _ in range(427):visits[order.next()]+=1
p=json.loads((OUT/'package-receipt.json').read_text());nb=json.loads((OUT/'NCA-G8-Exposure.ipynb').read_text());s='\n'.join(''.join(c['source']) for c in nb['cells'])
assert p['sha256'] in s and 'APPROVED_G8_JOB=False' in s
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
verification=dict(passed=True,run=run.name,seconds=res['wall_seconds'],verified_payloads=len(m),exact_recovery=recoveries,g7_prefix_pairing=paired,step_accounts=192,planned427_visits=dict(min=int(visits.min()),max=int(visits.max()),total=int(visits.sum())),model_quality_evaluated=False,fresh_reserved_inference=0)
save(OUT,'verification.json',verification)
r=json.loads((AUDIT/'result.json').read_text());alloc=json.loads((AUDIT/'allowance-analysis.json').read_text())
totals={}
for label in ['G6','G7']:
 groups=list(alloc[label].values());unused=sum(g['unused_total'] for g in groups);allowance=sum(g['allowance_total'] for g in groups);rejected=sum(g['unused_categories']['whole_cube_allowance_rejections']['unused_voxels'] for g in groups)
 totals[label]=dict(allowance=allowance,unused=unused,unused_fraction=unused/allowance,unused_without_rejection=unused-rejected,unused_without_rejection_fraction=(unused-rejected)/unused)
save(AUDIT,'aggregate-summary.json',totals)
report=f'''# G7 TRAIN-only diagnosis and next decision — 2026-10-04

The slowdown is **later proposal scarcity**, not delayed seed activation.
Both models grow their first full cube at step1 in all45 cases. The next bounded
experiment will increase training exposure only; it does not presume that longer
training fixes connection allocation.

## Scope and verification

Compared frozen G6 and G7 final256 weights on all45 G7 TRAIN examples:27 shared
original examples and18 added vertical examples. Those18 were not G6 training
data and are identified separately. Each used one64-step rollout, firing2101,
the unchanged0.5 threshold and quota. No optimizer updates, parameter searches,
new teacher construction, or development/reserved inference occurred.
All5760 detached admission transitions were replayed exactly against captured
fields/counts. Saved all90 rollouts, birth masks, terminal states, per-step
eligible indices, probabilities and firing, summaries, checkpoints and source.
Teacher membership and context distance were analysis labels only, never model
inputs. Connection here means contact with the opposite interface from a
connected legal field; this is not a new nine-family quality benchmark.

## Findings

|64-step TRAIN diagnostic|G6 original27|G7 original27|G6 added18|G7 added18|
|---|---:|---:|---:|---:|
|First cube step, every case|1|1|1|1|
|Opposite interface reached|25/27|21/27|15/18|14/18|
|Median absolute volume error, pp|0.237|2.970|0.157|3.107|
|Median target shortfall, voxels|0|108|0|153.5|

The quota schedule has enough theoretical capacity to reach the requested volume
by64 in every case given the observed step1 start. It does not guarantee that
the model offers the cubes needed to use that capacity.

Across non-seed steps where the full per-step quota applies (before the global
ceiling truncates it), G6 leaves{totals['G6']['unused']:,}/{totals['G6']['allowance']:,}
allowance cells unused ({100*totals['G6']['unused_fraction']:.2f}%). G7 leaves
{totals['G7']['unused']:,}/{totals['G7']['allowance']:,} unused
({100*totals['G7']['unused_fraction']:.2f}%). These are lost step opportunities,
not necessarily distinct missing final voxels.

Of G7's unused allowance,{totals['G7']['unused_without_rejection']:,} cells
({100*totals['G7']['unused_without_rejection_fraction']:.2f}%) occur on steps with
no allowance rejection: all eligible probabilities are below threshold, firing
misses the few above-threshold proposals, or the offered cubes are exhausted.
Only the remaining15.57% occurs alongside whole-cube allowance rejection. This
is a descriptive partition, not a causal estimate of changing the cap. It argues
against treating quota packing or delayed startup as the dominant diagnosis.

Before connection, on steps with fired teacher-positive candidates in both
progress and other groups, G7's mean progress probability is<=0.5 in113/720
shared-data steps and127/592 added-data steps, versus11/666 and28/503 for G6.
The models follow different trajectories, so these are conditional diagnostics,
not comparisons on identical hidden states or proof of calibrated confidence.
Progress means lowering minimum context cube-graph distance to the opposite
interface; other growth can still be useful building volume.

The last64 retained training updates also start adding at step1 in both start
modes. Their losses are not directly comparable as causal evidence because
examples and states differ. Reduced exposure is a plausible unresolved factor,
not an established sole cause. Some G6 fields already reach the cap without
connecting, showing that learning to fill faster alone will not guarantee access.

## Single next intervention: G8 exposure

Retain G7's exact45 data payloads, initialization, model, optimizer, teacher
stages, loss, firing, hard transition, nine families and64/128 review horizons.
Change only total retained updates from256 to427:
ceil(256*45/27)=427. This gives9-10 visits per row and approximately restores
G6's mean exposure. The number was derived from dataset sizes, not selected by
a checkpoint sweep. This tests exposure; it is not a promised solution or an
equal-compute comparison with G7.

Use a fresh same-seed run so the full lineage remains unambiguous. Compare the
new update256 numerical payload against the previous G7 update256 after return,
then judge only final427. Package/model provenance can differ; no checkpoint
selection is allowed. The existing model has not been modified or promoted.

G8 package is prepared at `{OUT}`. Its three-update local rehearsal and both
recovery replays passed; update0 andupdate3 match all G7 numerical payload keys,
with identity kept separate. Four fresh reserved scenes (12 requests) were
frozen before any G8 training, with no labels or inference. The33 consumed
legacy requests become regression evidence. Both cohorts retain the existing
all-nine, size and stability requirements. No architecture/loss/threshold change
is bundled in this proposal.

One T4 run of427 updates is proposed, capped600 controlled seconds; setup,
export/download and idle are extra. Based on G7, roughly310-320 controlled
seconds is a planning estimate, not a guarantee. Approval is still pending.
No paid retry, Drive operation, push, publication or MG7 replacement occurred.

## Resume and preservation

Use G8 RESUME.json for the ready package and next user action. This diagnostic,
all raw result arrays, exact dependencies and its verified same-disk archive
remain retained. Repository synchronization and off-device backup remain pending.
The original report and all historical runs are untouched.
'''
with (AUDIT/'FINDINGS.md').open('x',encoding='utf-8') as f:f.write(report)
save(AUDIT,'RESUME.json',dict(status='TRAIN-only diagnosis complete;G8 exposure-only package locally verified',previous=str(BASE/'G7-Final-Review-2026-10-04/RESUME.json'),next=str(OUT/'RESUME.json'),findings=str(AUDIT/'FINDINGS.md'),optimizer_updates=0,heldout_inference=False,repository_sync_pending=True,paid_run_authorized=False))
start=f'''# G8 — ready for one approved run

Purpose: keep G7 fixed and increase training to427 updates (9-10 visits per
example), testing the reduced-exposure hypothesis. Improvement is unproven.

1. Open NCA-G8-Exposure.ipynb in Colab; choose Tesla T4.
2. Upload NCA-G8-Exposure-Package.zip when prompted ({p['bytes']:,} bytes).
3. After explicit approval for this run, set APPROVED_G8_JOB=True and run once.
4. Download the full evidence ZIP and receipt, even after a failure. Share both.
5. Disconnect the runtime after downloads complete to avoid idle compute.

Proposed allowance: one fresh seed1201 T4 job,427 updates64 steps,max600
controlled seconds. Setup, export, downloads and idle are extra. No retries.
Strict runtime checks and two recovery replays remain. No Drive mounting.

Package SHA256: {p['sha256']}
Manifest SHA256: {p['manifest_sha256']}

Local rehearsal: passed in{res['wall_seconds']:.2f}s; two exact recoveries; G7/G8
update0 andupdate3 numerical payloads identical. This is engineering verification,
not evidence of improved G8 quality or a substitute for CUDA checks.

Review only final427 against33 regression and12 fresh reserved requests at64/128.
Use frozen-review.json; never select a better intermediate checkpoint. Check256
prefix reproducibility separately. MG7 remains live; repository sync is pending.
'''
with (OUT/'START-HERE.md').open('x',encoding='utf-8') as f:f.write(start)
save(OUT,'readiness.json',dict(ready=True,paid_run_authorized=False,package=p,verification=verification,allowance=dict(jobs=1,gpu='Tesla T4',updates=427,steps=64,seed=1201,max_controlled_seconds=600,setup_export_idle_extra=True,automatic_retry=False)))
save(OUT,'RESUME.json',dict(status='G8 ready;one paid allowance pending',previous=str(AUDIT/'RESUME.json'),notebook=str(OUT/'NCA-G8-Exposure.ipynb'),package=str(OUT/'NCA-G8-Exposure-Package.zip'),package_sha256=p['sha256'],manifest_sha256=p['manifest_sha256'],next='Ask explicit approval for one T4 seed1201 job427updates64steps,max600controlledseconds plus setup/export/idle. Return full ZIP+receipt. Verify all427traces and recovery;compare update256 numerical payload toG7;evaluate final427 only using frozen-review.json and visually review. No retry or checkpoint selection.',paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,off_device_backup_pending=True,live_model='MG7 unchanged'))
save(OUT,'project-record.json',dict(event='G7 diagnosis complete;G8 exposure-only package prepared',diagnosis=str(AUDIT/'FINDINGS.md'),data_model_loss_unchanged=True,updates_changed=[256,427],verification=verification,package=p,paid_run_authorized=False))
shutil.copyfile(__file__,OUT/'finalize-preparation.py')
for root in [AUDIT,OUT]:
 files={f.relative_to(root).as_posix():sha(f.read_bytes()) for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
 save(root,'milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((root/'milestone-manifest.json').read_bytes())
 archive=root.with_suffix('.verified.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for name in files:z.write(root/name,name)
 with zipfile.ZipFile(archive) as z:
  assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
 with archive.with_suffix('.receipt.json').open('x') as f:json.dump(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False),f,indent=2)
print(json.dumps(dict(package=p,verification=verification,totals=totals),indent=2))
