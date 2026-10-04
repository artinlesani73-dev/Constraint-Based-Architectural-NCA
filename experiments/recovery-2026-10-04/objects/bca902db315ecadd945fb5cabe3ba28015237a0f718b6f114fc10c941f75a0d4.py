from pathlib import Path
import sys,json,hashlib,zipfile,shutil
import torch,numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';ROOT=OUT/'package';OLD=BASE/'G8-Exposure-Training-2026-10-04'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from nca.generation_training import equal_tree
from nca.generation_package import verify
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
verify(ROOT);integration=json.loads((OUT/'integration-check.json').read_text());assert integration['passed']
runs=list((ROOT/'generation-runs').glob('*/result.json'));assert len(runs)==1
run=runs[0].parent;res=json.loads(runs[0].read_text());assert res['status']=='completed' and res['worker']['completed']==3 and res['cleanup']['active_processes_after_stop']==0
receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(run.with_suffix('.zip').read_bytes())==receipt['sha256']
with zipfile.ZipFile(run.with_suffix('.zip')) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'] and len(z.namelist())==len(set(z.namelist()))==len(m)+1 and set(z.namelist())==set(m)|{'evidence-manifest.json'}
 assert all(sha(z.read(k))==v for k,v in m.items())
recovery=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]]
assert all(r['full_payload_equal'] and r['state_equal'] for r in recovery)
oldrun=next((OLD/'package/generation-runs').glob('*/worker/checkpoint-0000.pt')).parent
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
a=load(oldrun/'checkpoint-0000.pt');b=load(run/'worker/checkpoint-0000.pt')
keys=[k for k in a if k!='identity'];assert all(equal_tree(a[k],b[k]) for k in keys)
a3=load(oldrun/'checkpoint-0003.pt');b3=load(run/'worker/checkpoint-0003.pt')
for k in ['sampler','rng','cuda_rng']:assert equal_tree(a3[k],b3[k]),k
assert not equal_tree(a3['model'],b3['model'])
phases={};active=0
for i in [1,2,3]:
 t=json.loads((run/f'worker/update-{i:04d}.json').read_text());old=json.loads((oldrun/f'update-{i:04d}.json').read_text())
 assert t['row_index']==old['row_index'] and t['start']==old['start']
 with np.load(run/f'worker/training-{i:04d}.npz',allow_pickle=False) as q:assert sha(q['start'].astype(np.uint8).tobytes())==t['start']['sha256']
 assert np.isfinite(t['loss']) and np.isfinite(t['pre_clip_gradient_norm'])
 cs=np.asarray(t['admission_counts']);caps=np.asarray(t['step_ceilings']);C=t['budget'][2];K=max(9,int(np.ceil((C-27)/63)))
 assert t['quota']==K and np.array_equal(caps,np.where(cs[:,0]==1,C,np.minimum(C,cs[:,0]+K))) and (cs[:,0]+cs[:,6]<=caps).all()
 assert len(t['access_phase_trace'])==64
 assert abs(t['loss']-(t['frontier_loss']+.25*t['volume_loss']+t['band_loss']+t['ranking_loss']))<1e-5
 for j,p in enumerate(t['access_phase_trace']):
  assert p['mass']==cs[j,0] and p['at_capacity']==(cs[j,0]==C)
  phases[p['phase']]=phases.get(p['phase'],0)+1;active+=p['ranking_active']
assert active>0
p=json.loads((OUT/'package-receipt.json').read_text());nb=json.loads((OUT/'NCA-G9-Access-Ranking.ipynb').read_text())
s='\n'.join(''.join(c['source']) for c in nb['cells']);assert p['sha256'] in s and 'APPROVED_G9_JOB=False' in s
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
verification=dict(passed=True,run=run.name,seconds=res['wall_seconds'],verified_payloads=len(m),exact_recovery=recovery,initial_numerical_keys_equal_g8=keys,three_starts_rows_and_rng_equal_g8=True,trained_weights_differ=True,step_accounts=192,access_phases=phases,ranking_active_steps=active,inference_parity=integration,quality_evaluated=False,fresh_reserved_inference=0)
save('verification.json',verification)
start=f'''# G9 — ready for one approved Colab run

Purpose: teach connection-building cubes to receive higher scores while retaining
G8's volume losses and pacing. Improvement is unproven.

1. Open NCA-G9-Access-Ranking.ipynb in Colab and select a Tesla T4 runtime.
2. Run the upload cell and select NCA-G9-Access-Ranking-Package.zip ({p['bytes']:,} bytes).
3. After approval of this exact allowance, change APPROVED_G9_JOB=False to True
   and run the training cell once.
4. Download the full results ZIP and matching receipt, including after failure.
5. Send both back here and disconnect the runtime after downloads finish.

Proposed allowance: ONE fresh seed1201 job,427updates64steps,maximum600 controlled
seconds; setup/export/download/idle extra. No automatic retry. G8 took336s;
G9 adds work and completion within the cap is not guaranteed. Do not bypass the
runtime guard if Colab changes. No Drive mounting.

Local checks passed:3retained updates, two exact checkpoint recoveries,
unchanged initialization/start schedule/random streams, finite gradients,
six fixed-weight64/128-step TRAIN inference comparisons, three exact
base-loss/forward comparisons. Rehearsal took{res['wall_seconds']:.2f}s.
This verifies engineering behavior, not improved model quality or CUDA parity.
GPU probes/recovery run inside the same bounded job.

Review only final427. Compare against45 prior requests and12 newly frozen cases;
G8 and G9 both run on those same new cases. Keep all old thresholds.

Package SHA256: {p['sha256']}
Manifest SHA256: {p['manifest_sha256']}

Everything remains local. Repository synchronization and off-device backup
remain pending. MG7 stays live.
'''
with (OUT/'START-HERE.md').open('x') as f:f.write(start)
save('readiness.json',dict(ready=True,paid_run_authorized=False,package=p,verification=verification,allowance=dict(jobs=1,gpu='Tesla T4',updates=427,steps=64,seed=1201,max_controlled_seconds=600,setup_export_idle_extra=True,automatic_retry=False)))
save('RESUME.json',dict(status='G9 integrated,locally verified and packaged;one specific paid allowance pending',previous=str(BASE/'G9-Access-Objective-Design-2026-10-04-v2/RESUME.json'),notebook=str(OUT/'NCA-G9-Access-Ranking.ipynb'),package=str(OUT/'NCA-G9-Access-Ranking-Package.zip'),package_sha256=p['sha256'],next='Obtain approval for ONE T4 seed1201 job427updates64steps,max600controlledseconds plus setup/export/idle. User runs notebook and returns fullZIP+receipt. Verify manifest/recovery/allstarts/all27328step accounts and ranking traces. Evaluate final427 only with frozen-review.json:45regression plus12fresh pairedG8/G9 at64/128. No retry or checkpoint selection.',paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,off_device_backup_pending=True,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G9 access ranking integrated and prepared',verification=verification,scientific_change='G8 loss plus margin1 weight1 TRAIN-only access ranking;original losses/model/data/inference unchanged',package=p,paid_run_authorized=False,failures_preserved=['Initial builder missing local massing_cases dependency;corrected path in separatev2','Initial integration diagnostic targetdtype Boolean;corrected to float with failed script retained;training package unchanged']))
with (OUT/'CHANGELOG.md').open('x') as f:f.write('''# G9 integration milestone
Added versioned TRAIN-only cached graph labels, ranking loss and ranked rollout.
Inference contains no route labels and matches G8 for fixed weights in checked cases.
Trainer logs per-step phase/group counts/cap status and mean ranking loss.
All45 training payloads and original paced source bytes unchanged.
Same427 updates and64-step training horizon, same numerical initialization and RNG.
Fresh4scenes x3requests frozen locally, excluded from training ZIP; paired G8/G9 review.
Local3-update packaged rehearsal and integration checks passed. No paid training.
Builder dependency failure and diagnostic dtype failure preserved. No training retry.
All evidence locally archived; repository synchronization and off-device backup pending.
No Drive/push/publication/live-model replacement. See RESUME.json for continuation.
''')
shutil.copyfile(__file__,OUT/'finalize-preparation.py')
shutil.copytree(BASE/'G9-Access-Ranking-Training-2026-10-04',OUT/'failed-build-attempt')
files={f.relative_to(OUT).as_posix():sha(f.read_bytes()) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False),f,indent=2)
print(json.dumps(dict(package=p,seconds=res['wall_seconds'],active=active,phases=phases,archive=str(archive)),indent=2))

