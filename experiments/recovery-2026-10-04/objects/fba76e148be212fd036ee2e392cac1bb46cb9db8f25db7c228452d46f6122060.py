from pathlib import Path
import sys,json,hashlib,zipfile,shutil
import torch,numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G10-One-Sided-Training-2026-10-04';ROOT=OUT/'package';OLD=BASE/'G9-Access-Ranking-Training-2026-10-04-v2'
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
p=json.loads((OUT/'package-receipt.json').read_text());nb=json.loads((OUT/'NCA-G10-One-Sided.ipynb').read_text())
s='\n'.join(''.join(c['source']) for c in nb['cells']);assert p['sha256'] in s and 'APPROVED_G10_JOB=False' in s
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
verification=dict(passed=True,run=run.name,seconds=res['wall_seconds'],verified_payloads=len(m),exact_recovery=recovery,initial_numerical_keys_equal_g9=keys,three_starts_rows_and_rng_equal_g9=True,trained_weights_differ=True,step_accounts=192,access_phases=phases,ranking_active_steps=active,inference_parity=integration,quality_evaluated=False,fresh_reserved_inference=0)
save('verification.json',verification)
start=f'''# G10 — one-sided ranking, ready for approval

1. Open NCA-G10-One-Sided.ipynb in Colab; select Tesla T4.
2. Upload NCA-G10-One-Sided-Package.zip when prompted ({p['bytes']:,} bytes).
3. After approval for this exact job, set APPROVED_G10_JOB=True and run once.
4. Download the complete results ZIP and receipt, including after any failure.
5. Send both here; disconnect the runtime after downloading.

Proposed allowance: ONE fresh seed1201 T4 run,427updates64steps,max600 controlled
seconds. Setup/export/download/idle extra. No automatic retry or cap extension.
G9 took469s; G10 completion within600s is not guaranteed. Runtime mismatch stops
before training; do not bypass it. No Drive mounting.

Only change: ranking raises advancing scores without directly lowering other
teacher-positive scores. Margin1/weight1,base losses,architecture,data,pacing and
training exposure unchanged. This is an explicit semi-gradient, not a guarantee
other scores remain unchanged through shared network parameters.

Local3-update rehearsal passed in{res['wall_seconds']:.2f}s; two exact recoveries.
Initial numerical payload,start choices and random streams match G9; trained
weights differ. Fixed-weight inference parity,loss decomposition,semi-gradient
behavior and finite gradients passed. No model-quality or CUDA parity claim.

Frozen review:57 regression requests plus12 new reserved requests at64/128.
G9 is the primary paired baseline; G8 is the stability reference; both evaluated
on the same fresh12 as G10. All original gates retained; final427 only.

Package SHA256: {p['sha256']}
Manifest SHA256: {p['manifest_sha256']}

Documented and archived locally. Repository sync and off-device backup pending.
MG7 remains live. No paid job,Drive,push,publication or live swap occurred.
'''
with (OUT/'START-HERE.md').open('x') as f:f.write(start)
save('readiness.json',dict(ready=True,paid_run_authorized=False,package=p,verification=verification,allowance=dict(jobs=1,gpu='Tesla T4',updates=427,steps=64,seed=1201,max_controlled_seconds=600,setup_export_idle_extra=True,automatic_retry=False)))
save('RESUME.json',dict(status='G10 ready;one specific paid allowance pending',previous=str(BASE/'G9-Training-Diagnosis-2026-10-04/RESUME.json'),notebook=str(OUT/'NCA-G10-One-Sided.ipynb'),package=str(OUT/'NCA-G10-One-Sided-Package.zip'),package_sha256=p['sha256'],next='Ask approval for ONE fresh seed1201 T4 job427updates64steps,max600controlledseconds plus setup/export/idle. User returns fullZIP+receipt. Verify hashes,recovery,427starts,27328step accounts,ranking traces. Frozen final427 review:57regression plus12fresh,paired G9 primary and G8 stability reference. No retry or checkpoint selection.',paid_run_authorized=False,repository_sync_pending=True,off_device_backup_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G10 one-sided ranking integrated and verified',scientific_change='Detach other reference logits in ranking only;retain advancing gradient and all base terms',verification=verification,package=p,paid_run_authorized=False))
with (OUT/'CHANGELOG.md').open('x') as f:f.write('''# G10 implementation milestone
Changed access ranking to detach only non-advancing teacher-positive comparison
scores. Same numerical loss and advancing logit gradient;explicit semi-gradient.
Updated identity/config version and seed-loss metadata. No change to45TRAIN
data,rollout,model,pacing,427updates64steps,seed1201 or random draws.
Fresh4scenes x3requests frozen before training;physical context uniqueness checked
against all earlier manifests. G9/G8 paired fresh baselines explicitly frozen.
Packaged3-update CPU rehearsal,exact recoveries at2/3 and integration checks pass.
One paid allowance remains pending. No automatic retry,Drive,publication,push
or live promotion. All evidence archived locally;repo sync/off-device backup pending.
''')
shutil.copyfile(__file__,OUT/'finalize-preparation.py')
files={f.relative_to(OUT).as_posix():sha(f.read_bytes()) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False),f,indent=2)
print(json.dumps(dict(package=p,seconds=res['wall_seconds'],active=active,phases=phases,archive=str(archive)),indent=2))

