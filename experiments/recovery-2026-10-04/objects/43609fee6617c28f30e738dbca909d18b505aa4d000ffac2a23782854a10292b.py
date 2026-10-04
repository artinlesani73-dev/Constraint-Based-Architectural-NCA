from pathlib import Path
import json,hashlib,shutil,zipfile
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G7-Vertical-Training-2026-10-04-v2';FAILED=BASE/'G7-Vertical-Training-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(root,name,v):
 with (root/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
v=json.loads((OUT/'verification.json').read_text());p=json.loads((OUT/'package-receipt.json').read_text());assert v['passed']
readme=f'''# G7 — ready for one approved Colab run

Use only the files in this **v2** folder. The original preparation folder is
historical evidence and contains a package that failed before training.

## Files to use

- Notebook: NCA-G7-Vertical.ipynb
- Upload when prompted: NCA-G7-Vertical-Package.zip
- ZIP size: {p['bytes']:,} bytes
- ZIP SHA256: {p['sha256']}
- Manifest SHA256: {p['manifest_sha256']}

## What this run tests

Broaden training geometry while keeping the G6 model, losses, pacing, grid and
nine constraint families fixed. Retain27 original examples byte-for-byte and
add18 new examples with upward/downward connection offsets, paired elevations
and unequal building heights. All18 new teachers pass the full geometry contract
and budget checks. All45 rows load in both single-seed and teacher-stage modes.

Six new training scenes have west/east connection origins at grid heights
(8,12),(12,8),(10,16),(16,10),(14,14),(16,16). Four fresh reserved scenes are
frozen locally before training, at (9,15),(15,9),(12,18),(18,12), with different
horizontal positions and building heights. No reserved teacher or model output
has been generated. Each scene has16%,24%,32% volume requests. These are related
synthetic variations, not an independent architectural dataset.

Training remains256 updates. Under the frozen shuffle, original examples get
155 updates and new examples101; each row is visited5 or6 times. Thus per-example
exposure changes relative to G6. This is a fixed-compute data-distribution study,
not a perfectly isolated causal estimate or an equal-epochs comparison.

## Local verification

Three retained CPU updates and two exact recovery replays completed in
{v['controlled_seconds']:.2f} controlled seconds. All23 exported payload hashes,
192 transition accounts/ceilings and both complete checkpoint/state recoveries
passed. All initial model parameters match G6 and the neural model/loss source
is byte-identical. No model-quality evaluation was performed. Local CPU recovery
does not certify CUDA; the GPU run retains its device and recovery checks.

The first local package failed at0 updates because the inherited loader expects
the scene seed under `damaged`. Version2 adds that compatibility alias with the
same single-seed values. All geometry arrays and labels remain unchanged. The
failed package, logs and receipt remain in the original folder. No paid compute
was used for either local rehearsal.

## Paid allowance — approval pending

One fresh Tesla T4 run, seed1201,256 updates of64 steps, at most600 controlled
seconds. Includes admission probes, exact recovery checks and per-update evidence.
Setup, upload, export, download and idle time are additional billed runtime.
Expected stack: Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, CUDA13.0,
cuDNN92700. Stop on mismatch or failed checks; do not bypass guards or retry.

1. After approving the allowance, upload the notebook to Colab and select T4.
2. Run the upload cell and select the package ZIP from this v2 folder.
3. Change APPROVED_G7_JOB=False to True only after this exact run is approved.
4. Run the training cell once. Then download the full evidence ZIP and receipt,
   including after a failure, and share both local files for review.
5. Disconnect the runtime after downloads finish to avoid idle compute use.

No Drive mounting, uploading or other Drive operation is included.

## Frozen review after returned evidence

Verify every artifact and completed update; review only final256. Evaluate the
21 legacy regression requests separately from12 fresh reserved requests, each
at64 and128 steps with firing2101 and fixed threshold/quota. Require every case
to pass all9 families at both horizons, median fraction error<=.02 and maximum
<=.04 in each cohort/horizon, and every mass change<=5%. Inspect geometry.
Do not reroll, tune, select intermediate checkpoints or hide failures.

This milestone prepares training; it does not establish improved G7 quality.
MG7 remains live. Repository synchronization and off-device backup are pending.
Use RESUME.json here; the checkout's old D098 resume is stale. The original
NCA next-phase report stays local and untouched.
'''
with (OUT/'START-HERE.md').open('x',encoding='utf-8') as f:f.write(readme)
save(OUT,'readiness.json',dict(ready_for_approval=True,paid_run_authorized=False,package=p,verification=v,allowance=dict(jobs=1,gpu='Tesla T4',seed=1201,updates=256,steps=64,max_controlled_seconds=600,setup_export_idle_extra=True,automatic_retry=False),fresh_reserved_inference=0,repository_sync_pending=True))
save(OUT,'RESUME.json',dict(status='G7 v2 ready;one paid run approval pending',previous=str(BASE/'G6-Reserved-Review-2026-10-04/RESUME.json'),failed_predecessor=str(FAILED),notebook=str(OUT/'NCA-G7-Vertical.ipynb'),package=str(OUT/'NCA-G7-Vertical-Package.zip'),package_sha256=p['sha256'],manifest_sha256=p['manifest_sha256'],verification=str(OUT/'verification.json'),next='Ask user for one T4 seed1201 job,256updates64steps,max600controlledseconds plus setup/export/idle. After approval use v2 notebook/ZIP. On returned full evidence ZIP+receipt verify all hashes,identity,recovery,trace/start/cap accounts and final256. Follow frozen-review.json:21legacy regression and12fresh reserved requests at64/128,report all failures;visual review before any integration. No checkpoint selection or paid retry.',paid_run_authorized=False,drive_operations=0,live_model='MG7 unchanged',repository_sync_pending=True,off_device_backup_pending=True))
save(OUT,'project-record.json',dict(event='G7 vertical-data package prepared and locally verified',date='2026-10-04',changes=['Froze6newTRAIN and4newreserved scenes before label construction','Built and verified18newTRAIN teachers;kept27original payloads','Changed package45row guard and provenance identifiers only;G6 model/loss unchanged','Recorded0-update packaging failure and corrected seed alias in separatev2','Passed3-update CPU rehearsal and2exact recovery replays','Frozen21regression plus12fresh request evaluation;no held-out inference','Notebook paid approval remainsFalse'],verification=v,package=p,decision='Ready for one bounded paid allowance;quality improvement unproven',repository_sync_pending=True,drive_operations=0))
save(FAILED,'SUPERSEDED.json',dict(status='Original package failed before training;do not upload',reason='Missing damaged compatibility alias',replacement=str(OUT),replacement_package_sha256=p['sha256'],old_evidence_preserved=True))
shutil.copyfile(__file__,OUT/'finalize-readiness.py')
for root in [FAILED,OUT]:
 files={f.relative_to(root).as_posix():sha(f.read_bytes()) for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
 save(root,'milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((root/'milestone-manifest.json').read_bytes())
 archive=root.with_suffix('.verified.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for name in files:z.write(root/name,name)
 with zipfile.ZipFile(archive) as z:
  assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==h for k,h in files.items())
 with archive.with_suffix('.receipt.json').open('x') as f:json.dump(dict(archive=str(archive),sha256=sha(archive.read_bytes()),payloads=len(files),bytes=archive.stat().st_size,verified=True,off_device_backup=False),f,indent=2)
print(json.dumps(dict(package=p,ready=True,archives_verified=2),indent=2))
