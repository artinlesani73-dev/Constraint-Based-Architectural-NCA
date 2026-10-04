from pathlib import Path
s=Path('finalize_g9.py').read_text()
head=s[:s.index("start=f'''")]
head=head.replace("OUT=BASE/'G9-Access-Ranking-Training-2026-10-04-v2'","OUT=BASE/'G10-One-Sided-Training-2026-10-04'").replace("OLD=BASE/'G8-Exposure-Training-2026-10-04'","OLD=BASE/'G9-Access-Ranking-Training-2026-10-04-v2'")
head=head.replace('NCA-G9-Access-Ranking.ipynb','NCA-G10-One-Sided.ipynb').replace('APPROVED_G9_JOB','APPROVED_G10_JOB').replace('equal_g8','equal_g9')
tail=s[s.index("files={f.relative_to(OUT)"):]
middle="""start=f'''# G10 — one-sided ranking, ready for approval

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
"""
Path('finalize_g10.py').write_text(head+middle+tail)
compile(head+middle+tail,'finalize_g10.py','exec')

