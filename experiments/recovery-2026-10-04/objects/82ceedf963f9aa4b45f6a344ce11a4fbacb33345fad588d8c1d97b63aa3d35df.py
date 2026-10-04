from pathlib import Path
import json,zipfile,hashlib,shutil,sys
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G4-Block-Training-2026-10-03-v2';FIRST=BASE/'G4-Block-Training-2026-10-03'
sys.path.insert(0,str(OUT/'package'))
from nca.block_generation import full_origins,connected
from nca.block_reference import cube_union
from nca.generation_package import verify
sha=lambda raw:hashlib.sha256(raw).hexdigest()
write=lambda path,value:path.write_text(json.dumps(value,indent=2),encoding='utf-8')
records=[]
for folder in [FIRST,OUT]:
    manifest,data=verify(folder/'package');runs=list((folder/'package/generation-runs').glob('*/result.json'));assert len(runs)==1
    run=runs[0].parent;result=json.loads(runs[0].read_text());assert result['status']=='completed' and result['worker']['completed']==3 and result['cleanup']['active_processes_after_stop']==0
    assert result['request']['manifest_sha256']==sha((folder/'package/manifest.json').read_bytes())
    archive=run.with_suffix('.zip');receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(archive.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        em=json.loads(z.read('evidence-manifest.json'));assert len(em)==receipt['files']
        assert len(z.namelist())==len(set(z.namelist()))==len(em)+1 and set(z.namelist())==set(em)|{'evidence-manifest.json'}
        assert all(sha(z.read(k))==v for k,v in em.items())
    traces=[]
    for i in range(1,4):
        trace=json.loads((run/f'worker/update-{i:04d}.json').read_text());counts=np.asarray(trace['admission_counts']);assert counts.shape==(64,7)
        assert (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[1:,0]==(counts[:-1,0]+counts[:-1,6])).all()
        with np.load(run/f'worker/training-{i:04d}.npz',allow_pickle=False) as a:start=a['start'].copy();field=a['state'][0].astype(bool)
        assert sha(start.tobytes(order='C'))==trace['start']['sha256']
        assert int(start.sum())==trace['start']['occupied']==counts[0,0]
        assert int(field.sum())==counts[-1,0]+counts[-1,6]<=trace['budget'][2]
        with np.load(folder/'package'/data['rows'][trace['row_index']]['arrays'],allow_pickle=False) as a:legal=a['condition'][0].astype(bool)
        assert not (field&~legal).any() and not (start.astype(bool)&~field).any() and connected(field)
        if field.sum()!=1:assert np.array_equal(cube_union(full_origins(field)),field)
        traces.append(dict(update=i,start_kind=trace['start']['kind'],start_mass=int(start.sum()),final_mass=int(field.sum()),loss=trace['loss'],accepted_blocks=int(counts[:,2].sum()),rejected_blocks=int(counts[:,3].sum())))
    recoveries=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]]
    assert all(r['full_payload_equal'] and r['state_equal'] for r in recoveries)
    records.append(dict(folder=str(folder),run=run.name,controlled_seconds=result['wall_seconds'],worker_seconds=result['worker']['wall_seconds'],evidence_files=len(em),evidence_sha256=receipt['sha256'],recoveries=recoveries,traces=traces))
write(OUT/'rehearsal-verification.json',records)

# The second package changes only removal of unused inherited loss metadata.
with zipfile.ZipFile(FIRST/'NCA-G4-Block-Package.zip') as a,zipfile.ZipFile(OUT/'NCA-G4-Block-Package.zip') as b:
    changed=[k for k in a.namelist() if a.read(k)!=b.read(k)]
assert set(changed)=={'nca/generation_training.py','manifest.json'}
source_a=(FIRST/'package/nca/generation_training.py').read_text()
source_b=(OUT/'package/nca/generation_training.py').read_text()
assert source_a.replace('LOSS={**BASE_LOSS,"global_band":1.0}','LOSS={"frontier_positive":1.0,"frontier_negative":1.0,"volume":.25,"cube":3,"global_band":1.0}')==source_b
for i in range(4):
    paths=[Path(r['folder'])/'package/generation-runs'/r['run']/f'worker/checkpoint-{i:04d}.pt' for r in records]
    states=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    assert all(torch.equal(states[0]['model'][k],states[1]['model'][k]) for k in states[0]['model'])
write(OUT/'metadata-correction.json',dict(predecessor=str(FIRST),changed_members=changed,reason='Removed inherited intact_negative=1.5 setting unused by G4; effective BCE is1:1.',retained_checkpoints_model_equal=True,original_attempt_preserved=True))

g3_checkpoints=list((BASE/'G3-Budget-Training-2026-10-03/package/generation-runs').glob('*/worker/checkpoint-0000.pt'));assert g3_checkpoints
g3=torch.load(g3_checkpoints[0],map_location='cpu',weights_only=False)
g4=torch.load(OUT/'package/generation-runs'/records[-1]['run']/'worker/checkpoint-0000.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(g3['model'][k],g4['model'][k]) for k in g3['model'])
write(OUT/'initialization-pairing.json',dict(g3_checkpoint=str(g3_checkpoints[0]),g3_checkpoint_sha256=sha(g3_checkpoints[0].read_bytes()),all_initial_parameters_equal=True,training_data_values_unchanged=True,stages_and_firing_rng_semantics_changed=True))
checks=json.loads((OUT/'implementation-checks.json').read_text());receipt=json.loads((OUT/'package-receipt.json').read_text())
nb=json.loads((OUT/'NCA-G4-Block.ipynb').read_text());source='\n'.join(''.join(c['source']) for c in nb['cells'])
assert 'APPROVED_G4_JOB = False' in source or 'APPROVED_G4_JOB=False' in source
assert receipt['sha256'] in source and 'files.upload()' in source and 'len(uploaded)!=1' in source
write(OUT/'verification-attempt-1.json',dict(status='verifier_assertion_corrected',reason='Verifier incorrectly required a literal archive filename. Notebook intentionally accepts one uploaded ZIP under any filename and checks exact SHA256; digest was already correct.',package_changed=False,training_rerun=False,script='verification-attempt-1.py'))
for c in nb['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
shutil.copyfile(Path(__file__).with_name('check_g4.py'),OUT/'check-implementation.py')
shutil.copyfile(__file__,OUT/'verify-and-record.py')
readiness=dict(status='ready_for_explicit_one_job_approval',paid_run_authorized=False,updates=256,steps=64,seed=1201,gpu='Tesla T4',max_controlled_seconds=600,setup_export_idle_extra=True,package=receipt,local_checks=checks,rehearsal=records[-1],development_or_reserved_access=False,quality_acceptance=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged')
write(OUT/'readiness.json',readiness)
write(OUT/'RESUME.json',dict(status=readiness['status'],previous=str(BASE/'G4-Block-Growth-Design-2026-10-03/RESUME.json'),latest_folder=str(OUT),superseded_package=str(FIRST),next='Ask approval for ONE seed1201 Tesla T4 job:256 updates,64 steps,600 controlled seconds, no retry. Open latest v2 notebook and upload v2 ZIP only after approval. Receive FULL evidence ZIP+receipt; verify all hashes,identity,recoveries,starts and7-column admission traces. If complete256,review final checkpoint on the same9G1 development requests at64 and128 steps using G4 BlockNCA,CPUfloat32,firing2101. Preserve frozen G3 gates and comparison. Do not use G3 four-column accounting for G4. Reserved labels remain unopened.',run_command='python scripts/colab_generation.py --device cuda:0 --seed 1201 --seconds 600 --approved-seed-job',command_requires_explicit_approval=True,review_source_parent=str(BASE/'G3-Final-Review-2026-10-03/source'),repository_sync_pending=True,repository='C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA',paid_run_authorized=False,drive_operations=0,live_model='MG7 unchanged'))
notes=f'''# G4 implementation and next run

G4 now proposes complete overlapping3x3x3 cubes. This addresses the thin additions
observed in G3 while keeping overall building mass and the same nine families.
The existing27TRAIN conditions,seed values and target fields are unchanged.
New teacher starts are connected unions of cubes; inference starts from the
same scene-defined single voxel. Origin firing changes the random-number stream.
See PROTOCOL.md for the complete frozen algorithm,losses,budget and review gates.

The differentiable union accounts for overlapping proposals without summing the
same empty voxel repeatedly. It is an independent-proposal surrogate before hard
ranking/cap,not the expected output of the constrained transition. First-seed
steps use only origin classification loss because at most one cube is admitted.
The budget band remains unchanged. A leftover gap too small for a full cube can
stall growth. Enforced thickness,legality and budget do not prove learned quality.

Hard admission currently runs on CPU,with one probability/firing transfer per
step. Measured CPU eligibility+admission median {checks['cpu_admission_seconds']['median']*1000:.3f}ms,
max {checks['cpu_admission_seconds']['max']*1000:.3f}ms over20synthetic32cubed cases.
These timings exclude GPU transfer and do not predict complete GPU job time.
The fixed600s supervisor remains the limit; partial/failure evidence is retained.

Local verification passed analytic two-cube union probabilities,finite gradients
including extreme logits,correct band-gradient direction,seed-only loss,27
independently reconstructed TRAIN cube graphs,12 reference cases,20 crowded
budget cases,connected/bulk/legal invariant checks and invalid-start rejection.
G3 checkpoints are rejected by identity. Fresh model parameters equal the G3
initialization,so G4 does not silently import trained weights.

The final packaged CPU rehearsal {records[-1]['run']} completed3retained updates
and2full-payload/state recovery replays in {records[-1]['controlled_seconds']:.3f}s controlled time.
All23evidence payload hashes verified. All192retained step accounts and saved
start hashes verified. Recovery covers a teacher-cube start and a true seed start.
These are engineering checks,not a G4 quality benchmark. No held-out or reserved
evaluation and no paid GPU work occurred. GPU compatibility is checked inside
the proposed job; it has not yet been demonstrated for this new operator.

The first package/rehearsal is preserved in {FIRST.name}. It passed; the v2 package
removes an unused inherited intact-negative loss setting from metadata so the
published objective exactly matches execution. No model math changed. Model
weights after each of the3updates match between the two rehearsals. Use v2 only.

Next action: approve one Tesla T4 job,seed1201,256updates64steps,maximum600
controlled seconds. Setup,export and idle time are extra. No automatic retry.
Open NCA-G4-Block.ipynb in Colab and upload NCA-G4-Block-Package.zip from THIS
folder. The notebook approval flag remains False. After explicit approval,
set APPROVED_G4_JOB=True,run once,and return FULL evidence ZIP plus receipt.
Do not upload the rehearsal evidence or previous G3/G4-v1 package in its place.

Results,changes and resumption instructions are saved locally here. Repository
sync remains pending because this session has no granted write access to the
project checkout. Its older RESUME is stale; use this folder's RESUME.json.
The milestone ZIP is a verified same-disk archive,not an off-device backup.
No Drive operation,push,publication or live-model replacement was performed.
'''
(OUT/'IMPLEMENTATION-NOTES.md').write_text(notes,encoding='utf-8')
write(OUT/'project-record.json',dict(milestone='G4 learned cube proposals implemented and locally rehearsed',changes=['cube-origin network output','full-cube TRAIN stages','overlap-aware differentiable voxel union','seed-only classification objective','detached CPU whole-cube budget admission','explicit7-column accounting','saved exact start fields and hashes','distinct semantic/checkpoint identity'],decisions=['Accept hybrid CPU admission for this bounded pilot; measure GPU cost inside capped job','Preserve same9families and frozen development gates','No separate paid preflight; checks embedded in proposed single job'],attempts=records,limitations=['No trained G4 quality result','GPU operator compatibility and speed unverified','Irreversible wrong additions and capacity stalls remain possible','CPU benchmark excludes transfers','Repository sync and off-device backup pending'],next=str(OUT/'RESUME.json')))
print(json.dumps(dict(folder=str(OUT),rehearsal=records[-1],package=receipt),indent=2))
