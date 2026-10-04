from pathlib import Path
import json,zipfile,hashlib,shutil,sys
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G5-Destination-Guidance-2026-10-04';ROOT=OUT/'package';sys.path.insert(0,str(ROOT))
from nca.generation_package import verify
from nca.block_generation import connected,full_origins
from nca.block_reference import cube_union
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
manifest,data=verify(ROOT);runs=list((ROOT/'generation-runs').glob('*/result.json'));assert len(runs)==1
run=runs[0].parent;result=json.loads(runs[0].read_text());assert result['status']=='completed' and result['worker']['completed']==3 and result['cleanup']['active_processes_after_stop']==0
assert result['request']['manifest_sha256']==sha((ROOT/'manifest.json').read_bytes())
archive=run.with_suffix('.zip');receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(archive.read_bytes())==receipt['sha256']
with zipfile.ZipFile(archive) as z:
    em=json.loads(z.read('evidence-manifest.json'));assert len(em)==receipt['files']
    assert len(z.namelist())==len(set(z.namelist()))==len(em)+1 and set(z.namelist())==set(em)|{'evidence-manifest.json'}
    assert all(sha(z.read(k))==v for k,v in em.items())
recoveries=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]]
assert all(r['full_payload_equal'] and r['state_equal'] for r in recoveries)
traces=[]
for i in range(1,4):
    trace=json.loads((run/f'worker/update-{i:04d}.json').read_text());counts=np.asarray(trace['admission_counts']);assert counts.shape==(64,7)
    assert (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
    with np.load(run/f'worker/training-{i:04d}.npz',allow_pickle=False) as a:start=a['start'].copy();field=a['state'][0].astype(bool)
    assert sha(start.tobytes(order='C'))==trace['start']['sha256'] and int(start.sum())==trace['start']['occupied']==counts[0,0]
    assert int(field.sum())==counts[-1,0]+counts[-1,6]<=trace['budget'][2]
    with np.load(ROOT/data['rows'][trace['row_index']]['arrays'],allow_pickle=False) as a:legal=a['condition'][0].astype(bool)
    assert not (field&~legal).any() and not (start.astype(bool)&~field).any() and connected(field)
    assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)
    traces.append(trace)
final=torch.load(run/'worker/checkpoint-0003.pt',map_location='cpu',weights_only=False)
cue_weights=final['model']['first.weight'][:,61:];channel_norms=cue_weights.abs().sum((0,2,3,4)).tolist();assert all(x>0 for x in channel_norms)
g4files=list((BASE/'G4-Block-Training-2026-10-03-v2/package/generation-runs').glob('*/worker/checkpoint-0003.pt'));assert len(g4files)==1
g4=torch.load(g4files[0],map_location='cpu',weights_only=False)
assert [(x['row_index'],x['start']) for x in traces]==[(x['row_index'],x['start']) for x in g4['trace']]
# Firing generator consumption remains unchanged by the added context channels.
assert torch.equal(final['rng']['firing'],g4['rng']['firing'])
record=dict(run=run.name,controlled_seconds=result['wall_seconds'],worker_seconds=result['worker']['wall_seconds'],evidence_payloads=len(em),evidence_sha256=receipt['sha256'],recoveries=recoveries,retained_updates=3,verified_step_accounts=192,all_start_hashes_verified=True,terminal_geometry_invariants_verified=True,cue_weight_absolute_sums_after3=channel_norms,g4_row_and_start_schedule_equal=True,g4_firing_rng_equal=True,quality_evaluated=False)
save('rehearsal-verification.json',record)
package=json.loads((OUT/'package-receipt.json').read_text());nb=json.loads((OUT/'NCA-G5-Destination.ipynb').read_text());source='\n'.join(''.join(c['source']) for c in nb['cells'])
assert package['sha256'] in source and 'APPROVED_G5_JOB=False' in source and 'len(uploaded)!=1' in source
for c in nb['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
checks=json.loads((OUT/'implementation-checks.json').read_text())
save('readiness.json',dict(status='ready_for_explicit_single_job_approval',paid_run_authorized=False,gpu='Tesla T4',seed=1201,updates=256,steps=64,max_controlled_seconds=600,setup_export_idle_extra=True,package=package,rehearsal=record,training_contexts_audited=27,development_or_reserved_access=False,quality_claim=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('RESUME.json',dict(status='G5 context-only destination cue implemented,TRAIN feasibility and CPU exact recovery passed;paid job pending',previous=str(BASE/'G4-Final-Review-2026-10-04/RESUME.json'),folder=str(OUT),next='Ask explicit allowance for ONE G5seed1201 Tesla T4 job,256updates64steps,600controlledseconds,setup/export/idle extra,no retry. User opens NCA-G5-Destination.ipynb and uploads NCA-G5-Destination-Package.zip;flag remainsFalse until approved. Receive FULL ZIP+receipt. Verify hashes,exact package identity,256updates,starts,7column accounts,recovery and device cue checks. Evaluate final256only with GuidedNCA on unchanged9development cases at64/128,CPUfloat32,firing2101;compare G4 and disclose development reuse. Preserve all failures;do not open reserved labels or promote live automatically.',model_class='nca.guided_generation.GuidedNCA',run_command='python scripts/colab_generation.py --device cuda:0 --seed 1201 --seconds 600 --approved-seed-job',command_requires_explicit_approval=True,review_parent=str(BASE/'G4-Final-Review-2026-10-04/review-script.py'),paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
witness_min=min(x['witness_voxels'] for x in checks['train_cases']);witness_max=max(x['witness_voxels'] for x in checks['train_cases'])
notes=f'''# G5: destination guidance — implementation and handoff

G4 solved thin-fringe growth but consumed its allowed volume within14–22steps,
leaving7of9development cases short of the far interface. G5tests whether giving
local cube proposals immediate information about that interface helps them
allocate volume more effectively. This is a hypothesis,not a demonstrated fix.

## The focused change

Two new network inputs describe each legal cube position:shortest cube-graph
distance to the destination and whether the destination is reachable. The map
is calculated once per context from legal space and interface masks. It has no
teacher target,current mass,requested volume or training-stage input. Its
destination follows the existing oppositeXinterface convention;general
arbitrary-interface support is outside this pilot.

The first layer expands from61 to63inputs,adding128trainable weights. Existing
freshly initialized core weights exactly match G4;both new inputs start with
zero weights. All dataset bytes,losses,teacher stages,start schedule,origin
firing and whole-cube budget admission are unchanged. No trained checkpoint is
used to initialize the paid pilot. G4 remains available as the reference.

The cue is global context preprocessing,so this remains a hybrid NCA. It is
information for the network,not a forced route or a new constraint family.
The hard budget can still lock in mistakes. Coverage,facade and other existing
families still need independent evaluation.

## Completed local checks

All27TRAIN cases were audited without reading development or reserved data.
An independent graph-distance implementation agrees exactly with the cue.
Analytic distances,unreachable regions,invalid interface inputs and immutable
cache reproducibility passed. Both new inputs receive finite nonzero gradients.
G4checkpoints are rejected by semantic identity.

Each TRAIN scene has a context-only path of full overlapping cubes connecting
both interfaces within the current volume ceiling. These witness paths contain
{witness_min}–{witness_max}voxels. They are feasibility witnesses,not model output,
not all-nine valid massing designs,and are not included as teacher routes in
the training package. Existing27targets are unchanged. The cue's cold compute
time had median{checks['median_cold_cue_seconds']*1000:.2f}ms locally;this is not a GPU
training-time prediction. Bounded caching reuses geometry across volume requests.

Packaged CPU rehearsal{run.name} completed3retained updates plus two exact recovery
replays in{result['wall_seconds']:.3f}s. All{len(em)}evidence payload hashes and192step
accounts verified. Both cue channels acquired nonzero weights. Training rows,
saved starting fields and firing-generator consumption match the G4CPU rehearsal.
No quality benchmark or held-out inference was performed. GPU compatibility for
the expanded model is checked inside the proposed capped job,not assumed proven.

## Ready job and user action

One Tesla T4 job,seed1201,256updates,64steps,maximum600controlledseconds. Setup,
export,download and idle time are extra. Runtime guard and12admission probes,
cue-device check,union backward and exact recovery are embedded in that job.
No separate paid preflight and no automatic retry. See PROTOCOL.md for all
effective settings and unchanged final-checkpoint evaluation gates.

After explicit approval,open NCA-G5-Destination.ipynb in Colab,upload this
folder's NCA-G5-Destination-Package.zip,set APPROVED_G5_JOB=True and run once.
Return the full evidenceZIP and receipt,including after failure. The distributed
notebook still has its approval flagFalse. No paid job was launched here.

## Persistence and limits

Source,package hashes,individual TRAIN witness arrays,checks,rehearsal evidence,
decisions and exact continuation instructions are saved locally. Earlier stages
are untouched. Repository synchronization remains pending because this session
has no granted write access to the checkout;use this folder's RESUME.json.
The verified milestone archive is on the same disk,not an off-device backup.
No Drive operation,push,publication or live-model replacement was performed.
'''
with (OUT/'IMPLEMENTATION-NOTES.md').open('x',encoding='utf-8') as f:f.write(notes)
save('project-record.json',dict(date='2026-10-04',event='G5 destination cue implemented and locally verified',parent='G4-Final-Review-2026-10-04',decisions=['Test context-only destination distance and reachability as one focused feature intervention','Keep G4data,losses,cube stages,cap,threshold,firing and review gates','Use fresh initialization;cue weights startzero','Keep checks embedded in one proposed600second job'],results=record,limitations=['No trained G5quality result','Cue only supports existing oppositeXinterface convention','Route witnesses do not demonstrate complete massing validity','New information does not force correct allocation','Repository sync/off-device backup pending'],paid_job_started=False))
for filename,target in [('check_g5.py','check-implementation.py'),('finalize_g5.py','verify-and-record.py')]:shutil.copyfile(Path(__file__).with_name(filename),OUT/target)
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',{'files':files});files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
archive_record=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
archive.with_suffix('.receipt.json').write_text(json.dumps(archive_record,indent=2));print(json.dumps(dict(record=record,archive=archive_record,package=package),indent=2))
