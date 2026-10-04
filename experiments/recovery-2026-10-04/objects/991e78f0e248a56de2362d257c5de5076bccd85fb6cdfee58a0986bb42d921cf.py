from pathlib import Path
import io,json,zipfile,hashlib,shutil
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G3-Budget-Training-2026-10-03';DESIGN=BASE/'G4-Block-Growth-Design-2026-10-03'
OUT=BASE/'G4-Block-Training-2026-10-03-v2';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(OLD/'NCA-G3-Budget-Package.zip') as z:original={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
payload=dict(original)
payload['nca/block_generation.py']=Path(__file__).with_name('block_generation.py').read_bytes()
payload['nca/block_reference.py']=(DESIGN/'block_reference.py').read_bytes()
data=json.loads(payload['data.json']);lineage=[]
for row in data['rows']:
    parent=payload[row['arrays']];assert sha(parent)==row['arrays_sha256']
    with np.load(io.BytesIO(parent),allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    stage_file=DESIGN/(row['id']+'.npz')
    with np.load(stage_file,allow_pickle=False) as a:
        arrays['block_distance']=a['stage_distance'].copy();arrays['target_origins']=a['target_origins'].copy()
        assert np.array_equal(a['oracle_field'].astype(bool),arrays['target'].astype(bool))
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);payload[row['arrays']]=stream.getvalue()
    lineage.append(dict(id=row['id'],parent_arrays_sha256=row['arrays_sha256'],audit_npz_sha256=sha(stage_file.read_bytes()),condition_target_seed_unchanged=True))
    row['arrays_sha256']=sha(payload[row['arrays']])
data['version']='g4_full_cube_stages_v1';payload['data.json']=json.dumps(data,indent=2).encode()
payload['dataset-lineage.json']=json.dumps(lineage,indent=2).encode()
s=payload['nca/generation_training.py'].decode()
s=s.replace('LOSS={**BASE_LOSS,"global_band":1.0}','LOSS={"frontier_positive":1.0,"frontier_negative":1.0,"volume":.25,"cube":3,"global_band":1.0}')
s=s.replace('G3 global-budget seed generation','G4 cube-proposal generation').replace('from nca.budget_generation import BudgetNCA','from nca.block_generation import BlockNCA,COUNT_COLUMNS\nfrom nca.block_reference import cube_union')
s=s.replace("VERSION='seed_generation_training_v3_budget'","VERSION='seed_generation_training_v4_blocks'")
s=s.replace("teacher_stages='alternating; sha256(update:row) modulo max teacher distance'","teacher_stages='alternating seed / cube union; sha256(update:row) modulo maximum origin distance'")
s=s.replace('admission="stable_rank_budget_v1",budget_width=3','admission="cpu_sequential_cube_overlap_v1",budget_width=3,proposal_shape=[30,30,30],seed_loss="frontier_only",union_surrogate="independent_before_hard_selection",admission_count_columns=COUNT_COLUMNS')
s=s.replace('model=BudgetNCA()','model=BlockNCA()')
s=s.replace("distance=a['distance'].copy()","distance=a['block_distance'].copy()")
s=s.replace("depth=1+int.from_bytes(key[:8],'little')%(maximum-1)","depth=int.from_bytes(key[:8],'little')%maximum")
start="""  start=training_start(distance,depth,'train')
  if depth==0 and not np.array_equal(start,x['occupancy']):raise ValueError('Seed differs')
  self.last_start=dict(kind='seed' if depth==0 else 'teacher_stage',depth=depth,occupied=int(start.sum()))"""
replacement="""  seed_start=self.completed%2==0
  start=x['occupancy'].copy() if seed_start else cube_union((distance>=0)&(distance<=depth)).astype(np.float32)
  if not np.all(start[x['occupancy'].astype(bool)]==1):raise ValueError('Seed lost')
  self.last_start=dict(kind='seed' if seed_start else 'cube_teacher_stage',depth=None if seed_start else depth,occupied=int(start.sum()),sha256=hashlib.sha256(start.astype(np.uint8).tobytes(order='C')).hexdigest())
  self.last_start_field=start.astype(np.uint8).copy()"""
assert start in s;s=s.replace(start,replacement)
payload['nca/generation_training.py']=s.encode()
s=payload['scripts/colab_generation.py'].decode().replace('G3','G4').replace('from nca.budget_generation import device_probe','from nca.block_generation import device_probe')
s=s.replace("np.savez_compressed(f,state=state)","np.savez_compressed(f,state=state,start=session.last_start_field)")
payload['scripts/colab_generation.py']=s.encode()
protocol='''# G4 bounded cube-growth pilot

Purpose: fix G3 thin-fringe growth by proposing complete overlapping 3x3x3 cubes.
Occupied cells still mean overall building volume. Same nine constraint families,
same 32cubed grid,0.8m spacing and same27TRAIN contexts/teachers. Derived TRAIN
origin-graph stages replace incompatible voxel-distance stages. No development
or reserved labels included. This is an integrated architectural candidate.

Fresh seed1201,61->64->8 network initialized by the same G3 procedure. No trained
weights imported. Logit at each cube centre becomes a30cubed origin proposal.
Firing is Bernoulli0.5 on the origin grid; pad one zero cell around this mask
for the hidden-state update. This changes RNG consumption relative to G3.
First admitted cube must contain the independently chosen scene seed; at most
one first cube per step. Later eligible origins are face-neighbours of existing
full cube origins. Eligibility is frozen for each step. Sort fired scores>0.5
descending,flat ZYX ascending ties. Admit whole cubes only,subtracting their
actual newly occupied cells after earlier admissions. Never trim cubes.

Hybrid CPU/GPU implementation: neural network and differentiable loss use Torch
on selected device; probability/firing copy to CPU once per step,NumPy performs
detached sorting and overlap admission,then masks return to the device. No claim
of GPU-native or strictly local computation. B=ceil(request*domain cells),
C=min(B+8,floor(0.40*domain cells));same requests16/24/32%. Budget and full-cube
support are enforced,not learned quality. A seed can remain stalled when no
proposal fires/exceeds threshold; leftover capacity can remain unfillable.

256updates64steps,batch1,float32,Adam0.001,clip1. Alternate true single-seed
starts and cube unions of TRAIN teacher origin BFS depths. Depth is SHA256 of
completed_update:row_index,first8bytes little-endian modulo maximum origin
distance (0 through maximum-1). Save exact starts+hashes with every update.
TRAIN root is lex-first teacher cube containing context seed; inference never
reads root,stages or teacher. Conditions,targets and seed arrays preserved.

Frontier origin BCE positive1,negative1. After a cube exists,add0.25 local
3-cube mean-volume error and1.0 global band error. Differentiable voxel union
is1-product(1-p) over eligible fired cubes covering each empty voxel; existing
occupancy remains1. This independent-proposal surrogate avoids double counting
but does not model hard score ranking or budget selection. Seed phase uses
frontier BCE only: no misleading multi-cube volume surrogate when only one
cube can be admitted. Hidden recurrence gradients retained; hard births detached.

ONE Tesla T4 job capped600controlledseconds:12 device/reference probes plus
union backward check,256retainedupdates,two exact full-payload recovery replays,
per-update checkpoints/states/starts/traces,and final seed-only diagnostic.
Setup/upload/export/download/idle extra. ExpectedPython3.13.15,Torch2.11.0+cu130,
NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on mismatch,nonfinite values,probe/recovery
failure or GPU reserved memory>80%. No automatic retry or extension. Completed-
update recovery only,not cross-runtime/mid-rollout. Full evidenceZIP+receipt.
CPU timing is not a GPU runtime guarantee; the600s guard still applies.

Trace columns:initial_mass,offered_blocks,accepted_blocks,budget_rejected_blocks,
redundant_blocks,deferred_seed_blocks,added_voxels. Block counts are not voxel
counts. Pre-admission candidate combines all offers and can exceed cap; seed
candidate can contain multiple cubes. It is not an unguarded rollout.

Frozen review:final256only,CPUfloat32,firing2101,the same9 reused development
requests,seed-only64steps,then fixed128steps. Gates unchanged:9/9all-nine at64;
median absolute requested-fraction error<=0.02,max<=0.04;9/9all-nine at128;each
mass change<=5% of64-step mass. Report every case and family,IoU diagnostic,
budget events,origin/voxel counts,stalls and bulk coverage. No checkpoint/horizon
search,threshold tuning or clipping. Compare existing G3 evidence,not a new
control GPU job. Reserved targets remain unopened. Passing this pilot is not
generalization or automatic deployment approval. MG7 remains live.

Keep APPROVED_G4_JOB=False until this exact one-job budget is approved. Run once
and download fullZIP+receipt even after failure. No Drive operation,automatic
retry,extra seed,push,publication or model promotion is authorized.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
    if k.endswith('.py'):compile(v,k,'exec')
manifest={'version':'g4_block_package_v1','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G4-Block-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for k,v in payload.items():z.writestr(k,v)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
    assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G3-Budget.ipynb').read_text())
for c in nb['cells']:
    text=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace(oldreceipt['archive'],archive.name).replace('APPROVED_G3_JOB','APPROVED_G4_JOB').replace('one G3 job','one G4 job').replace('nca-g3-','nca-g4-').replace('supplied G1 package ZIP','supplied G4 package ZIP')
    if c['cell_type']=='markdown':text=protocol
    else:compile(text,'notebook','exec')
    c['source']=text.splitlines(True)
(OUT/'NCA-G4-Block.ipynb').write_text(json.dumps(nb,indent=2));(OUT/'PROTOCOL.md').write_text(protocol)
(OUT/'package-receipt.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive.read_bytes()),payloads=len(payload),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())),indent=2))
(OUT/'change-audit.json').write_text(json.dumps(dict(changed_members=[k for k in payload if k in original and payload[k]!=original[k]],added_members=[k for k in payload if k not in original],original_context_target_seed_values_unchanged=True,derived_dataset_version=data['version']),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py');print(OUT)
