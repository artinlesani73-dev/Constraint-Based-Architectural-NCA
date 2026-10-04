from pathlib import Path
import json,zipfile,hashlib,shutil,difflib
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G4-Block-Training-2026-10-03-v2';OUT=BASE/'G6-Paced-Growth-2026-10-04';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(OLD/'NCA-G4-Block-Package.zip') as z:original={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
payload=dict(original);s=payload['nca/block_generation.py'].decode()
s=s.replace('G4: learned cube proposals','G6: paced learned cube proposals').replace('import numpy as np','import numpy as np\nimport math').replace("VERSION='block_generation_v1'","VERSION='paced_block_generation_v1'").replace("'budget_rejected_blocks'","'allowance_rejected_blocks'").replace('class BlockNCA(BudgetNCA):','class PacedNCA(BudgetNCA):')
s=s.replace('        valid=full_origins(legal)','        quota=max(9,math.ceil((C-27)/63))\n        step_ceilings=[]\n        valid=full_origins(legal)')
needle="            new_field,count=admit(field,eligible_np,copied[0],copied[1].astype(bool),C,seed_phase);counts.append(count)"
assert needle in s
s=s.replace(needle,"            effective_cap=C if seed_phase else min(C,int(field.sum())+quota)\n            step_ceilings.append(effective_cap)\n            new_field,count=admit(field,eligible_np,copied[0],copied[1].astype(bool),effective_cap,seed_phase);counts.append(count)")
needle='budget=torch.tensor([D,B,C],device=m.device))';assert needle in s
s=s.replace(needle,'budget=torch.tensor([D,B,C],device=m.device),quota=torch.tensor(quota,device=m.device),step_ceilings=torch.tensor(step_ceilings,device=m.device))')
s=s.replace('def device_probe(device):','def original_device_probe(device):')
s+='''
def device_probe(device):
    from nca.block_reference import transition
    result=original_device_probe(device)
    for seed in range(8):
        rng=np.random.default_rng(seed);field=np.zeros((9,9,9),bool);field[3:6,3:6,3:6]=True;legal=np.ones_like(field)
        q=np.round(rng.random((7,7,7)),2).astype(np.float32);fire=rng.random(q.shape)<.5
        copied=torch.as_tensor(np.stack((q,fire)),device=device).float().cpu().numpy()
        C=[35,60,200,900][seed%4];quota=max(9,math.ceil((C-27)/63));cap=min(C,int(field.sum())+quota)
        e,phase=eligibility(field,full_origins(legal));actual,count=admit(field,e,copied[0],copied[1].astype(bool),cap,phase)
        expected,_=transition(field,legal,q,fire,cap)
        assert np.array_equal(actual,expected) and int(actual.sum())-int(field.sum())<=quota
    result.update(paced_reference_cases=8,pacing_horizon_constant=64,passed=True)
    return result
'''
payload['nca/paced_generation.py']=s.encode()
(OUT/'model-change.diff').write_text(''.join(difflib.unified_diff(original['nca/block_generation.py'].decode().splitlines(True),s.splitlines(True),fromfile='G4/block_generation.py',tofile='G6/paced_generation.py')))
s=payload['nca/generation_training.py'].decode().replace('G4 cube-proposal','G6 paced cube-proposal').replace('from nca.block_generation import BlockNCA,COUNT_COLUMNS','from nca.paced_generation import PacedNCA,COUNT_COLUMNS').replace("VERSION='seed_generation_training_v4_blocks'","VERSION='seed_generation_training_v6_paced'").replace('model=BlockNCA()','model=PacedNCA()').replace('admission="cpu_sequential_cube_overlap_v1"','admission="cpu_paced_cube_overlap_v1",quota_rule="max(9,ceil((C-27)/63))",pacing_horizon_constant=64')
needle='budget=r["budget"].detach().cpu().tolist())';assert needle in s
s=s.replace(needle,'budget=r["budget"].detach().cpu().tolist(),quota=int(r["quota"].detach().cpu()),step_ceilings=r["step_ceilings"].detach().cpu().tolist())')
payload['nca/generation_training.py']=s.encode()
payload['scripts/colab_generation.py']=payload['scripts/colab_generation.py'].decode().replace('G4','G6').replace('from nca.block_generation import device_probe','from nca.paced_generation import device_probe').encode()
protocol='''# G6: one paced-growth training pilot

Change only the per-step admission allowance relative to G4. Keep the same fresh
seed1201,61->64->8 network,dataset bytes,teacher stages,losses,origin firing,
threshold and ordering. No G5destination channels and no trained warm start.
Same nine families,32cubed grid,0.8m voxels and overall building-volume meaning.

Global band unchanged:B=ceil(request*domain cells),C=min(B+8,floor(0.40*D)).
Set K=max(9,ceil((C-27)/63)) once per rollout. This formula uses the fixed64step
training horizon;it is unchanged for128step review. First seed-containing cube
uses global C and admits at most one cube. After that,effective per-step ceiling
is min(C,current mass+K). Apply the existing exact sequential overlap admission
to that ceiling. Unused per-step allowance does not carry over. Never trim cubes.
The floor9 permits the maximum new mass of a face-adjacent3cube. This schedule
does not guarantee target volume by64,connection,facade compliance or stability.

The teacher BCE,0.25local cube-volume term and1.0global band term are unchanged.
The differentiable union and global band still use global B,C before hard
admission;the new temporary cap is detached. This is a focused timing experiment,
not a claim that the surrogate equals the expected paced transition. Seed-phase
loss remains BCE only. Keep all training failures and post-cap trace evidence.

Seven count columns:initial_mass,offered_blocks,accepted_blocks,
allowance_rejected_blocks,redundant_blocks,deferred_seed_blocks,added_voxels.
Column3 now records rejections by the effective allowance,not solely global
budget rejection. Save quota and every step ceiling separately. Do not compare
this rejection count to G4's global-only count as if the definitions were equal.

Local prior evidence:existing G4weights with this single fixed schedule improved
TRAIN64 all-nine validity2/27to12/27 and access2/27to22/27,but facade26/27to14/27
and only18/27were stable within5% at128. This is a changed-inference diagnostic,
not trained G6performance or held-out evidence. No quota search was performed.

One Tesla T4 run:256updates64steps,batch1,float32,Adam0.001,clip1,maximum600
controlledseconds. Includes12original admission probes,8paced reference probes,
union backward check,two exact full-payload/state recovery replays,every completed
update checkpoint/start/state/trace,and final seed-only diagnostic. Setup,
upload/export/download/idle extra. ExpectedPython3.13.15,Torch2.11.0+cu130,
NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on runtime/probe/recovery failure,nonfinite
values or reserved GPU memory>80%. No automatic retry or extension. Recovery
is same-runtime and completed-update only. Full evidenceZIP+receipt required.

Frozen review:final256checkpoint only,CPUfloat32,firing2101,same9 reused G1
development requests,single scene seed,64primary and128stability steps with
the same K. Require9/9all-nine atboth,median absolute volume-fraction error
<=0.02,max<=0.04,and each mass change<=5%. Report every case and family,IoU,
global-cap hits,per-step allowance usage and stalls. No threshold,checkpoint,
horizon or quota search. Existing G4run is the comparison;no extra GPU control.
Reserved labels stay unopened. Development reuse must be disclosed. No live
promotion follows automatically. MG7 remains live.

Keep APPROVED_G6_JOB=False until this exact one-job allowance is approved.
No Drive operation,push,publication,extra seed,paid retry or model replacement.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
    if k.endswith('.py'):compile(v,k,'exec')
assert payload['data.json']==original['data.json']
for row in json.loads(payload['data.json'])['rows']:assert payload[row['arrays']]==original[row['arrays']]
manifest={'version':'g6_paced_package_v1','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G6-Paced-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for k,v in payload.items():z.writestr(k,v)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
    assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G4-Block.ipynb').read_text())
for c in nb['cells']:
    text=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace('G4','G6').replace('nca-g4-','nca-g6-')
    if c['cell_type']=='markdown':text=protocol
    else:compile(text,'notebook','exec')
    c['source']=text.splitlines(True)
(OUT/'NCA-G6-Paced.ipynb').write_text(json.dumps(nb,indent=2));(OUT/'PROTOCOL.md').write_text(protocol)
(OUT/'package-receipt.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive.read_bytes()),payloads=len(payload),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())),indent=2))
(OUT/'change-audit.json').write_text(json.dumps(dict(changed_members=[k for k in payload if k in original and payload[k]!=original[k]],added_members=[k for k in payload if k not in original],dataset_bytes_unchanged=True,parent_manifest_sha256=oldreceipt['manifest_sha256']),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py');print(OUT)
