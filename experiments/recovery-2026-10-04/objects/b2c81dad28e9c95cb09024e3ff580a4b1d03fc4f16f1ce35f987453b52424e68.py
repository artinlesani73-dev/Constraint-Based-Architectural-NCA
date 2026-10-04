from pathlib import Path
from copy import deepcopy
import json,zipfile,hashlib,shutil,sys
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G7-Vertical-Training-2026-10-04-v2';OUT=BASE/'G8-Exposure-Training-2026-10-04';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
sys.path.insert(0,str(OLD/'source'));sys.dont_write_bytecode=True
from nca.massing_cases import target_context
from nca.repair_benchmark import context_hash
oldsplit=json.loads((OLD/'split-manifest.json').read_text());g1=json.loads((BASE/'G1-Preparation-2026-10-03/split-manifest.json').read_text());config=json.loads((OLD/'environment.json').read_text())['config']
seen={e['context_sha256'] for e in oldsplit['entries']+g1['entries']};base=oldsplit['entries'][0]['scene'];entries=[]
for i,(west,east,wh,eh,y) in enumerate([(11,18,22,29,13),(18,11,29,22,17),(13,19,24,30,14),(19,13,30,24,16)]):
 s=deepcopy(base);s['scene_id']=f'g8-reserved-{i}';s['description']='G8 fresh reserved geometry;no labels or inference'
 s['buildings'][0]['z'][1]=wh;s['buildings'][1]['z'][1]=eh
 for e,z in zip(s['entrances'],[west,east]):e['z']=z;e['y']=y
 f,d,_=target_context(s,config);h=context_hash(s,f,d);assert h not in seen;seen.add(h)
 entries.append(dict(id=s['scene_id'],split='reserved',scene=s,context_sha256=h))
save('fresh-split-manifest.json',dict(entries=entries,requests=[.16,.24,.32],frozen_before_new_training=True,labels_generated=0,model_inference=0))
shutil.copyfile(OLD/'environment.json',OUT/'environment.json')
with zipfile.ZipFile(OLD/'NCA-G7-Vertical-Package.zip') as z:
 m=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in m['files'].items());original={k:z.read(k) for k in m['files']}
payload=dict(original)
s=payload['nca/generation_training.py'].decode().replace('G7 vertical-data','G8 exposure-matched').replace('seed_generation_training_v7_vertical','seed_generation_training_v8_exposure')
assert 'updates=256' in s;s=s.replace('updates=256','updates=427');payload['nca/generation_training.py']=s.encode()
s=payload['scripts/colab_generation.py'].decode().replace('G7','G8');assert 'end=3 if a.cpu_rehearsal else 256' in s
s=s.replace('end=3 if a.cpu_rehearsal else 256','end=3 if a.cpu_rehearsal else 427').replace("'updates':3 if a.cpu_rehearsal else 256","'updates':3 if a.cpu_rehearsal else 427")
payload['scripts/colab_generation.py']=s.encode()
protocol='''# G8: one training-exposure experiment

G7's broader data did not pass acceptance. A TRAIN-only comparison found both
models begin growth at step1, while G7 leaves much more per-step allowance unused
through sparse above-threshold proposals. This supports testing optimization
exposure before changing the model, objective, threshold or admission mechanism.
It does not prove that additional training will solve connection failures.

Change only retained training updates from256 to427. Calculation:
ceil(256*45/27)=427, approximately restoring G6's mean visits per example.
Use the exact45 G7 TRAIN payloads and order, fresh seed1201, original optimizer,
61->64->8 model, G6 paced transition, 64 training steps, batch1, float32,
Adam0.001, clip1, alternating seed/teacher stages, firing0.5, threshold0.5,
32cubed grid and0.8m voxels. No warm start or learned checkpoint selection.
Losses and teacher construction unchanged. Same nine constraint families.

Global B=ceil(request*D), C=min(B+8,floor(.4D)), K=max(9,ceil((C-27)/63)).
Do not change inference horizon or quota. Neural/cube-admission source and all
data bytes remain identical to G7. The427 target is exposure-derived, not tuned
from a checkpoint sweep. Compare final427 with the already-frozen G7 final256.

One fresh Tesla T4 job, at most600 controlled seconds,427 updates64 steps.
Estimated controlled time roughly310-320s from G7, not a guarantee. Setup,
upload/export/download/idle are extra. Stop at600s, never extend or auto-retry.
Expected Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Keep device probes, full recovery checks at2/3, per-update checkpoints, finite
guards and80% reserved-memory guard. Runtime mismatch stops the run.

After return, verify the427 final and every trace. Also compare update256 model,
optimizer, sampler, trace and RNG numerically against the prior G7 update256,
with run identity kept separate. This is a prefix reproducibility check, not
checkpoint selection. Record any mismatch; do not silently assert equivalence.

Frozen review: final427 only, CPUfloat32, firing2101, single scene seed,
64 and128 steps. Separately report33 exposed regression requests (all prior
G1 development/reserved plus G7 reserved), and12 new frozen G8 reserved requests.
All nine families must pass every request at both horizons. Each cohort/horizon
requires median absolute requested-fraction error<=.02 and maximum<=.04;
each case's mass change must be<=5%. Visually review volumes. No tuning,
rerolls, dropped cases, best-checkpoint choice or postprocessing.
Fresh reserved geometry stays local outside the TRAIN package; no labels or
inference have been produced. Synthetic relatives are not external validation.

Keep APPROVED_G8_JOB=False until this exact one-job paid allowance is approved.
No Drive operation, retry, extra seed, publication, push or live promotion.
MG7 remains live; all G6/G7 evidence remains intact.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
assert payload['data.json']==original['data.json'] and payload['nca/paced_generation.py']==original['nca/paced_generation.py']
assert all(payload[r['arrays']]==original[r['arrays']] for r in json.loads(payload['data.json'])['rows'])
manifest=dict(version='g8_exposure_package_v1',files={k:sha(v) for k,v in payload.items()});archive=OUT/'NCA-G8-Exposure-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1 and all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
nb=json.loads((OLD/'NCA-G7-Vertical.ipynb').read_text());prior=json.loads((OLD/'package-receipt.json').read_text())
for c in nb['cells']:
 s=''.join(c['source']).replace(prior['sha256'],sha(archive.read_bytes())).replace('G7','G8').replace('nca-g7-','nca-g8-')
 if c['cell_type']=='markdown':s=protocol
 else:compile(s,'notebook','exec')
 c['source']=s.splitlines(True)
save('NCA-G8-Exposure.ipynb',nb);(OUT/'PROTOCOL.md').write_text(protocol,encoding='utf-8')
save('package-receipt.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes()),payloads=len(payload)))
save('change-audit.json',dict(changed_members=[k for k in original if original[k]!=payload[k]],added_members=[k for k in payload if k not in original],all45_data_bytes_unchanged=True,model_loss_pacing_unchanged=True,update_target=[256,427],fresh_initialization=True))
save('frozen-review.json',dict(checkpoint=427,prefix_check=256,firing_seed=2101,horizons=[64,128],threshold=.5,device='cpu',dtype='float32',regression_expected=33,regression_sources=[dict(path=str(BASE/'G1-Preparation-2026-10-03/split-manifest.json'),sha256=sha((BASE/'G1-Preparation-2026-10-03/split-manifest.json').read_bytes()),splits=['development','reserved']),dict(path=str(OLD/'split-manifest.json'),sha256=sha((OLD/'split-manifest.json').read_bytes()),splits=['reserved'])],fresh_manifest_sha256=sha((OUT/'fresh-split-manifest.json').read_bytes()),fresh_expected=12,requests=[.16,.24,.32],gates=dict(all_nine_every_case_both_horizons=True,median_error_max=.02,max_error_max=.04,per_case_mass_change_max=.05),visual_review_required=True,executed=False))
save('RESUME-preparation.json',dict(status='G8 package prepared;local rehearsal pending;paid approvalFalse',previous=str(BASE/'G7-Training-Diagnosis-2026-10-04'),next='Run one3-update CPU rehearsal and verify equality with G7 local initial andupdate3;archive before requesting paid allowance'))
shutil.copyfile(__file__,OUT/'build-package.py');print(archive)
