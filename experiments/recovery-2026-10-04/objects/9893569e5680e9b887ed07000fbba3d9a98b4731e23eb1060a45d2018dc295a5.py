from pathlib import Path
from copy import deepcopy
import json,zipfile,hashlib,shutil,sys
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G8-Exposure-Training-2026-10-04';DESIGN=BASE/'G9-Access-Objective-Design-2026-10-04-v2';OUT=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
with zipfile.ZipFile(OLD/'NCA-G8-Exposure-Package.zip') as z:
 m=json.loads(z.read('manifest.json'));original={k:z.read(k) for k in m['files']};assert all(sha(original[k])==v for k,v in m['files'].items())
payload=dict(original)
helper=(DESIGN/'source/access_priority_v1.py').read_text().split('\ndef temporal_loss(')[0]
helper=helper.replace('from torch.nn import functional as F','from functools import lru_cache')
helper+='''
@lru_cache(maxsize=64)
def cached_teacher_graph(shape,target_bytes,interface_bytes):
    target=np.frombuffer(target_bytes,dtype=bool).reshape(shape)
    interface=np.frombuffer(interface_bytes,dtype=bool).reshape(shape)
    return teacher_graph(target,interface,split='train')
'''
payload['nca/access_labels.py']=helper.encode()
ranking=(DESIGN/'source/access_ranking_v1.py').read_text().replace('from access_priority_v1 import teacher_graph, priority','from nca.access_labels import teacher_graph, priority')
payload['nca/access_ranking.py']=ranking.encode()
s=payload['nca/paced_generation.py'].decode()
s=s.replace("VERSION='paced_block_generation_v1'","VERSION='ranked_paced_block_generation_v1'\nfrom nca.access_labels import cached_teacher_graph,priority\nfrom nca.access_ranking import ranked_loss")
s=s.replace('class PacedNCA(BudgetNCA):','class RankedNCA(BudgetNCA):')
s=s.replace('target=None,capture=False','target=None,capture=False,training_split=None')
s=s.replace('if target is not None:\n            if target.shape',"if target is not None:\n            if training_split!='train':raise ValueError('Explicit TRAIN supervision required')\n            if target.shape",1)
s=s.replace('origins=torch.as_tensor(origins_np,device=occupancy.device)[None,None]',"origins=torch.as_tensor(origins_np,device=occupancy.device)[None,None]\n            interface=static_features[0,5].bool().detach().cpu().numpy()\n            graph=cached_teacher_graph(label.shape,label.tobytes(),interface.tobytes())")
s=s.replace('losses=[];fronts=[];volumes=[];bands=[];counts=[];', 'ranks=[];phase_trace=[]\n        losses=[];fronts=[];volumes=[];bands=[];counts=[];')
s=s.replace('loss,front,volume,band=block_loss(logits,m,eligible,target,origins,seed_phase,D,B,C)', '''positive,phase=priority(field,legal,graph)
                positive_t=tensor(positive)
                loss,front,volume,band,rank=ranked_loss(logits,m,eligible,target,origins,seed_phase,D,B,C,positive_t,phase)
                ranks.append(rank)
                n_progress=int((eligible & origins & positive_t).sum().detach().cpu())
                n_other=int((eligible & origins & ~positive_t).sum().detach().cpu())
                phase_trace.append(dict(phase=phase,progress_fired=n_progress,other_teacher_fired=n_other,ranking_active=phase in ('seed_access','advance_access') and n_progress>0 and n_other>0,mass=int(field.sum()),at_capacity=int(field.sum())==C))''')
s=s.replace('band_loss=torch.stack(bands).mean())','band_loss=torch.stack(bands).mean(),ranking_loss=torch.stack(ranks).mean(),access_phase_trace=phase_trace)')
payload['nca/ranked_generation.py']=s.encode()
s=payload['nca/generation_training.py'].decode().replace('G8 exposure-matched','G9 access-ranking').replace('from nca.paced_generation import PacedNCA,COUNT_COLUMNS','from nca.ranked_generation import RankedNCA,COUNT_COLUMNS').replace('model=PacedNCA()','model=RankedNCA()').replace('seed_generation_training_v8_exposure','seed_generation_training_v9_access_ranking')
s=s.replace('"global_band":1.0}', '"global_band":1.0,"access_ranking":1.0,"ranking_margin":1.0}')
s=s.replace('64,target=target)','64,target=target,training_split="train")')
s=s.replace('band_loss=float(r["band_loss"].detach()),','band_loss=float(r["band_loss"].detach()),ranking_loss=float(r["ranking_loss"].detach()),access_phase_trace=r["access_phase_trace"],')
payload['nca/generation_training.py']=s.encode()
payload['scripts/colab_generation.py']=payload['scripts/colab_generation.py'].decode().replace('G8','G9').encode()
protocol='''# G9: access-ranking loss experiment

Only scientific change versus G8: add a TRAIN-only access-ranking loss, margin1,
weight1. All original membership/volume/band terms remain active. Teacher graph
labels never enter inference. Exact specification: G9 access-objective proposal.
Same45 TRAIN payloads,61-64-8 network, fresh paired seed1201,427updates64steps,
Adam0.001,clip1,teacher-stage schedule,firing0.5,threshold0.5,32cubed at0.8m.
Same cube admission, global cap and per-step quota. Nine families unchanged.
No G8 warm start, checkpoint selection, route input, bridge or postprocessing.

One proposed T4 job, maximum600 controlled seconds including startup/probes/
recovery; setup/export/download/idle extra. Runtime must match Python3.13.15,
Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on mismatch or timeout;
no automatic extension/retry. G8 took336s; G9 has additional overhead, so this
is not a completion-time guarantee. Full recovery at updates2/3 remains required.
Keep all checkpoints,starts,trace and failures. Evaluate final427 only.

Log ranking loss and every step's supervision phase, fired comparison counts,
active ranking and cap status. Report no-route fallback separately; capped
states can have no route. Cache graphs deterministically; cache is derivable,
not learned state or random-number state. No new random draws in supervision.

Frozen local review:45 exposed regression requests and12 new reserved requests.
Evaluate G8 and G9 final427 on the SAME new12 requests, firing2101,CPUfloat32,
64/128steps. All nine families every case/horizon; cohort median absolute volume
fraction error<=.02,max<=.04; per-case mass change<=5%; visual review required.
Report regressions and individual failures. Fresh geometry and all heldout
labels are excluded from this training ZIP. No heldout inference before return.
No threshold/seed/epoch sweep. Synthetic one-seed result is not broad validation.

APPROVED_G9_JOB=False until this exact one-job allowance is approved. No Drive,
publication,push,automatic retry or live replacement; MG7 remains live.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
assert payload['nca/paced_generation.py']==original['nca/paced_generation.py']
assert payload['data.json']==original['data.json']
assert all(payload[r['arrays']]==original[r['arrays']] for r in json.loads(payload['data.json'])['rows'])
manifest=dict(version='g9_access_ranking_package_v1',files={k:sha(v) for k,v in payload.items()})
archive=OUT/'NCA-G9-Access-Ranking-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1 and all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
nb=json.loads((OLD/'NCA-G8-Exposure.ipynb').read_text());prior=json.loads((OLD/'package-receipt.json').read_text())
for c in nb['cells']:
 s=''.join(c['source']).replace(prior['sha256'],sha(archive.read_bytes())).replace('G8','G9').replace('nca-g8-','nca-g9-')
 if c['cell_type']=='markdown':s=protocol
 else:compile(s,'notebook','exec')
 c['source']=s.splitlines(True)
save('NCA-G9-Access-Ranking.ipynb',nb);(OUT/'PROTOCOL.md').write_text(protocol)
save('package-receipt.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes()),payloads=len(payload)))
save('change-audit.json',dict(changed_members=[k for k in original if original[k]!=payload[k]],added_members=[k for k in payload if k not in original],all45_data_bytes_unchanged=True,base_model_and_pacing_bytes_unchanged=True,updates=427,loss_change_only=True))
sys.path.insert(0,str(BASE/'G7-Vertical-Training-2026-10-04-v2/source'));sys.dont_write_bytecode=True
from nca.massing_cases import target_context
from nca.repair_benchmark import context_hash
splits=[BASE/'G1-Preparation-2026-10-03/split-manifest.json',BASE/'G7-Vertical-Training-2026-10-04-v2/split-manifest.json',OLD/'fresh-split-manifest.json']
loaded=[json.loads(p.read_text()) for p in splits];seen={e['context_sha256'] for m in loaded for e in m['entries']}
config=json.loads((OLD/'environment.json').read_text())['config'];base=loaded[1]['entries'][0]['scene'];entries=[]
for i,(west,east,wh,eh,y) in enumerate([(10,17,23,28,12),(17,10,28,23,18),(12,20,25,31,13),(20,12,31,25,17)]):
 s=deepcopy(base);s['scene_id']=f'g9-reserved-{i}';s['description']='G9 fresh reserved paired G8/G9;no labels or inference'
 s['buildings'][0]['z'][1]=wh;s['buildings'][1]['z'][1]=eh
 for e,z in zip(s['entrances'],[west,east]):e['z']=z;e['y']=y
 f,d,_=target_context(s,config);h=context_hash(s,f,d);assert h not in seen;seen.add(h)
 entries.append(dict(id=s['scene_id'],split='reserved',scene=s,context_sha256=h))
save('fresh-split-manifest.json',dict(entries=entries,requests=[.16,.24,.32],frozen_before_new_training=True,labels_generated=0,model_inference=0,paired_models=['G8-final427','G9-final427']))
shutil.copyfile(OLD/'environment.json',OUT/'environment.json')
save('frozen-review.json',dict(checkpoint=427,paired_baseline_checkpoint_sha256='2bca26ab350fe6a318d15b12fd0f3ee1da9d6f96d524df103a0b5249743df5ff',firing_seed=2101,horizons=[64,128],threshold=.5,device='cpu',dtype='float32',regression_expected=45,regression_sources=[dict(path=str(p),sha256=sha(p.read_bytes()),splits=['development','reserved'] if i==0 else ['reserved']) for i,p in enumerate(splits)],fresh_manifest_sha256=sha((OUT/'fresh-split-manifest.json').read_bytes()),fresh_expected=12,fresh_paired_baseline=True,requests=[.16,.24,.32],gates=dict(all_nine_every_case_both_horizons=True,median_error_max=.02,max_error_max=.04,per_case_mass_change_max=.05),visual_review_required=True,executed=False))
save('RESUME-preparation.json',dict(status='G9 integrated package;local checks pending',previous=str(DESIGN/'RESUME.json'),next='Run consolidated fixed-weight inference parity and3-update CPU recovery rehearsal;verify then finalize local archive',paid_run_authorized=False))
shutil.copyfile(__file__,OUT/'build-package.py');print(archive)

