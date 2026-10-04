from pathlib import Path
from copy import deepcopy
import json,zipfile,hashlib,shutil,sys
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';OUT=BASE/'G10-One-Sided-Training-2026-10-04';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
with zipfile.ZipFile(OLD/'NCA-G9-Access-Ranking-Package.zip') as z:
 m=json.loads(z.read('manifest.json'));original={k:z.read(k) for k in m['files']};assert all(sha(original[k])==v for k,v in m['files'].items())
payload=dict(original)
s=payload['nca/access_ranking.py'].decode()
assert s.count('torch.logsumexp(other,0)')==1
s=s.replace('torch.logsumexp(other,0)','torch.logsumexp(other.detach(),0)').replace('train_access_ranking_v1','train_access_ranking_v2_one_sided')
s=s.replace('# A=advancing teacher origins, B=other teacher origins. Both remain BCE positives.','# A=advancing teacher origins, B=other teacher origins. Both remain BCE positives.\n    # Explicit semi-gradient: other scores are a detached comparison reference.')
payload['nca/access_ranking.py']=s.encode()
s=payload['nca/generation_training.py'].decode().replace('G9 access-ranking','G10 one-sided access-ranking').replace('seed_generation_training_v9_access_ranking','seed_generation_training_v10_one_sided')
s=s.replace('"ranking_margin":1.0}', '"ranking_margin":1.0,"detach_other_reference":True}')
s=s.replace('seed_loss="frontier_only"','seed_loss="frontier_plus_one_sided_access_ranking"')
payload['nca/generation_training.py']=s.encode()
payload['scripts/colab_generation.py']=payload['scripts/colab_generation.py'].decode().replace('G9','G10').encode()
protocol='''# G10: one-sided access ranking

The only scientific change from G9 is stopping the auxiliary ranking gradient
through the non-advancing teacher-positive reference scores. Numerical ranking
value and advancing logit gradient are unchanged at a fixed state. This is an
explicit semi-gradient, not the full derivative of symmetric ranking. Margin1,
weight1 and all original membership/volume/band losses remain. The other logits
can still change through shared parameters: improvement is not guaranteed.

Same45 TRAIN examples,61-64-8 network,fresh seed1201,427updates64steps,Adam0.001,
clip1,teacher-stage schedule,0.5firing/threshold,32cubed at0.8m,original quota
and global cap. No new inputs,teacher routes at inference or postprocessing.
The metadata seed-loss label is corrected to include the ranking term.
No warm start or checkpoint selection.

One proposed fresh TeslaT4 job,max600 controlled seconds; setup/export/download/
idle extra. G9 took469s; this is not a guarantee G10 finishes in the cap.
Strict runtime:Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Keep full recovery at2/3,finite/memory guards,per-update evidence and all failures.
No automatic retry,extension or runtime-guard bypass.

Frozen review: final427,CPUfloat32,firing2101,64/128steps. Evaluate57 prior
regression requests and12 new reserved requests. Run G9 final427 on the SAME new
12 cases as the primary causal comparison; also run fixed G8 as the stability
reference. No fresh labels in package. Report models/cohorts separately.
All nine families every case/horizon; median absolute volume-fraction error<=.02,
max<=.04; per-case mass growth<=5%; raw geometry visual review required.
No rerolls,coefficient sweep or changing gates after results. One seeded
synthetic comparison is not broad architectural validation.

Keep APPROVED_G10_JOB=False until this exact one-job allowance is approved.
No Drive mount,push,publication or live promotion. MG7 remains live.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
for k in original:
 if k not in ['nca/access_ranking.py','nca/generation_training.py','scripts/colab_generation.py','PROTOCOL.md']:assert payload[k]==original[k]
manifest=dict(version='g10_one_sided_package_v1',files={k:sha(v) for k,v in payload.items()})
archive=OUT/'NCA-G10-One-Sided-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1 and all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
nb=json.loads((OLD/'NCA-G9-Access-Ranking.ipynb').read_text());prior=json.loads((OLD/'package-receipt.json').read_text())
for c in nb['cells']:
 s=''.join(c['source']).replace(prior['sha256'],sha(archive.read_bytes())).replace('G9','G10').replace('nca-g9-','nca-g10-')
 if c['cell_type']=='markdown':s=protocol
 else:compile(s,'notebook','exec')
 c['source']=s.splitlines(True)
save('NCA-G10-One-Sided.ipynb',nb);(OUT/'PROTOCOL.md').write_text(protocol)
save('package-receipt.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes()),payloads=len(payload)))
save('change-audit.json',dict(changed_members=[k for k in original if original[k]!=payload[k]],all45_data_bytes_unchanged=True,model_rollout_and_pacing_bytes_unchanged=True,scientific_change='Detach other reference logits in ranking only',metadata_correction='seed loss includes ranking'))
sys.path.insert(0,str(BASE/'G7-Vertical-Training-2026-10-04-v2/source'));sys.dont_write_bytecode=True
from nca.massing_cases import target_context
from nca.repair_benchmark import context_hash,condition
splits=[BASE/'G1-Preparation-2026-10-03/split-manifest.json',BASE/'G7-Vertical-Training-2026-10-04-v2/split-manifest.json',BASE/'G8-Exposure-Training-2026-10-04/fresh-split-manifest.json',OLD/'fresh-split-manifest.json']
loaded=[json.loads(p.read_text()) for p in splits];config=json.loads((OLD/'environment.json').read_text())['config'];base=loaded[1]['entries'][0]['scene'];entries=[]
seen=set()
for m in loaded:
 for e in m['entries']:
  f,d,_=target_context(e['scene'],config);seen.add(sha(condition(e['scene'],f,d,.16).tobytes()))
for i,(west,east,wh,eh,y) in enumerate([(9,18,23,30,13),(18,9,30,23,17),(14,21,26,31,12),(21,14,31,26,18)]):
 s=deepcopy(base);s['scene_id']=f'g10-reserved-{i}';s['description']='G10 frozen fresh paired G8/G9/G10;no labels or inference'
 s['buildings'][0]['z'][1]=wh;s['buildings'][1]['z'][1]=eh
 for e,z in zip(s['entrances'],[west,east]):e['z']=z;e['y']=y
 f,d,_=target_context(s,config);h=context_hash(s,f,d);physical=sha(condition(s,f,d,.16).tobytes());assert physical not in seen;seen.add(physical)
 entries.append(dict(id=s['scene_id'],split='reserved',scene=s,context_sha256=h,condition_v16_sha256=physical))
save('fresh-split-manifest.json',dict(entries=entries,requests=[.16,.24,.32],frozen_before_new_training=True,labels_generated=0,model_inference=0,paired_models=['G8-final427','G9-final427','G10-final427'],physical_context_uniqueness_checked=True))
shutil.copyfile(OLD/'environment.json',OUT/'environment.json')
baselines={}
for label,folder in [('G8','G8-Final-Review-2026-10-04-v2'),('G9','G9-Final-Review-2026-10-04')]:
 p=BASE/folder/'import/worker/checkpoint-0427.pt';baselines[label]=dict(path=str(p),sha256=sha(p.read_bytes()))
save('frozen-review.json',dict(checkpoint=427,baselines=baselines,primary_paired_comparison='G9',stability_reference='G8',firing_seed=2101,horizons=[64,128],threshold=.5,device='cpu',dtype='float32',regression_expected=57,regression_sources=[dict(path=str(p),sha256=sha(p.read_bytes()),splits=['development','reserved'] if i==0 else ['reserved']) for i,p in enumerate(splits)],fresh_manifest_sha256=sha((OUT/'fresh-split-manifest.json').read_bytes()),fresh_expected=12,requests=[.16,.24,.32],gates=dict(all_nine_every_case_both_horizons=True,median_error_max=.02,max_error_max=.04,per_case_mass_change_max=.05),visual_review_required=True,executed=False))
save('RESUME-preparation.json',dict(status='G10 package built;local parity/recovery pending',previous=str(BASE/'G9-Training-Diagnosis-2026-10-04/RESUME.json'),next='Run three-update CPU rehearsal and fixed-weight parity/semi-gradient checks,verify archive;then ask one bounded paid allowance',paid_run_authorized=False))
shutil.copyfile(__file__,OUT/'build-package.py');print(archive)

