from pathlib import Path
import json,zipfile,hashlib,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G6-Paced-Growth-2026-10-04';OUT=BASE/'G7-Vertical-Training-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
prep=json.loads((OUT/'data-preparation.json').read_text());assert prep['ready'] and prep['admitted']==18
with zipfile.ZipFile(OLD/'NCA-G6-Paced-Package.zip') as z:
 m=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in m['files'].items());original={k:z.read(k) for k in m['files']}
payload=dict(original);data=json.loads(payload['data.json']);assert len(data['rows'])==27
for row in prep['rows']:
 payload[row['arrays']]=(OUT/row['arrays']).read_bytes()
 data['rows'].append({k:row[k] for k in ['id','split','arrays','arrays_sha256']})
data['version']='g7_vertical_train45_v1';payload['data.json']=json.dumps(data,indent=2).encode()
assert len(data['rows'])==len({r['id'] for r in data['rows']})==45
s=payload['nca/generation_package.py'].decode();assert "len(data['rows'])!=27" in s
s=s.replace("len(data['rows'])!=27","len(data['rows'])!=45").replace('TRAIN27 only','TRAIN45 only')
payload['nca/generation_package.py']=s.encode()
s=payload['nca/generation_training.py'].decode().replace('G6 paced cube-proposal generation','G7 vertical-data paced cube-proposal generation').replace('seed_generation_training_v6_paced','seed_generation_training_v7_vertical')
payload['nca/generation_training.py']=s.encode()
payload['scripts/colab_generation.py']=payload['scripts/colab_generation.py'].decode().replace('G6','G7').encode()
split=json.loads((OUT/'split-manifest.json').read_text())
payload['vertical-train-scenes.json']=json.dumps([e for e in split['entries'] if e['split']=='train'],indent=2).encode()
protocol='''# G7: vertical training diversity — one bounded run

G6 passed 9/9 reused development requests and 10/12 first-use reserved requests.
The two failures missed a higher east connection. Every original training pair
had connection origins at z=8/8. G7 tests broader vertical training data.

Retain all 27 original TRAIN payloads byte-for-byte and add 18 TRAIN examples
from six scenes, each at 16%,24%,32% requested volume. Connection heights vary
in both directions, with paired elevations and unequal building heights.
The same anchored teacher, seed=0, 15-second generation cap, contact weight=12
and full-cube origin BFS stages are used. All18 new labels pass all9 families
and the unchanged budget band; none was filtered or rerolled.

Fresh seed1201, same G6 initialization, 61->64->8 model, loss, Adam0.001,
clip1, batch1, float32, 64 training steps, alternating seed/teacher stages,
origin firing0.5 and proposal threshold0.5. Keep 32cubed grid and 0.8m cells.
Nine existing families; occupancy means overall building volume.
Neural model and loss implementation bytes are unchanged. Global admission:
B=ceil(request*D); C=min(B+8,floor(.4*D)); K=max(9,ceil((C-27)/63)). First cube
uses C; later allowance=min(C,current_mass+K). Same K at128 review steps.
This remains a hybrid learned/algorithmic method; thickness and budget are
partly enforced. It is not architectural or structural certification.

45 uniformly shuffled rows, 256 updates, same fresh seed as G6. Mean visits
fall from256/27 to256/45. This is a fixed-compute data-diversity intervention,
not equal per-example exposure or a multi-seed causal result. Do not alter
updates to rescue poor results or warm-start from the G6 checkpoint.

ONE Tesla T4 job, at most600 controlled seconds, including device probes,
256 retained updates, exact recovery replays at2/3, and checkpoint/evidence
writes. Setup/upload/export/download/idle time is extra. Expected runtime:
Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, CUDA13.0, cuDNN92700.
Stop on mismatch, recovery/probe failure, nonfinite values, or GPU reserved
memory above80%. Same-runtime completed-update recovery only. No automatic
retry or extension. Always download the full evidence ZIP and receipt.

Frozen review: final256 only, CPUfloat32, firing2101, scene-defined single
seed, horizons64/128, unchanged thresholds/quota. First report 21 legacy
regression requests (9 reused development +12 consumed G6 reserved), separately
from12 newly frozen G7 reserved requests. Require21/21 regression and12/12 fresh
all-nine passes at both horizons, median absolute fraction error<=.02,
maximum<=.04 in each cohort/horizon, and each mass change<=5%. Report every
failure; do not select checkpoints or tune from these evaluations. No fresh
reserved teachers or inference have been generated in preparation. All fresh
reserved scene/request specifications remain local, outside this TRAIN package.
These are related synthetic scenes, not external validation. Passing does not
automatically replace MG7; visual review and a separate integration decision follow.

Keep APPROVED_G7_JOB=False until the user approves this exact paid allowance.
No Drive access, paid retry, extra seed, push, publication or live promotion.
'''
payload['PROTOCOL.md']=protocol.encode();(OUT/'PROTOCOL.md').write_text(protocol,encoding='utf-8')
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
assert payload['nca/paced_generation.py']==original['nca/paced_generation.py']
assert all(payload[r['arrays']]==original[r['arrays']] for r in json.loads(original['data.json'])['rows'])
manifest=dict(version='g7_vertical_package_v1',files={k:sha(v) for k,v in payload.items()})
archive=OUT/'NCA-G7-Vertical-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
 assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G6-Paced.ipynb').read_text())
for c in nb['cells']:
 s=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace('G6','G7').replace('nca-g6-','nca-g7-')
 if c['cell_type']=='markdown':s=protocol
 else:compile(s,'notebook','exec')
 c['source']=s.splitlines(True)
with (OUT/'NCA-G7-Vertical.ipynb').open('x') as f:json.dump(nb,f,indent=2)
save('package-receipt.json',dict(archive=archive.name,sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,payloads=len(payload),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())))
save('change-audit.json',dict(parent_manifest_sha256=oldreceipt['manifest_sha256'],changed_members=[k for k in original if payload[k]!=original[k]],added_members=[k for k in payload if k not in original],paced_model_loss_bytes_unchanged=True,original27_payloads_unchanged=True,train_rows=45,new_labels=18,fresh_reserved_inference=0))
save('frozen-review.json',dict(checkpoint=256,device='cpu',dtype='float32',firing_seed=2101,horizons=[64,128],threshold=.5,quota='max(9,ceil((C-27)/63))',regression_manifest=str(BASE/'G1-Preparation-2026-10-03/split-manifest.json'),regression_manifest_sha256=sha((BASE/'G1-Preparation-2026-10-03/split-manifest.json').read_bytes()),regression_splits=['development','reserved'],regression_expected=21,fresh_split_sha256=sha((OUT/'split-manifest.json').read_bytes()),fresh_reserved_expected=12,requests=[.16,.24,.32],gates=dict(all_nine_each_cohort_both_horizons=True,median_error_max=.02,max_error_max=.04,per_case_mass_change_max=.05),no_checkpoint_selection=True,visual_review_required=True,executed=False))
shutil.copyfile(__file__,OUT/'build-package.py');print(archive)
