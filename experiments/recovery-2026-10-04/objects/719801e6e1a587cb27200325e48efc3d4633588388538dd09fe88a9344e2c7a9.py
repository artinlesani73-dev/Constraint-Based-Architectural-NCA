from pathlib import Path
import json,zipfile,hashlib,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G2-Balanced-Growth-2026-10-03';OUT=BASE/'G3-Budget-Training-2026-10-03';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(OLD/'NCA-G2-Balanced-Growth-Package.zip') as z:original={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
payload=dict(original)
payload['nca/budget_generation.py']=Path(__file__).with_name('budget_generation.py').read_bytes()
payload['nca/budget_reference.py']=(BASE/'G3-Budget-Design-2026-10-03/budget_reference.py').read_bytes()
s=payload['nca/generation_training.py'].decode().replace('G2 positive-balanced seed generation','G3 global-budget seed generation').replace('from nca.connected_repair import ConnectedSession,LOSS','from nca.connected_repair import ConnectedSession,LOSS as BASE_LOSS\nfrom nca.budget_generation import BudgetNCA\nLOSS={**BASE_LOSS,"global_band":1.0}')
s=s.replace("VERSION='seed_generation_training_v2_positive_balance'","VERSION='seed_generation_training_v3_budget'")
s=s.replace('objective=LOSS,review_firing_seed=2101,automatic_retry=False)','objective=LOSS,review_firing_seed=2101,automatic_retry=False,architecture="61-64-8",admission="stable_rank_budget_v1",budget_width=3)')
old='  self.identity.update(model_semantics=VERSION,train_steps=64,generation_settings=SETTINGS)'
new='''  model=BudgetNCA().float().to(self.device);model.import_base(self.model);self.model=model
  self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
  self.identity.update(model_semantics=VERSION,train_steps=64,generation_settings=SETTINGS,objective=LOSS,global_budget_operations=True,admission_is_detached=True)'''
assert old in s;s=s.replace(old,new)
s=s.replace('start=self.last_start)','start=self.last_start,band_loss=float(r["band_loss"].detach()),admission_counts=r["admission_counts"].detach().cpu().tolist(),budget=r["budget"].detach().cpu().tolist())')
payload['nca/generation_training.py']=s.encode()
s=payload['scripts/colab_generation.py'].decode().replace('G2 bounded','G3 bounded').replace('G2 generation training evidence','G3 generation training evidence')
needle="        session.save(out/'checkpoint-0000.pt')"
s=s.replace(needle,"        from nca.budget_generation import device_probe\n        write_once(out/'device-admission-check.json',device_probe(session.device))\n"+needle)
payload['scripts/colab_generation.py']=s.encode()
protocol='''# G3 bounded budget-feedback pilot

Fresh seed1201 model:61->64->8 pointwise network with existing60 inputs copied
from an identically seeded fresh G2-size initialization and the new input weights
zero. No trained checkpoint warm start. Extra channel broadcasts(B-M)/D each step.
B=ceil(request*domain count);C=min(B+8,floor(0.40*D)). Only16/24/32% requests,
32cubed,0.8m spacing,3-cell physical bulk. Reject starts/teachers outside the band.
All27 original TRAIN arrays and row/start schedules unchanged. Same50/50 seed
and teacher stages,256updates64steps,batch1,float32,Adam lr0.001,clip1.
Keep G2 frontier weights1:1 and local cube-volume0.25. Add coefficient1 global
band error:distance of current mass+sum fired-frontier sigmoid proposals from
[B,C],divided byD. Compute loss before hard admission. Hard births/ranking detached.

Intrinsic cap selects at most C-M above0.5 fired legal frontier proposals,descending
probability,stable ascending ZYX index ties. GPU tensor sort; no NumPy conversion
in the admission loop. Global count,broadcast,ranking make this a hybrid NCA.
This is an integrated architectural candidate,not a single-factor ablation.
Budget obedience and stability at capacity are enforced,not learned achievements.
Wrong early additions remain irreversible;geometry may fail before budget is spent.

ONE Tesla T4 run capped600controlledseconds including8 device/reference admission
probes,256retainedupdates,two recovery replays,checkpoint/state/trace writes and
final seed-only diagnostic. Setup/upload/export/download/idle extra. Expected:
Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Stop on mismatch,device-probe or recovery failure,nonfinite values or reserved
GPU memory>80%. No automatic retry. Every completed update saved. FullZIP+receipt.
Recovery replays teacher-stage update2 and seed-start update3;full payload+state
must match. Completed-update recovery only,not cross-runtime or mid-rollout.

Frozen review:final256only,CPUfloat32,firing2101,the same9G1/G2 development requests,
single-seed64steps. Fixed128steps for stability,never best-horizon selection.
Same gates:9/9 all-nine at64;median absolute requested fraction error<=0.02,max<=0.04;
9/9all-nine at128 and mass changes<=5% of64-step count. These are pilot engineering
gates,not generalization or deployment approval. Report all failures,per-family
metrics,teacherIoU diagnostic,volume errors,raw proposed/admitted/rejected counts,
ceiling-hit steps and pre-admission candidates. A pre-admission candidate is not
a full no-guard rollout. No post-hoc clipping,reroll,threshold or checkpoint search.
G2 completed run20261003T193511Z_56901714f0be is existing baseline,no extra control
GPU job. Reused development clearly labeled;reserved targets stay unopened.

Keep APPROVED_G3_JOB=False until this exact one-job budget is approved. After
approval run once,download FULL evidenceZIP+receipt even after failure. No Drive,
automatic retry,extra seed,public deployment or model promotion. MG7 stays live.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
assert payload['data.json']==original['data.json']
for row in json.loads(payload['data.json'])['rows']:assert '\\' not in row['arrays'] and payload[row['arrays']]==original[row['arrays']]
manifest={'version':'g3_budget_package_v1','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G3-Budget-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
 assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G2-Balanced-Growth.ipynb').read_text())
for c in nb['cells']:
 text=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace(oldreceipt['archive'],archive.name).replace('APPROVED_G2_JOB','APPROVED_G3_JOB').replace('one G2 job','one G3 job').replace('nca-g2-','nca-g3-')
 if c['cell_type']=='markdown':text=protocol
 else:compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
(OUT/'NCA-G3-Budget.ipynb').write_text(json.dumps(nb,indent=2));(OUT/'PROTOCOL.md').write_text(protocol)
(OUT/'package-receipt.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive.read_bytes()),payloads=len(payload)),indent=2))
(OUT/'change-audit.json').write_text(json.dumps(dict(changed_members=[k for k in payload if k in original and payload[k]!=original[k]],added_members=[k for k in payload if k not in original],data_and_arrays_unchanged=True,manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py');print(OUT)
