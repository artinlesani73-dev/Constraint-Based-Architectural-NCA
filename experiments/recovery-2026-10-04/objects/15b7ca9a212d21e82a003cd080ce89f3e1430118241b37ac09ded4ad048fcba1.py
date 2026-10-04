from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time,os
BASE=Path('C:/Users/artin/Documents/Codex/outputs');PREP=BASE/'G1-Preparation-2026-10-03';PACKAGE=BASE/'G2-Balanced-Growth-2026-10-03';OUT=BASE/'G2-Final-Review-2026-10-03';OUT.mkdir(exist_ok=False)
RUN='20261003T193511Z_56901714f0be';DOWNLOAD=Path('C:/Users/artin/Downloads')
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
receipt=json.loads((DOWNLOAD/(RUN+'.receipt.json')).read_text());assert sha((DOWNLOAD/(RUN+'.zip')).read_bytes())==receipt['sha256']
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'];assert set(z.namelist())==set(m)|{'evidence-manifest.json'};assert len(z.namelist())==len(set(z.namelist()));assert all(sha(z.read(k))==v for k,v in m.items())
 for k in m:
  if k.startswith('worker/checkpoint-') and k not in ('worker/checkpoint-0256.pt','worker/checkpoint-0256.json'):continue
  if k.startswith('worker/training-'):continue
  p=(OUT/'import'/k).resolve();assert p.is_relative_to((OUT/'import').resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(k))
for ext in ('.zip','.receipt.json'):shutil.copyfile(DOWNLOAD/(RUN+ext),OUT/(RUN+ext))
# Evaluate against saved source, not mutable working-tree code.
source=OUT/'source';source.mkdir()
with zipfile.ZipFile(PREP/'source.zip') as z:z.extractall(source)
with zipfile.ZipFile(PACKAGE/'NCA-G2-Balanced-Growth-Package.zip') as z:
 manifest=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in manifest['files'].items())
 manifest_hash=sha(z.read('manifest.json'));train_data=json.loads(z.read('data.json'))
 for k in manifest['files']:
  if k.endswith('.py'):
   p=source/k;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(k))
sys.path.insert(0,str(source))
import numpy as np
import torch
from nca.repair_portable import read_portable,runtime
from nca.connected_repair import ConnectedRepair
from nca.generation_data import generate
from nca.massing_targets import evaluate_targets
from nca.massing_cases import target_context
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
identity=json.loads((OUT/'import/worker/identity.json').read_text());request=json.loads((OUT/'import/request.json').read_text());run_result=json.loads((OUT/'import/result.json').read_text())
assert request['manifest_sha256']==manifest_hash==identity['experiment']['manifest_sha256']
assert run_result['status']=='completed' and run_result['worker']['completed']==256
p=read_portable(OUT/'import/worker/checkpoint-0256.pt',identity)
assert p['completed']==256 and p['sampler']['consumed']==256
assert all(int(s['step'])==256 for s in p['optimizer']['state'].values())
assert identity['ordered_training_rows']==[{'arrays':r['arrays'],'sha256':r['arrays_sha256']} for r in train_data['rows']]
starts={'seed':0,'teacher_stage':0}
with zipfile.ZipFile(PACKAGE/'NCA-G2-Balanced-Growth-Package.zip') as z:
 import io
 for i,t in enumerate(p['trace']):
  assert t==json.loads((OUT/f'import/worker/update-{i+1:04d}.json').read_text())
  row=train_data['rows'][t['row_index']]
  with np.load(io.BytesIO(z.read(row['arrays'])),allow_pickle=False) as a:
   distance=a['distance'];depth=0
   if i%2:depth=1+int.from_bytes(hashlib.sha256(f'{i}:{t["row_index"]}'.encode()).digest()[:8],'little')%(int(distance.max())-1)
   expected={'kind':'seed' if depth==0 else 'teacher_stage','depth':depth,'occupied':int(((distance>=0)&(distance<=depth)).sum())}
   assert t['start']==expected;starts[expected['kind']]+=1
recovery=[json.loads((OUT/f'import/worker/recovery-{i:04d}.json').read_text()) for i in (2,3)];assert all(r['full_payload_equal'] and r['state_equal'] for r in recovery)
model=ConnectedRepair().float();model.load_state_dict(p['model'],strict=True);model.eval()
data=json.loads((PREP/'dataset.json').read_text());split=json.loads((PREP/'split-manifest.json').read_text());scenes={r['id']:r['scene'] for r in split['entries']};config=json.loads((PREP/'environment.json').read_text())['config']
rows=[r for r in data['rows'] if r['split']=='development'];assert len(rows)==9
save('request.json',dict(run=RUN,checkpoint=256,horizons=[64,128],firing_seed=2101,split='development',rows=[r['id'] for r in rows],postprocessing=False,reserved_evaluation=False))
save('imports.json',dict(receipt=receipt,verified_payloads=len(m),identity=identity,result=run_result,recovery=recovery,verified_start_counts=starts,evaluation_runtime=runtime(torch.device('cpu'))))
observations=[]
for row in rows:
 path=PREP/row['arrays'];assert sha(path.read_bytes())==row['arrays_sha256']
 with np.load(path,allow_pickle=False) as a:c=a['context'].copy();teacher=a['target'].copy()
 scene=scenes[row['id'].rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config);outputs={}
 for steps in (64,128):
  tick=time.perf_counter();r=generate(model,c,2101,steps);seconds=time.perf_counter()-tick;field=r['field'].numpy()[0,0].astype(bool)
  score,masks=evaluate_targets(field,scene,fields,domain);outputs[steps]=field
  record=dict(case=row['id'],steps=steps,request=row['request'],score=score,absolute_fraction_error=abs(score['volume_fraction']-row['request']),teacher_iou=float((field&teacher).sum()/(field|teacher).sum()),forward_seconds=seconds)
  name=f'observations/{row["id"]}-{steps}'
  (OUT/'observations').mkdir(exist_ok=True)
  with (OUT/(name+'.npz')).open('xb') as f:np.savez_compressed(f,field=field,bulk=masks['bulk'],state=r['state'].numpy(),proposal=r['proposal'].numpy())
  save(name+'.json',record);observations.append(record)
  print(row['id'],steps,score['contract_pass'],score['occupied_voxels'],[k for k,v in score['family_pass'].items() if not v],flush=True)
 assert not (outputs[64]&~outputs[128]).any()
stability=[]
for row in rows:
 a,b=[next(r for r in observations if r['case']==row['id'] and r['steps']==s) for s in (64,128)]
 stability.append(dict(case=row['id'],relative_mass_change=(b['score']['occupied_voxels']-a['score']['occupied_voxels'])/a['score']['occupied_voxels']))
summary={}
for steps in (64,128):
 selected=[r for r in observations if r['steps']==steps]
 summary[str(steps)]=dict(valid=sum(r['score']['contract_pass'] for r in selected),count=len(selected),
  median_absolute_fraction_error=float(np.median([r['absolute_fraction_error'] for r in selected])),max_absolute_fraction_error=max(r['absolute_fraction_error'] for r in selected),
  median_teacher_iou=float(np.median([r['teacher_iou'] for r in selected])),family_pass={k:sum(r['score']['family_pass'][k] for r in selected) for k in selected[0]['score']['family_pass']})
gates=dict(valid_at64=summary['64']['valid']==9,median_fraction_error=summary['64']['median_absolute_fraction_error']<=.02,max_fraction_error=summary['64']['max_absolute_fraction_error']<=.04,valid_at128=summary['128']['valid']==9,stable_mass=all(r['relative_mass_change']<=.05 for r in stability))
save('result.json',dict(summary=summary,gates=gates,accepted=all(gates.values()),stability=stability,observations=observations))
shutil.copyfile(__file__,OUT/'review-script.py');print(json.dumps(dict(summary=summary,gates=gates),indent=2))
