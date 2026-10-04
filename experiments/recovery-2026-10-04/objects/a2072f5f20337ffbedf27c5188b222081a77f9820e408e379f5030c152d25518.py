from pathlib import Path
import sys,json,hashlib,zipfile,shutil
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G7-Vertical-Training-2026-10-04-v2';ROOT=OUT/'package';OLD=BASE/'G6-Paced-Growth-2026-10-04'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from nca.generation_package import verify
from nca.generation_training import GenerationSession,SETTINGS
from nca.generation_data import seed_inputs
from nca.block_reference import cube_union,full_origins
from nca.repair_portable import TrainingOrder
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
manifest,data=verify(ROOT);paths=list((ROOT/'generation-runs').glob('*/result.json'));assert len(paths)==1
run=paths[0].parent;result=json.loads(paths[0].read_text());assert result['status']=='completed' and result['worker']['completed']==3 and result['cleanup']['active_processes_after_stop']==0
receipt=json.loads(run.with_suffix('.receipt.json').read_text());assert sha(run.with_suffix('.zip').read_bytes())==receipt['sha256']
with zipfile.ZipFile(run.with_suffix('.zip')) as z:
 em=json.loads(z.read('evidence-manifest.json'));assert set(z.namelist())==set(em)|{'evidence-manifest.json'} and len(z.namelist())==len(set(z.namelist()))==len(em)+1
 assert len(em)==receipt['files'] and all(sha(z.read(k))==v for k,v in em.items())
recovery=[json.loads((run/f'worker/recovery-{i:04d}.json').read_text()) for i in [2,3]];assert all(r['full_payload_equal'] and r['state_equal'] for r in recovery)
session=GenerationSession(ROOT,data['rows'],{'manifest_sha256':sha((ROOT/'manifest.json').read_bytes()),'settings':SETTINGS},device='cpu',seed=1201)
for i,row in enumerate(data['rows']):
 with np.load(ROOT/row['arrays'],allow_pickle=False) as a:
  expected=seed_inputs(a['condition']);assert np.array_equal(a['damaged'],expected['occupancy']);assert np.array_equal(cube_union(full_origins(a['target'])),a['target'])
  for completed in [0,1]:
   session.completed=completed;start,features,allowed,target=session.tensors(i)
   assert features.shape==(1,28,32,32,32) and start.shape==target.shape==(1,1,32,32,32)
   assert torch.isfinite(features).all() and torch.isfinite(target).all() and (start<=target).all()
session.completed=0
for i in [1,2,3]:
 t=json.loads((run/f'worker/update-{i:04d}.json').read_text());cs=np.array(t['admission_counts']);caps=np.array(t['step_ceilings']);D,B,C=t['budget'];K=max(9,int(np.ceil((C-27)/63)))
 assert t['quota']==K and cs.shape==(64,7) and np.array_equal(caps,np.where(cs[:,0]==1,C,np.minimum(C,cs[:,0]+K)))
 assert (cs[:,0]+cs[:,6]<=caps).all() and (cs[:,1]==cs[:,2:6].sum(1)).all()
 with np.load(run/f'worker/training-{i:04d}.npz') as a:
  assert sha(a['start'].tobytes())==t['start']['sha256'] and int(a['state'][0].sum())==int(cs[-1,0]+cs[-1,6])
g6init=list((OLD/'package/generation-runs').glob('*/worker/checkpoint-0000.pt'));assert len(g6init)==1
old=torch.load(g6init[0],map_location='cpu',weights_only=False);new=torch.load(run/'worker/checkpoint-0000.pt',map_location='cpu',weights_only=False)
assert old['model'].keys()==new['model'].keys() and all(torch.equal(old['model'][k],new['model'][k]) for k in old['model'])
assert (ROOT/'nca/paced_generation.py').read_bytes()==(OLD/'package/nca/paced_generation.py').read_bytes()
order=TrainingOrder(45,1203);visits=np.zeros(45,int)
for _ in range(256):visits[order.next()]+=1
nb=json.loads((OUT/'NCA-G7-Vertical.ipynb').read_text());pr=json.loads((OUT/'package-receipt.json').read_text());s='\n'.join(''.join(c['source']) for c in nb['cells'])
assert 'APPROVED_G7_JOB=False' in s and pr['sha256'] in s
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
save('verification.json',dict(passed=True,run=run.name,controlled_seconds=result['wall_seconds'],worker_seconds=result['worker']['wall_seconds'],retained_local_updates=3,exact_recoveries=recovery,verified_payloads=len(em),all45_inputs_checked_in_both_start_modes=True,paced_model_and_loss_source_bytes_equal_g6=True,all_initial_parameters_equal_g6=True,verified_step_accounts=192,planned256visits=dict(min=int(visits.min()),max=int(visits.max()),original27=int(visits[:27].sum()),new18=int(visits[27:].sum())),new_label_passes=18,fresh_reserved_teachers=0,fresh_reserved_inference=0,quality_evaluated=False,gpu_runtime_compatibility_not_retested_locally=True))
shutil.copyfile(__file__,OUT/'verify-package.py');print((OUT/'verification.json').read_text())
