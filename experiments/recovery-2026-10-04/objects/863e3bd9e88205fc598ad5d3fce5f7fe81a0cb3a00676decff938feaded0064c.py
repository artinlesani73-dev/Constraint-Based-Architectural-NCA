from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time,os
BASE=Path('C:/Users/artin/Documents/Codex/outputs');PREP=BASE/'G1-Preparation-2026-10-03';PACKAGE=BASE/'G6-Paced-Growth-2026-10-04';OUT=BASE/'G6-Final-Review-2026-10-04';OUT.mkdir(exist_ok=False)
RUN='20261004T081739Z_95ddbf523738';DOWNLOAD=Path('C:/Users/artin/Downloads')
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
with zipfile.ZipFile(PACKAGE/'NCA-G6-Paced-Package.zip') as z:
 manifest=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in manifest['files'].items())
 manifest_hash=sha(z.read('manifest.json'));train_data=json.loads(z.read('data.json'))
 for k in manifest['files']:
  if k.endswith('.py'):
   p=source/k;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(k))
sys.path.insert(0,str(source))
import numpy as np
import torch
from nca.repair_portable import read_portable,runtime
from nca.paced_generation import PacedNCA
from nca.block_generation import full_origins,connected
from nca.block_reference import cube_union
from nca.budget_reference import budget
from nca.generation_training import SETTINGS
from nca.repair_portable import TrainingOrder
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
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
starts={'seed':0,'cube_teacher_stage':0};training_events={'accepted_blocks':0,'allowance_rejected_blocks':0,'redundant_blocks':0,'deferred_seed_blocks':0,'added_voxels':0};training_step_accounts=0
assert identity['model_semantics']=='seed_generation_training_v6_paced'
assert identity['generation_settings']==request['settings']==SETTINGS
assert request['device']=='cuda:0' and request['seed']==1201 and request['updates']==256 and request['max_seconds']==600
assert run_result['wall_seconds']<=600
probe=json.loads((OUT/'import/worker/device-admission-check.json').read_text());assert probe['passed'] and probe['reference_cases']==12 and probe['union_backward_finite']
assert probe['paced_reference_cases']==8 and probe['pacing_horizon_constant']==64
order=TrainingOrder(27,1203)
with zipfile.ZipFile(PACKAGE/'NCA-G6-Paced-Package.zip') as z,zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as evidence:
 import io
 for i,t in enumerate(p['trace']):
  assert t==json.loads((OUT/f'import/worker/update-{i+1:04d}.json').read_text()) and t['update']==i+1 and t['row_index']==order.next()
  row=train_data['rows'][t['row_index']];raw=z.read(row['arrays']);assert sha(raw)==row['arrays_sha256']
  with np.load(io.BytesIO(raw),allow_pickle=False) as a:
   c=a['condition'].copy();target=a['target'].astype(bool);distance=a['block_distance'].copy();x=seed_inputs(c)
   depth=None if i%2==0 else int.from_bytes(hashlib.sha256(f'{i}:{t["row_index"]}'.encode()).digest()[:8],'little')%int(distance.max())
   expected_start=x['occupancy'].astype(np.uint8) if depth is None else cube_union((distance>=0)&(distance<=depth)).astype(np.uint8)
   expected={'kind':'seed' if depth is None else 'cube_teacher_stage','depth':depth,'occupied':int(expected_start.sum()),'sha256':sha(expected_start.tobytes(order='C'))}
   assert t['start']==expected;starts[expected['kind']]+=1
  with np.load(io.BytesIO(evidence.read(f'worker/training-{i+1:04d}.npz')),allow_pickle=False) as a:
   start_field=a['start'];state=a['state'];assert np.array_equal(start_field,expected_start) and np.isfinite(state).all()
   assert np.isin(state[0],[0,1]).all();field=state[0].astype(bool)
  D=int(x['allowed'].sum());B,C=budget(D,float(c[6,0,0,0]),3);assert t['budget']==[D,B,C]
  counts=np.asarray(t['admission_counts']);assert counts.shape==(64,7) and (counts>=0).all()
  K=max(9,int(np.ceil((C-27)/63)));caps=np.asarray(t['step_ceilings'])
  assert t['quota']==K and caps.shape==(64,) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
  assert (counts[:,0]+counts[:,6]<=caps).all()
  assert (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[:,0]+counts[:,6]<=C).all()
  assert (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
  assert counts[0,0]==int(start_field.sum()) and counts[-1,0]+counts[-1,6]==int(field.sum())
  assert not (start_field.astype(bool)&~field).any() and not (field&~x['allowed']).any() and connected(field)
  assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)
  training_step_accounts+=len(counts)
  for k,column in [('accepted_blocks',2),('allowance_rejected_blocks',3),('redundant_blocks',4),('deferred_seed_blocks',5),('added_voxels',6)]:training_events[k]+=int(counts[:,column].sum())
save('training-verification.json',dict(updates=256,step_accounts=training_step_accounts,starts=starts,events=training_events,all_start_hashes_and_saved_fields_verified=True,all_terminal_fields_legal_connected_cube_unions_or_single_seed=True,sampler_order_verified=True,device_probe=probe,full_training_rollouts_replayed=False,all_16384_effective_ceilings_verified=True))
recovery=[json.loads((OUT/f'import/worker/recovery-{i:04d}.json').read_text()) for i in (2,3)];assert all(r['full_payload_equal'] and r['state_equal'] for r in recovery)

# Verify pairing against the existing G4GPU run without another control job.
g4_dir=BASE/'G4-Final-Review-2026-10-04'
g4_identity=json.loads((g4_dir/'import/worker/identity.json').read_text())
g4=read_portable(g4_dir/'import/worker/checkpoint-0256.pt',g4_identity)
assert identity['ordered_training_rows']==g4_identity['ordered_training_rows']
assert [(t['row_index'],t['start']) for t in p['trace']]==[(t['row_index'],t['start']) for t in g4['trace']]
assert torch.equal(p['rng']['firing'],g4['rng']['firing'])
runtime_keys=['python','torch','numpy','gpu_name','cuda_build','cudnn','dtype','threads','deterministic','tf32_matmul','tf32_cudnn']
assert all(identity['runtime'][k]==g4_identity['runtime'][k] for k in runtime_keys)
g4archive=g4_dir/'20261004T065608Z_176639ac1bc5.zip'
g4receipt=json.loads(g4archive.with_suffix('.receipt.json').read_text());assert sha(g4archive.read_bytes())==g4receipt['sha256']
initial=[]
for path in [g4archive,DOWNLOAD/(RUN+'.zip')]:
 with zipfile.ZipFile(path) as z:
  em=json.loads(z.read('evidence-manifest.json'));raw=z.read('worker/checkpoint-0000.pt');assert sha(raw)==em['worker/checkpoint-0000.pt']
  initial.append(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)['model'])
assert initial[0].keys()==initial[1].keys() and all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
save('pairing-with-g4.json',dict(dataset_hashes_equal=True,all256row_and_start_schedules_equal=True,final_firing_rng_equal=True,all_initial_parameters_equal=True,runtime_keys_equal=runtime_keys,extra_parameters=0,limitation='One seeded pacing intervention,not a multi-seed causal estimate. Rejection column now means effective allowance,not global cap alone.'))

model=PacedNCA().float();model.load_state_dict(p['model'],strict=True);model.eval()
data=json.loads((PREP/'dataset.json').read_text());split=json.loads((PREP/'split-manifest.json').read_text());scenes={r['id']:r['scene'] for r in split['entries']};config=json.loads((PREP/'environment.json').read_text())['config']
rows=[r for r in data['rows'] if r['split']=='development'];assert len(rows)==9
save('request.json',dict(run=RUN,checkpoint=256,horizons=[64,128],firing_seed=2101,split='development',rows=[r['id'] for r in rows],postprocessing=False,reserved_evaluation=False))
save('imports.json',dict(receipt=receipt,verified_payloads=len(m),identity=identity,result=run_result,recovery=recovery,verified_start_counts=starts,evaluation_runtime=runtime(torch.device('cpu'))))
def generate(model,c,firing_seed,steps):
 x=seed_inputs(c)
 with torch.no_grad():
  return model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(firing_seed),steps,capture=True)
observations=[]
for row in rows:
 path=PREP/row['arrays'].replace('\\','/');assert sha(path.read_bytes())==row['arrays_sha256']
 with np.load(path,allow_pickle=False) as a:c=a['context'].copy();teacher=a['target'].copy()
 scene=scenes[row['id'].rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config);outputs={}
 for steps in (64,128):
  tick=time.perf_counter();r=generate(model,c,2101,steps);seconds=time.perf_counter()-tick;field=r['field'].numpy()[0,0].astype(bool)
  score,masks=evaluate_targets(field,scene,fields,domain);outputs[steps]=field
  record=dict(case=row['id'],steps=steps,request=row['request'],score=score,absolute_fraction_error=abs(score['volume_fraction']-row['request']),teacher_iou=float((field&teacher).sum()/(field|teacher).sum()),forward_seconds=seconds)
  counts=r['admission_counts'].numpy();D,B,C=r['budget'].tolist();candidate=r['pre_admission_candidates'][-1].numpy()[0,0]
  K=int(r['quota']);caps=r['step_ceilings'].numpy()
  assert K==max(9,int(np.ceil((C-27)/63))) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
  assert (counts[:,0]+counts[:,6]<=caps).all()
  assert (counts[:,0]+counts[:,6]<=C).all() and (counts[:,1]==counts[:,2:6].sum(1)).all()
  assert (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
  hits=np.flatnonzero(counts[:,0]+counts[:,6]>=C)
  assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)
  assert connected(field) and not (field&~domain).any()
  raw_score,_=evaluate_targets(candidate,scene,fields,domain)
  record.update(budget={'domain':D,'target':B,'ceiling':C},first_ceiling_step=int(hits[0]+1) if len(hits) else None,allowance_rejected_events=int(counts[:,3].sum()),quota=K,effective_allowance_hit_steps=int(((counts[:,0]+counts[:,6]==caps)&(caps<C)).sum()),accepted_block_events=int(counts[:,2].sum()),redundant_block_events=int(counts[:,4].sum()),deferred_seed_block_events=int(counts[:,5].sum()),zero_growth_steps=int((counts[:,6]==0).sum()),unused_capacity=int(C-field.sum()),pre_admission_final_candidate_score=raw_score)
  name=f'observations/{row["id"]}-{steps}'
  (OUT/'observations').mkdir(exist_ok=True)
  with (OUT/(name+'.npz')).open('xb') as f:np.savez_compressed(f,field=field,bulk=masks['bulk'],state=r['state'].numpy(),proposal=r['proposal'].numpy(),admission_counts=counts,step_ceilings=caps,quota=K,pre_admission_final_candidate=candidate)
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
