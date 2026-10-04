from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time,os
BASE=Path('C:/Users/artin/Documents/Codex/outputs');PREP=BASE/'G1-Preparation-2026-10-03';PACKAGE=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';OUT=BASE/'G9-Final-Review-2026-10-04';OUT.mkdir(exist_ok=False)
RUN='20261004T120338Z_60498d5f0838';DOWNLOAD=Path('C:/Users/artin/Downloads')
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
receipt=json.loads((DOWNLOAD/(RUN+'.receipt.json')).read_text());assert sha((DOWNLOAD/(RUN+'.zip')).read_bytes())==receipt['sha256']
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'];assert set(z.namelist())==set(m)|{'evidence-manifest.json'};assert len(z.namelist())==len(set(z.namelist()));assert all(sha(z.read(k))==v for k,v in m.items())
 for k in m:
  if k.startswith('worker/checkpoint-') and k not in ('worker/checkpoint-0427.pt','worker/checkpoint-0427.json'):continue
  if k.startswith('worker/training-'):continue
  p=(OUT/'import'/k).resolve();assert p.is_relative_to((OUT/'import').resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(k))
for ext in ('.zip','.receipt.json'):shutil.copyfile(DOWNLOAD/(RUN+ext),OUT/(RUN+ext))
# Evaluate against saved source, not mutable working-tree code.
source=OUT/'source';source.mkdir()
with zipfile.ZipFile(PREP/'source.zip') as z:z.extractall(source)
with zipfile.ZipFile(PACKAGE/'NCA-G9-Access-Ranking-Package.zip') as z:
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
from nca.ranked_generation import RankedNCA
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
assert run_result['status']=='completed' and run_result['worker']['completed']==427
p=read_portable(OUT/'import/worker/checkpoint-0427.pt',identity)
assert p['completed']==427 and p['sampler']['consumed']==427
assert all(int(s['step'])==427 for s in p['optimizer']['state'].values())
assert identity['ordered_training_rows']==[{'arrays':r['arrays'],'sha256':r['arrays_sha256']} for r in train_data['rows']]
starts={'seed':0,'cube_teacher_stage':0};training_events={'accepted_blocks':0,'allowance_rejected_blocks':0,'redundant_blocks':0,'deferred_seed_blocks':0,'added_voxels':0};training_step_accounts=0
assert identity['model_semantics']=='seed_generation_training_v9_access_ranking'
assert identity['generation_settings']==request['settings']==SETTINGS
assert request['device']=='cuda:0' and request['seed']==1201 and request['updates']==427 and request['max_seconds']==600
assert run_result['wall_seconds']<=600
probe=json.loads((OUT/'import/worker/device-admission-check.json').read_text());assert probe['passed'] and probe['reference_cases']==12 and probe['union_backward_finite']
assert probe['paced_reference_cases']==8 and probe['pacing_horizon_constant']==64
phase_totals={};active_ranking=0;no_route_open=0
order=TrainingOrder(45,1203)
with zipfile.ZipFile(PACKAGE/'NCA-G9-Access-Ranking-Package.zip') as z,zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as evidence:
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
  assert len(t['access_phase_trace'])==64
  assert np.isfinite([t[k] for k in ['loss','frontier_loss','volume_loss','band_loss','ranking_loss','pre_clip_gradient_norm']]).all()
  assert abs(t['loss']-(t['frontier_loss']+.25*t['volume_loss']+t['band_loss']+t['ranking_loss']))<1e-5
  for j,event in enumerate(t['access_phase_trace']):
   assert event['phase'] in ('seed_access','advance_access','connected','no_teacher_route')
   assert event['mass']==int(counts[j,0]) and event['at_capacity']==(counts[j,0]==C)
   assert event['ranking_active']==(event['phase'] in ('seed_access','advance_access') and event['progress_fired']>0 and event['other_teacher_fired']>0)
   assert event['progress_fired']>=0 and event['other_teacher_fired']>=0
   phase_totals[event['phase']]=phase_totals.get(event['phase'],0)+1
   active_ranking+=event['ranking_active']
   no_route_open+=event['phase']=='no_teacher_route' and not event['at_capacity']
  training_step_accounts+=len(counts)
  for k,column in [('accepted_blocks',2),('allowance_rejected_blocks',3),('redundant_blocks',4),('deferred_seed_blocks',5),('added_voxels',6)]:training_events[k]+=int(counts[:,column].sum())
save('training-verification.json',dict(phase_totals=phase_totals,active_ranking_steps=active_ranking,no_route_below_capacity=no_route_open,updates=427,step_accounts=training_step_accounts,starts=starts,events=training_events,all_start_hashes_and_saved_fields_verified=True,all_terminal_fields_legal_connected_cube_unions_or_single_seed=True,sampler_order_verified=True,device_probe=probe,full_training_rollouts_replayed=False,all_27328_effective_ceilings_verified=True))
recovery=[json.loads((OUT/f'import/worker/recovery-{i:04d}.json').read_text()) for i in (2,3)];assert all(r['full_payload_equal'] and r['state_equal'] for r in recovery)


sys.dont_write_bytecode=True
from nca.repair_benchmark import condition,context_hash
from nca.contract import entrance_masks
from nca.generation_training import generate
protocol=json.loads((PACKAGE/'frozen-review.json').read_text())
for dependency in protocol['regression_sources']:
 assert sha(Path(dependency['path']).read_bytes())==dependency['sha256']
assert sha((PACKAGE/'fresh-split-manifest.json').read_bytes())==protocol['fresh_manifest_sha256']
for name in ['frozen-review.json','fresh-split-manifest.json','environment.json','package-receipt.json']:
 shutil.copyfile(PACKAGE/name,OUT/name)
oldsplit=json.loads(Path(protocol['regression_sources'][0]['path']).read_text())
g7split=json.loads(Path(protocol['regression_sources'][1]['path']).read_text())
g8split=json.loads(Path(protocol['regression_sources'][2]['path']).read_text())
newsplit=json.loads((PACKAGE/'fresh-split-manifest.json').read_text())
save('regression-splits.json',dict(g1=oldsplit,g7=g7split,g8=g8split))
entries=[{**e,'cohort':'regression'} for e in oldsplit['entries'] if e['split'] in ['development','reserved']]+[{**e,'cohort':'regression'} for e in g7split['entries'] if e['split']=='reserved']+[{**e,'cohort':'regression'} for e in g8split['entries']]+[{**e,'cohort':'fresh_reserved'} for e in newsplit['entries']]+[{**e,'id':'g8-baseline-'+e['id'],'cohort':'baseline_fresh'} for e in newsplit['entries']]
assert len(entries)==23
config=json.loads((PACKAGE/'environment.json').read_text())['config']
# Input construction checked against every packaged TRAIN fixture before reserved inference.
allscenes={e['id']:e['scene'] for e in oldsplit['entries']+g7split['entries']+g8split['entries']+newsplit['entries']}
with zipfile.ZipFile(PACKAGE/'NCA-G9-Access-Ranking-Package.zip') as z:
 for row in train_data['rows']:
  scene=allscenes[row['id'].rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config)
  with np.load(io.BytesIO(z.read(row['arrays'])),allow_pickle=False) as a:
   c=condition(scene,fields,domain,float(a['condition'][6,0,0,0]))
   assert c.tobytes()==a['condition'].tobytes() and np.array_equal(seed_inputs(c)['occupancy'],a['damaged'])
save('context-verification.json',dict(count=45,all_context_bytes_and_seeds_equal=True,before_fresh_reserved_inference=True))
from nca.generation_training import equal_tree
previous=BASE/'G8-Final-Review-2026-10-04-v2'
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:initial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)
with zipfile.ZipFile(previous/'20261004T111257Z_f5eb0598cf17.zip') as z:oldinitial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)
assert initial.keys()==oldinitial.keys()
assert all(equal_tree(initial[k],oldinitial[k]) for k in initial if k!='identity')
g8payload=torch.load(previous/'import/worker/checkpoint-0427.pt',map_location='cpu',weights_only=False)
assert sha((previous/'import/worker/checkpoint-0427.pt').read_bytes())==protocol['paired_baseline_checkpoint_sha256']
assert all(t['start']==old['start'] and t['row_index']==old['row_index'] for t,old in zip(p['trace'],g8payload['trace']))
assert all(equal_tree(p[k],g8payload[k]) for k in ['sampler','rng','cuda_rng'])
assert (source/'nca/paced_generation.py').read_bytes()==(previous/'source/nca/paced_generation.py').read_bytes()
save('pairing.json',dict(initial_numerical_payload_equal_g8=True,all427_starts_and_rows_equal=True,final_rng_and_sampler_equal=True,original_pacing_source_equal=True,objective_changed=True))
model=RankedNCA().float();model.load_state_dict(p['model'],strict=True);model.eval()
baseline=PacedNCA().float();baseline.load_state_dict(g8payload['model'],strict=True);baseline.eval()
expected_runtime={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
assert all(identity['runtime'][k]==v for k,v in expected_runtime.items())
save('execution.json',dict(protocol=protocol,runtime=runtime(torch.device('cpu')),receipt=receipt,verified_payloads=len(m),checkpoint_sha256=sha((OUT/'import/worker/checkpoint-0427.pt').read_bytes()),run_result=run_result,recovery=recovery))
observations=[];stability=[]
for e in entries:
 scene=e['scene'];fields,domain,_=target_context(scene,config)
 evaluator=baseline if e['cohort']=='baseline_fresh' else model
 assert context_hash(scene,fields,domain)==e['context_sha256']
 for request in protocol['requests']:
  case=e['id']+f'-v{round(request*100)}';c=condition(scene,fields,domain,request);x=seed_inputs(c);outputs={}
  (OUT/'contexts').mkdir(exist_ok=True)
  with (OUT/f'contexts/{case}.npz').open('xb') as f:np.savez_compressed(f,context=c,seed=x['occupancy'])
  for steps in protocol['horizons']:
   tick=time.perf_counter()
   with torch.no_grad():r=evaluator.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(2101),steps,capture=True)
   field=r['field'].numpy()[0,0].astype(bool);score,masks=evaluate_targets(field,scene,fields,domain);outputs[steps]=field
   counts=r['admission_counts'].numpy();caps=r['step_ceilings'].numpy();D,B,C=r['budget'].tolist();K=int(r['quota'])
   assert K==max(9,int(np.ceil((C-27)/63))) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
   assert (counts[:,0]+counts[:,6]<=caps).all() and (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
   assert connected(field) and not(field&~x['allowed']).any() and (field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field))
   coords=np.argwhere(field);hits=np.flatnonzero(counts[:,0]+counts[:,6]>=C)
   record=dict(case=case,scene=e['id'],cohort=e['cohort'],request=request,steps=steps,status='evaluated',score=score,absolute_fraction_error=abs(score['volume_fraction']-request),extent_zyx_cells=(coords.max(0)-coords.min(0)+1).tolist(),direct_interface_contact_voxels={k:int((field&v).sum()) for k,v in entrance_masks(scene).items()},budget=dict(domain=D,target=B,ceiling=C),quota=K,first_ceiling_step=int(hits[0]+1) if len(hits) else None,unused_capacity=int(C-field.sum()),seconds=time.perf_counter()-tick)
   (OUT/'observations').mkdir(exist_ok=True)
   with (OUT/f'observations/{case}-{steps}.npz').open('xb') as f:np.savez_compressed(f,field=field,bulk=masks['bulk'],state=r['state'].numpy(),proposal=r['proposal'].numpy(),admission_counts=counts,step_ceilings=caps,births=r['births'].numpy())
   save(f'observations/{case}-{steps}.json',record);observations.append(record)
   print(e['cohort'],case,steps,score['contract_pass'],[k for k,v in score['family_pass'].items() if not v],flush=True)
  a,b=outputs[64],outputs[128];assert not(a&~b).any()
  stability.append(dict(case=case,cohort=e['cohort'],relative_mass_change=float((int(b.sum())-int(a.sum()))/int(a.sum())),identical_field=bool(np.array_equal(a,b))))
summary={};gates={}
for cohort,expected in [('regression',45),('fresh_reserved',12),('baseline_fresh',12)]:
 summary[cohort]={}
 for steps in [64,128]:
  selected=[o for o in observations if o['cohort']==cohort and o['steps']==steps];assert len(selected)==expected
  errors=[o['absolute_fraction_error'] for o in selected]
  s=dict(valid=sum(o['score']['contract_pass'] for o in selected),expected=expected,median_absolute_fraction_error=float(np.median(errors)),max_absolute_fraction_error=max(errors),family_pass={k:sum(o['score']['family_pass'][k] for o in selected) for k in selected[0]['score']['family_pass']});summary[cohort][str(steps)]=s
  gates[f'{cohort}_{steps}_all_nine']=s['valid']==expected
  gates[f'{cohort}_{steps}_volume_error']=s['median_absolute_fraction_error']<=.02 and s['max_absolute_fraction_error']<=.04
 gates[cohort+'_stable_mass']=all(s['relative_mass_change']<=.05 for s in stability if s['cohort']==cohort)
save('result.json',dict(summary=summary,gates=gates,accepted=all(v for k,v in gates.items() if not k.startswith('baseline_fresh')),observations=observations,stability=stability))
save('scene-index.json',dict(entries=entries))
shutil.copyfile(__file__,OUT/'review-script.py');print(json.dumps(dict(summary=summary,gates=gates),indent=2))
