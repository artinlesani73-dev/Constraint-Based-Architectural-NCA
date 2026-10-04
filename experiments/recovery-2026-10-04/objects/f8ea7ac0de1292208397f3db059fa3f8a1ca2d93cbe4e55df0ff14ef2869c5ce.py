from pathlib import Path
import sys,json,hashlib,shutil,time,traceback
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
PREP=BASE/'G1-Preparation-2026-10-03'
OLD=BASE/'G6-Final-Review-2026-10-04'
OUT=BASE/'G6-Reserved-Review-2026-10-04'
OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
shutil.copyfile(__file__,OUT/'evaluation-script.py')
protocol=json.loads((OLD/'next-generalization-protocol.json').read_text())
candidate=json.loads((OLD/'candidate-freeze.json').read_text())
for name in ['next-generalization-protocol.json','candidate-freeze.json']:
 shutil.copyfile(OLD/name,OUT/name)
for name in ['split-manifest.json','environment.json','dataset.json']:
 shutil.copyfile(PREP/name,OUT/name)
assert sha((OUT/'split-manifest.json').read_bytes())==protocol['split_manifest_sha256']
shutil.copytree(OLD/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['checkpoint-0256.pt','checkpoint-0256.json','identity.json']:
 shutil.copyfile(OLD/'import/worker'/name,OUT/name)
assert sha((OUT/'checkpoint-0256.pt').read_bytes())==protocol['candidate']==candidate['checkpoint_sha256']
assert sha((OUT/'source/nca/massing_targets.py').read_bytes())==candidate['metric_source_sha256']
sys.path.insert(0,str(OUT/'source'));sys.dont_write_bytecode=True
import numpy as np
import torch
from nca.massing_cases import target_context
from nca.repair_benchmark import condition,context_hash
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_portable import read_portable,runtime
from nca.paced_generation import PacedNCA,connected,full_origins
from nca.block_reference import cube_union
from nca.massing_targets import evaluate_targets
from nca.contract import entrance_masks
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
data=json.loads((OUT/'dataset.json').read_text())
split=json.loads((OUT/'split-manifest.json').read_text())
scenes={r['id']:r for r in split['entries']}
config=json.loads((OUT/'environment.json').read_text())['config']
checks=[]
for row in data['rows']:
 if row['split']!='train':continue
 path=PREP/row['arrays'].replace('\\','/');assert sha(path.read_bytes())==row['arrays_sha256']
 scene=scenes[row['id'].rsplit('-v',1)[0]]['scene']
 fields,domain,_=target_context(scene,config);c=condition(scene,fields,domain,row['request'])
 with np.load(path,allow_pickle=False) as a:
  assert c.shape==a['context'].shape and c.dtype==a['context'].dtype and c.tobytes()==a['context'].tobytes()
  assert np.array_equal(seed_inputs(c)['occupancy'],a['seed'])
 assert context_hash(scene,fields,domain)==row['context_sha256']
 checks.append(dict(case=row['id'],context_bytes_equal=True,seed_equal=True,fixture_sha256=row['arrays_sha256']))
assert len(checks)==27
save('context-validation.json',dict(passed=True,count=len(checks),checks=checks,completed_before_reserved_inference=True))
print('27 TRAIN contexts and seeds match archived fixtures exactly.',flush=True)
identity=json.loads((OUT/'identity.json').read_text());p=read_portable(OUT/'checkpoint-0256.pt',identity)
model=PacedNCA().float();model.load_state_dict(p['model'],strict=True);model.eval()
save('execution.json',dict(protocol=protocol,runtime=runtime(torch.device('cpu')),postprocessing=False,teacher_geometry_used=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
observations=[];stability=[]
for sid in protocol['reserved_scene_ids']:
 entry=scenes[sid];assert entry['split']=='reserved';scene=entry['scene']
 for request in protocol['requests']:
  case=f'{sid}-v{round(100*request)}';outputs={}
  for steps in protocol['horizons']:
   tick=time.perf_counter();record=dict(case=case,scene=sid,request=request,steps=steps,status='failed')
   try:
    fields,domain,_=target_context(scene,config);c=condition(scene,fields,domain,request);x=seed_inputs(c)
    assert context_hash(scene,fields,domain)==entry['context_sha256']
    if steps==64:
     (OUT/'contexts').mkdir(exist_ok=True)
     with (OUT/f'contexts/{case}.npz').open('xb') as f:np.savez_compressed(f,context=c,seed=x['occupancy'])
    with torch.no_grad():
     r=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(protocol['firing_seed']),steps,capture=True)
    field=r['field'].numpy()[0,0].astype(bool);outputs[steps]=field
    score,masks=evaluate_targets(field,scene,fields,domain)
    counts=r['admission_counts'].numpy();D,B,C=r['budget'].tolist();K=int(r['quota']);caps=r['step_ceilings'].numpy()
    assert K==max(9,int(np.ceil((C-27)/63)))
    assert np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
    assert (counts[:,0]+counts[:,6]<=caps).all() and (counts[:,1]==counts[:,2:6].sum(1)).all()
    assert (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
    assert connected(field) and not (field&~x['allowed']).any()
    assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)
    assert counts[0,0]==1 and counts[-1,0]+counts[-1,6]==field.sum()
    hits=np.flatnonzero(counts[:,0]+counts[:,6]>=C)
    zyx=np.argwhere(field);extent=(zyx.max(0)-zyx.min(0)+1).tolist()
    record.update(status='evaluated',score=score,absolute_fraction_error=abs(score['volume_fraction']-request),budget=dict(domain=D,target=B,ceiling=C),quota=K,first_ceiling_step=int(hits[0]+1) if len(hits) else None,unused_capacity=int(C-field.sum()),extent_zyx_cells=extent,direct_interface_contact_voxels={k:int((field&mask).sum()) for k,mask in entrance_masks(scene).items()},field_sha256=sha(field.tobytes()))
    (OUT/'observations').mkdir(exist_ok=True)
    with (OUT/f'observations/{case}-{steps}.npz').open('xb') as f:
     np.savez_compressed(f,field=field,bulk=masks['bulk'],state=r['state'].numpy(),proposal=r['proposal'].numpy(),admission_counts=counts,step_ceilings=caps,births=r['births'].numpy(),quota=K)
   except Exception:
    record['error']=traceback.format_exc()
   record['seconds']=time.perf_counter()-tick
   save(f'observations/{case}-{steps}.json',record);observations.append(record)
   print(case,steps,record.get('score',{}).get('contract_pass'),[k for k,v in record.get('score',{}).get('family_pass',{}).items() if not v],record.get('error',''),flush=True)
  if set(outputs)=={64,128}:
   a,b=outputs[64],outputs[128]
   stability.append(dict(case=case,relative_mass_change=float(abs(int(b.sum())-int(a.sum()))/int(a.sum())),identical_field=bool(np.array_equal(a,b)),removed_voxels=int((a&~b).sum())))
  else:stability.append(dict(case=case,relative_mass_change=None,identical_field=False))
summary={}
for steps in protocol['horizons']:
 selected=[r for r in observations if r['steps']==steps and r['status']=='evaluated']
 errors=[r['absolute_fraction_error'] for r in selected]
 summary[str(steps)]=dict(evaluated=len(selected),expected=12,valid=sum(r['score']['contract_pass'] for r in selected),family_pass={k:sum(r['score']['family_pass'][k] for r in selected) for k in ('access','coverage','facade','ground','legality','sparsity','spill','support','thickness')},median_absolute_fraction_error=float(np.median(errors)) if errors else None,max_absolute_fraction_error=max(errors) if errors else None)
g=protocol['gates']
gates=dict(all_nine_at64=summary['64']['valid']==12,all_nine_at128=summary['128']['valid']==12,median_fraction_error=all(s['evaluated']==12 and s['median_absolute_fraction_error']<=g['median_absolute_fraction_error_max'] for s in summary.values()),max_fraction_error=all(s['evaluated']==12 and s['max_absolute_fraction_error']<=g['max_absolute_fraction_error_max'] for s in summary.values()),stable_mass=all(s['relative_mass_change'] is not None and s['relative_mass_change']<=g['per_case_relative_mass_change_max'] for s in stability))
save('result.json',dict(summary=summary,gates=gates,accepted=all(gates.values()),stability=stability,observations=observations,reserved_first_use=True))
print(json.dumps(dict(summary=summary,gates=gates),indent=2))
