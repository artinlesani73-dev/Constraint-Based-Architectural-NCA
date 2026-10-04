from pathlib import Path
import json,sys,hashlib,shutil,time
import numpy as np,torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G11-R3-Independent-Review-2026-10-04';OLD=BASE/'G10-Final-Review-2026-10-04'
sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
from context_route import route_from_context
from g11_reservation import witness,rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_benchmark import condition
from nca.paced_generation import PacedNCA,connected,full_origins
from nca.block_reference import cube_union
from nca.repair_portable import read_portable,runtime
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets,neighbors
from nca.contract import entrance_masks
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
shutil.copyfile(__file__,OUT/'evaluate.py')
save('frozen-execution-hashes.json',{p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()})
protocol=json.loads((OUT/'protocol.json').read_text())
assert sha((OUT/'source/g11_reservation.py').read_bytes())==protocol['adapter_sha256']
assert sha((OUT/'model/checkpoint-0427.pt').read_bytes())==protocol['checkpoint_sha256']
model=PacedNCA();model.load_state_dict(read_portable(OUT/'model/checkpoint-0427.pt',json.loads((OUT/'model/identity.json').read_text()))['model']);model.eval()
config=json.loads((OUT/'config.json').read_text());entries=json.loads((OUT/'scene-index.json').read_text())['entries']
oldmanifest=json.loads((OLD/'milestone-manifest.json').read_text())['files']
save('runtime.json',runtime(torch.device('cpu')))
observations=[];stability=[];certificates=[];started=time.perf_counter()
for entry in entries:
 scene=entry['scene'];fields,domain,_=target_context(scene,config)
 for request in [.16,.24,.32]:
  case=entry['id']+f'-v{round(request*100)}';p=OUT/'cases'/case;p.mkdir(parents=True)
  c=condition(scene,fields,domain,request);x=seed_inputs(c);D=int(domain.sum());B,C=budget(D,request,3);K=max(9,int(np.ceil((C-27)/63)))
  np.savez_compressed(p/'context.npz',condition=c)
  outputs={}
  if entry['cohort']=='regression':
   cp=OLD/f'contexts/{case}.npz';assert sha(cp.read_bytes())==oldmanifest[cp.relative_to(OLD).as_posix()]
   with np.load(cp) as a:assert np.array_equal(a['context'],c)
   for step in [64,128]:
    path=OLD/f'observations/{case}-{step}.npz';assert sha(path.read_bytes())==oldmanifest[path.relative_to(OLD).as_posix()]
    with np.load(path) as a:outputs['G10',step]=a['field'].copy()
   rawseconds=None
  else:
   with torch.no_grad():
    tick=time.perf_counter();raw=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(2101),steps=128,capture=True);rawseconds=time.perf_counter()-tick
   births=raw['births'].numpy()[:,0,0];np.savez_compressed(p/'raw-trajectory.npz',births=births,state128=raw['state'].numpy())
   for step in [64,128]:outputs['G10',step]=x['occupancy'].astype(bool)|births[:step].any(0)
  tick=time.perf_counter();route=route_from_context(c);wr=dict(case=case,cohort=entry['cohort'],certified=False,reason='no_route')
  if route is not None:
   W,plan=witness(route,x['allowed'],scene,fields,domain,C);wscore,_=evaluate_targets(W,scene,fields,domain)
   wr.update(certified=bool(wscore['contract_pass'] and W.sum()<=C),reason='evaluated',score=wscore,plan=plan)
   np.savez_compressed(p/'witness.npz',route=route,field=W)
  wr['seconds']=time.perf_counter()-tick;certificates.append(wr);save(f'cases/{case}/witness.json',wr)
  if wr['certified']:
   contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(scene)
   for e in scene['entrances']:
    if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
   tick=time.perf_counter();hy=rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact);hyseconds=time.perf_counter()-tick
   np.savez_compressed(p/'hybrid-trajectory.npz',births=hy['births'],provenance=hy['provenance'],state64=hy['states'][64],state128=hy['states'][128])
   save(f'cases/{case}/hybrid-trace.json',hy['trace'])
   assert np.array_equal(hy['births'],hy['provenance']>0)
   f=x['occupancy'].astype(bool)
   for i,born in enumerate(hy['births']):
    before=int(f.sum());assert not(f&born).any();f|=born
    cap=C if before==1 else min(C,27+i*K)
    assert int(f.sum())<=cap and int((f|W).sum())<=C and not(f&~x['allowed']).any()
    assert hy['trace'][i]['mass']==f.sum() and hy['trace'][i]['cap']==cap
   for step in [64,128]:
    assert np.isfinite(hy['states'][step]).all()
    outputs['R3',step]=hy['states'][step][0,0].astype(bool)
  for label in ['G10','R3']:
   for step in [64,128]:
    rec=dict(case=case,scene=entry['id'],cohort=entry['cohort'],model=label,steps=step,request=request,status='certificate_failed')
    if (label,step) in outputs:
     field=outputs[label,step];score,_=evaluate_targets(field,scene,fields,domain)
     assert connected(field) and np.array_equal(cube_union(full_origins(field)),field) and field.sum()<=C
     rec.update(status='evaluated',score=score,volume_error=abs(int(field.sum())/D-request),seconds128=rawseconds if label=='G10' else hyseconds)
     if label=='R3':rec.update(planner_voxels=int((hy['provenance'][:step]==2).sum()),learned_voxels=int((hy['provenance'][:step]==1).sum()),witness_missing=int((W&~field).sum()))
     np.savez_compressed(p/f'{label}-{step}.npz',field=field)
    observations.append(rec);save(f'cases/{case}/{label}-{step}.json',rec)
   if (label,64) in outputs:
    f,g=outputs[label,64],outputs[label,128];assert not(f&~g).any()
    st=dict(case=case,cohort=entry['cohort'],model=label,growth=float((g.sum()-f.sum())/f.sum()),passed=bool((g.sum()-f.sum())/f.sum()<=.05))
   else:st=dict(case=case,cohort=entry['cohort'],model=label,growth=None,passed=False)
   stability.append(st)
  print(case,wr['certified'],observations[-2].get('score',{}).get('contract_pass'),flush=True)
summary={};gates={}
for cohort,expected in [('regression',69),('fresh',12)]:
 for label in ['G10','R3']:
  key=cohort+'_'+label;summary[key]={}
  for step in [64,128]:
   rs=[o for o in observations if o['cohort']==cohort and o['model']==label and o['steps']==step];assert len(rs)==expected
   evaluated=[o for o in rs if o['status']=='evaluated'];errors=[o['volume_error'] for o in evaluated]
   v=dict(expected=expected,evaluated=len(evaluated),valid=sum(o['score']['contract_pass'] for o in evaluated),median_error=float(np.median(errors)) if errors else None,max_error=max(errors,default=None))
   summary[key][str(step)]=v
   gates[f'{key}_{step}_families']=v['valid']==expected
   gates[f'{key}_{step}_volume']=v['evaluated']==expected and v['median_error']<=.02 and v['max_error']<=.04
  gates[key+'_stable']=all(s['passed'] for s in stability if s['cohort']==cohort and s['model']==label)
save('result.json',dict(summary=summary,gates=gates,accepted=all(v for k,v in gates.items() if '_R3_' in k),observations=observations,stability=stability,certificates=certificates,seconds=time.perf_counter()-started))
print(json.dumps(dict(summary=summary,gates=gates),indent=2))

