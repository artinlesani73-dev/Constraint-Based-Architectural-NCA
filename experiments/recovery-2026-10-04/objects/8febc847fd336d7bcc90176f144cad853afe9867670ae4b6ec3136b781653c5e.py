from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np,torch
B=Path('C:/Users/artin/Documents/Codex/outputs');OLD=B/'G11-R3-Revisit-2026-10-04';P=B/'G11-R3-Route-Options-2026-10-04';P.mkdir(exist_ok=False)
parent=OLD/'runs/bbf64c65230c4531bf50a887ade4f59d'
for n in ['source','model']:shutil.copytree(OLD/n,P/n)
shutil.copyfile('r3_route_options.py',P/'source/r3_route_options.py');shutil.copyfile(__file__,P/'run.py');shutil.copytree(parent,P/'baseline')
def save(n,v):(P/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
save('RESUME.json',dict(status='running exploratory route comparison',next='Inspect result.json and logs before retry; never overwrite this attempt.'))
save('protocol.json',dict(change='two geometry-planner waypoint alternatives; downstream R3 unchanged',variants=['low_y','high_y'],site='fixed custom site from parent bbf64c65230c4531bf50a887ade4f59d',request=.24,seed=2102,horizons=[64,128],gate='all nine families, absolute volume error<=.04, growth64-128<=.05; no retry or tuning on these results',scope='single-site exploratory design probe; not held-out acceptance',interpretation='Jaccard difference is geometric variation, not architectural merit'))
save('freeze.json',{p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file()})
sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'source'))
from r3_route_options import route_via
from g11_reservation import witness,rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.paced_generation import PacedNCA
from nca.repair_portable import read_portable
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets,neighbors
from nca.contract import entrance_masks
from nca.repair_benchmark import condition
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
model=PacedNCA();model.load_state_dict(read_portable(P/'model/checkpoint-0427.pt',json.loads((P/'model/identity.json').read_text()))['model']);model.eval()
scene=json.loads((parent/'scene.json').read_text());cfg=json.loads((OLD/'config.json').read_text());save('config.json',cfg)
fields,domain,_=target_context(scene,cfg);c=condition(scene,fields,domain,.24);x=seed_inputs(c);_,C=budget(int(domain.sum()),.24,3)
with np.load(parent/'context.npz') as a:assert np.array_equal(c,a['condition'])
contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(scene)
for e in scene['entrances']:
 if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
results=[];arrays={}
with np.load(parent/'R3-128.npz') as a:arrays['baseline']=a['field'].astype(bool)
for side in ['low_y','high_y']:
 route,meta=route_via(c,side);row=dict(variant=side,route=meta,status='no_route')
 if route is not None:
  W,plan=witness(route,x['allowed'],scene,fields,domain,C);score,_=evaluate_targets(W,scene,fields,domain);np.savez_compressed(P/f'{side}-witness.npz',route=route,field=W)
  row.update(status='certificate_failed',witness_score=score,plan=plan)
  if score['contract_pass'] and W.sum()<=C:
   hy=rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact,firing_seed=2102)
   np.savez_compressed(P/f'{side}-trajectory.npz',births=hy['births'],provenance=hy['provenance'],state64=hy['states'][64],state128=hy['states'][128]);save(side+'-trace.json',hy['trace']);row.update(status='evaluated',outputs={})
   for n in [64,128]:
    f=hy['states'][n][0,0].astype(bool);score,_=evaluate_targets(f,scene,fields,domain);row['outputs'][str(n)]=dict(score=score,volume_error=float(abs(f.sum()/domain.sum()-.24)),planner_share=float((hy['provenance'][:n]==2).sum()/(f.sum()-1)));np.savez_compressed(P/f'{side}-{n}.npz',field=f)
   arrays[side]=f;row['growth']=float((hy['states'][128][0,0].sum()-hy['states'][64][0,0].sum())/hy['states'][64][0,0].sum());row['passes']=all(v['score']['contract_pass'] and v['volume_error']<=.04 for v in row['outputs'].values()) and row['growth']<=.05
 results.append(row);save(side+'-result.json',row);print(side,row['status'],row.get('passes'),flush=True)
pairwise=[]
for i,a in enumerate(arrays):
 for b in list(arrays)[i+1:]:pairwise.append(dict(a=a,b=b,jaccard_distance=float(1-(arrays[a]&arrays[b]).sum()/(arrays[a]|arrays[b]).sum())))
save('result.json',dict(variants=results,pairwise=pairwise));print(pairwise)
