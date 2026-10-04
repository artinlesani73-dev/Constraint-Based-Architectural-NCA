"""Bounded exploratory variety assessment; never replaces frozen R3 evidence."""
from pathlib import Path
import json,hashlib,shutil,sys,copy,itertools,time
import numpy as np,torch
ROOT=Path('C:/Users/artin/Documents/Codex/outputs');SRC=ROOT/'G11-R3-Independent-Review-2026-10-04';OUT=ROOT/'G11-R3-Variety-2026-10-04'
OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
save('RESUME.json',dict(status='preparing',next='Inspect progress and result.json before retry. Do not overwrite this attempt.',source=str(SRC)))
shutil.copytree(SRC/'source',OUT/'source');shutil.copytree(SRC/'model',OUT/'model');shutil.copyfile(__file__,OUT/'run.py')
adapter=OUT/'source/g11_reservation.py';old=adapter.read_text();new=old.replace('contact,steps=128):','contact,steps=128,firing_seed=2101):').replace('manual_seed(2101)','manual_seed(firing_seed)');assert new!=old
adapter.write_text(new,encoding='utf-8')
sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
from context_route import route_from_context
from g11_reservation import witness,rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_benchmark import condition
from nca.paced_generation import PacedNCA
from nca.repair_portable import read_portable
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets,neighbors
from nca.contract import entrance_masks,validate_scene
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
config=json.loads((SRC/'config.json').read_text());model=PacedNCA();model.load_state_dict(read_portable(OUT/'model/checkpoint-0427.pt',json.loads((OUT/'model/identity.json').read_text()))['model']);model.eval()
base=json.loads((SRC/'fresh-splits.json').read_text())['entries'][0]['scene'];scenes=[]
for name in ['wide-gap','narrow-gap','offset-y','partial-obstacle']:
 s=copy.deepcopy(base);s['scene_id']='r3-variety-'+name;s['description']='Exploratory transfer probe: '+name;s['notes']=['Not a held-out acceptance set; one constructed site per geometry change.']
 if name=='wide-gap':
  s['buildings'][0]['x']=[0,6];s['buildings'][0]['gap_facing_x']=6;s['buildings'][1]['x']=[26,32];s['buildings'][1]['gap_facing_x']=26;s['entrances'][0]['x']=6;s['entrances'][1]['x']=24
 if name=='narrow-gap':
  s['buildings'][0]['x']=[0,10];s['buildings'][0]['gap_facing_x']=10;s['buildings'][1]['x']=[22,32];s['buildings'][1]['gap_facing_x']=22;s['entrances'][0]['x']=10;s['entrances'][1]['x']=20
 if name=='offset-y':s['entrances'][0]['y']=9;s['entrances'][1]['y']=21
 if name=='partial-obstacle':s['buildings'].append(dict(id='B_obstacle',x=[15,18],y=[10,19],z=[0,17],side=None,gap_facing_x=None))
 scenes.append(validate_scene(s))
save('scenes.json',scenes)
save('protocol.json',dict(intent='exploratory geometry transfer and firing-seed variation; no tuning or release gate',request=.24,seeds=[2101,2102,2103],horizons=[64,128],models=['G10','R3'],geometry_changes=['gap width larger','gap width smaller','entrance Y offset','partial obstacle'],unchanged='weights, nine families, evaluator, grid, budget, planner ordering',adapter_change='Expose fixed firing seed as optional parameter; default2101 exact parity required',checkpoint_sha256=sha(OUT/'model/checkpoint-0427.pt'),adapter_sha256=sha(adapter),variation_metric='pairwise occupied-voxel Jaccard distance (1-IoU); geometric difference, not architectural quality'))
save('freeze.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
def setup(s):
 fields,domain,_=target_context(s,config);c=condition(s,fields,domain,.24);x=seed_inputs(c);B,C=budget(int(domain.sum()),.24,3);route=route_from_context(c)
 if route is None:return fields,domain,c,x,None,None,None
 W,plan=witness(route,x['allowed'],s,fields,domain,C);score,_=evaluate_targets(W,s,fields,domain)
 contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(s)
 for e in s['entrances']:
  if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
 return fields,domain,c,x,W,contact,dict(score=score,plan=plan,certified=bool(score['contract_pass'] and W.sum()<=C))
def hybrid(c,x,W,contact,seed):
 return rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact,firing_seed=seed)
# One exact replay to ensure the optional seed did not change the frozen default.
fields,domain,c,x,W,contact,cert=setup(base);h=hybrid(c,x,W,contact,2101)
with np.load(SRC/'cases/g11r3-reserved-0-v24/hybrid-trajectory.npz') as ref:
 checks={k:bool(np.array_equal(h[k],ref[k])) for k in ['births','provenance']}
 checks.update({f'state{n}':bool(np.array_equal(h['states'][n],ref[f'state{n}'])) for n in [64,128]})
save('default-seed-parity.json',checks);assert all(checks.values())
records=[];diversity=[];started=time.time()
for s in scenes:
 sid=s['scene_id'];folder=OUT/'cases'/sid;folder.mkdir(parents=True)
 fields,domain,c,x,W,contact,cert=setup(s)
 np.savez_compressed(folder/'context.npz',condition=c)
 save(f'cases/{sid}/certificate.json',cert or dict(certified=False,reason='no route'))
 if W is not None:np.savez_compressed(folder/'witness.npz',field=W)
 allfields={}
 for seed in [2101,2102,2103]:
  with torch.no_grad():raw=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(seed),steps=128,capture=True)
  rb=raw['births'].numpy()[:,0,0];np.savez_compressed(folder/f'G10-{seed}-trajectory.npz',births=rb,state128=raw['state'].numpy())
  hy=hybrid(c,x,W,contact,seed) if cert and cert['certified'] else None
  if hy is not None:
   np.savez_compressed(folder/f'R3-{seed}-trajectory.npz',births=hy['births'],provenance=hy['provenance'],state64=hy['states'][64],state128=hy['states'][128]);save(f'cases/{sid}/R3-{seed}-trace.json',hy['trace'])
  for label in ['G10','R3']:
   outputs={}
   for step in [64,128]:
    rec=dict(scene=sid,seed=seed,model=label,steps=step,status='certificate_failed')
    if label=='G10' or hy is not None:
     f=x['occupancy'].astype(bool)|rb[:step].any(0) if label=='G10' else hy['states'][step][0,0].astype(bool)
     score,_=evaluate_targets(f,s,fields,domain);rec.update(status='evaluated',score=score,volume_error=abs(f.sum()/domain.sum()-.24));outputs[step]=f
     if label=='R3':rec['planner_share']=int((hy['provenance'][:step]==2).sum())/(int(f.sum())-1)
     np.savez_compressed(folder/f'{label}-{seed}-{step}.npz',field=f)
    records.append(rec)
   if outputs:
    growth=float((outputs[128].sum()-outputs[64].sum())/outputs[64].sum());records[-1]['growth64_128']=growth;allfields[label,seed]=outputs[128]
  save('progress.json',dict(completed_scene=sid,seed=seed,records=records));print(sid,seed,flush=True)
 for label in ['G10','R3']:
  pairs=[]
  for a,b in itertools.combinations([2101,2102,2103],2):
   if (label,a) in allfields and (label,b) in allfields:
    f,g=allfields[label,a],allfields[label,b];pairs.append(dict(seeds=[a,b],jaccard_distance=float(1-(f&g).sum()/(f|g).sum())))
  diversity.append(dict(scene=sid,model=label,pairs=pairs))
summary={}
for label in ['G10','R3']:
 rs=[r for r in records if r['model']==label and r['steps']==128];ok=[r for r in rs if r['status']=='evaluated'];summary[label]=dict(total=len(rs),evaluated=len(ok),family_pass=sum(r['score']['contract_pass'] for r in ok),stable=sum(r['growth64_128']<=.05 for r in ok),max_volume_error=max((r['volume_error'] for r in ok),default=None))
save('result.json',dict(summary=summary,records=records,diversity=diversity,seconds=time.time()-started))
save('RESUME.json',dict(status='assessment complete; review results and preserve archive next',next='Inspect failures and diversity without retuning on these cases.',source=str(SRC)))
print(json.dumps(summary),flush=True)
