from pathlib import Path
import sys,json,hashlib,shutil,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G11-R3-Ledger-2026-10-04'
OUT.mkdir(exist_ok=False);sys.dont_write_bytecode=True
SRC=BASE/'G11-Allocation-Design-2026-10-04/source'
shutil.copytree(SRC,OUT/'source')
shutil.copyfile(Path(__file__).with_name('g11_reservation_r3.py'),OUT/'source/g11_reservation.py')
shutil.copyfile(__file__,OUT/'run.py')
sys.path.insert(0,str(OUT/'source'))
import numpy as np,torch
from g11_reservation import cumulative_cap
from g11_reservation import witness,rollout
from nca.massing_targets import evaluate_targets,neighbors
from nca.massing_cases import target_context
from nca.contract import entrance_masks
from nca.generation_data import seed_inputs
from nca.paced_generation import PacedNCA,full_origins,connected
from nca.block_reference import cube_union
from nca.repair_training import perceive
from nca.repair_portable import read_portable,runtime
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
protocol=dict(version='G11-R3',training_cases=45,horizons=[64,128],firing_seed=2101,checkpoint='G10 final427',proposal_threshold=.5,quota='K=max(9,ceil((C-27)/63)); CHANGED non-seed cap=min(C,27+(step-1)*K), unused allowance carried; original single-cube seed phase',procedural_priority='R1 order restored: witness lexicographic then learned score; witness bypass unchanged',learned_admission='Original score/firing outside witness plus exact union-cap and union-facade guards',witness='G11 legal shortest-hop seed/east route then deterministic coverage/contact growth; minimum witness, not requested volume',no_teacher=True,optimizer_updates=0,heldout_cases=0,baseline='Cached unchanged G10 paired outputs from R1, not re-inferred',required='All nine witness checks before inference; no certificate means recorded failure, no silent fallback',limitations='Single witness may dominate morphology; no completion deadline guarantee; volume band assessed on output not mandatory witness')
# Pre-inference arithmetic boundary checks, including a delayed seed and saturation.
checks=[]
for C in [27,28,569,1859]:
 K=max(9,int(np.ceil((C-27)/63)))
 caps=[cumulative_cap(C,K,t) for t in range(1,129)]
 assert caps[0]==27 and caps[63]==C and caps[-1]==C
 assert all(a<=b<=C for a,b in zip(caps,caps[1:]))
 # A seed that waits until step20 still admits only one27-voxel cube.
 assert 27<=caps[19] and cumulative_cap(C,K,21)>=27
 checks.append(dict(C=C,K=K,cap64=caps[63],late_seed_step20_cap=caps[19]))
save('boundary-checks.json',dict(arithmetic_cases=checks,interpretation='Cap arithmetic and single-cube-seed premise; not a complete late-seed model replay'))
save('protocol.json',protocol);save('RESUME.json',dict(status='running',next='Inspect result and running process; preserve completed case files; no rerun into same folder'))
previous=BASE/'G10-Final-Review-2026-10-04'
splits=json.loads((previous/'regression-splits.json').read_text())
scenes={e['id']:e['scene'] for v in splits.values() for e in v['entries']}
config=json.loads((BASE/'G10-One-Sided-Training-2026-10-04/environment.json').read_text())['config']
save('scenes.json',scenes);save('config.json',config)
identity=json.loads((previous/'import/worker/identity.json').read_text())
checkpoint=previous/'import/worker/checkpoint-0427.pt'
expected=json.loads((previous/'execution.json').read_text())['checkpoint_sha256']
assert sha(checkpoint.read_bytes())==expected
(OUT/'model').mkdir()
for n in ['checkpoint-0427.pt','checkpoint-0427.json','identity.json']:shutil.copyfile(previous/'import/worker'/n,OUT/'model'/n)
model=PacedNCA();model.load_state_dict(read_portable(checkpoint,identity)['model']);model.eval()
save('runtime.json',dict(runtime=runtime(torch.device('cpu')),checkpoint_sha256=expected))
data=json.loads((BASE/'G11-Allocation-Design-2026-10-04/input-data.json').read_text())
results=[];witnesses=[];started=time.perf_counter()
for row in data['rows']:
 assert row['split']=='train'
 case=row['id'];folder=OUT/'cases'/case;folder.mkdir(parents=True)
 path=BASE/'G11-Allocation-Design-2026-10-04/contexts'/f'{case}.npz'
 with np.load(path) as a:c=a['condition'].copy()
 shutil.copyfile(path,folder/'context.npz')
 x=seed_inputs(c);scene=scenes[case.rsplit('-v',1)[0]]
 fields,domain,_=target_context(scene,config)
 assert np.array_equal(domain,x['allowed'])
 with np.load(BASE/'G11-Allocation-Design-2026-10-04/routes'/f'{case}.npz') as a:route=a['route'].copy()
 D=int(domain.sum());B,C=budget(D,float(c[6,0,0,0]),3)
 tick=time.perf_counter();W,plan=witness(route,x['allowed'],scene,fields,domain,C)
 score,_=evaluate_targets(W,scene,fields,domain)
 wr=dict(case=case,seconds=time.perf_counter()-tick,mass=int(W.sum()),B=B,C=C,plan=plan,score=score,certified=bool(score['contract_pass'] and W.sum()<=C))
 witnesses.append(wr);save(f'cases/{case}/witness.json',wr);np.savez_compressed(folder/'witness.npz',field=W,route=route)
 print(case,'witness',wr['certified'],int(W.sum()),flush=True)
 if not wr['certified']:continue
 # Context-only certification: contact union check preserves facade;
 # all additions legal/connected/full-cubes, preserving other witness families.
 contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool)
 face=neighbors(fields['existing'].astype(bool));endpoints=entrance_masks(scene)
 for e in scene['entrances']:
  if e['kind']=='facade':contact &= ~(endpoints[e['id']]&face&x['allowed'])
 occ=torch.from_numpy(x['occupancy'])[None,None];allowed=torch.from_numpy(x['allowed'])[None,None]
 static=perceive(torch.from_numpy(c)[None])
 with torch.no_grad():
  parent=BASE/'G11-R1-Prototype-2026-10-04-v2/cases'/case
  with np.load(parent/'G10-trajectory.npz') as a:cached_births=a['births'].copy()
  with np.load(parent/'raw-terminal.npz') as a:cached_state=a['state'].copy()
  raw=dict(births=torch.from_numpy(cached_births[:,None,None]),state=torch.from_numpy(cached_state))
  rawsec=json.loads((parent/'G10-128.json').read_text())['seconds128']
 tick=time.perf_counter();hybrid=rollout(model,occ,static,allowed,W,contact);hysec=time.perf_counter()-tick
 rb=raw['births'].numpy()[:,0,0];hb=hybrid['births']
 assert np.array_equal(hybrid['provenance']>0,hb)
 for label,births,seconds in [('G10',rb,rawsec),('G11-R3',hb,hysec)]:
  np.savez_compressed(folder/f'{label}-trajectory.npz',births=births,**({'provenance':hybrid['provenance']} if label=='G11-R3' else {}))
  volumes={}
  for step in [64,128]:
   field=x['occupancy'].astype(bool)|births[:step].any(0)
   assert connected(field) and not(field&~x['allowed']).any()
   assert np.array_equal(cube_union(full_origins(field)),field) and field.sum()<=C
   score,_=evaluate_targets(field,scene,fields,domain);volumes[step]=int(field.sum())
   rec=dict(case=case,model=label,steps=step,score=score,volume_error=abs(int(field.sum())/D-float(c[6,0,0,0])),seconds128=seconds)
   if label=='G11-R3':
    rec.update(procedural_voxels=int((hybrid['provenance'][:step]==2).sum()),learned_voxels=int((hybrid['provenance'][:step]==1).sum()),witness_missing=int((W&~field).sum()))
   results.append(rec);save(f'cases/{case}/{label}-{step}.json',rec)
   np.savez_compressed(folder/f'{label}-{step}.npz',field=field)
  save(f'cases/{case}/{label}-stability.json',dict(growth=(volumes[128]-volumes[64])/volumes[64],passed=(volumes[128]-volumes[64])/volumes[64]<=.05))
 save(f'cases/{case}/hybrid-trace.json',hybrid['trace'])
 np.savez_compressed(folder/'hybrid-states.npz',state64=hybrid['states'][64],state128=hybrid['states'][128])
 np.savez_compressed(folder/'raw-terminal.npz',state=raw['state'].numpy())
 print(case,'evaluated',flush=True)
save('witnesses.json',witnesses)
summary={}
for label in ['G10','G11-R3']:
 summary[label]={}
 for step in [64,128]:
  rr=[r for r in results if r['model']==label and r['steps']==step]
  summary[label][str(step)]=dict(evaluated=len(rr),valid=sum(r['score']['contract_pass'] for r in rr),median_error=float(np.median([r['volume_error'] for r in rr])) if rr else None,max_error=max([r['volume_error'] for r in rr],default=None))
save('result.json',dict(summary=summary,certified_witnesses=sum(w['certified'] for w in witnesses),cases=45,observations=results,wall_seconds=time.perf_counter()-started,accepted=False,interpretation='TRAIN prototype comparison; no deployment admission'))
print(json.dumps(summary),flush=True)

