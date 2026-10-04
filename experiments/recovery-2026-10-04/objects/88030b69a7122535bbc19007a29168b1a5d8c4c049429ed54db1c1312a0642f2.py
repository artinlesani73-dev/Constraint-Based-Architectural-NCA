from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G9-Training-Diagnosis-2026-10-04';OUT.mkdir(exist_ok=False)
PACKAGE=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
shutil.copyfile(__file__,OUT/'diagnosis-script.py');shutil.copytree(BASE/'G9-Final-Review-2026-10-04/source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(BASE/'G6-Objective-Audit-2026-10-04/source/nca/destination_cue.py',OUT/'source/nca/destination_cue.py')
with zipfile.ZipFile(PACKAGE/'NCA-G9-Access-Ranking-Package.zip') as z:
 m=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in m['files'].items());data=json.loads(z.read('data.json'))
 assert len(data['rows'])==45 and all(r['split']=='train' for r in data['rows'])
 for r in data['rows']:
  p=OUT/'inputs'/r['arrays'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(r['arrays']))
save('input-data.json',data)
sys.path.insert(0,str(OUT/'source'));sys.dont_write_bytecode=True
import numpy as np
import torch
from nca.paced_generation import PacedNCA,eligibility,full_origins,admit
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.destination_cue import destination_cue
from nca.repair_portable import read_portable,runtime
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
models={};checkpoints={}
for label in ['G8','G9']:
 root=BASE/('G8-Final-Review-2026-10-04-v2' if label=='G8' else 'G9-Final-Review-2026-10-04');identity=json.loads((root/'import/worker/identity.json').read_text())
 expected=json.loads((root/'execution.json').read_text())['checkpoint_sha256']
 assert sha((root/'import/worker/checkpoint-0427.pt').read_bytes())==expected
 p=OUT/'inputs'/label;p.mkdir()
 for name in ['checkpoint-0427.pt','checkpoint-0427.json','identity.json']:shutil.copyfile(root/'import/worker'/name,p/name)
 checkpoint=read_portable(p/'checkpoint-0427.pt',identity);model=PacedNCA().float();model.load_state_dict(checkpoint['model']);model.eval();models[label]=model;checkpoints[label]=checkpoint
save('protocol.json',dict(rows=[r['id'] for r in data['rows']],groups={'shared_original_train':27,'added_g7_train':18},models=['G8','G9'],horizon=64,firing_seed=2101,threshold=.5,quota_unchanged=True,optimizer_updates=0,heldout_inference=False,teacher_use='TRAIN-only graph labels and objective gradients after fixed-weight inference;never fed to model',distance_use='Context cube-graph distance used only after rollout for analysis;no destination cue input',runtime=runtime(torch.device('cpu')),gradient_probe_steps=[1,8,16,32,48,64],gradient_interpretation='Instantaneous true-logit gradients with field and firing fixed;not parameter gradients or causal retraining effect',intent='Separate score suppression from allowance rejection;no coefficient sweep'))
save('RESUME-preparation.json',dict(status='TRAIN-only diagnostic in progress',next='Inspect running process and cases before resuming;do not overwrite or duplicate completed outputs',paid_training=False))
from nca.access_labels import teacher_graph,priority
from nca.access_ranking import access_ranking
from nca.paced_generation import block_loss
gradient_records=[]
def gradient_probe(logits,field,eligible,positive,origins,target,phase,seed_phase,D,B,C):
 tensor=lambda a:torch.from_numpy(a)[None,None]
 z=logits.detach().clone().requires_grad_(True)
 _,front,volume,band=block_loss(z,tensor(field),tensor(eligible),tensor(target.astype(np.float32)),tensor(origins),seed_phase,D,B,C)
 rank=access_ranking(z,tensor(eligible),tensor(origins),tensor(positive),phase)
 terms={'membership':front,'volume':.25*volume,'band':band,'ranking':rank}
 grads={k:torch.autograd.grad(v,z,retain_graph=True)[0].detach().numpy()[0,0] for k,v in terms.items()}
 grads['base']=grads['membership']+grads['volume']+grads['band'];grads['combined']=grads['base']+grads['ranking']
 masks={'progress':eligible&origins&positive,'other_teacher':eligible&origins&~positive,'nonteacher':eligible&~origins}
 summary={}
 for name,mask in masks.items():
  summary[name]={'count':int(mask.sum()),'terms':{k:dict(mean=float(g[mask].mean()) if mask.any() else None,l1=float(np.abs(g[mask]).sum()),raises=int((g[mask]<0).sum()),lowers=int((g[mask]>0).sum())) for k,g in grads.items()},'base_raise_to_combined_lower':int((mask&(grads['base']<0)&(grads['combined']>0)).sum())}
 return summary,grads,{k:float(v.detach()) for k,v in terms.items()}

cases=[]
for index,row in enumerate(data['rows']):
 with np.load(OUT/'inputs'/row['arrays'],allow_pickle=False) as a:c=a['condition'].copy();target=a['target'].astype(bool)
 graph=teacher_graph(target,c[5].astype(bool),split='train')
 x=seed_inputs(c);valid=full_origins(x['allowed']);target_origins=full_origins(target);_,distance=destination_cue(x['allowed'],c[5].astype(bool))
 goal=(c[5].astype(bool)&x['allowed']);coords=np.argwhere(goal);goal&=np.arange(goal.shape[2])[None,None,:]==coords[:,2].max()
 for label,model in models.items():
  tick=time.perf_counter()
  captured=[]
  hook=model.last.register_forward_hook(lambda module,args,output:captured.append(output[:,:1,1:-1,1:-1,1:-1].detach().cpu().clone()))
  with torch.no_grad():r=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(2101),64,capture=True)
  hook.remove();assert len(captured)==64
  cs=r['admission_counts'].numpy();caps=r['step_ceilings'].numpy();D,B,C=r['budget'].tolist();K=int(r['quota']);field=x['occupancy'].astype(bool);g=torch.Generator().manual_seed(2101)
  steps=[];ids=[];scores=[];firing=[];offsets=[0]
  for t in range(64):
   q=r['proposals'][t,0,0].numpy();fire=(torch.rand((1,1,*q.shape),generator=g)<.5).numpy()[0,0]
   e,seed_phase=eligibility(field,valid);mass=int(field.sum());full=full_origins(field)
   available=distance[e&(distance>=0)] if seed_phase else distance[full&(distance>=0)]
   best=int(available.min())+(1 if seed_phase else 0) if len(available) else None
   progress=e&(distance>=0)&(distance<best) if best is not None else np.zeros_like(e)
   fired=e&fire;offered=fired&(q>.5);pp=fired&progress&target_origins;other=fired&~progress&target_origins
   born=r['births'][t,0,0].numpy().astype(bool)
   # Re-run detached admission only; all90x64 transitions must match captured inference.
   actual,count=admit(field,e,q,fire,int(caps[t]),seed_phase)
   assert np.array_equal(actual,field|born) and np.array_equal(count,cs[t])
   rec=dict(step=t+1,seed_phase=seed_phase,mass_before=mass,added=int(born.sum()),best_distance=best,connected_before=bool((field&goal).any()),eligible=int(e.sum()),fired=int(fired.sum()),above_threshold=int((e&(q>.5)).sum()),offered=int(offered.sum()),max_eligible_probability=float(q[e].max()) if e.any() else None,positive_progress=int(pp.sum()),positive_other=int(other.sum()),progress_positive_mean_q=float(q[pp].mean()) if pp.any() else None,other_positive_mean_q=float(q[other].mean()) if other.any() else None,progress_offered=int((offered&progress).sum()),other_offered=int((offered&~progress).sum()),quota=K,cap=C,effective_cap=int(caps[t]),unused_step_allowance=int(caps[t]-actual.sum()),allowance_rejected=int(cs[t,3]))
   route_positive,phase=priority(field,x['allowed'],graph)
   # These supervised labels never change rollout: hook captured true logits,not inverse probabilities.
   if t+1 in [1,8,16,32,48,64]:
    grad_summary,grads,values=gradient_probe(captured[t],field,fired,route_positive,target_origins,target,phase,seed_phase,D,B,C)
    snap=dict(model=label,case=row['id'],step=t+1,phase=phase,mass=mass,ceiling=C,groups=grad_summary,loss_terms=values)
    gradient_records.append(snap)
    sp=OUT/f'gradient-snapshots/{label}-{row["id"]}-{t+1:02d}.npz';sp.parent.mkdir(exist_ok=True)
    with sp.open('xb') as f:np.savez_compressed(f,field=field,logits=captured[t].numpy(),fired=fired,positive=route_positive,**grads)
   rec['supervision_phase']=phase
   rec['route_progress_fired']=int((fired&route_positive).sum())
   rec['route_progress_offered']=int((offered&route_positive).sum())
   # Full quota rejection statistic below is descriptive; record actual accepted route progress.
   accepted_origins=full_origins(actual)&~full_origins(field)
   rec['route_progress_completed_origins']=int((accepted_origins&route_positive).sum())
   steps.append(rec);ei=np.flatnonzero(e).astype(np.uint16);ids.append(ei);scores.append(q.ravel()[ei]);firing.append(fire.ravel()[ei]);offsets.append(offsets[-1]+len(ei));field=actual
  assert np.array_equal(field,r['field'][0,0].numpy())
  growth=np.flatnonzero(cs[:,6]>0);first=int(growth[0]+1) if len(growth) else None
  # Absolute reachable-mass ceiling under observed first-cube timing,even if later quota fully used.
  temporal_upper=min(C,27+(64-first)*K) if first else 1
  rec=dict(model=label,case=row['id'],group='shared_original_train' if index<27 else 'added_g7_train',first_birth=first,final_mass=int(field.sum()),target=B,cap=C,quota=K,temporal_upper_given_first_birth=temporal_upper,upper_shortfall=max(0,B-temporal_upper),target_shortfall=max(0,B-int(field.sum())),connected=bool((field&goal).any()),fraction_error=abs(field.sum()/D-float(c[6,0,0,0])),zero_growth_before_cap=sum(s['added']==0 and s['mass_before']<C for s in steps),seed_steps_no_eligible_above_threshold=sum(s['seed_phase'] and s['above_threshold']==0 for s in steps),seconds=time.perf_counter()-tick,steps=steps)
  name=f'cases/{label}-{row["id"]}';save(name+'.json',rec)
  with (OUT/(name+'.npz')).open('xb') as f:np.savez_compressed(f,field=field,state=r['state'].numpy(),births=r['births'].numpy(),admission_counts=cs,step_ceilings=caps,eligible_offsets=np.array(offsets),eligible_ids=np.concatenate(ids),eligible_probabilities=np.concatenate(scores),eligible_firing=np.concatenate(firing))
  cases.append(rec);print(label,row['id'],'first',first,'mass',int(field.sum()),'target',B,'connected',rec['connected'],flush=True)
save('gradient-records.json',gradient_records)
summary={}
for label in models:
 summary[label]={}
 for group in ['shared_original_train','added_g7_train']:
  a=[c for c in cases if c['model']==label and c['group']==group]
  summary[label][group]=dict(count=len(a),median_first_birth=float(np.median([c['first_birth'] for c in a])),range_first_birth=[min(c['first_birth'] for c in a),max(c['first_birth'] for c in a)],connected=sum(c['connected'] for c in a),median_fraction_error=float(np.median([c['fraction_error'] for c in a])),median_target_shortfall=float(np.median([c['target_shortfall'] for c in a])),timing_alone_precludes_target=sum(c['upper_shortfall']>0 for c in a),median_temporal_upper_shortfall=float(np.median([c['upper_shortfall'] for c in a])),median_zero_growth_before_cap=float(np.median([c['zero_growth_before_cap'] for c in a])))
training={}
for label,checkpoint in checkpoints.items():
 traces=checkpoint['trace'];starts={}
 for kind in ['seed','cube_teacher_stage']:
  rows=[t for t in traces if t['start']['kind']==kind];last=[t for t in rows if t['update']>363];firsts=[]
  for t in last:
   count=np.array(t['admission_counts']);ii=np.flatnonzero(count[:,6]>0);firsts.append(int(ii[0]+1) if len(ii) else 65)
  starts[kind]=dict(count=len(rows),last64_updates_count=len(last),last64_median_first_addition=float(np.median(firsts)),last64_median_loss=float(np.median([t['loss'] for t in last])))
 training[label]=starts
save('result.json',dict(summary=summary,training=training,optimizer_updates=0,heldout_inference=False,transition_replays_verified=90*64,cases=[{k:v for k,v in c.items() if k!='steps'} for c in cases]))
print(json.dumps(dict(summary=summary,training=training),indent=2))
