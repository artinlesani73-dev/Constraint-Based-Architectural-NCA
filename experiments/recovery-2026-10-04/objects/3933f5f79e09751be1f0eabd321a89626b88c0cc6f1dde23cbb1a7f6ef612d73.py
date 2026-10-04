from pathlib import Path
import sys,json,hashlib,io,zipfile,shutil,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs');PRIOR=BASE/'G1-Final-Review-2026-10-03';OUT=BASE/'G1-Seed-Diagnosis-2026-10-03-v2';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(PRIOR/'source'))
import numpy as np
import torch
from torch.nn import functional as F
from nca.connected_repair import ConnectedRepair,neighbors6,step_loss
from nca.repair_training import perceive
from nca.generation_data import seed_inputs,training_start
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2)
package=BASE/'G1-Portable-Paths-2026-10-03/NCA-G1-Generation-Portable-Package.zip'
with zipfile.ZipFile(package) as z:
 data=json.loads(z.read('data.json'));rows=[r for r in data['rows'] if '-y2-v24' in r['id']]
 assert len(rows)==3
 samples=[]
 for row in rows:
  raw=z.read(row['arrays']);assert hashlib.sha256(raw).hexdigest()==row['arrays_sha256']
  with np.load(io.BytesIO(raw),allow_pickle=False) as a:samples.append((row,{k:a[k].copy() for k in a.files}))
checkpoint=PRIOR/'import/worker/checkpoint-0256.pt';p=torch.load(checkpoint,map_location='cpu',weights_only=True)
model=ConnectedRepair().float();model.load_state_dict(p['model']);model.eval()
save('protocol.json',dict(purpose='Diagnostic only; no acceptance changes or training',cases=[r['id'] for r in rows],stages=['seed','distance3','half_max_teacher_distance'],steps=64,firing_seed=2101,
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reserved_access=False,oracle='target-restricted growth with identical firing, no model; feasibility diagnostic only'))
records=[];began=time.perf_counter()
for row,a in samples:
 c=torch.from_numpy(a['condition'])[None];target=torch.from_numpy(a['target'].astype(bool))[None,None]
 x=seed_inputs(a['condition']);allowed=torch.from_numpy(x['allowed'])[None,None];features=perceive(c)
 for label,depth in [('seed',0),('distance3',3),('half',int(a['distance'].max())//2)]:
  initial=torch.from_numpy(training_start(a['distance'],depth,'train'))[None,None];m=initial.bool();hidden=torch.zeros(1,7,*m.shape[2:]);oracle=m.clone()
  g=torch.Generator().manual_seed(2101);hist=[];rejections=torch.zeros_like(m,dtype=torch.int32);ever=torch.zeros_like(m)
  for t in range(64):
   with torch.no_grad():out=model.last(F.relu(model.first(torch.cat((perceive(torch.cat((m.float(),hidden),1)),features),1))));q=torch.sigmoid(out[:,:1])
   fire=torch.rand(m.shape,generator=g)<.5;eligible=allowed & ~m & neighbors6(m) & fire
   positive=eligible & target;negative=eligible & ~target
   rejected=positive & (q<=.5);rejections+=rejected.int();ever|=positive
   logits=out[:,:1].detach().requires_grad_(True)
   loss,front,volume=step_loss(logits,m,eligible,target.float(),False)
   grad=torch.autograd.grad(loss,logits)[0]
   # Local output-bias pressure for class-balanced BCE, before network Jacobian.
   positive_bias=float((.5*(q[positive]-1)).mean()) if positive.any() else 0.
   negative_bias=float(q[negative].mean()) if negative.any() else 0.
   hist.append(dict(step=t+1,mass=int(m.sum()),positive_fired=int(positive.sum()),positive_accepted=int((positive&(q>.5)).sum()),negative_fired=int(negative.sum()),negative_accepted=int((negative&(q>.5)).sum()),
    positive_q_mean=float(q[positive].mean()) if positive.any() else None,positive_bias_gradient=positive_bias,negative_bias_gradient=negative_bias,
    total_logit_gradient_sum=float(grad.sum()),positive_wrong_sign=int((grad[positive]>=0).sum()),positive_grad_abs_sum=float(grad[positive].abs().sum())))
   with torch.no_grad():
    m=m|(eligible&(q>.5));hidden=(hidden+out[:,1:]*fire)*allowed
    oracle|=allowed & target & ~oracle & neighbors6(oracle) & fire
  if label=='seed':
   with torch.no_grad():reference=model.rollout(initial,features,allowed,torch.Generator().manual_seed(2101),64)
   assert torch.equal(reference['state'],torch.cat((m.float(),hidden),1))
  missing=target&~m;remaining=target&~initial.bool();tp=int((m&target).sum())
  r=dict(case=row['id'],stage=label,depth=depth,start_cells=int(initial.sum()),target_cells=int(target.sum()),final_cells=int(m.sum()),teacher_recall=tp/int(target.sum()),
   missing_cells=int(missing.sum()),remaining_target_recovered=int((m&remaining).sum())/int(remaining.sum()),false_positive=int((m&~target).sum()),
   oracle_missing=int((target&~oracle).sum()),missing_never_fired_frontier=int((missing&~ever).sum()),missing_rejected_8plus=int((missing&(rejections>=8)).sum()),
   positive_fired_events=sum(h['positive_fired'] for h in hist),positive_accepted_events=sum(h['positive_accepted'] for h in hist),positive_wrong_sign_events=sum(h['positive_wrong_sign'] for h in hist),history=hist)
  records.append(r);save(row['id']+'-'+label+'.json',r)
  with (OUT/(row['id']+'-'+label+'.npz')).open('xb') as f:np.savez_compressed(f,initial=initial.numpy(),field=m.numpy(),oracle=oracle.numpy(),rejections=rejections.numpy())
  print(row['id'],label,'remaining recovery',round(r['remaining_target_recovered'],3),'missing',r['missing_cells'],'never frontier',r['missing_never_fired_frontier'],'repeat reject',r['missing_rejected_8plus'],'oracle missing',r['oracle_missing'],flush=True)
summary={}
for stage in ('seed','distance3','half'):
 selected=[r for r in records if r['stage']==stage]
 summary[stage]=dict(median_remaining_recovery=float(np.median([r['remaining_target_recovered'] for r in selected])),missing=sum(r['missing_cells'] for r in selected),missing_never_frontier=sum(r['missing_never_fired_frontier'] for r in selected),missing_rejected8=sum(r['missing_rejected_8plus'] for r in selected),
  positive_acceptance=sum(r['positive_accepted_events'] for r in selected)/sum(r['positive_fired_events'] for r in selected),wrong_sign=sum(r['positive_wrong_sign_events'] for r in selected),oracle_missing=sum(r['oracle_missing'] for r in selected))
save('result.json',dict(summary=summary,wall_seconds=time.perf_counter()-began,records=records,reference_seed_matches=3,optimizer_updates=0));shutil.copyfile(__file__,OUT/'diagnose-script.py');print(json.dumps(summary,indent=2))
