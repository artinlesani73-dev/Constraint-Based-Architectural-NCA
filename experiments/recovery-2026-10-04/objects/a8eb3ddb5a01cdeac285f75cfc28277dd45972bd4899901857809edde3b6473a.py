from pathlib import Path
s=Path('C:/Users/artin/Documents/Codex/outputs/G7-Training-Diagnosis-2026-10-04/diagnosis-script.py').read_text()
s=s.replace("OUT=BASE/'G7-Training-Diagnosis-2026-10-04'","OUT=BASE/'G9-Training-Diagnosis-2026-10-04'")
s=s.replace("PACKAGE=BASE/'G7-Vertical-Training-2026-10-04-v2'","PACKAGE=BASE/'G9-Access-Ranking-Training-2026-10-04-v2'")
s=s.replace("BASE/'G7-Final-Review-2026-10-04/source'","BASE/'G9-Final-Review-2026-10-04/source'")
s=s.replace('NCA-G7-Vertical-Package.zip','NCA-G9-Access-Ranking-Package.zip')
s=s.replace("['G6','G7']","['G8','G9']")
s=s.replace("root=BASE/f'{label}-Final-Review-2026-10-04'","root=BASE/('G8-Final-Review-2026-10-04-v2' if label=='G8' else 'G9-Final-Review-2026-10-04')")
s=s.replace("('candidate-freeze.json' if label=='G6' else 'execution.json')","'execution.json'")
s=s.replace('checkpoint-0256','checkpoint-0427')
s=s.replace("t['update']>192","t['update']>363")
s=s.replace("cases=[]\n","""from nca.access_labels import teacher_graph,priority
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
""")
s=s.replace("x=seed_inputs(c);valid=", "graph=teacher_graph(target,c[5].astype(bool),split='train')\n x=seed_inputs(c);valid=")
s=s.replace("  with torch.no_grad():r=model.rollout","  captured=[]\n  hook=model.last.register_forward_hook(lambda module,args,output:captured.append(output[:,:1,1:-1,1:-1,1:-1].detach().cpu().clone()))\n  with torch.no_grad():r=model.rollout")
s=s.replace("  cs=r['admission_counts']", "  hook.remove();assert len(captured)==64\n  cs=r['admission_counts']")
needle="   steps.append(rec);ei="
replacement="""   route_positive,phase=priority(field,x['allowed'],graph)
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
   steps.append(rec);ei="""
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("summary={}\nfor label in models:","save('gradient-records.json',gradient_records)\nsummary={}\nfor label in models:")
s=s.replace("teacher_use='Post-hoc diagnostic membership only;never fed to model'","teacher_use='TRAIN-only graph labels and objective gradients after fixed-weight inference;never fed to model'")
s=s.replace("intent='Separate initial latency,frontier scores,quota waste and terminal connection misses;no parameter search or causal optimization claim'","gradient_probe_steps=[1,8,16,32,48,64],gradient_interpretation='Instantaneous true-logit gradients with field and firing fixed;not parameter gradients or causal retraining effect',intent='Separate score suppression from allowance rejection;no coefficient sweep'")
Path('diagnose_g9.py').write_text(s);compile(s,'diagnose_g9.py','exec')
ss=Path('C:/Users/artin/Documents/Codex/outputs/G7-Training-Diagnosis-2026-10-04/summarize-diagnosis.py').read_text().replace("OUT=BASE/'G7-Training-Diagnosis-2026-10-04'","OUT=BASE/'G9-Training-Diagnosis-2026-10-04'").replace("['G6','G7']","['G8','G9']")
Path('summarize_g9_diagnosis.py').write_text(ss)

