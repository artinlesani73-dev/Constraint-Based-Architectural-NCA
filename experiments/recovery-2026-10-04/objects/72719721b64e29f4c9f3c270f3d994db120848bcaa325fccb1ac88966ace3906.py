from pathlib import Path
import sys,json,zipfile,hashlib,shutil,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Objective-Audit-2026-10-04';OUT.mkdir(exist_ok=False)
PARENT=BASE/'G4-Final-Review-2026-10-04';PACKAGE=BASE/'G4-Block-Training-2026-10-03-v2';GUIDANCE=BASE/'G5-Destination-Guidance-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
source=OUT/'source';source.mkdir();inputs=OUT/'inputs';inputs.mkdir()
with zipfile.ZipFile(PACKAGE/'NCA-G4-Block-Package.zip') as z:
    manifest=json.loads(z.read('manifest.json'));assert all(sha(z.read(k))==v for k,v in manifest['files'].items())
    data=json.loads(z.read('data.json'));assert len(data['rows'])==27 and all(x['split']=='train' for x in data['rows'])
    for name in manifest['files']:
        if name.endswith('.py') or name=='data.json' or name.startswith('examples/'):
            p=(source if name.endswith('.py') else inputs)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    save('input-package-manifest.json',manifest)
cue=GUIDANCE/'package/nca/destination_cue.py';shutil.copyfile(cue,source/'nca/destination_cue.py')
for name in ['checkpoint-0256.pt','checkpoint-0256.json','identity.json']:shutil.copyfile(PARENT/'import/worker'/name,inputs/name)
sys.path.insert(0,str(source))
import numpy as np
import torch
from torch.nn import functional as F
from nca.block_generation import BlockNCA,eligibility,full_origins,block_loss,admit,soft_union
from nca.block_reference import transition
from nca.destination_cue import destination_cue
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_portable import read_portable,runtime
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
identity=json.loads((inputs/'identity.json').read_text());checkpoint=read_portable(inputs/'checkpoint-0256.pt',identity)
model=BlockNCA().float();model.load_state_dict(checkpoint['model']);model.eval();started=time.perf_counter()
save('protocol.json',dict(parent='G5 rejected;G4 research reference',model_checkpoint=str(PARENT/'import/worker/checkpoint-0256.pt'),model_sha256=sha((inputs/'checkpoint-0256.pt').read_bytes()),rows=[x['id'] for x in data['rows']],split='train',horizon=32,firing_seed=2101,snapshots=[1,4,8,12,16,24,32],probes=['trained logits','neutral logits0 with identical masks'],optimizer_updates=0,progress_definition='Eligible origin strictly lowers minimum context cube-graph distance among existing full origins. Seed diagnostic uses minimum seed-containing origin distance plus1.',gradient_interpretation='Instantaneous dLoss/dLogit with state/eligibility frozen;not parameter gradient or full rollout counterfactual.',development_or_reserved_access=False,runtime=runtime(torch.device('cpu'))))

def mean(a,mask):return float(a[mask].mean()) if mask.any() else None
def fraction(a,mask):return float(a[mask].mean()) if mask.any() else None
def gradients(logits,m,e,target,origins,seed_phase,D,B,C):
    z=logits.detach().clone().requires_grad_(True)
    loss,front,volume,band=block_loss(z,m,e,target,origins,seed_phase,D,B,C)
    terms={'frontier':front,'weighted_local_volume':.25*volume,'global_band':band}
    grads={k:torch.autograd.grad(v,z,retain_graph=True)[0].detach().numpy()[0,0] for k,v in terms.items()}
    grads['total']=sum(grads.values())
    return grads,{k:float(v.detach()) for k,v in terms.items()}

all_steps=[];all_snapshots=[];case_results=[]
for row in data['rows']:
    p=inputs/row['arrays'];assert sha(p.read_bytes())==row['arrays_sha256']
    with np.load(p,allow_pickle=False) as a:c=a['condition'].copy();teacher=a['target'].astype(bool)
    x=seed_inputs(c);field=x['occupancy'].astype(bool);legal=x['allowed'];valid=full_origins(legal);_,distance=destination_cue(legal,c[5].astype(bool))
    features=perceive(torch.from_numpy(c)[None]);allowed=torch.from_numpy(legal)[None,None];target=torch.from_numpy(teacher.astype(np.float32))[None,None];origins=torch.from_numpy(full_origins(teacher))[None,None]
    D=int(legal.sum());B,C=budget(D,float(c[6,0,0,0]),3);hidden=torch.zeros((1,7,*field.shape));g=torch.Generator().manual_seed(2101);counts=[];steps=[]
    tensor=lambda a:torch.from_numpy(a)[None,None]
    for step in range(1,33):
        mass=int(field.sum());m=tensor(field.copy());e,seed_phase=eligibility(field,valid)
        current=distance[e&(distance>=0)] if seed_phase else distance[full_origins(field)&(distance>=0)]
        best=int(current.min())+(1 if seed_phase else 0) if len(current) else None
        progress=e&(distance>=0)&(distance<best) if best is not None else np.zeros_like(e)
        with torch.no_grad():
            inputs_t=torch.cat((perceive(torch.cat((m.float(),hidden),1)),features,torch.tensor((B-mass)/D).expand(1,1,*field.shape)),1)
            output=model.last(F.relu(model.first(inputs_t)));logits=output[:,:1,1:-1,1:-1,1:-1];q=torch.sigmoid(logits)
            fire=torch.rand(q.shape,generator=g)<.5
        fired=fire.numpy()[0,0];ef=e&fired;positive=ef&origins.numpy()[0,0];negative=ef&~origins.numpy()[0,0];pp=positive&progress;po=positive&~progress;qn=q.numpy()[0,0]
        new_field,details=transition(field,legal,qn,fired,C);fast,count=admit(field,e,qn,fired,C,seed_phase)
        assert np.array_equal(new_field,fast);counts.append(count)
        accepted_progress=sum(bool(progress[tuple(t['origin'])]) for t in details['trace'])
        voxels_progress=sum(t['new_cells'] for t in details['trace'] if progress[tuple(t['origin'])])
        rec=dict(case=row['id'],step=step,seed_phase=seed_phase,mass_before=mass,mass_after=int(new_field.sum()),ceiling=C,best_distance=best,open_unreached=mass<C and best is not None and best>0 and not seed_phase,progress_available=int((ef&progress).sum()),teacher_positive_progress=int(pp.sum()),teacher_positive_other=int(po.sum()),positive_progress_mean_q=mean(qn,pp),positive_other_mean_q=mean(qn,po),positive_progress_above_threshold=fraction(qn>.5,pp),positive_other_above_threshold=fraction(qn>.5,po),accepted_blocks=len(details['trace']),accepted_progress_blocks=accepted_progress,added_voxels=int(new_field.sum())-mass,progress_added_voxels=voxels_progress,budget_rejected_blocks=len(details['rejected_budget']))
        steps.append(rec);all_steps.append(rec)
        if step in [1,4,8,12,16,24,32]:
            gradients_actual,values=gradients(logits,m,tensor(ef),target,origins,seed_phase,D,B,C)
            gradients_neutral,values_neutral=gradients(torch.zeros_like(logits),m,tensor(ef),target,origins,seed_phase,D,B,C)
            parts={}
            for label,grads in [('actual',gradients_actual),('neutral',gradients_neutral)]:
                parts[label]={}
                for name,gradient in grads.items():
                    parts[label][name]=dict(l1=float(np.abs(gradient[ef]).sum()),progress_positive_mean=mean(gradient,pp),other_positive_mean=mean(gradient,po),progress_positive_raise_fraction=fraction(gradient<0,pp),other_positive_raise_fraction=fraction(gradient<0,po),negative_raise_fraction=fraction(gradient<0,negative))
            snap=dict(**rec,loss_terms=values,neutral_loss_terms=values_neutral,gradients=parts)
            name=f'snapshots/{row["id"]}-{step:02d}';save(name+'.json',snap)
            with (OUT/(name+'.npz')).open('xb') as f:np.savez_compressed(f,field=field,logits=logits.numpy()[0,0],eligible=ef,progress=progress,**{'actual_'+k:v for k,v in gradients_actual.items()},**{'neutral_'+k:v for k,v in gradients_neutral.items()})
            all_snapshots.append(snap)
        field=new_field;hidden=(hidden+output[:,1:]*F.pad(fire.float(),(1,1,1,1,1,1)))*allowed
    # One complete reference replay verifies that instrumentation did not change inference.
    if row==data['rows'][0]:
        with torch.no_grad():reference=model.rollout(tensor(x['occupancy']),features,allowed,torch.Generator().manual_seed(2101),32)
        assert np.array_equal(field,reference['field'].numpy()[0,0]) and torch.equal(torch.cat((tensor(field).float(),hidden),1),reference['state']) and np.array_equal(counts,reference['admission_counts'].numpy())
        save('instrumentation-equivalence.json',dict(case=row['id'],steps=32,field_state_and_counts_exact=True))
    save(f'cases/{row["id"]}.json',dict(steps=steps,final_mass=int(field.sum()),ceiling=C))
    with (OUT/f'cases/{row["id"]}.npz').open('xb') as f:np.savez_compressed(f,field=field,admission_counts=np.asarray(counts))
    case_results.append(dict(case=row['id'],final_mass=int(field.sum()),ceiling=C,first_cap_step=next((t['step'] for t in steps if t['mass_after']==C),None)))
    print(row['id'],'complete',flush=True)

training={}
for label,folder in [('G4',PARENT),('G5',BASE/'G5-Final-Review-2026-10-04')]:
    all_counts=[];hit=[]
    for i in range(1,257):
        trace=json.loads((folder/f'import/worker/update-{i:04d}.json').read_text());cs=np.asarray(trace['admission_counts']);cap=trace['budget'][2]
        all_counts.extend((cs[:,0]==cap).tolist());positions=np.flatnonzero(cs[:,0]+cs[:,6]==cap)
        if len(positions):hit.append(int(positions[0]+1))
    training[label]=dict(total_steps=len(all_counts),steps_starting_at_capacity=sum(all_counts),fraction_starting_at_capacity=float(np.mean(all_counts)),updates_reaching_capacity=len(hit),median_first_cap_step=float(np.median(hit)) if hit else None)
save('training-capacity-accounting.json',training)
opportunities=[x for x in all_steps if x['open_unreached'] and x['teacher_positive_progress'] and x['teacher_positive_other']]
snapshots=[x for x in all_snapshots if x['open_unreached'] and x['teacher_positive_progress'] and x['teacher_positive_other']]
neutral_equal=[abs(x['gradients']['neutral']['frontier']['progress_positive_mean']-x['gradients']['neutral']['frontier']['other_positive_mean'])<1e-8 for x in snapshots]
result=dict(cases=case_results,steps=len(all_steps),snapshot_count=len(all_snapshots),open_unreached_matched_opportunities=len(opportunities),matched_gradient_snapshots=len(snapshots),other_positive_q_higher_than_progress_steps=sum(x['positive_other_mean_q']>x['positive_progress_mean_q'] for x in opportunities),both_groups_mean_above_threshold_steps=sum(x['positive_progress_mean_q']>.5 and x['positive_other_mean_q']>.5 for x in opportunities),matched_steps_progress_added_voxels=sum(x['progress_added_voxels'] for x in opportunities),matched_steps_all_added_voxels=sum(x['added_voxels'] for x in opportunities),neutral_bce_treats_both_positive_groups_equally=all(neutral_equal),neutral_bce_equal_snapshots=sum(neutral_equal),snapshots_global_band_increases_other_positive_q=sum(x['gradients']['actual']['global_band']['other_positive_mean']<0 for x in snapshots),snapshots_total_increases_other_positive_q=sum(x['gradients']['actual']['total']['other_positive_mean']<0 for x in snapshots),median_component_l1={k:float(np.median([x['gradients']['actual'][k]['l1'] for x in snapshots])) for k in ['frontier','weighted_local_volume','global_band']},training_capacity=training,optimizer_updates=0,development_or_reserved_access=False,elapsed_seconds=time.perf_counter()-started)
save('result.json',result);shutil.copyfile(__file__,OUT/'audit-script.py');print(json.dumps(result,indent=2))
