"""One frozen pacing policy, existing G4 weights, TRAIN-only paired rollout."""
from pathlib import Path
import sys,json,hashlib,shutil,math,time
import numpy as np
import torch
from torch.nn import functional as F
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Objective-Audit-2026-10-04';SOURCE=OUT/'source';sys.path.insert(0,str(SOURCE))
# Add only missing frozen evaluation modules; never replace audited training code.
parent=BASE/'G4-Final-Review-2026-10-04/source'
added=[]
for p in parent.rglob('*.py'):
    dest=SOURCE/p.relative_to(parent)
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);added.append(dest.relative_to(SOURCE).as_posix())
from nca.block_generation import BlockNCA,full_origins,eligibility,admit
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_portable import read_portable
from nca.budget_reference import budget
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
checkpoint=read_portable(OUT/'inputs/checkpoint-0256.pt',json.loads((OUT/'inputs/identity.json').read_text()))
model=BlockNCA().float();model.load_state_dict(checkpoint['model']);model.eval()
data=json.loads((OUT/'inputs/data.json').read_text());prep=BASE/'G1-Preparation-2026-10-03'
split=json.loads((prep/'split-manifest.json').read_text());train_ids={r['id'].rsplit('-v',1)[0] for r in data['rows']};scenes={x['id']:x['scene'] for x in split['entries'] if x['id'] in train_ids}
config=json.loads((prep/'environment.json').read_text())['config'];records=[];started=time.perf_counter()
destination=OUT/'paced-rollouts';destination.mkdir(exist_ok=False)
for row in data['rows']:
    with np.load(OUT/'inputs'/row['arrays'],allow_pickle=False) as a:c=a['condition'].copy()
    x=seed_inputs(c);field=x['occupancy'].astype(bool);legal=x['allowed'];valid=full_origins(legal);D=int(legal.sum());B,C=budget(D,float(c[6,0,0,0]),3)
    quota=max(9,math.ceil((C-27)/63));allowed=torch.from_numpy(legal)[None,None];features=perceive(torch.from_numpy(c)[None]);hidden=torch.zeros((1,7,*field.shape));g=torch.Generator().manual_seed(2101);counts=[];fields_at={}
    with torch.no_grad():
        for step in range(1,129):
            m=torch.from_numpy(field)[None,None];e,seed_phase=eligibility(field,valid);mass=int(field.sum())
            inputs=torch.cat((perceive(torch.cat((m.float(),hidden),1)),features,torch.tensor((B-mass)/D).expand_as(m)),1)
            output=model.last(F.relu(model.first(inputs)));q=torch.sigmoid(output[:,:1,1:-1,1:-1,1:-1]);fire=torch.rand(q.shape,generator=g)<.5
            per_step_cap=C if seed_phase else min(C,mass+quota)
            field,count=admit(field,e,q.numpy()[0,0],fire.numpy()[0,0],per_step_cap,seed_phase);counts.append(count)
            hidden=(hidden+output[:,1:]*F.pad(fire.float(),(1,1,1,1,1,1)))*allowed
            if step in [64,128]:fields_at[step]=field.copy()
    scene=scenes[row['id'].rsplit('-v',1)[0]];context,domain,_=target_context(scene,config)
    with np.load(OUT/f'cases/{row["id"]}.npz',allow_pickle=False) as a:baseline=a['field'].copy()
    assert int(baseline.sum())==C # Monotone full-cap G4 occupancy cannot change after its saved32step state.
    original_score,_=evaluate_targets(baseline,scene,context,domain)
    scores={str(s):evaluate_targets(f,scene,context,domain)[0] for s,f in fields_at.items()}
    change=(int(fields_at[128].sum())-int(fields_at[64].sum()))/int(fields_at[64].sum())
    r=dict(case=row['id'],request=float(c[6,0,0,0]),quota=quota,ceiling=C,baseline=original_score,paced=scores,relative_mass_change64_128=change)
    with (destination/(row['id']+'.json')).open('x') as f:json.dump(r,f,indent=2)
    with (destination/(row['id']+'.npz')).open('xb') as f:np.savez_compressed(f,field64=fields_at[64],field128=fields_at[128],admission_counts=np.asarray(counts))
    records.append(r);print(row['id'],original_score['contract_pass'],'->',scores['64']['contract_pass'],scores['128']['contract_pass'],round(change,4),flush=True)
summary={}
for label in ['baseline','64','128']:
    selected=[r['baseline'] if label=='baseline' else r['paced'][label] for r in records]
    errors=[abs(s['volume_fraction']-r['request']) for s,r in zip(selected,records)]
    summary[label]=dict(valid=sum(s['contract_pass'] for s in selected),count=len(selected),family_pass={k:sum(s['family_pass'][k] for s in selected) for k in selected[0]['family_pass']},median_volume_error=float(np.median(errors)),max_volume_error=max(errors))
result=dict(summary=summary,stable_within5percent=sum(r['relative_mass_change64_128']<=.05 for r in records),max_relative_mass_change=max(r['relative_mass_change64_128'] for r in records),elapsed_seconds=time.perf_counter()-started,optimizer_updates=0,development_or_reserved_evaluation=False,quota_search=False,interpretation='Existing G4 weights evaluated under one changed admission schedule on TRAIN only. Not a trained G6 result. Baseline occupancy reuses32step full-cap output,which is frozen for all later steps by its monotone global cap.',added_evaluation_modules=added,records=records)
with (OUT/'paced-rollout-result.json').open('x') as f:json.dump(result,f,indent=2)
with (OUT/'train-scene-contexts.json').open('x') as f:json.dump(dict(scenes=scenes,config=config),f,indent=2)
shutil.copyfile(__file__,OUT/'paced-rollout-script.py');print(json.dumps({k:v for k,v in result.items() if k not in ['records','added_evaluation_modules']},indent=2))
