"""Fixed-state TRAIN admission replay; not a rollout or quality benchmark."""
from pathlib import Path
import sys,json,math,hashlib,shutil
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Objective-Audit-2026-10-04';sys.path.insert(0,str(OUT/'source'))
from nca.block_generation import admit
from nca.budget_reference import budget

def growth_quota(ceiling,horizon=64):
    # First seed-containing cube has27cells; any adjacent cube adds at most9.
    if type(ceiling) is not int or type(horizon) is not int or horizon<2 or ceiling<27:raise ValueError('Full first cube and at least two steps required')
    return max(9,math.ceil((ceiling-27)/(horizon-1)))

data=json.loads((OUT/'inputs/data.json').read_text());rows={x['id']:x for x in data['rows']}
records=[]
for metadata in sorted((OUT/'snapshots').glob('*.json')):
    record=json.loads(metadata.read_text())
    if not record['open_unreached'] or not record['progress_available']:continue
    with np.load(metadata.with_suffix('.npz'),allow_pickle=False) as a:
        field=a['field'].copy();e=a['eligible'].copy();progress=a['progress'].copy();z=a['logits'].copy()
    # Torch sigmoid used in the original transition for matching float32 rounding.
    import torch
    q=torch.sigmoid(torch.from_numpy(z)).numpy()
    ceiling=record['ceiling'];quota=growth_quota(ceiling)
    original,original_counts=admit(field,e,q,np.ones_like(e),ceiling,False)
    paced,paced_counts=admit(field,e,q,np.ones_like(e),min(ceiling,int(field.sum())+quota),False)
    # Independent detailed reference verifies exact overlap selection.
    from nca.block_reference import transition
    with np.load(OUT/'inputs'/rows[record['case']]['arrays'],allow_pickle=False) as a:legal=a['condition'][0].astype(bool)
    reference,details=transition(field,legal,q,e,min(ceiling,int(field.sum())+quota))
    assert np.array_equal(reference,paced)
    old_reference,old_details=transition(field,legal,q,e,ceiling);assert np.array_equal(old_reference,original)
    assert original_counts[6]==record['added_voxels'] and original_counts[2]==record['accepted_blocks']
    assert paced_counts[6]<=quota and not (field&~paced).any() and not (paced&~legal).any()
    def tally(details):
        total=details['final_mass']-details['initial_mass']
        advanced=sum(t['new_cells'] for t in details['trace'] if progress[tuple(t['origin'])])
        return dict(added_voxels=total,progress_added_voxels=advanced,progress_share=advanced/total if total else None,accepted_blocks=len(details['trace']))
    records.append(dict(case=record['case'],step=record['step'],quota=quota,original=tally(old_details),paced=tally(details)))

usable=[x for x in records if x['original']['added_voxels'] and x['paced']['added_voxels']]
summary=dict(states=len(records),states_with_growth_both=len(usable),progress_share_higher=sum(x['paced']['progress_share']>x['original']['progress_share']+1e-12 for x in usable),progress_share_equal=sum(abs(x['paced']['progress_share']-x['original']['progress_share'])<=1e-12 for x in usable),progress_share_lower=sum(x['paced']['progress_share']<x['original']['progress_share']-1e-12 for x in usable),original_added_voxels=sum(x['original']['added_voxels'] for x in usable),original_progress_added_voxels=sum(x['original']['progress_added_voxels'] for x in usable),paced_added_voxels=sum(x['paced']['added_voxels'] for x in usable),paced_progress_added_voxels=sum(x['paced']['progress_added_voxels'] for x in usable),quota_range=[min(x['quota'] for x in records),max(x['quota'] for x in records)])
result=dict(protocol='One predetermined quota rule:max(9,ceil((C-27)/63));only fixed pre-cap,unreached TRAIN snapshots with fired progress candidates. Same logits,field,firing,probability threshold and ordering. No quota search.',summary=summary,records=records,optimizer_updates=0,full_rollouts=0,development_or_reserved_access=False,caveat='This changes temporary admission allowance in a frozen state. It does not establish sufficient total mass,connection,coverage,64/128 stability or learned quality under the resulting state distribution. Greater progress share can coexist with fewer absolute progress voxels.')
with (OUT/'fixed-state-pacing-probe.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
shutil.copyfile(__file__,OUT/'pacing-probe.py');print(json.dumps(summary,indent=2))
