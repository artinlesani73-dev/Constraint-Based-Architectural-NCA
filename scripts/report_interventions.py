"""Verify and render the completed L2/R1 evidence without overwriting reports."""
from collections import defaultdict
from pathlib import Path
import math,sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json
from nca.recovery import tree_equal
L2='20260923T075113Z_d95fabaf3776'
R1='20260923T075727Z_2233c0e51b9a'

def load(run,roles):
    store=RunStore(REPO/'.local-artifacts/runs')
    if store.verify(run):raise ValueError('Artifact verification failed: '+run)
    directory=store.path(run)
    if read_json(directory/'result.json')['status']!='completed':raise ValueError('Incomplete run')
    records=defaultdict(list)
    for path in sorted((directory/'events').glob('*.json')):
        event=read_json(path)
        if event['kind']=='artifact' and event['details']['role'] in roles:
            records[event['details']['role']].append(read_json(directory/event['details']['path']))
    return directory,records

def render():
    directory,records=load(L2,('budget_record','gradient_record','summary','protocol'))
    budgets=records['budget_record'];cases=records['gradient_record'];cfg=records['protocol'][0]['config']
    assert len(budgets)==108 and len(cases)==63
    keys=lambda r:(r['scene_id'],r['seed'],r['steps'],r['zero_scaffold'])
    assert len({keys(r)+(r['arm'],) for r in cases})==63
    assert len({(r['scene_id'],r['envelope'],r['contract']) for r in budgets})==108
    hard={};matched=0
    for row in cases:
        with np.load(directory/row['fields']['path'],allow_pickle=False) as data:
            for name in ('coverage','access'):
                for suffix,key in [('raw','raw_l2'),('parameters','parameter_l2')]:
                    vector=data[name+'_'+suffix].astype(np.float64)
                    assert np.isfinite(vector).all()
                    assert math.isclose(float(np.linalg.norm(vector)),row['gradients'][name][key],rel_tol=1e-9,abs_tol=1e-12)
            assert np.array_equal(data['state'][:,:cfg['n_frozen']],data['seed'][:,:cfg['n_frozen']])
            if row['arm']=='hard_projected':hard[keys(row)]=data['state'].copy()
            elif row['arm']=='hard_preclamp':
                assert np.array_equal(hard[keys(row)],data['state']);matched+=1
    assert matched==21
    lines=['# L2 material intervention and R1 recovery evidence','',f'L2 `{L2}`; R1 `{R1}`.',
        '', 'All registered artifacts verified. Saved gradients independently reproduce recorded norms; all 21 hard-forward pairs and frozen fields match exactly. These are local diagnostics, not trained-model quality benchmarks.',
        '', '## Explicit budget comparison','',
        'Counts include 17 feasible scenes and one intentionally sealed reference. Compatibility checks only envelope capacity against minimum mass and full-guide mass against maximum mass; they do not establish simultaneous feasibility of all nine objectives.', '',
        '| Region | Denominator | Necessary-valid contexts |','|---|---|---:|']
    for ename in ('scaffold','radius3','radius6'):
        for contract in ('site','envelope'):
            selected=[r for r in budgets if r['envelope']==ename and r['contract']==contract]
            lines.append(f'| {ename} | {contract} | {sum(r["valid_context"] for r in selected)}/18 |')
    lines+=['','Both fractions remain 3%-12%. Changing the denominator changes the physical allowance. Radius-six comparison below uses equivalent fully occupied voxel counts; fractional density is a proxy, not a constructed material specification. Per-scene cubic-metre values are also preserved in raw records.','',
        '| Scene | Site voxels | Envelope voxels | Site minimum / maximum | Envelope minimum / maximum | Envelope / site |',
        '|---|---:|---:|---:|---:|---:|']
    ratios=[]
    for site in sorted([r for r in budgets if r['envelope']=='radius6' and r['contract']=='site'],key=lambda r:r['scene_id']):
        env=next(r for r in budgets if r['scene_id']==site['scene_id'] and r['envelope']=='radius6' and r['contract']=='envelope')
        ratio=env['denominator_voxels'][0]/site['denominator_voxels'][0];ratios.append(ratio)
        lines.append(f'| {site["scene_id"]} | {site["denominator_voxels"][0]} | {env["denominator_voxels"][0]} | {site["minimum_mass"][0]:.3f} / {site["maximum_mass"][0]:.3f} | {env["minimum_mass"][0]:.3f} / {env["maximum_mass"][0]:.3f} | {ratio:.2%} |')
    lines+=['',f'Across all scenes the radius-six allowance becomes {min(ratios):.2%}-{max(ratios):.2%} of the site allowance. This is a substantive objective change, selected only for the recovery test.',
        '', '## Material gradient comparison','',
        'Each ordinary arm has 18 cases (three scenes, three seeds, two horizons). Each absent-scaffold arm has three cases (seed zero, four steps). Binary connectivity uses independent six-neighbor traversal at material threshold 0.5. Nonzero derivative counts alone do not demonstrate an effective training objective.','',
        '| Scaffold | Arm | Cases | Coverage weight derivative nonzero | Access weight derivative nonzero | Binary connected |',
        '|---|---|---:|---:|---:|---:|']
    for zero in (False,True):
        for arm in ('hard_projected','hard_preclamp','smooth_projected'):
            selected=[r for r in cases if r['zero_scaffold']==zero and r['arm']==arm]
            lines.append(f'| {"absent" if zero else "present"} | {arm} | {len(selected)} | {sum(r["gradients"]["coverage"]["parameter_l2"]>0 for r in selected)} | {sum(r["gradients"]["access"]["parameter_l2"]>0 for r in selected)} | {sum(r["metrics"]["connectivity"]["all_connected"] for r in selected)} |')
    lines+=['','The known ground case (seed zero, four steps) retains raw material -0.001279894 and -0.009346317 at the two failed cells under both hard arms. Their coverage derivatives change from zero to -1/36 with pre-clamp guidance. Its projected access weight derivative is still zero. This changes coverage guidance; it does not repair the access derivative.','',
        'Smooth clipping at beta=20 assigns about 0.03466 material at raw zero, before recurrent effects. Its soft access improvement can therefore reflect diffuse background. No straight-through estimator was used.','',
        '### Individual gradient and geometry cases','',
        '| Scene | Seed | Steps | No scaffold | Arm | Coverage weight norm | Access weight norm | Soft mass | Binary voxels | Connected |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---|']
    for r in cases:
        lines.append(f'| {r["scene_id"]} | {r["seed"]} | {r["steps"]} | {r["zero_scaffold"]} | {r["arm"]} | {r["gradients"]["coverage"]["parameter_l2"]:.8g} | {r["gradients"]["access"]["parameter_l2"]:.8g} | {r["soft_mass"]:.5f} | {r["metrics"]["legality"]["material_voxels"]} | {r["metrics"]["connectivity"]["all_connected"]} |')
    rd,rr=load(R1,('summary','recovery_branch'))
    branches={r['branch']:r for r in rr['recovery_branch']};assert len(branches)==4
    checkpoint=lambda b,i=-1:torch.load(rd/branches[b]['checkpoints'][i]['checkpoint']['path'],weights_only=True,map_location='cpu')
    reference=checkpoint('uninterrupted')
    assert tree_equal(checkpoint('prefix'),checkpoint('uninterrupted',1))
    for name in ('resumed','repeat'):
        assert tree_equal(reference,checkpoint(name))
        assert branches['uninterrupted']['trace']==branches['prefix']['trace']+branches[name]['trace']
        for a,b in zip(branches['uninterrupted']['checkpoints'],branches['prefix']['checkpoints']+branches[name]['checkpoints']):
            with np.load(rd/a['fields']['path'],allow_pickle=False) as left,np.load(rd/b['fields']['path'],allow_pickle=False) as right:
                assert left.files==right.files and all(np.array_equal(left[k],right[k]) for k in left.files)
    lines+=['','## Separate-process optimizer recovery','',
        'Four logical updates; ten executed updates across uninterrupted, prefix, resumed and repeated-resume branches. All seven comparisons pass and are independently rechecked here: full checkpoint trees, traces and field arrays, plus the update-two boundary. CPU Adam, StepLR, model buffers and Python/NumPy/global PyTorch/explicit firing RNG are saved.','',
        '| Update | Sampled scene | Steps | Loss | Gradient norm before clipping | Learning rate after step |',
        '|---:|---|---:|---:|---:|---:|']
    for r in branches['uninterrupted']['trace']:
        lines.append(f'| {r["update"]} | {r["scene"]} | {r["steps"]} | {r["total_loss"]:.8f} | {r["gradient_norm_before_clip"]:.8f} | {r["learning_rate_after_step"]:.8f} |')
    lines+=['','The three-scene sampling pool happened to draw only two scenes in four updates; this does not test legacy-scene optimization. All nine terms used unit weights strictly for recovery mechanics. Loss values from differing scenes/horizons are not a learning curve.',
        '', 'Recovery is tested at a completed-update boundary with orderly process exit. It is not a test of CUDA determinism, Colab disconnection, mid-write power loss, or mid-backward continuation. No paid compute, cloud access or deployment occurred.','']
    return '\n'.join(lines)

if __name__=='__main__':
    text=render()
    output=REPO/'experiments/reports/L2-R1-interventions.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:stream.write(text)
    print(output)
