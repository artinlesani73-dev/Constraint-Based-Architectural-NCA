"""Verify and report D1 saved direct fields, gradients, checkpoints and controls."""
from pathlib import Path
import sys,json,math,hashlib,zipfile
from collections import defaultdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.direct import DirectField
from nca.sensitivity import contexts
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.e0 import evaluate
from nca.recovery import metadata_hash
from nca.experiments import read_json,write_once
from scripts.diagnostic_inputs import load_inputs
from scripts.run_sensitivity import STORE,records
from scripts.run_direct import recovery_check,admission


def verify(run):
    torch.set_num_threads(2);d=STORE.path(run);assert not STORE.verify(run)
    assert read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];mode=protocol['mode']
    if mode=='recovery':return protocol,records(run,'direct_case'),recovery_check(run)
    recovery_check(protocol['recovery_run'])
    cfg=next(iter(protocol['members'].values()))['config'];_,inputs=load_inputs(REPO)
    scenes=sorted({m['scene'] for m in protocol['members'].values()});ctxs=contexts(inputs,cfg,scenes)
    events=[read_json(p) for p in (d/'events').glob('*.json')]
    src=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    meta=next(iter(protocol['members'].values()))
    with zipfile.ZipFile(d/src) as z:
        for name,expected in meta['code_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    cases=records(run,'direct_case');updates=records(run,'direct_update')
    expected_steps=protocol['config']['pilot_updates'] if mode=='pilot' else protocol['config']['full_updates']
    assert len(cases)==len(protocol['members']) and len(updates)==len(cases)*expected_steps
    assert {r['branch'] for r in cases}==set(protocol['members'])
    assert len({(r['branch'],r['trace']['update']) for r in updates})==len(updates)
    with torch.no_grad():
        for case in cases:
            branch=case['branch'];m=protocol['members'][branch];item=inputs[m['scene']];ctx,allow=ctxs[m['scene']]
            assert item['scene_hash']==m['scene_hash'] and case['completed_updates']==expected_steps
            for scored in (case['initial'],case['final']):
                with np.load(d/scored['fields']['path'],allow_pickle=False) as f:
                    raw=torch.from_numpy(f['raw'].copy());material=torch.from_numpy(f['material'].copy())
                assert torch.equal(raw.clamp(0,1)*ctx.permitted,material)
                if scored['completed_updates']==0:
                    expected=(item['seed'][:,cfg['ch_structure']]+.15*item['scaffold']).clamp(0,1)
                    assert torch.equal(expected,raw)
                state=item['seed'].clone();state[:,cfg['ch_structure']]=material
                v=research_terms(state,raw,ctx,cfg,allow,LossSpec(**m['loss_spec']))
                assert bool(v['context_valid'][0])
                assert {k:float(x[0]) for k,x in v['terms'].items()}==scored['terms']
                assert {k:float(x[0]) for k,x in v['regularizers'].items()}==scored['regularizers']
                assert float(v['mass_ratio'][0])==scored['mass_ratio']
                assert evaluate(state,cfg,item['scene'])==scored['metrics']
                for recipe,c in protocol['config']['recipes'].items():
                    assert float(weighted_total(v,c['family_weights'],c['regularizer_weights']))==scored['totals'][recipe]
            branch_rows=sorted([r for r in updates if r['branch']==branch],key=lambda r:r['trace']['update'])
            assert [r['trace']['update'] for r in branch_rows]==list(range(1,expected_steps+1))
            for row in branch_rows:
                payload=torch.load(d/row['checkpoint']['path'],weights_only=True,map_location='cpu')
                assert payload['metadata']==m and payload['metadata_hash']==metadata_hash(m)
                assert payload['completed_updates']==row['trace']['update']
                assert payload['optimizer']['param_groups'][0]['lr']==.05
                assert float(next(iter(payload['optimizer']['state'].values()))['step'])==row['trace']['update']
                with np.load(d/row['fields']['path'],allow_pickle=False) as f:
                    raw=torch.from_numpy(f['raw'].copy());material=torch.from_numpy(f['material'].copy());grad=f['gradient_before_update']
                    assert torch.equal(raw,payload['model']['raw']) and torch.equal(payload['model']['permitted'],ctx.permitted)
                    assert torch.isfinite(raw).all() and torch.equal(raw.clamp(0,1)*ctx.permitted,material)
                    assert np.isfinite(grad).all() and np.count_nonzero(grad[~ctx.permitted.numpy()])==0
                    assert math.isclose(np.linalg.norm(grad.astype(np.float64)),row['trace']['gradient_norm_before_clip'],rel_tol=2e-5,abs_tol=1e-7)
            with np.load(d/branch_rows[-1]['fields']['path'],allow_pickle=False) as x,np.load(d/case['final']['fields']['path'],allow_pickle=False) as y:
                assert np.array_equal(x['raw'],y['raw']) and np.array_equal(x['material'],y['material'])
        for probe in records(run,'gradient_probe'):
            with np.load(d/probe['fields']['path'],allow_pickle=False) as f:
                for name,norm in probe['norms'].items():assert math.isclose(np.linalg.norm(f[name].astype(np.float64)),norm,rel_tol=1e-9,abs_tol=1e-12)
    controls=records(protocol['config']['source_comparison_run'],'evaluation_record')
    assert not STORE.verify(protocol['config']['source_comparison_run'])
    controls=[r for r in controls if r['scene'] in scenes]
    assert all(r['scene_hash']==inputs[r['scene']]['scene_hash'] for r in controls)
    return protocol,cases,{'saved_field_projections_verified':len(updates),'checkpoint_boundaries_verified':len(updates),
        'initial_final_scores_recomputed':2*len(cases),'gradient_norms_verified':len(updates)+12*len(cases),
        'controls_reused':len(controls),'scope':'Intermediate projections/checkpoints/norms verified; full objective recomputation at initial/final states.'}


def aggregate(cases,which='final'):
    groups=defaultdict(list)
    for c in cases:groups[c['recipe']].append(c[which])
    out=[]
    for recipe,items in sorted(groups.items()):
        inside=lambda r:.029999<=r['mass_ratio']<=.120001
        out.append({'recipe':recipe,'states':which,'cases':len(items),
            'connected':sum(r['metrics']['connectivity']['all_connected'] is True for r in items),
            'in_budget':sum(inside(r) for r in items),
            'connected_in_budget':sum(inside(r) and r['metrics']['connectivity']['all_connected'] is True for r in items),
            'mean_mass_ratio':float(np.mean([r['mass_ratio'] for r in items])),
            'illegal_voxels':sum(r['metrics']['legality']['illegal_voxels'] for r in items),
            'blocked_voxels':sum(r['metrics']['ground']['blocked_voxels'] for r in items),
            'unsupported_voxels':sum(r['metrics']['support']['unsupported_voxels'] for r in items),
            'mean_terms':{k:float(np.mean([r['terms'][k] for r in items])) for k in items[0]['terms']},
            'mean_regularizers':{k:float(np.mean([r['regularizers'][k] for r in items])) for k in items[0]['regularizers']},
            'mean_totals':{k:float(np.mean([r['totals'][k] for r in items])) for k in ('mapped_30','mass_3')}})
    return out


def render(run,protocol,cases,verification):
    mode=protocol['mode'];lines=['# D1 '+mode+' direct-field control','',f'Run `{run}`. Per-scene raw voxel optimization, not NCA training.', '']
    if mode=='recovery':
        lines+=['Four logical/ten executed updates in four fresh processes. All seven exact checkpoint/trace/field comparisons pass. Ground-pair, mass_3, actual direct optimizer. Completed CPU update boundaries only; no CUDA or abrupt-write certification.','',str(verification)]
        return '\n'.join(lines)+'\n'
    lines+=['Both recipes start from the same weak0.15 scaffold. No solved W1 initialization. Every case uses its own voxel parameters, Adam0.05 and the fixed nine-family/three-regularizer objective. Optimizer steps are not NCA growth steps and the compute budgets are not matched.','',
        '## Initial and final summary','', '| Recipe | State | Cases | Connected | In3%-12% budget | Connected AND in budget | Mean material/envelope | Illegal / blocked / unsupported voxels |','|---|---|---:|---:|---:|---:|---:|---|']
    for g in aggregate(cases,'initial')+aggregate(cases):
        lines.append(f'| {g["recipe"]} | {g["states"]} | {g["cases"]} | {g["connected"]} | {g["in_budget"]} | {g["connected_in_budget"]} | {g["mean_mass_ratio"]:.7g} | {g["illegal_voxels"]} / {g["blocked_voxels"]} / {g["unsupported_voxels"]} |')
    lines+=['','Budget tolerance1e-6; connectivity uses material>0.5. Connected-and-in-budget is a limited conjunction, not a full nine-family or architectural success claim.','',
        '## Final per-family means','', '| Recipe | '+' | '.join(cases[0]['final']['terms'])+' | Total mapped_30 | Total mass_3 |','|---|'+ '|'.join(['---:']*11)+'|']
    for g in aggregate(cases):lines.append('| '+g['recipe']+' | '+' | '.join(f'{g["mean_terms"][k]:.7g}' for k in cases[0]['final']['terms'])+f' | {g["mean_totals"]["mapped_30"]:.7g} | {g["mean_totals"]["mass_3"]:.7g} |')
    lines+=['','## Every optimized case','', '| Recipe | Scene | Updates | Connected | Mass/envelope | Coverage | Access | Sparsity | Total mapped_30 | Total mass_3 | Worker seconds |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        r=c['final'];lines.append(f'| {c["recipe"]} | {c["scene"]} | {c["completed_updates"]} | {r["metrics"]["connectivity"]["all_connected"]} | {r["mass_ratio"]:.7g} | {r["terms"]["coverage"]:.7g} | {r["terms"]["access"]:.7g} | {r["terms"]["sparsity"]:.7g} | {r["totals"]["mapped_30"]:.7g} | {r["totals"]["mass_3"]:.7g} | {c["seconds"]:.3f} |')
    controls=records(protocol['config']['source_comparison_run'],'evaluation_record');scenes={c['scene'] for c in cases}
    lines+=['','## Preserved K2 and W1 controls on these same scenes','', '| Model | Scene | NCA growth steps | Connected | Mass/envelope | Total mapped_30 | Total mass_3 |','|---|---|---:|---|---:|---:|---:|']
    for r in controls:
        if r['scene'] in scenes:lines.append(f'| {r["branch"]} | {r["scene"]} | {r["steps"] or "static"} | {r["metrics"]["connectivity"]["all_connected"]} | {r["mass_ratio"]:.7g} | {r["totals_under_both_recipes"]["mapped_30"]:.7g} | {r["totals_under_both_recipes"]["mass_3"]:.7g} |')
    lines+=['','No control was rerun or selected by appearance. These are existing development scenes; direct per-scene fitting does not establish learned generalization, physical usability or safety.','',
        '## Verification and limits','',json.dumps(verification,indent=2),'',
        'Every update has a raw/projected field, pre-update gradient, objective trace and complete optimizer checkpoint. Field/checkpoint records are AFTER updates; traces describe BEFORE updates. All initial/final objectives and binary metrics were recomputed. Intermediate objectives were not all independently recomputed. No production checkpoint/default, paid compute or cloud operation.']
    if mode=='pilot':lines+=['','## Cost admission for the frozen full comparison','',json.dumps(admission(run,protocol['config']),indent=2),'','Admission uses timing only, not quality-based tuning. Full comparison is not an outcome of this pilot.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    run=sys.argv[1];protocol,cases,checks=verify(run)
    prefix={'recovery':'D1R','pilot':'D1P','full':'D1'}[protocol['mode']]
    base=REPO/'experiments/reports'
    with (base/(prefix+'-direct.md')).open('x',encoding='utf-8') as f:f.write(render(run,protocol,cases,checks))
    write_once(base/(prefix+'-verification.json'),{'run_id':run,**checks})
    if protocol['mode']!='recovery':
        out={'run_id':run,'groups':aggregate(cases),'initial_groups':aggregate(cases,'initial')}
        if protocol['mode']=='pilot':out['admission']=admission(run,protocol['config'])
        write_once(base/(prefix+'-aggregates.json'),out);print(json.dumps(out,indent=2))
    else:print(checks)
