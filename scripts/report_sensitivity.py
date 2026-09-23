"""Recompute K2 saved-field outcomes and report the entire paired comparison."""
from pathlib import Path
import json, math, sys, zipfile, hashlib
from collections import defaultdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
from nca.sensitivity import contexts
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.e0 import evaluate
from nca.recovery import metadata_hash
from scripts.diagnostic_inputs import load_inputs
from scripts.run_sensitivity import records,verify_recovery


def verify(run):
    torch.set_num_threads(2);store=RunStore(REPO/'.local-artifacts/runs');d=store.path(run)
    assert not store.verify(run)
    assert read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];proposal=protocol['proposal'];meta=next(iter(protocol['members'].values()))
    gate=protocol['recovery_gate'];checks=verify_recovery(gate)
    gate_meta=records(gate,'protocol')[0]['members']['whole']
    assert gate_meta==protocol['members']['mass_3-s0']
    events=[read_json(p) for p in (d/'events').glob('*.json')]
    source=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/source) as z:
        for name,expected in meta['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
    _,inputs=load_inputs(REPO);cfg=meta['config'];ctxs=contexts(inputs,cfg,proposal['training_scenes'])
    training=records(run,'training_update');evaluation=records(run,'evaluation_record')
    assert len(training)==68 and len(evaluation)==187
    assert len({(r['branch'],r['trace']['update']) for r in training})==68
    assert len({(r['branch'],r['scene'],r['steps']) for r in evaluation})==187
    expected={(b,s,h) for b in [*protocol['members'],'original_checkpoint'] for s in proposal['training_scenes'] for h in (16,50)}
    expected|={('W1_procedural',s,None) for s in proposal['training_scenes']}
    assert {(r['branch'],r['scene'],r['steps']) for r in evaluation}==expected
    for branch,m in protocol['members'].items():
        rows=sorted([r for r in training if r['branch']==branch],key=lambda r:r['trace']['update'])
        assert [r['trace']['scene'] for r in rows]==m['scene_order']
        for row in rows:
            trace=row['trace'];payload=torch.load(d/row['checkpoint']['path'],weights_only=True,map_location='cpu')
            assert payload['metadata']==m and payload['metadata_hash']==metadata_hash(m)
            assert payload['completed_updates']==trace['update'] and trace['steps']==16
            assert trace['learning_rate']==.0001
            assert payload['optimizer']['param_groups'][0]['lr']==.0001
            assert all(float(v['step'])==trace['update'] for v in payload['optimizer']['state'].values())
    with torch.no_grad():
        for row in training+evaluation:
            is_train='trace' in row;v=row['trace'] if is_train else row
            item=inputs[v['scene']];ctx,allowance=ctxs[v['scene']]
            with np.load(d/row['fields']['path'],allow_pickle=False) as f:
                material=torch.from_numpy(f['material'].copy());raw=torch.from_numpy(f['raw'].copy())
            assert torch.isfinite(material).all() and torch.isfinite(raw).all()
            assert ((material>=0)&(material<=1)).all()
            state=item['seed'].clone();state[:,cfg['ch_structure']]=material
            values=research_terms(state,raw,ctx,cfg,allowance,LossSpec(**meta['loss_spec']))
            assert bool(values['context_valid'][0])
            assert {k:float(x[0]) for k,x in values['terms'].items()}==v['terms']
            assert {k:float(x[0]) for k,x in values['regularizers'].items()}==v['regularizers']
            assert float(values['mass_ratio'][0])==v['mass_ratio']
            if is_train:
                coeff=protocol['members'][row['branch']]['coefficients']
                assert float(weighted_total(values,coeff['family_weights'],coeff['regularizer_weights']))==v['total_loss']
            else:
                assert evaluate(state,cfg,item['scene'])==v['metrics']
                for recipe,coeff in proposal['recipes'].items():
                    assert float(weighted_total(values,coeff['family_weights'],coeff['regularizer_weights']))==v['totals_under_both_recipes'][recipe]
                assert v['firing_seed']==(None if row['branch']=='W1_procedural' else 2)
    return protocol,training,evaluation,checks


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['branch'],r['steps'])].append(r)
    result=[]
    for (branch,steps),items in sorted(groups.items(),key=lambda p:(p[0][0],p[0][1] or 0)):
        result.append({'branch':branch,'steps':steps,'cases':len(items),
            'connected':sum(r['metrics']['connectivity']['all_connected'] is True for r in items),
            'scorable':sum(r['metrics']['connectivity']['status']=='scored' for r in items),
            'illegal_voxels':sum(r['metrics']['legality']['illegal_voxels'] for r in items),
            'blocked_voxels':sum(r['metrics']['ground']['blocked_voxels'] for r in items),
            'empty':sum(r['metrics']['legality']['material_voxels']==0 for r in items),
            'unsupported_voxels':sum(r['metrics']['support']['unsupported_voxels'] for r in items),
            'mean_threshold_counts':{t:float(np.mean([r['metrics']['threshold_material_counts'][t] for r in items])) for t in ('0.3','0.5','0.7')},
            'eroded_core_voxels':sum(r['metrics']['thickness_proxy']['core_voxels'] for r in items),
            'over_budget':sum(r['mass_ratio']>.120001 for r in items),
            'under_budget':sum(r['mass_ratio']<.029999 for r in items),
            'mean_mass_ratio':float(np.mean([r['mass_ratio'] for r in items])),
            'mean_terms':{k:float(np.mean([r['terms'][k] for r in items])) for k in items[0]['terms']},
            'mean_regularizers':{k:float(np.mean([r['regularizers'][k] for r in items])) for k in items[0]['regularizers']},
            'mean_totals':{k:float(np.mean([r['totals_under_both_recipes'][k] for r in items])) for k in ('mapped_30','mass_3')}})
    return result


def render(run,protocol,training,evaluation):
    groups=aggregate(evaluation)
    lines=['# K2 coefficient sensitivity: complete local comparison','',f'Run `{run}`; recovery gate `{protocol["recovery_gate"]}`.',
        '', '68 logical optimizer updates: two fixed recipes x two training seeds x17 scenes once each. 187 evaluation records:136 trained-model,34 original-checkpoint and17 static W1 controls. All development geometry; no unseen-scene claim. Material-budget coefficient30 versus3 is the only difference between paired training arms.',
        '', 'Registered hashes and source snapshot checked. All68 completed checkpoints have matching metadata/update counters and constant learning rate. All255 saved training/evaluation fields have recomputed objective values; every evaluation binary metric and both weighted totals were checked. Actual-loop recovery comparisons pass. This checks saved results, not an independent reimplementation of each formula.',
        '', '## Geometry and budget outcomes','', '| Model | Steps | Cases | Connected | Scorable | Empty | Over budget | Under budget | Mean mass/envelope | Illegal / blocked voxels |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["branch"]} | {g["steps"] or "static"} | {g["cases"]} | {g["connected"]} | {g["scorable"]} | {g["empty"]} | {g["over_budget"]} | {g["under_budget"]} | {g["mean_mass_ratio"]:.6g} | {g["illegal_voxels"]} / {g["blocked_voxels"]} |')
    lines+=['','Budget tolerance is1e-6 around the3%-12% limits. Connected uses binary_v1 at material>0.5. Unscorable cases remain explicit; neither empty nor unscorable is counted connected. W1 is static, not a recurrent trajectory.','',
        '## Per-family means and common scoring','', '| Model | Steps | '+ ' | '.join(evaluation[0]['terms'])+' | Total mapped_30 | Total mass_3 |', '|---|---:|'+ '|'.join(['---:']*(len(evaluation[0]['terms'])+2))+'|']
    for g in groups:lines.append(f'| {g["branch"]} | {g["steps"] or "static"} | '+' | '.join(f'{g["mean_terms"][k]:.6g}' for k in evaluation[0]['terms'])+f' | {g["mean_totals"]["mapped_30"]:.6g} | {g["mean_totals"]["mass_3"]:.6g} |')
    lines+=['','Totals use the SAME coefficient set down each column. Lowering a coefficient alone must not be called an improvement. Some zero family residuals arise from hard projection or inactivity.','',
        '## Retained regularizers','', '| Model | Steps | Density/binarization | TV | Boundary cantilever |','|---|---:|---:|---:|---:|']
    for g in groups:lines.append(f'| {g["branch"]} | {g["steps"] or "static"} | {g["mean_regularizers"]["density_binary"]:.6g} | {g["mean_regularizers"]["tv"]:.6g} | {g["mean_regularizers"]["cantilever_boundary"]:.6g} |')
    lines+=['','## Binary support and threshold sensitivity','', '| Model | Steps | Unsupported voxels total | Radius1 eroded core voxels total | Mean material count >.3 | >.5 | >.7 |','|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f'| {g["branch"]} | {g["steps"] or "static"} | {g["unsupported_voxels"]} | {g["eroded_core_voxels"]} | {g["mean_threshold_counts"]["0.3"]:.6g} | {g["mean_threshold_counts"]["0.5"]:.6g} | {g["mean_threshold_counts"]["0.7"]:.6g} |')
    lines+=['','Binary radius1 erosion is an independent bulk proxy; the training thickness term uses radius2. Neither is a minimum-thickness or mechanical certification. Unsupported means disconnected from the declared geometric support boundary.']
    lines+=['','## Paired per-scene tradeoffs: mass_3 versus mapped_30','', '| Training seed | Steps | Coverage lower / higher / equal | Sparsity lower / higher / equal | Connectivity gained / lost |','|---:|---:|---|---|---|']
    lookup={(r['branch'],r['scene'],r['steps']):r for r in evaluation}
    for seed in (0,1):
        for steps in (16,50):
            pairs=[(lookup[(f'mass_3-s{seed}',s,steps)],lookup[(f'mapped_30-s{seed}',s,steps)]) for s in protocol['proposal']['training_scenes']]
            values=[]
            for term in ('coverage','sparsity'):
                diff=[a['terms'][term]-b['terms'][term] for a,b in pairs]
                values.append(f'{sum(x < -1e-7 for x in diff)} / {sum(x > 1e-7 for x in diff)} / {sum(abs(x)<=1e-7 for x in diff)}')
            gained=sum(a['metrics']['connectivity']['all_connected'] is True and b['metrics']['connectivity']['all_connected'] is not True for a,b in pairs)
            lost=sum(b['metrics']['connectivity']['all_connected'] is True and a['metrics']['connectivity']['all_connected'] is not True for a,b in pairs)
            lines.append(f'| {seed} | {steps} | {values[0]} | {values[1]} | {gained} / {lost} |')
    lines+=['','## Every evaluation case','', '| Model | Scene | Steps | Connected | Mass/envelope | Coverage | Access | Sparsity | Support | Total mapped_30 | Total mass_3 |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(evaluation,key=lambda r:(r['branch'],r['scene'],r['steps'] or 0)):
        t=r['terms'];lines.append(f'| {r["branch"]} | {r["scene"]} | {r["steps"] or "static"} | {r["metrics"]["connectivity"]["all_connected"]} | {r["mass_ratio"]:.6g} | {t["coverage"]:.6g} | {t["access"]:.6g} | {t["sparsity"]:.6g} | {t["support"]:.6g} | {r["totals_under_both_recipes"]["mapped_30"]:.6g} | {r["totals_under_both_recipes"]["mass_3"]:.6g} |')
    lines+=['','## Every training update','', '| Model | Update | Scene | Objective before update | Norm before clipping | Seconds incl. evidence |','|---|---:|---|---:|---:|---:|']
    for r in sorted(training,key=lambda r:(r['branch'],r['trace']['update'])):
        t=r['trace'];lines.append(f'| {r["branch"]} | {t["update"]} | {t["scene"]} | {t["total_loss"]:.6g} | {t["gradient_norm_before_clip"]:.6g} | {r["seconds"]:.3f} |')
    lines+=['','Different scenes appear at different updates; this table is not a fixed-scene learning curve. Timing includes local contention/evidence overhead and is not a deployment benchmark. Seventeen updates do not establish convergence. No model architecture, production serving defaults or original checkpoint was changed. No paid compute, Drive operation or deployment.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    run=sys.argv[1];protocol,training,evaluation,checks=verify(run)
    out=REPO/'experiments/reports'
    with (out/'K2-sensitivity.md').open('x',encoding='utf-8') as f:f.write(render(run,protocol,training,evaluation))
    write_once(out/'K2-verification.json',{'run_id':run,'training_checkpoints_verified':68,'fields_recomputed':255,'evaluation_metrics_recomputed':187,'recovery_checks':checks,'source_snapshot_hashes_verified':len(next(iter(protocol['members'].values()))['code_sha256'])})
    write_once(out/'K2-aggregates.json',{'run_id':run,'groups':aggregate(evaluation)})
    print(json.dumps(aggregate(evaluation),indent=2))
