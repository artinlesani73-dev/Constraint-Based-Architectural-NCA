"""Verify every F1 saved score/checkpoint and report all evaluation boundaries."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_fitting import STORE,records,verify_recovery,pilot_gate
from scripts.diagnostic_inputs import load_inputs
from nca.experiments import read_json,write_once
from nca.sensitivity import contexts
from nca.losses import LossSpec
from nca.objective import research_terms,weighted_total
from nca.e0 import evaluate
from nca.recovery import metadata_hash
from nca.fitting import Session


def verify(run, replay=True):
    torch.set_num_threads(2)
    d=STORE.path(run);assert not STORE.verify(run)
    assert read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];p=protocol['proposal'];mode=protocol['mode']
    members=protocol['members'];meta=next(iter(members.values()))
    checks=verify_recovery(run if mode=='recovery' else protocol['recovery_gate'])
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    source=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/source) as z:
        for name,expected in meta['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    _,inputs=load_inputs(REPO);cfg=meta['config']
    ctxs=contexts(inputs,cfg,sorted({m['scene'] for m in members.values()}))
    for m in members.values():
        name=m['scene'];assert inputs[name]['scene_hash']==m['scene_hashes'][name]
        assert inputs[name]['source_fields']['sha256']==m['input_field_hashes'][name]
    training=records(run,'training_update');evaluation=records(run,'evaluation_record')
    if mode=='recovery':
        schedule={'whole':[1,2,3],'prefix':[1],'resumed':[2,3],'repeat':[2,3]}
        bounds={'whole':[0,1,3],'prefix':[0,1],'resumed':[3],'repeat':[3]}
    else:
        stop=2 if mode=='pilot' else 64
        schedule={b:list(range(1,stop+1)) for b in members}
        bounds={b:([0,1,2] if mode=='pilot' else p['evaluation']['boundaries']) for b in members}
    expected={(b,u) for b,updates in schedule.items() for u in updates}
    assert len(training)==len(expected) and {(r['branch'],r['trace']['update']) for r in training}==expected
    expected_eval={(b,u,h) for b,updates in bounds.items() for u in updates for h in (16,50)}
    assert len(evaluation)==len(expected_eval) and {(r['branch'],r['trace']['update'],r['trace']['steps']) for r in evaluation}==expected_eval
    with torch.no_grad():
        for row in training+evaluation:
            m=members[row['branch']];t=row['trace'];u=t['update'];name=m['scene']
            assert t['scene']==name
            ckpt=torch.load(d/row['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert ckpt['metadata']==m and ckpt['metadata_hash']==metadata_hash(m)
            assert ckpt['completed_updates']==u
            assert ckpt['optimizer']['param_groups'][0]['lr']==m['optimizer']['lr']
            assert all(float(s['step'])==u for s in ckpt['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
            with np.load(d/row['fields']['path'],allow_pickle=False) as f:
                raw=torch.from_numpy(f['raw'].copy());material=torch.from_numpy(f['material'].copy())
            item=inputs[name];ctx,allow=ctxs[name]
            assert torch.isfinite(raw).all() and torch.isfinite(material).all()
            assert torch.equal(raw.clamp(0,1)*ctx.permitted,material)
            state=item['seed'].clone();state[:,cfg['ch_structure']]=material
            v=research_terms(state,raw,ctx,cfg,allow,LossSpec(**m['loss_spec']))
            assert bool(v['context_valid'][0])
            assert {k:float(x[0]) for k,x in v['terms'].items()}==t['terms']
            assert {k:float(x[0]) for k,x in v['regularizers'].items()}==t['regularizers']
            assert float(v['mass_ratio'][0])==t['mass_ratio']
            if 'total_loss' in t:
                c=m['coefficients']
                assert float(weighted_total(v,c['family_weights'],c['regularizer_weights']))==t['total_loss']
                assert t['steps']==16 and np.isfinite(t['gradient_norm_before_clip'])
            else:
                assert t['firing_seed']==2 and evaluate(state,cfg,item['scene'])==t['metrics']
                for recipe,c in p['recipes'].items():
                    assert float(weighted_total(v,c['family_weights'],c['regularizer_weights']))==t['totals_under_both_recipes'][recipe]
                assert t['raw_saturation']=={'permitted_below_zero':int((raw[ctx.permitted]<0).sum()),
                    'permitted_above_one':int((raw[ctx.permitted]>1).sum()),
                    'guide_below_zero':int((raw[ctx.coverage]<0).sum()),'guide_voxels':int(ctx.coverage.sum())}
    replayed=0
    if replay and mode=='study':
        for branch,m in members.items():
            rows=[r for r in evaluation if r['branch']==branch and r['trace']['update']==64]
            session=Session(m,d/rows[0]['checkpoint']['path'])
            for row in rows:
                trace,fields=session.score(row['trace']['steps'],2)
                assert trace==row['trace']
                with np.load(d/row['fields']['path'],allow_pickle=False) as f:
                    assert all(np.array_equal(fields[k],f[k]) for k in fields)
                replayed+=1
    controls=[]
    for source in (p['source_sensitivity_run'],p['source_direct_run']):
        assert not STORE.verify(source) and read_json(STORE.path(source)/'result.json')['status']=='completed'
    for row in records(p['source_sensitivity_run'],'evaluation_record'):
        if row['scene'] in ctxs:
            assert row['scene_hash']==inputs[row['scene']]['scene_hash']
            controls.append(dict(row,source_run=p['source_sensitivity_run']))
    for case in records(p['source_direct_run'],'direct_case'):
        if case['scene'] in ctxs:
            row=case['final']
            controls.append(dict(row,branch='D1_'+case['recipe'],scene=case['scene'],steps=None,
                totals_under_both_recipes=row['totals'],source_run=p['source_direct_run']))
    checks.update(source_hashes_verified=len(meta['code_sha256']),saved_fields_rescored=len(training)+len(evaluation),
        training_checkpoints_verified=len(training),evaluation_metrics_verified=len(evaluation),
        final_checkpoint_rollouts_replayed=replayed,controls_reused=len(controls))
    if mode in ('pilot','study'):
        checks['cost_admission']=pilot_gate(run if mode=='pilot' else protocol['pilot_gate'])
    return protocol,training,evaluation,controls,checks


def render(run,protocol,training,evaluation,controls,checks):
    lines=['# F1 '+protocol['mode']+': repeated single-scene fitting','',f'Run `{run}`.',
        '', 'Unchanged K2 update rule and original checkpoint; independent models repeat one development scene. One training seed, weak scaffold initialization, 16 growth steps per optimizer update. The only paired coefficient change is sparsity30 versus3. Evaluations use firing seed2, separate from training.',
        '',f'{len(training)} executed optimizer updates; {len(evaluation)} evaluations. All saved objectives and evaluation metrics rescored; complete checkpoint metadata/counters checked. Verification uses the shared formulas, not an independent implementation. Final study rollouts are replayed from all four final checkpoints at both horizons.',
        '', '## Every evaluation boundary','',
        '| Member | Update | Growth steps | Connected | Mass/envelope | Coverage | Access | Sparsity | Support | Total weight30 | Total weight3 | Guide raw<0 / count |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(evaluation,key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'])):
        t=r['trace'];s=t['raw_saturation']
        lines.append(f'| {r["branch"]} | {t["update"]} | {t["steps"]} | {t["metrics"]["connectivity"]["all_connected"]} | {t["mass_ratio"]:.7g} | '+
            ' | '.join(f'{t["terms"][k]:.7g}' for k in ('coverage','access','sparsity','support'))+
            f' | {t["totals_under_both_recipes"]["mapped_30"]:.7g} | {t["totals_under_both_recipes"]["mass_3"]:.7g} | {s["guide_below_zero"]} / {s["guide_voxels"]} |')
    lines+=['','r0 is ground-pair; r1 is minimal-smoke. Connected uses material>0.5. Mass is continuous; budget3%-12%, tolerance1e-6. Connectivity and budget do not certify all nine constraints, usable architecture or mechanical safety. Raw saturation counts do not prove a blocked parameter gradient.','',
        '## Complete objective values','', '| Member | Update | Growth | '+ ' | '.join(evaluation[0]['trace']['terms'])+' | '+' | '.join(evaluation[0]['trace']['regularizers'])+' |',
        '|---|---:|---:|'+ '|'.join(['---:']*12)+'|']
    for r in evaluation:
        t=r['trace'];lines.append(f'| {r["branch"]} | {t["update"]} | {t["steps"]} | '+' | '.join(f'{v:.7g}' for v in [*t['terms'].values(),*t['regularizers'].values()])+' |')
    lines+=['','## Saved controls on the same scenes','',
        '| Method | Scene | Growth steps | Connected | Mass/envelope | Total weight30 | Total weight3 |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in controls:
        lines.append(f'| {r["branch"]} | {r["scene"]} | {r["steps"] or "static"} | {r["metrics"]["connectivity"]["all_connected"]} | {r["mass_ratio"]:.7g} | {r["totals_under_both_recipes"]["mapped_30"]:.7g} | {r["totals_under_both_recipes"]["mass_3"]:.7g} |')
    lines+=['','K2 controls used17 shared-weight updates across17 scenes. D1 used32 per-scene raw-voxel updates at learning rate0.05. F1 uses64 per-scene NCA updates at0.0001. W1 is a static procedural witness. Degrees of freedom and effort differ; these are diagnostic comparisons, not matched-cost model rankings.','',
        '## Every training update','', '| Member | Update | Objective before update | Gradient norm before clip | Seconds incl. checkpoint/fields |',
        '|---|---:|---:|---:|---:|']
    for r in training:
        t=r['trace'];lines.append(f'| {r["branch"]} | {t["update"]} | {t["total_loss"]:.7g} | {t["gradient_norm_before_clip"]:.7g} | {r["seconds"]:.3f} |')
    lines+=['','Training fields describe the forward pass BEFORE the update; named checkpoints are AFTER it. Evaluation fields are AFTER the named update. Recovery certifies completed CPU boundaries only, not CUDA or abrupt write interruption. Every failed/interrupted attempt must remain archived. No checkpoint is promoted automatically.','',
        '## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'All per-case metrics (including binary legality, blocked ground, unsupported material, thickness proxy and threshold counts) are retained in the machine-readable evaluation report and original immutable records. One seed/two scenes cannot establish generalization or architecture failure. Timing includes local overhead; it is not deployment performance.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];protocol,training,evaluation,controls,checks=verify(run)
    prefix={'recovery':'F1R','pilot':'F1P','study':'F1'}[protocol['mode']]
    out=REPO/'experiments/reports'
    with (out/(prefix+'-fitting.md')).open('x',encoding='utf-8') as f:
        f.write(render(run,protocol,training,evaluation,controls,checks))
    write_once(out/(prefix+'-verification.json'),dict(run_id=run,**checks))
    write_once(out/(prefix+'-evaluations.json'),{'run_id':run,'evaluations':evaluation,'controls':controls})
    print(json.dumps(checks,indent=2))
