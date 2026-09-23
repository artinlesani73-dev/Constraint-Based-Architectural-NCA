"""Verify F5 fields, schedule/checkpoint identity, controls and final rollouts."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_guide_training import STORE,records,verify_parity,verify_recovery,parity_current,pilot_gate
from scripts.growth_common import score
from scripts.diagnostic_inputs import load_inputs
from nca.guide_training import Session,next_horizon,RAW,GUIDED,objective_pair
from nca.sensitivity import contexts
from nca.losses import LossSpec
from nca.objective import weighted_total
from nca.recovery import metadata_hash
from nca.experiments import read_json,write_once,digest


def normalized(t):
    """Compare the same scored quantities across H1 and F2/F5 storage layouts."""
    out={k:v for k,v in t.items() if k not in ('update','scene','steps','firing_seed','raw_access','raw_details','raw_totals')}
    out['totals_v1']=out.pop('totals_under_both_recipes')
    out['totals_v2']=out.pop('candidate_totals')
    return out


def verify(run):
    torch.set_num_threads(2);d=STORE.path(run)
    assert not STORE.verify(run) and read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];p=protocol['proposal'];mode=protocol['mode'];members=protocol['members']
    meta=next(iter(members.values()));cfg=meta['config']
    parity=run if mode=='parity' else protocol['parity_gate'];checks=verify_parity(parity)
    if mode!='parity':
        if mode=='recovery':
            base=records(parity,'protocol')[0]['members']['mass_3-r0']
            assert all(m==dict(base,architecture_version=GUIDED) for m in members.values())
        else:assert parity_current(parity,members)
        checks.update(verify_recovery(run if mode=='recovery' else protocol['recovery_gate']))
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    src=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/src) as z:
        for name,value in meta['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==value,name
            assert digest(REPO/name)==value,'Current source differs: '+name
    _,inputs=load_inputs(REPO);ctxs=contexts(inputs,cfg,p['training_scenes'])
    for m in members.values():
        scene=m['scene'];assert inputs[scene]['scene_hash']==m['scene_hashes'][scene]
        assert inputs[scene]['source_fields']['sha256']==m['input_field_hashes'][scene]
    training=records(run,'training_update');evaluation=records(run,'evaluation_record');grid=records(run,'horizon_evaluation')
    if mode=='recovery':
        updates={'whole':[1,2,3],'prefix':[1],'resumed':[2,3],'repeat':[2,3]}
        bounds={'whole':[0,1,3],'prefix':[0,1],'resumed':[3],'repeat':[3]}
    else:
        stop={'parity':3,'pilot':2,'study':64}[mode]
        updates={b:list(range(1,stop+1)) for b in members}
        bounds={b:([0,1,3] if mode=='parity' else [0,1,2] if mode=='pilot' else p['evaluation']['boundaries']) for b in members}
    assert len(training)==sum(map(len,updates.values()))
    assert {(r['branch'],r['trace']['update']) for r in training}=={(b,u) for b,us in updates.items() for u in us}
    key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'],r['trace']['firing_seed'])
    expected={(b,u,h,2) for b,us in bounds.items() for u in us for h in (16,50)}
    assert len(evaluation)==len(expected) and {key(r) for r in evaluation}==expected
    if mode in ('pilot','study'):
        expected={(b,stop,h,s) for b in members for h in p['final_evaluation']['horizons'] for s in p['final_evaluation']['firing_seeds']}
        assert len(grid)==72 and {key(r) for r in grid}==expected
        boundary={key(r):r for r in evaluation}
        for r in grid:
            assert r['reused_boundary']==(key(r) in boundary)
            if r['reused_boundary']:assert {k:v for k,v in r.items() if k!='reused_boundary'}==boundary[key(r)]
        assert sum(r['reused_boundary'] for r in grid)==8
    else:assert not grid
    all_eval={key(r):r for r in evaluation+grid}
    cursors=records(run,'schedule_cursor');checkpoint_count=0
    expected_cursors={(b,u) for b,us in updates.items() for u in us}|{(b,0) for b,us in bounds.items() if 0 in us}
    assert len(cursors)==len(expected_cursors) and {(r['branch'],r['update']) for r in cursors}==expected_cursors
    for r in cursors:
        m=members[r['branch']];u=r['update']
        assert r['next_horizon']==next_horizon(m,u) and r['schedule_version']==m['schedule_version']
        ckpt=torch.load(d/r['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert ckpt['metadata']==m and ckpt['metadata_hash']==metadata_hash(m) and ckpt['completed_updates']==u
        assert ckpt['optimizer']['param_groups'][0]['lr']==m['optimizer']['lr']
        assert all(float(s['step'])==u for s in ckpt['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
        checkpoint_count+=1
    cursor_map={(r['branch'],r['update']):r for r in cursors}
    def rescore(r,folder,m,is_train=False):
        t=r['trace'];scene=m['scene'];ctx,allow=ctxs[scene];assert t['scene']==scene
        with np.load(folder/r['fields']['path'],allow_pickle=False) as f:
            material=torch.from_numpy(f['material'].copy());raw=torch.from_numpy(f['raw'].copy())
        state=inputs[scene]['seed'].clone();state[:,cfg['ch_structure']]=material
        actual=score(state,raw,inputs[scene],ctx,allow,cfg,p['recipes'])
        if m.get('objective_version')==RAW or 'raw_access' in t:
            _,raw_values,raw_details=objective_pair(state,raw,ctx,cfg,allow,LossSpec(),RAW)
            raw_access=float(raw_values['terms']['access'][0])
            raw_totals={name:float(weighted_total(raw_values,c['family_weights'],c['regularizer_weights'])) for name,c in p['recipes'].items()}
        if 'raw_access' in t:
            assert t['raw_access']==raw_access and t['raw_details']==raw_details[0] and t['raw_totals']==raw_totals
        if is_train:
            assert t['steps']==next_horizon(m,t['update']-1)
            is_raw=m['objective_version']==RAW
            terms=dict(actual['terms'],access=raw_access if is_raw else actual['candidate_access'])
            assert t['terms']==terms and t['regularizers']==actual['regularizers']
            assert t['legacy_access']==actual['terms']['access'] and t['candidate_details']==(raw_details[0] if is_raw else actual['candidate_details'])
            assert t['total_loss']==(raw_totals if is_raw else actual['totals_v2'])[m['recipe']] and t['mass_ratio']==actual['mass_ratio']
            assert np.isfinite(t['gradient_norm_before_clip']) and t['gradient_norm_before_clip']>=0
            assert t['learning_rate']==m['optimizer']['lr']
        else:assert normalized(t)==actual
        return actual
    with torch.no_grad():
        for is_train,rows in ((True,training),(False,list(all_eval.values()))):
            for r in rows:
                assert r['checkpoint']==cursor_map[r['branch'],r['trace']['update']]['checkpoint']
                rescore(r,d,members[r['branch']],is_train)
    control_run=p['source_raw_training_run'];old_d=STORE.path(control_run)
    assert not STORE.verify(control_run) and read_json(old_d/'result.json')['status']=='completed'
    old_p=records(control_run,'protocol')[0];controls=records(control_run,'evaluation_record')
    with torch.no_grad():
        for r in controls:rescore(r,old_d,old_p['members'][r['branch']])
    initial=0
    for r in evaluation:
        if r['trace']['update']!=0:continue
        old=next(x for x in controls if x['trace']==r['trace'])
        with np.load(d/r['fields']['path']) as a,np.load(old_d/old['fields']['path']) as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
        initial+=1
    h1=records(control_run,'horizon_evaluation');assert len(h1)==72
    with torch.no_grad():
        for r in h1:rescore(r,old_d,old_p['members'][r['branch']])
    replayed=0
    if mode=='study':
        for branch,m in members.items():
            rows=[r for r in evaluation if r['branch']==branch and r['trace']['update']==64]
            session=Session(m,d/rows[0]['checkpoint']['path'])
            assert next_horizon(session.metadata,session.completed) is None
            for r in rows:
                t,fields=session.score(r['trace']['steps'],2);assert t==r['trace']
                with np.load(d/r['fields']['path']) as f:assert all(np.array_equal(fields[k],f[k]) for k in fields)
                replayed+=1
    processes=records(run,'process_record');assert len(processes)==4
    assert all(r['returncode']==0 and not r['timed_out'] and not r['elapsed_cap_exceeded'] and r['seconds']<=r['cap_seconds'] for r in processes)
    assert records(run,'summary')[0]['seconds']<=p['phase_caps_seconds'][mode]
    checks.update(source_hashes_verified=len(meta['code_sha256']),unique_saved_fields_rescored=len(training)+len(all_eval),
        unique_evaluations=len(all_eval),boundary_evaluations=len(evaluation),final_grid_evaluations=len(grid),
        checkpoint_schedule_cursors_verified=checkpoint_count,F4_boundary_controls_rescored=len(controls),
        F4_final_controls_rescored=len(h1),initial_F4_fields_exact=initial,final_checkpoint_rollouts_exact=replayed,
        elapsed_caps_met=True)
    if mode in ('pilot','study'):checks['admission']=pilot_gate(run if mode=='pilot' else protocol['pilot_gate'])
    return protocol,training,evaluation,grid,controls,h1,checks


def aggregate(rows):
    result={}
    for h in sorted({r['trace']['steps'] for r in rows}):
        ts=[r['trace'] for r in rows if r['trace']['steps']==h]
        result[str(h)]={'cases':len(ts),'connected':sum(t['candidate_metrics']['all_connected'] for t in ts),
            'in_budget':sum(.03-1e-6<=t['mass_ratio']<=.12+1e-6 for t in ts),
            'joint':sum(t['candidate_metrics']['all_connected'] and .03-1e-6<=t['mass_ratio']<=.12+1e-6 for t in ts),
            'mass_min':min(t['mass_ratio'] for t in ts),'mass_max':max(t['mass_ratio'] for t in ts)}
    return result


def render(run,p,training,evaluation,grid,controls,h1,checks):
    lines=['# F5 '+p['mode']+': raw-access training','',f'Run `{run}`.',
        '', 'One architecture change: persistent same-scaffold input; unchanged F4 raw-access objective. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.',
        '', 'All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.',
        '', '## Evaluation results','', '| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |', '|---|---|---:|---:|---:|---|---:|---|']
    for arm,rows in [('F5',list({(r['branch'],r['trace']['update'],r['trace']['steps'],r['trace']['firing_seed']):r for r in evaluation+grid}.values())),('F4',controls)]:
        for r in sorted(rows,key=lambda r:(r['branch'],r['trace']['update'],r['trace']['firing_seed'],r['trace']['steps'])):
            t=r['trace'];c=t['candidate_metrics']['all_connected'];j=c and .03-1e-6<=t['mass_ratio']<=.12+1e-6
            lines.append(f'| {arm} | {r["branch"]} | {t["update"]} | {t["steps"]} | {t["firing_seed"]} | {c} | {t["mass_ratio"]:.8g} | {j} |')
    lines+=['','Budget3%-12%, tolerance1e-6. Strict material>0.5 component connectivity. Final grids reuse eight boundary evaluations explicitly; unique counts do not count them twice. F4 historical final-grid controls and all individual losses/metrics are in the JSON report.','',
        '## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];result=verify(run);p,training,evaluation,grid,controls,h1,checks=result
    prefix=sys.argv[2] if len(sys.argv)>2 else {'parity':'F5B','recovery':'F5R','pilot':'F5P','study':'F5'}[p['mode']]
    if not prefix.replace('-','').isalnum():raise ValueError('Invalid report prefix')
    out=REPO/'experiments/reports'
    with (out/(prefix+'-guide-training.md')).open('x',encoding='utf-8') as f:f.write(render(run,*result))
    write_once(out/(prefix+'-verification.json'),dict(run_id=run,**checks))
    write_once(out/(prefix+'-evidence.json'),dict(run_id=run,training=training,evaluation=evaluation,final_grid=grid,F4_controls=controls,F4_final_controls=h1))
    write_once(out/(prefix+'-summary.json'),dict(run_id=run,boundaries=aggregate(evaluation),final_grid=aggregate(grid),
        F4_boundaries=aggregate(controls),F4_final=aggregate([r for r in controls if r['trace']['update']==64])))
    print(json.dumps(checks,indent=2))
