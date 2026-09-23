"""Verify F3 fields, schedule/checkpoint identity, controls and final rollouts."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_horizon_training import STORE,records,verify_parity,verify_recovery,parity_current,pilot_gate
from scripts.growth_common import score
from scripts.diagnostic_inputs import load_inputs
from nca.horizon_training import Session,next_horizon,MIXED
from nca.sensitivity import contexts
from nca.recovery import metadata_hash
from nca.experiments import read_json,write_once,digest


def normalized(t):
    """Compare the same scored quantities across H1 and F2/F3 storage layouts."""
    out={k:v for k,v in t.items() if k not in ('update','scene','steps','firing_seed')}
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
            assert all(m==dict(base,schedule_version=MIXED,horizon_schedule=[16,50]*32) for m in members.values())
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
        if is_train:
            assert t['steps']==next_horizon(m,t['update']-1)
            terms=dict(actual['terms'],access=actual['candidate_access'])
            assert t['terms']==terms and t['regularizers']==actual['regularizers']
            assert t['legacy_access']==actual['terms']['access'] and t['candidate_details']==actual['candidate_details']
            assert t['total_loss']==actual['totals_v2'][m['recipe']] and t['mass_ratio']==actual['mass_ratio']
            assert np.isfinite(t['gradient_norm_before_clip']) and t['gradient_norm_before_clip']>=0
            assert t['learning_rate']==m['optimizer']['lr']
        else:assert normalized(t)==actual
        return actual
    with torch.no_grad():
        for is_train,rows in ((True,training),(False,list(all_eval.values()))):
            for r in rows:
                assert r['checkpoint']==cursor_map[r['branch'],r['trace']['update']]['checkpoint']
                rescore(r,d,members[r['branch']],is_train)
    control_run=p['source_access_training_run'];old_d=STORE.path(control_run)
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
    growth_run=p['source_growth_run'];assert not STORE.verify(growth_run)
    assert read_json(STORE.path(growth_run)/'result.json')['status']=='completed'
    h1=[r for r in records(growth_run,'growth_case') if r['source_id'].startswith('F2-')]
    assert len(h1)==72
    with torch.no_grad():
        for r in h1:
            m=old_p['members'][r['source_id'][3:]];scene=m['scene'];ctx,allow=ctxs[scene]
            with np.load(STORE.path(growth_run)/r['fields']['path']) as f:
                state=inputs[scene]['seed'].clone();state[:,cfg['ch_structure']]=torch.from_numpy(f['material'].copy())
                assert score(state,torch.from_numpy(f['raw'].copy()),inputs[scene],ctx,allow,cfg,p['recipes'])==r['score']
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
        checkpoint_schedule_cursors_verified=checkpoint_count,F2_boundary_controls_rescored=len(controls),
        H1_F2_growth_controls_rescored=len(h1),initial_F2_fields_exact=initial,final_checkpoint_rollouts_exact=replayed,
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
    lines=['# F3 '+p['mode']+': mixed-horizon training','',f'Run `{run}`.',
        '', 'One schedule change: alternate16/50 instead of constant16 growth. Original initialization, F2 objectives, two development scenes, one training seed and two recipes. Proposed64 updates/model means2112 recurrent training steps versus1024 for F2: update counts match, compute does not.',
        '', 'All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.',
        '', '## Evaluation results','', '| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |', '|---|---|---:|---:|---:|---|---:|---|']
    for arm,rows in [('F3',list({(r['branch'],r['trace']['update'],r['trace']['steps'],r['trace']['firing_seed']):r for r in evaluation+grid}.values())),('F2',controls)]:
        for r in sorted(rows,key=lambda r:(r['branch'],r['trace']['update'],r['trace']['firing_seed'],r['trace']['steps'])):
            t=r['trace'];c=t['candidate_metrics']['all_connected'];j=c and .03-1e-6<=t['mass_ratio']<=.12+1e-6
            lines.append(f'| {arm} | {r["branch"]} | {t["update"]} | {t["steps"]} | {t["firing_seed"]} | {c} | {t["mass_ratio"]:.8g} | {j} |')
    lines+=['','Budget3%-12%, tolerance1e-6. Strict material>0.5 component connectivity. Final grids reuse eight boundary evaluations explicitly; unique counts do not count them twice. H1 historical final-grid controls and all individual losses/metrics are in the JSON report.','',
        '## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];result=verify(run);p,training,evaluation,grid,controls,h1,checks=result
    prefix={'parity':'F3B','recovery':'F3R','pilot':'F3P','study':'F3'}[p['mode']];out=REPO/'experiments/reports'
    with (out/(prefix+'-horizon-training.md')).open('x',encoding='utf-8') as f:f.write(render(run,*result))
    write_once(out/(prefix+'-verification.json'),dict(run_id=run,**checks))
    write_once(out/(prefix+'-evidence.json'),dict(run_id=run,training=training,evaluation=evaluation,final_grid=grid,F2_controls=controls,H1_F2_controls=h1))
    write_once(out/(prefix+'-summary.json'),dict(run_id=run,boundaries=aggregate(evaluation),final_grid=aggregate(grid),
        F2_boundaries=aggregate(controls),F2_final=aggregate([r for r in controls if r['trace']['update']==64])))
    print(json.dumps(checks,indent=2))
