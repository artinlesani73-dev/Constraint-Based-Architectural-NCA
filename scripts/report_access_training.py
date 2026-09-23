"""Recompute F2 evidence and compare both definitions on immutable F1 fields."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_access_training import STORE,records,verify_parity,verify_recovery,pilot_gate,parity_current
from scripts.diagnostic_inputs import load_inputs
from nca.access_training import Session,objective_pair,LEGACY,CANDIDATE,EXTRA_EVALUATION_KEYS
from nca.access import component_connectivity
from nca.sensitivity import contexts
from nca.losses import LossSpec
from nca.objective import weighted_total
from nca.e0 import evaluate
from nca.recovery import metadata_hash
from nca.experiments import read_json,write_once


def verify(run,replay=True):
    torch.set_num_threads(2);d=STORE.path(run)
    assert not STORE.verify(run) and read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];p=protocol['proposal'];mode=protocol['mode']
    members=protocol['members'];meta=next(iter(members.values()));cfg=meta['config']
    parity=run if mode=='parity' else protocol['parity_gate']
    checks=verify_parity(parity)
    if mode!='parity':
        current=records(parity,'protocol')[0]['members']
        assert all(m==dict(current['mass_3-r0'],objective_version=CANDIDATE) for m in members.values()) if mode=='recovery' else parity_current(parity,members)
        checks.update(verify_recovery(run if mode=='recovery' else protocol['recovery_gate']))
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    source=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/source) as z:
        for name,expected in meta['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        if 'continuation' in protocol:
            assert hashlib.sha256(z.read('scripts/continue_access_training.py')).hexdigest()==protocol['continuation']['wrapper_sha256']
    _,inputs=load_inputs(REPO);ctxs=contexts(inputs,cfg,p['training_scenes'])
    for m in members.values():
        name=m['scene'];assert inputs[name]['scene_hash']==m['scene_hashes'][name]
        assert inputs[name]['source_fields']['sha256']==m['input_field_hashes'][name]
    training=records(run,'training_update');evaluation=records(run,'evaluation_record')
    if 'continuation' in protocol:
        parent=protocol['continuation']['parent_run']
        assert not STORE.verify(parent) and read_json(STORE.path(parent)/'result.json')['status']=='interrupted'
        previous=records(parent,'protocol')[0]
        assert previous=={k:v for k,v in protocol.items() if k!='continuation'}
        for role,rows in [('training_update',training),('evaluation_record',evaluation)]:
            key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'])
            current={key(r):r for r in rows}
            inherited=records(parent,role)
            for old in inherited:
                row=current[key(old)]
                assert {k:v for k,v in row.items() if k not in ('fields','checkpoint')}=={k:v for k,v in old.items() if k not in ('fields','checkpoint')}
                assert all(row[k]['sha256']==old[k]['sha256'] for k in ('fields','checkpoint'))
            checks['imported_'+role+'_verified']=len(inherited)
        checks['continuation_parent']=parent
        checks['newly_executed_training_updates']=len(training)-len(records(parent,'training_update'))
    if mode=='recovery':
        schedule={'whole':[1,2,3],'prefix':[1],'resumed':[2,3],'repeat':[2,3]}
        bounds={'whole':[0,1,3],'prefix':[0,1],'resumed':[3],'repeat':[3]}
    else:
        stop={'parity':3,'pilot':2,'study':64}[mode]
        schedule={b:list(range(1,stop+1)) for b in members}
        bounds={b:([0,1,3] if mode=='parity' else [0,1,2] if mode=='pilot' else p['evaluation']['boundaries']) for b in members}
    expected={(b,u) for b,updates in schedule.items() for u in updates}
    assert len(training)==len(expected) and {(r['branch'],r['trace']['update']) for r in training}==expected
    expected_eval={(b,u,h) for b,updates in bounds.items() for u in updates for h in (16,50)}
    assert len(evaluation)==len(expected_eval) and {(r['branch'],r['trace']['update'],r['trace']['steps']) for r in evaluation}==expected_eval
    control_run=p['source_fitting_run'];control_d=STORE.path(control_run)
    assert not STORE.verify(control_run) and read_json(control_d/'result.json')['status']=='completed'
    f1=records(control_run,'protocol')[0]
    controls=[]

    def rescore(row,folder,m,training_row=False):
        t=row['trace'];name=m['scene'];ctx,allow=ctxs[name];item=inputs[name]
        assert t['scene']==name
        with np.load(folder/row['fields']['path'],allow_pickle=False) as f:
            raw=torch.from_numpy(f['raw'].copy());material=torch.from_numpy(f['material'].copy())
        assert torch.isfinite(raw).all() and torch.isfinite(material).all()
        assert torch.equal(raw.clamp(0,1)*ctx.permitted,material)
        state=item['seed'].clone();state[:,cfg['ch_structure']]=material
        old,new,details=objective_pair(state,raw,ctx,cfg,allow,LossSpec(**m['loss_spec']))
        v=new if training_row and m.get('objective_version')==CANDIDATE else old
        assert bool(v['context_valid'][0])
        assert {k:float(x[0]) for k,x in v['terms'].items()}==t['terms']
        assert {k:float(x[0]) for k,x in v['regularizers'].items()}==t['regularizers']
        assert float(v['mass_ratio'][0])==t['mass_ratio']
        extra={'candidate_access':float(new['terms']['access'][0]),'candidate_details':details[0],
            'candidate_metrics':component_connectivity(material[0].numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0]),
            'candidate_totals':{r:float(weighted_total(new,c['family_weights'],c['regularizer_weights'])) for r,c in p['recipes'].items()}}
        if training_row:
            c=m['coefficients'];assert float(weighted_total(v,c['family_weights'],c['regularizer_weights']))==t['total_loss']
            assert t['steps']==16 and np.isfinite(t['gradient_norm_before_clip'])
            assert t['learning_rate']==m['optimizer']['lr']
            if m['objective_version']==CANDIDATE:
                assert t['legacy_access']==float(old['terms']['access'][0]) and t['candidate_details']==details[0]
        else:
            assert t['firing_seed']==2 and evaluate(state,cfg,item['scene'])==t['metrics']
            for r,c in p['recipes'].items():
                assert float(weighted_total(old,c['family_weights'],c['regularizer_weights']))==t['totals_under_both_recipes'][r]
            assert t['raw_saturation']=={'permitted_below_zero':int((raw[ctx.permitted]<0).sum()),
                'permitted_above_one':int((raw[ctx.permitted]>1).sum()),
                'guide_below_zero':int((raw[ctx.coverage]<0).sum()),'guide_voxels':int(ctx.coverage.sum())}
            if m.get('objective_version'):
                assert all(t[k]==extra[k] for k in EXTRA_EVALUATION_KEYS)
        return extra

    with torch.no_grad():
        for is_train,rows in ((True,training),(False,evaluation)):
            for row in rows:
                m=members[row['branch']];u=row['trace']['update']
                ckpt=torch.load(d/row['checkpoint']['path'],map_location='cpu',weights_only=True)
                assert ckpt['metadata']==m and ckpt['metadata_hash']==metadata_hash(m)
                assert ckpt['completed_updates']==u
                assert ckpt['optimizer']['param_groups'][0]['lr']==m['optimizer']['lr']
                assert all(float(s['step'])==u for s in ckpt['optimizer']['state'].values())
                assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
                rescore(row,d,m,is_train)
        for row in records(control_run,'evaluation_record'):
            extra=rescore(row,control_d,f1['members'][row['branch']])
            controls.append(dict(row,trace=dict(row['trace'],**extra),source_run=control_run))
    initial_matches=0
    for row in evaluation:
        t=row['trace']
        if t['update']!=0:continue
        original=next(c for c in controls if c['trace']['update']==0 and c['trace']['scene']==t['scene'] and c['trace']['steps']==t['steps'])
        assert original['trace']==t
        with np.load(d/row['fields']['path'],allow_pickle=False) as a,np.load(control_d/original['fields']['path'],allow_pickle=False) as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
        initial_matches+=1
    replayed=0
    if replay and mode=='study':
        for branch,m in members.items():
            rows=[r for r in evaluation if r['branch']==branch and r['trace']['update']==64]
            session=Session(m,d/rows[0]['checkpoint']['path'])
            for row in rows:
                trace,fields=session.score(row['trace']['steps'],2);assert trace==row['trace']
                with np.load(d/row['fields']['path'],allow_pickle=False) as f:
                    assert all(np.array_equal(fields[k],f[k]) for k in fields)
                replayed+=1
    processes=records(run,'process_record')
    assert len(processes)==4 and all(r['returncode']==0 and not r['timed_out'] for r in processes)
    summary=records(run,'summary')[0]
    overruns=[{'member':r['label'],'seconds':r['seconds'],'cap_seconds':r['cap_seconds']}
        for r in processes if r['seconds']>r['cap_seconds']]
    total_ok=mode!='study' or summary['seconds']<=p['study_cap_seconds']
    checks.update(source_hashes_verified=len(meta['code_sha256']),saved_fields_rescored=len(training)+len(evaluation),
        training_checkpoints_verified=len(training),evaluation_metrics_verified=len(evaluation),
        final_checkpoint_rollouts_replayed=replayed,F1_controls_rescored_both_definitions=len(controls),
        initial_fields_exactly_match_F1=initial_matches,successful_workers=len(processes),
        worker_elapsed_cap_overruns=overruns,overall_elapsed_cap_met=total_ok,
        elapsed_caps_compliant=not overruns and total_ok)
    if mode in ('pilot','study'):checks['cost_admission']=pilot_gate(run if mode=='pilot' else protocol['pilot_gate'])
    return protocol,training,evaluation,controls,checks


def aggregate(rows):
    out={}
    for h in (16,50):
        selected=[r['trace'] for r in rows if r['trace']['steps']==h]
        out[str(h)]={'evaluations':len(selected),
            'old_connected':sum(t['metrics']['connectivity']['all_connected'] for t in selected),
            'candidate_connected':sum(t['candidate_metrics']['all_connected'] for t in selected),
            'in_budget':sum(.03-1e-6<=t['mass_ratio']<=.12+1e-6 for t in selected),
            'joint_candidate':sum(t['candidate_metrics']['all_connected'] and .03-1e-6<=t['mass_ratio']<=.12+1e-6 for t in selected)}
    return out


def render(run,protocol,training,evaluation,controls,checks):
    lines=['# F2 '+protocol['mode']+': access-only training','',f'Run `{run}`.',
        '', 'Only the access family changes. Same original checkpoint, two development scenes, one training seed, optimizer, 16-step rollout and coefficients as F1. Both access definitions and both common recipe totals are retained. F1 controls are reused only after exact short baseline parity; this is not a new 64-update old-objective run.',
        '',f'{len(training)} recorded optimizer updates; {len(evaluation)} evaluations. Imported versus newly executed updates are distinguished in verification for linked continuations. All saved fields rescored and checkpoint metadata/counters checked. Final study checkpoints are replayed at both growth horizons. Formula rescoring shares the implementation; binary component connectivity uses independent BFS.',
        '', '## Every evaluation and paired historical control','',
        '| Arm | Member | Update | Growth | Old connected | Component connected | Mass/envelope | Joint component/budget | Old access | New access | Coverage | Sparsity | Old total30 | New total30 | Old total3 | New total3 |',
        '|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,rows in [('F2',evaluation),('F1',controls)]:
        for row in sorted(rows,key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'])):
            t=row['trace'];connected=t['candidate_metrics']['all_connected'];joint=connected and .03-1e-6<=t['mass_ratio']<=.12+1e-6
            lines.append(f'| {arm} | {row["branch"]} | {t["update"]} | {t["steps"]} | {t["metrics"]["connectivity"]["all_connected"]} | {connected} | {t["mass_ratio"]:.8g} | {joint} | '+
                ' | '.join(f'{v:.8g}' for v in [t['terms']['access'],t['candidate_access'],t['terms']['coverage'],t['terms']['sparsity'],t['totals_under_both_recipes']['mapped_30'],t['candidate_totals']['mapped_30'],t['totals_under_both_recipes']['mass_3'],t['candidate_totals']['mass_3']])+' |')
    lines+=['','r0 is ground-pair; r1 is minimal-smoke. Strict material>0.5 connectivity; continuous mass budget3%-12%, tolerance1e-6. Both can pass without satisfying all graded terms or usable/structurally safe architecture. All nine terms, three regularizers, binary metrics and candidate critical voxels are preserved in the machine-readable report.','',
        '## Every training update','', '| Member | Update | Loss before update | Gradient norm before clip | Access used | Seconds incl. evidence |', '|---|---:|---:|---:|---:|---:|']
    for row in training:
        t=row['trace'];lines.append(f'| {row["branch"]} | {t["update"]} | {t["total_loss"]:.8g} | {t["gradient_norm_before_clip"]:.8g} | {t["terms"]["access"]:.8g} | {row["seconds"]:.3f} |')
    lines+=['','Training fields precede the update; checkpoints follow it. Evaluation follows its named update. CPU recovery covers completed update boundaries, not GPU/AMP or abrupt writes. Three early recovery steps may not exercise a nonzero access parameter gradient; do not overstate that gate.','',
        ('Linked completion of interrupted parent '+protocol['continuation']['parent_run']+'. The parent remains interrupted; copied records are hash-verified and only missing planned updates are executed. Cumulative elapsed includes the parent; no clean timing claim.' if 'continuation' in protocol else 'No imported training records in this run.'), '',
        'Timing compliance is checked separately from process completion. A completed worker can exceed its elapsed cap if its timeout does not fire during a system pause. Any overrun below is a protocol deviation; do not relabel elapsed time as active compute or claim a clean timing benchmark.', '',
        '## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'One seed/two development scenes cannot establish generalization. Semantic rescoring alone is not geometry improvement. No production model or coefficient is promoted by this diagnostic.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];result=verify(run);protocol,training,evaluation,controls,checks=result
    prefix={'parity':'F2B','recovery':'F2R','pilot':'F2P','study':'F2'}[protocol['mode']];out=REPO/'experiments/reports'
    with (out/(prefix+'-access-training.md')).open('x',encoding='utf-8') as f:f.write(render(run,*result))
    write_once(out/(prefix+'-verification.json'),dict(run_id=run,**checks))
    write_once(out/(prefix+'-evaluations.json'),dict(run_id=run,evaluations=evaluation,controls=controls))
    write_once(out/(prefix+'-summary.json'),dict(run_id=run,candidate=aggregate(evaluation),F1=aggregate(controls),
        candidate_final=aggregate([r for r in evaluation if r['trace']['update']==64]),
        F1_final=aggregate([r for r in controls if r['trace']['update']==64])))
    print(json.dumps(checks,indent=2))
