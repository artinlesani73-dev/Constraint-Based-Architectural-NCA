"""Verify every H1 field, transition, anchor and gradient-vector statistic."""
from pathlib import Path
import hashlib,json,math,sys,zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.growth_common import load_source,score,transition,vector_summary
from scripts.run_growth_audit import pilot_gate
from scripts.run_sensitivity import STORE,records
from nca.experiments import read_json,write_once


def verify(run):
    torch.set_num_threads(2);d=STORE.path(run)
    assert not STORE.verify(run) and read_json(d/'result.json')['status']=='completed'
    p=records(run,'protocol')[0];c=p['config'];sources=p['sources']
    rows=records(run,'growth_case');new=records(run,'growth_gradient');summary=records(run,'summary')[0]
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    src=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/src) as z:
        for name,value in p['code_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==value,name
    for source_run in {s['source_run'] for s in sources.values()}|{c['access_audit_run']}:
        assert not STORE.verify(source_run)
    selected=c['pilot_sources'] if p['mode']=='pilot' else list(sources)
    seeds=c['pilot_firing_seeds'] if p['mode']=='pilot' else c['firing_seeds']
    key=lambda r:(r['source_id'],r['firing_seed'],r['steps'])
    assert len(rows)==len(selected)*len(seeds)*len(c['horizons'])
    assert {key(r) for r in rows}=={(s,k,h) for s in selected for k in seeds for h in c['horizons']}
    gradients=c['pilot_gradient_sources'] if p['mode']=='pilot' else [s for s in sources if sources[s]['arm']=='F2']
    assert len(new)==len(gradients)*2 and {(r['source_id'],r['steps']) for r in new}=={(s,h) for s in gradients for h in (16,50)}
    lookup={key(r):r for r in rows};loaded={s:load_source(source) for s,source in sources.items()}
    anchor_count=0;changes=0
    with torch.no_grad():
        for sid in selected:
            model,_,item,ctx,allow=loaded[sid];source=sources[sid]
            for seed in seeds:
                previous=None;previous_horizon=None
                for h in c['horizons']:
                    r=lookup[sid,seed,h];assert r['scene']==source['scene']
                    with np.load(d/r['fields']['path'],allow_pickle=False) as f:
                        material=f['material'].copy();raw=torch.from_numpy(f['raw'].copy())
                    state=item['seed'].clone();state[:,model.config['ch_structure']]=torch.from_numpy(material)
                    assert score(state,raw,item,ctx,allow,model.config,p['recipes'])==r['score']
                    assert r['previous_horizon']==previous_horizon
                    assert r['change']==(transition(previous,material) if previous is not None else None)
                    changes+=previous is not None
                    if seed==2 and str(h) in source['anchors']:
                        anchor=source['anchors'][str(h)]
                        with np.load(STORE.path(source['source_run'])/anchor['fields']['path'],allow_pickle=False) as f:
                            assert np.array_equal(material,f['material']) and np.array_equal(raw.numpy(),f['raw'])
                        assert r['anchor_exact'] is True;anchor_count+=1
                    else:assert r['anchor_exact'] is None
                    previous=material;previous_horizon=h
    weights=records(run,'weight_check');assert len(weights)==len(selected)
    assert {r['source_id'] for r in weights}==set(selected) and all(r['frozen_weights_unchanged'] for r in weights)
    combined=[dict(r,evidence_run=run,reused=False) for r in new]
    audit=c['access_audit_run']
    assert p['reused_gradient_records']==records(audit,'access_gradient')
    for r in p['reused_gradient_records']:
        case=r['case'];sid=('original-r'+str(c['scenes'].index(case['scene']))) if case['label']=='original' else 'F1-'+case['label']
        source=sources[sid]
        if case['checkpoint'] is not None:assert case['checkpoint']==source['checkpoint']
        combined.append(dict(r,source_id=sid,steps=case['steps'],firing_seed=2,scene=case['scene'],evidence_run=audit,reused=True))
    for r in combined:
        source=sources[r['source_id']];h=r['steps'];folder=STORE.path(r['evidence_run'])
        assert r['saved_forward_exact'] and r['frozen_weights_unchanged']
        anchor=lookup.get((r['source_id'],2,h))
        reference=anchor['fields'] if anchor else source['anchors'][str(h)]['fields']
        reference_folder=d if anchor else STORE.path(source['source_run'])
        with np.load(folder/r['fields']['path'],allow_pickle=False) as f,np.load(reference_folder/reference['path'],allow_pickle=False) as original:
            assert np.array_equal(f['material'],original['material']) and np.array_equal(f['raw'],original['raw'])
            vectors={name:f[name] for name in r['norms']};norms,cosines=vector_summary(vectors)
            elements=sum(x['elements'] for x in r['parameter_layout'])
            for name,info in r['norms'].items():
                assert vectors[name].size==elements and np.isfinite(vectors[name]).all()
                raw=f['raw_gradient_'+name];assert raw.shape==f['raw'].shape and np.isfinite(raw).all()
                assert math.isclose(info['parameter_l2'],norms[name],rel_tol=1e-10,abs_tol=1e-12)
                assert math.isclose(info['last_raw_l2'],float(np.linalg.norm(raw.astype(float))),rel_tol=1e-10,abs_tol=1e-12)
                for other,value in cosines[name].items():
                    old=r['cosines'][name][other]
                    assert old is None if value is None else math.isclose(value,old,rel_tol=1e-10,abs_tol=1e-12)
            model,_,item,ctx,allow=loaded[r['source_id']]
            state=item['seed'].clone();state[:,model.config['ch_structure']]=torch.from_numpy(f['material'])
            with torch.no_grad():t=score(state,torch.from_numpy(f['raw']),item,ctx,allow,model.config,p['recipes'])
            expected={'access_v1':t['terms']['access'],'access_v2':t['candidate_access'],'coverage':t['terms']['coverage'],
                'sparsity':t['terms']['sparsity'],'total_v1':t['totals_v1'][source['recipe']],'total_v2':t['totals_v2'][source['recipe']]}
            assert {k:v['value'] for k,v in r['norms'].items()}==expected
            assert r['candidate']==t['candidate_details']
    processes=records(run,'process_record');assert len(processes)==len(selected)+len(new)
    assert all(r['returncode']==0 and not r['timed_out'] and not r['elapsed_cap_exceeded'] and r['seconds']<=r['cap_seconds'] for r in processes)
    assert summary['seconds']<=summary['cap_seconds'] and summary['optimizer_updates']==0
    admission=pilot_gate(run if p['mode']=='pilot' else p['pilot_run'])
    checks={'source_hashes_verified':len(p['code_sha256']),'growth_fields_rescored':len(rows),'sampled_transitions_verified':changes,
        'exact_historical_anchors':anchor_count,'new_gradient_cases':len(new),'reused_gradient_cases':12,
        'parameter_vectors_verified':6*len(combined),'last_raw_vectors_verified':6*len(combined),
        'cosines_verified':36*len(combined),'gradient_forward_fields_exact':len(combined),
        'frozen_model_weight_checks':len(weights),'elapsed_caps_met':True,'optimizer_updates':0,'admission':admission,
        'scope':'Shared formulas rescore all fields; independent BFS evaluates connectivity. Backprop vectors recorded; norms/cosines recomputed, not independent differentiation.'}
    return p,rows,combined,checks


def summarize(p,rows,gradients):
    grouped=[];trajectories=[]
    for sid,source in p['sources'].items():
        selected=[r for r in rows if r['source_id']==sid]
        for h in p['config']['horizons']:
            batch=[r['score'] for r in selected if r['steps']==h]
            if not batch:continue
            joint=lambda t:t['candidate_metrics']['all_connected'] and .03-1e-6<=t['mass_ratio']<=.12+1e-6
            grouped.append({'source_id':sid,'steps':h,'cases':len(batch),
                'connected':sum(t['candidate_metrics']['all_connected'] for t in batch),
                'in_budget':sum(.03-1e-6<=t['mass_ratio']<=.12+1e-6 for t in batch),'joint':sum(joint(t) for t in batch),
                'mass_min':min(t['mass_ratio'] for t in batch),'mass_max':max(t['mass_ratio'] for t in batch)})
        for seed in sorted({r['firing_seed'] for r in selected}):
            batch=sorted([r for r in selected if r['firing_seed']==seed],key=lambda r:r['steps'])
            connected=[r['steps'] for r in batch if r['score']['candidate_metrics']['all_connected']]
            valid=[r['steps'] for r in batch if joint(r['score'])]
            trajectories.append({'source_id':sid,'firing_seed':seed,'connected_horizons':connected,'joint_horizons':valid,
                'consecutive_sampled_joint_pairs':sum(a['steps'] in valid and b['steps'] in valid for a,b in zip(batch,batch[1:])),
                'mass_change_first_to_last':batch[-1]['score']['mass_ratio']-batch[0]['score']['mass_ratio'],
                'connection_lost_between_samples':any(a['score']['candidate_metrics']['all_connected'] and not b['score']['candidate_metrics']['all_connected'] for a,b in zip(batch,batch[1:]))})
    probes=[]
    for r in gradients:
        source=p['sources'][r['source_id']];recipe=p['recipes'][source['recipe']]['family_weights']
        probes.append({'source_id':r['source_id'],'steps':r['steps'],'reused':r['reused'],'norms':r['norms'],
            'weighted_parameter_norms':{term:r['norms'][name]['parameter_l2']*recipe[term] for term,name in [('access','access_v2'),('coverage','coverage'),('sparsity','sparsity')]},
            'access_sparsity_cosine':r['cosines']['access_v2']['sparsity'],
            'coverage_sparsity_cosine':r['cosines']['coverage']['sparsity'],
            'total_sparsity_cosine':r['cosines']['total_v2']['sparsity'],
            'total_old_new_cosine':r['cosines']['total_v1']['total_v2']})
    return {'by_source_and_horizon':grouped,'trajectories':trajectories,'gradients':probes}


def render(run,p,rows,gradients,checks,summary):
    lines=['# H1 growth stability: '+p['mode'],'',f'Run `{run}`. No optimizer updates.',
        '', 'Original, F1 and F2 models on two development scenes. Six fixed growth durations and three firing seeds in the full study. Each horizon restarts the same seed, so neighboring samples share their random prefix. Pilot uses two F2 models and firing seed2 only. F1/original gradients reuse verified A2 evidence.',
        '', '## Growth and budget across firing seeds','', '| Model | Growth steps | Cases | Connected | In budget | Joint | Mass range |', '|---|---:|---:|---:|---:|---:|---|']
    for r in summary['by_source_and_horizon']:
        lines.append(f'| {r["source_id"]} | {r["steps"]} | {r["cases"]} | {r["connected"]} | {r["in_budget"]} | {r["joint"]} | {100*r["mass_min"]:.3f}%–{100*r["mass_max"]:.3f}% |')
    lines+=['','Strict material>0.5 connectivity; continuous3%-12% mass/envelope budget, tolerance1e-6. Consecutive sampled successes would not prove stability between samples or beyond64steps. No fresh holdout scenes or statistical independence claim. All metrics, losses, transitions and individual failures are in the JSON and raw evidence.','',
        '## Actual parameter-gradient tradeoffs','', '| Model | Growth | Reused | Access value | Access norm x15 | Coverage norm x25 | Sparsity weighted norm | Access/sparsity cosine | Coverage/sparsity | Total/sparsity |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'undefined (zero norm)' if x is None else f'{x:.6g}'
    for r in summary['gradients']:
        n=r['weighted_parameter_norms'];lines.append(f'| {r["source_id"]} | {r["steps"]} | {r["reused"]} | {r["norms"]["access_v2"]["value"]:.6g} | '+
            ' | '.join(fmt(x) for x in [n['access'],n['coverage'],n['sparsity'],r['access_sparsity_cosine'],r['coverage_sparsity_cosine'],r['total_sparsity_cosine']])+' |')
    lines+=['','Negative cosine means the two local gradient-descent directions conflict. Above the upper budget, sparsity is monotone in continuous mass; this does not forecast an Adam update or prove the cause of learned growth. Weights, gradients and raw-field derivatives are distinct. Zero gradients are retained.','',
        '## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'No model, stopping rule or objective is promoted from this diagnostic. No production, paid-compute or Drive action.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];p,rows,gradients,checks=verify(run);summary=summarize(p,rows,gradients)
    prefix='H1P' if p['mode']=='pilot' else 'H1';out=REPO/'experiments/reports'
    with (out/(prefix+'-growth.md')).open('x',encoding='utf-8') as f:f.write(render(run,p,rows,gradients,checks,summary))
    write_once(out/(prefix+'-verification.json'),dict(run_id=run,**checks))
    write_once(out/(prefix+'-summary.json'),dict(run_id=run,**summary))
    write_once(out/(prefix+'-evidence.json'),dict(run_id=run,growth=rows,gradients=gradients))
    print(json.dumps(checks,indent=2))
