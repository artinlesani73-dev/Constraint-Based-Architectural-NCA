"""Verify A3 saved evidence, all fields, vector statistics and source snapshots."""
from pathlib import Path
import hashlib,math,sys,zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_raw_access_audit import selected_indices,field_score,cost_gate
from scripts.run_sensitivity import STORE,records
from scripts.growth_common import load_source,vector_summary
from scripts.diagnostic_inputs import load_inputs
from deploy.checkpoints import load_model_c
from nca.experiments import read_json,write_once
from nca.sensitivity import contexts
from nca.access_training import objective_pair
from nca.raw_access import raw_component_access
from nca.objective import weighted_total
from nca.losses import LossSpec


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif a is None:assert b is None
    else:assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11),(a,b)


def verify(run):
    torch.set_num_threads(2);d=STORE.path(run)
    assert not STORE.verify(run) and read_json(d/'result.json')['status']=='completed'
    p=records(run,'protocol')[0];c=p['config'];summary=records(run,'summary')[0]
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    source=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/source) as z:
        for n,v in p['code_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==v,n
    for key in ('F3_run','H1_run','W1_run','D1_run'):assert not STORE.verify(c[key])
    cfg,_,_=load_model_c();_,inputs=load_inputs(REPO)
    ctxs=contexts(inputs,cfg,sorted({s['scene'] for s in p['fields']}))
    rows=records(run,'raw_replay');indices=selected_indices(p)
    assert len(rows)==len(indices) and {r['index'] for r in rows}==set(indices)
    with torch.no_grad():
        for r in rows:
            s=p['fields'][r['index']];assert s==r['source']
            assert r['scene_hash']==inputs[s['scene']]['scene_hash']
            assert field_score(s,ctxs[s['scene']][0])==r['score']
    gradients=records(run,'raw_gradient')
    selected=[c['pilot_branch']] if p['mode']=='pilot' else list(p['sources'])
    assert len(gradients)==len(selected)*2
    assert {(r['source_id'],r['steps']) for r in gradients}=={(s,h) for s in selected for h in c['gradient_horizons']}
    for r in gradients:
        s=p['sources'][r['source_id']];model,_,item,ctx,allow=load_source(s)
        recipe=p['recipes'][s['recipe']];fw,rw=recipe['family_weights'],recipe['regularizer_weights']
        anchor=s['anchors'][str(r['steps'])]
        assert r['saved_forward_exact'] and r['frozen_weights_unchanged'] and r['optimizer_updates']==0
        with np.load(d/r['fields']['path'],allow_pickle=False) as f,np.load(STORE.path(s['source_run'])/anchor['fields']['path'],allow_pickle=False) as a:
            assert np.array_equal(f['raw'],a['raw']) and np.array_equal(f['material'],a['material'])
            vectors={k:f[k] for k in r['norms']};norms,cosines=vector_summary(vectors)
            close(cosines,r['cosines'])
            weighted={k:fw[k.split('_')[0]]*vectors[k] for k in ('access_v2','access_v3','coverage','sparsity')}
            weighted['other_terms']=vectors['total_v3']-weighted['access_v3'];wn,wc=vector_summary(weighted)
            close(wn,r['weighted_norms']);close(wc,r['weighted_cosines'])
            size=sum(v['elements'] for v in r['parameter_layout'])
            for k,info in r['norms'].items():
                assert vectors[k].size==size and np.isfinite(vectors[k]).all()
                close(norms[k],info['parameter_l2']);close(float(np.linalg.norm(f['raw_gradient_'+k].astype(float))),info['last_raw_l2'])
                if k.startswith('access_'):
                    trace=f['trace_raw_gradient_'+k];assert trace.shape==(r['steps'],*f['raw'].shape)
                    assert np.array_equal(trace[-1],f['raw_gradient_'+k])
                    for v,n in zip(trace,r['traces'][k]['raw_l2']):close(float(np.linalg.norm(v.astype(float))),n)
            assert np.array_equal(f['trajectory_raw'][-1],f['raw'])
            assert f['trajectory_fired'].dtype==bool
            for key in ('v2_critical','v3_critical'):
                t=r['traces'].get(key)
                if t:
                    cell=tuple(t['zyx'])
                    assert list(f['trajectory_raw'][(slice(None),0,*cell)].astype(float))==t['raw']
                    assert list(f['trajectory_fired'][(slice(None),0,0,*cell)])==t['fired']
            with torch.no_grad():
                state=item['seed'].clone();state[:,cfg['ch_structure']]=torch.from_numpy(f['material'])
                raw=torch.from_numpy(f['raw']);_,v2,d2=objective_pair(state,raw,ctx,cfg,allow,LossSpec())
                a3,d3=raw_component_access(raw,ctx.permitted,ctx.endpoints)
                expected={'access_v2':float(v2['terms']['access'][0]),'access_v3':float(a3[0]),
                    'coverage':float(v2['terms']['coverage'][0]),'sparsity':float(v2['terms']['sparsity'][0]),
                    'total_v2':float(weighted_total(v2,fw,rw)),
                    'total_v3':float(weighted_total({**v2,'terms':{**v2['terms'],'access':a3}},fw,rw))}
                assert expected=={k:v['value'] for k,v in r['norms'].items()}
                assert r['projected_details']==d2[0] and r['raw_details']==d3[0]
                assert len(r['probes'])==(2 if norms['access_v3'] else 0)
                for probe in r['probes']:
                    sign=str(probe['sign']);raw=torch.from_numpy(f['probe_raw_'+sign]);state[:,cfg['ch_structure']]=torch.from_numpy(f['probe_material_'+sign])
                    assert torch.equal(raw.clamp(0,1)*ctx.permitted,state[:,cfg['ch_structure']])
                    _,v,_=objective_pair(state,raw,ctx,cfg,allow,LossSpec());a3,_=raw_component_access(raw,ctx.permitted,ctx.endpoints)
                    assert float(a3[0])==probe['access_v3'] and float(v['terms']['access'][0])==probe['access_v2']
                    assert float(v['mass_ratio'][0])==probe['mass_ratio']
                    assert float(weighted_total({**v,'terms':{**v['terms'],'access':a3}},fw,rw))==probe['total_v3']
                    assert math.isclose(probe['actual_parameter_l2'],c['parameter_probe_l2'],rel_tol=.01)
    processes=records(run,'process_record');assert len(processes)==1+len(gradients)
    assert all(r['returncode']==0 and not r['timed_out'] and not r['elapsed_cap_exceeded'] and r['seconds']<=r['cap_seconds'] for r in processes)
    assert summary['seconds']<=summary['cap_seconds'] and summary['optimizer_updates']==0
    checks={'source_hashes':len(p['code_sha256']),'fields_rescored':len(rows),'gradient_fields_exact':len(gradients),
        'parameter_vectors_verified':6*len(gradients),'raw_gradient_vectors_verified':6*len(gradients),
        'bounded_probes_rescored':sum(len(r['probes']) for r in gradients),'caps_met':True,'optimizer_updates':0,
        'admission':cost_gate(run if p['mode']=='pilot' else p['pilot_run']),
        'limit':'Recomputes stored statistics and objective values; recorded backprop/trace delta norms are not independent differentiation.'}
    return p,rows,gradients,checks


def main():
    run=sys.argv[1];p,rows,gradients,checks=verify(run);prefix='A3P' if p['mode']=='pilot' else 'A3'
    report=REPO/'experiments/reports'
    write_once(report/(prefix+'-'+run+'-verification.json'),dict(run_id=run,**checks))
    write_once(report/(prefix+'-'+run+'-evidence.json'),{'run_id':run,'replay':rows,'gradients':gradients})
    print(checks)


if __name__=='__main__':main()
