"""Verified K1 report from immutable gradients; no automatic coefficient fitting."""
from collections import defaultdict
from pathlib import Path
import sys,math,itertools
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
RUN='20260923T092355Z_f657f2f3bdb9'


def load(run=RUN):
    store=RunStore(REPO/'.local-artifacts/runs');issues=store.verify(run)
    if issues:raise ValueError(issues)
    d=store.path(run);assert read_json(d/'result.json')['status']=='completed';r=defaultdict(list)
    for path in sorted((d/'events').glob('*.json')):
        e=read_json(path)
        if e['kind']=='artifact' and e['details']['role'] in ('model_record','probe_record','protocol','summary'):
            r[e['details']['role']].append(read_json(d/e['details']['path']))
    assert len(r['model_record'])==71 and len(r['probe_record'])==51 and len(r['protocol'])==1 and len(r['summary'])==1
    assert len({(v['scene_id'],v['seed'],v['steps']) for v in r['model_record']})==71
    assert len({(v['scene_id'],v['ratio']) for v in r['probe_record']})==51
    names=r['protocol'][0]['names']
    expected_size=sum(x['elements'] for x in r['protocol'][0]['parameter_layout'])
    for row in r['model_record']:
        with np.load(d/row['fields']['path'],allow_pickle=False) as f:
            vectors={name:f[name].astype(np.float64) for name in names}
            for name,v in vectors.items():
                assert v.size==expected_size and np.isfinite(v).all()
                assert math.isclose(np.linalg.norm(v),row['norms'][name]['parameter_l2'],rel_tol=1e-9,abs_tol=1e-12)
            for a in names:
                for b in names:
                    norm=np.linalg.norm(vectors[a])*np.linalg.norm(vectors[b]);value=row['cosines'][a][b]
                    if norm==0:assert value is None
                    else:assert math.isclose(np.dot(vectors[a],vectors[b])/norm,value,rel_tol=1e-8,abs_tol=1e-11)
            with np.load(d/row['input_fields']['path'],allow_pickle=False) as source:
                guide=source['guide'];raw=f['raw'][guide];gradient=f['coverage_raw_gradient']
                expected=np.where(guide & (f['raw']<1),-1/guide.sum(),0.)
                assert np.allclose(expected,gradient,rtol=1e-6,atol=1e-10)
                sat=row['coverage_saturation']
                assert math.isclose((raw<0).mean(),sat['below_zero_fraction'],abs_tol=1e-7)
                assert math.isclose((raw>=1).mean(),sat['at_or_above_one_fraction'],abs_tol=1e-7)
    for row in r['probe_record']:
        ratio=row['ratio'];expected=150*max(ratio-.12,0)**2+max(.03-ratio,0)
        assert math.isclose(expected,row['terms']['sparsity'],rel_tol=1e-5,abs_tol=1e-7)
        g=row['sparsity_gradient_sum']
        assert g<0 if ratio<.03 else (g>0 if ratio>.12 else g==0)
    return r


def verify_composition(records):
    import torch
    from scripts.diagnostic_inputs import load_inputs
    from nca.losses import LossSpec,context_from_scenes,material_envelope
    from nca.facade import endpoint_allowance
    from nca.objective import research_terms
    torch.set_num_threads(2)
    _,inputs=load_inputs(REPO);cfg=records['protocol'][0]['config'];spec=LossSpec()
    d=RunStore(REPO/'.local-artifacts/runs').path(records['summary'][0]['run_id'])
    contexts={sid:context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],material_envelope(item['guide'],item['permitted'],6),item['feasible']) for sid,item in inputs.items() if bool(item['feasible'][0])}
    for row in records['model_record']:
        item=inputs[row['scene_id']];ctx=contexts[row['scene_id']]
        with np.load(d/row['fields']['path'],allow_pickle=False) as f:
            material=torch.from_numpy(f['material'].copy());raw=torch.from_numpy(f['raw'].copy())
        state=item['seed'].clone();state[:,cfg['ch_structure']]=material
        allowance=endpoint_allowance(item['scene'],item['permitted'])[0]
        result=research_terms(state,raw,ctx,cfg,allowance,spec)
        assert bool(result['context_valid'][0])
        for name,value in {**result['terms'],**result['regularizers']}.items():
            assert float(value[0])==row['norms'][name]['value'],name
    return len(records['model_record'])


def render(r):
    protocol=r['protocol'][0];models=r['model_record'];probes=r['probe_record'];names=protocol['names']
    lines=['# K1 actual model-gradient calibration','',f'Run `{r["summary"][0]["run_id"]}`; source `{r["summary"][0]["provenance"]["commit"]}`.',
        '', '71 model cases and51 controlled material-budget probes. Zero optimizer updates. Registered hashes, all parameter-vector norms/cosines and pre-clamp hinge derivatives independently checked. Every one of17 feasible scenes has seeds0/1 at4/16 steps; three named scenes additionally have seed0 at50 steps. The sealed reference is explicitly excluded from this feasible-scene calibration, not reclassified.',
        '', '## Gradient scales by horizon','', '| Steps | Term | Nonzero / cases | Median value | Median parameter norm | Maximum norm |', '|---:|---|---:|---:|---:|---:|']
    for steps in (4,16,50):
        rows=[x for x in models if x['steps']==steps]
        for name in names:
            values=[x['norms'][name]['value'] for x in rows];norms=[x['norms'][name]['parameter_l2'] for x in rows]
            lines.append(f'| {steps} | {name} | {sum(v>0 for v in norms)}/{len(rows)} | {np.median(values):.7g} | {np.median(norms):.7g} | {max(norms):.7g} |')
    lines+=['','Zero gradients may indicate correct inactivity/saturation or a blocked derivative. They must be interpreted alongside raw values and state. Hard legality/ground are enforced by projection. Density and TV retain notebook definitions; both cantilever variants are diagnostic, never implicitly added together.','',
        '## Budget probes','', '| Envelope occupancy | Expected mass penalty | Measured range | Derivative direction |','|---:|---:|---:|---|']
    for ratio in (.015,.075,.20):
        rows=[x for x in probes if x['ratio']==ratio];v=[x['terms']['sparsity'] for x in rows]
        expected=150*max(ratio-.12,0)**2+max(.03-ratio,0)
        lines.append(f'| {ratio} | {expected:.6g} | {min(v):.7g} - {max(v):.7g} | {"increase mass" if ratio<.03 else "decrease mass" if ratio>.12 else "inactive"} |')
    lines+=['','All17 scenes reproduce each branch. These synthetic fields are intentionally below/in/above budget, not NCA outputs or successful designs.','',
        '## Pre-clamp coverage saturation and binary outcomes','', '| Scene | Seed | Steps | Guide raw below0 | Guide raw >=1 | Maximum raw | Mass/envelope | Binary connected |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in models:
        v=row['coverage_saturation'];lines.append(f'| {row["scene_id"]} | {row["seed"]} | {row["steps"]} | {v["below_zero_fraction"]:.3%} | {v["at_or_above_one_fraction"]:.3%} | {v["maximum_raw"]:.5f} | {row["mass_ratio"]:.6f} | {row["metrics"]["connectivity"]["all_connected"]} |')
    lines+=['','The hinge has no gradient where raw material >=1; the saved raw derivatives match that definition. Continued coverage pressure elsewhere does not imply prevention of all overshoot or correction of projected-access gradient failure.','',
        '## Pairwise opposition, common4/16-step matrix','', '| Pair | Cosine <-0.1 / defined | Median cosine |','|---|---:|---:|']
    rows=[x for x in models if x['steps'] in (4,16)]
    for a,b in itertools.combinations(names,2):
        vals=[x['cosines'][a][b] for x in rows if x['cosines'][a][b] is not None]
        if vals and any(v<-.1 for v in vals):lines.append(f'| {a} / {b} | {sum(v<-.1 for v in vals)}/{len(vals)} | {np.median(vals):.5f} |')
    lines+=['','Cosines are local parameter derivatives, not proof of global incompatibility. Inactive pairs are undefined and excluded. Longer50-step cases are not pooled into this paired17-scene comparison.','',
        '## Original checkpoint recipe provenance','', 'The saved checkpoint weights equal the notebook trainer table exactly. K1 stores the exact cell19 regularizer class sources and notebook/checkpoint hashes. Historical coefficients are evidence, not calibration of corrected objective scales.', '',
        '| Historical key | Coefficient |','|---|---:|']
    for key,value in protocol['checkpoint_weights'].items():lines.append(f'| {key} | {value} |')
    lines+=['','## All individual values and parameter norms','', '| Scene | Seed | Steps | Term | Value | Parameter norm |','|---|---:|---:|---|---:|---:|']
    for row in models:
        for name in names:lines.append(f'| {row["scene_id"]} | {row["seed"]} | {row["steps"]} | {name} | {row["norms"][name]["value"]:.8g} | {row["norms"][name]["parameter_l2"]:.8g} |')
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    records=load();verified_compositions=verify_composition(records);p=REPO/'experiments/reports/K1-calibration.md'
    with p.open('x',encoding='utf-8') as f:f.write(render(records))
    write_once(REPO/'experiments/reports/K1-verification.json',{'run_id':RUN,'artifacts_verified':True,'parameter_cases_rechecked':71,'budget_branches_rechecked':51,'coverage_derivatives_rechecked':71,'composed_objective_cases_recomputed':verified_compositions})
    print(p)
