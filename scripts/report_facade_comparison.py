"""Verify and report A1 matched facade evidence, without overwriting history."""
from collections import defaultdict
from pathlib import Path
import hashlib,sys,math
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
from nca.facade import endpoint_allowance
from scripts.diagnostic_inputs import load_inputs
RUN='20260923T084341Z_e37699e31f26'


def load():
    store=RunStore(REPO/'.local-artifacts/runs')
    if store.verify(RUN):raise ValueError('Artifact verification failed')
    d=store.path(RUN);assert read_json(d/'result.json')['status']=='completed';r=defaultdict(list)
    for path in sorted((d/'events').glob('*.json')):
        e=read_json(path)
        if e['kind']=='artifact' and e['details']['role'] in ('target_record','control_record','bound_record','gradient_record','summary','annotations'):
            r[e['details']['role']].append(read_json(d/e['details']['path']))
    for role,count in [('target_record',864),('control_record',144),('bound_record',144),('gradient_record',72),('annotations',1),('summary',1)]:assert len(r[role])==count
    _,inputs=load_inputs(REPO);annotations=r['annotations'][0];arrays={}
    for sid,item in inputs.items():
        ref=next(row['fields'] for row in r['target_record'] if row['scene_id']==sid)
        with np.load(d/ref['path'],allow_pickle=False) as f:arrays[sid]={k:f[k].copy() for k in f.files}
        mask,ann=endpoint_allowance(item['scene'],item['permitted'])
        ann.update(source_scene_hash=item['scene_hash'],mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest())
        assert ann==annotations[sid] and np.array_equal(mask.numpy(),arrays[sid]['allowance'])
    for role,keys in [('target_record',('scene_id','candidate','envelope_radius','budget')),('control_record',('scene_id','control'))]:
        pairs=defaultdict(dict)
        for row in r[role]:
            key=tuple(row[k] for k in keys);assert row['arm'] not in pairs[key];pairs[key][row['arm']]=row
        assert len(pairs)==len(r[role])//2
        for pair in pairs.values():
            old,new=pair['original'],pair['endpoint_allowance']
            assert all(old['terms'][k]==new['terms'][k] for k in old['terms'] if k!='facade')
            assert old['mass_voxels']==new['mass_voxels'] and old['metrics']==new['metrics']
            f=arrays[old['scene_id']]
            if role=='target_record':p=f[old['candidate']]
            else:
                blanket=f['facade']&f['permitted'];p={'empty':np.zeros_like(blanket),'allowance_only':f['allowance'],'facade_blanket':blanket,'guide_and_blanket':blanket|f['guide']}[old['control']]
            for row,charged in [(old,f['facade']),(new,f['facade']&~f['allowance'])]:
                value=max(float((p*charged).sum())/max(float(p.sum()),1)-.15,0.)
                assert math.isclose(value,row['terms']['facade'],abs_tol=1e-7)
    for row in r['bound_record']:
        f=arrays[row['scene_id']];env=f['envelope'+str(row['envelope_radius'])];charged=f['facade']
        if row['arm']=='endpoint_allowance':charged=charged&~f['allowance']
        contact=int((f['guide']&charged).sum());capacity=int((env&f['permitted']).sum())
        denominator=int(f['budget'].sum()) if row['budget']=='site' else capacity
        lower=max(.03*denominator,int(f['guide'].sum()),contact/.15);upper=min(.12*denominator,capacity)
        free=int((env&f['permitted']&~charged).sum());fraction=contact/(contact+free) if contact+free else 0.
        assert (lower<=upper+1e-8 and fraction<=.15+1e-8)==row['joint_necessary_compatible'][0]
        assert math.isclose(lower,row['joint_minimum_mass'][0],rel_tol=1e-6,abs_tol=1e-6)
    for row in r['gradient_record']:
        f=arrays[row['scene_id']]
        with np.load(d/row['gradient_fields']['path'],allow_pickle=False) as data:
            p=data['occupancy'].astype(np.float64);g=data['gradient'].astype(np.float64)
        charged=f['facade'] if row['arm']=='original' else f['facade']&~f['allowance']
        mass=p.sum();contact=(p*charged).sum();ratio=contact/max(mass,1)
        assert math.isclose(max(ratio-.15,0),row['value'],abs_tol=1e-7)
        # Independent quotient derivative: T1 probes have mass>1, away from hinge.
        assert mass>1 and abs(ratio-.15)>1e-6
        expected=charged/mass-contact/mass**2 if ratio>.15 else np.zeros_like(p)
        assert np.allclose(g,expected,rtol=1e-5,atol=1e-8)
        assert math.isclose(np.linalg.norm(g),row['full_l2'],rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(g*f['permitted']),row['legal_l2'],rel_tol=1e-10,abs_tol=1e-12)
    return r


def render(r):
    lines=['# A1 facade endpoint allowance comparison','',f'Run `{RUN}`; source `{r["summary"][0]["provenance"]["commit"]}`.',
        '', '864 matched target-arm records,144 control-arm records,144 bound-arm records and72 gradient-arm records. All hashes verified. Scene-derived allowances, raw facade values, other-eight-term equality, paired mass/metrics, joint bounds and independent analytical quotient gradients rechecked. No optimizer updates.',
        '', '## Necessary compatibility on 17 feasible scenes','',
        '| Envelope radius | Budget | Original | Endpoint allowance |','|---:|---|---:|---:|']
    for radius in (3,6):
        for budget in ('site','envelope'):
            counts=[sum(x['joint_necessary_compatible'][0] for x in r['bound_record'] if x['route_feasible'] and x['envelope_radius']==radius and x['budget']==budget and x['arm']==arm) for arm in ('original','endpoint_allowance')]
            lines.append(f'| {radius} | {budget} | {counts[0]}/17 | {counts[1]}/17 |')
    lines+=['','The sealed reference is retained and route-invalid. Passing is necessary, not proof that all losses admit an acceptable common solution. Budgets/regions/fractions did not change.','',
        '### Previously conflicting radius-six scenes','',
        '| Scene | Arm | Charged mandatory contact | Joint minimum mass | Maximum mass |','|---|---|---:|---:|---:|']
    for row in r['bound_record']:
        if row['scene_id'] in ('legacy-easy-seed-007','legacy-easy-seed-008') and row['envelope_radius']==6 and row['budget']=='envelope':
            lines.append(f'| {row["scene_id"]} | {row["arm"]} | {row["mandatory_facade_voxels"][0]:.0f} | {row["joint_minimum_mass"][0]:.4f} | {row["joint_maximum_mass"][0]:.4f} |')
    lines+=['','## Controls','', '| Control | Arm | Cases | Facade penalty range |','|---|---|---:|---:|']
    for name in ('empty','allowance_only','facade_blanket','guide_and_blanket'):
        for arm in ('original','endpoint_allowance'):
            selected=[x for x in r['control_record'] if x['control']==name and x['arm']==arm];v=[x['terms']['facade'] for x in selected]
            lines.append(f'| {name} | {arm} | {len(v)} | {min(v):.6g} - {max(v):.6g} |')
    lines+=['','Allowance-only has zero facade penalty by definition; it is not a successful architecture. Ground-only reference scenes have no facade allowance. Empty and attachment-only fields still expose failures through other objectives. Facade blankets remain penalized on all18 scenes.',
        '', '## Static candidates: zero-term witnesses at radius six / envelope budget','',
        '| Candidate | Original | Endpoint allowance |','|---|---:|---:|']
    for name in ('empty','guide','scaffold','radius1','radius3','radius6'):
        counts=[sum(x['all_terms_zero'] and x['context_valid'] for x in r['target_record'] if x['route_feasible'] and x['candidate']==name and x['envelope_radius']==6 and x['budget']=='envelope' and x['arm']==arm) for arm in ('original','endpoint_allowance')]
        lines.append(f'| {name} | {counts[0]}/17 | {counts[1]}/17 |')
    lines+=['','Simple-route zero-loss examples remain. This intervention fixes one accounting conflict; it does not create minimum thickness, require attachment or demonstrate learned value. The denominator still permits facade-ratio dilution by unrelated material.','',
        '## Exact allowance patches','', '| Scene | Allowed cells | Named patches |','|---|---:|---:|']
    for sid,ann in sorted(r['annotations'][0].items()):lines.append(f'| {sid} | {ann["allowance_voxels"]} | {len(ann["patches"])} |')
    lines+=['','Allowances come from facade-typed entrance geometry intersected with direct building-face neighbors and permitted space. No dilation or target-derived expansion. At0.8m per voxel an entrance block is1.6m per axis; each four-cell face patch is2.56m2. These are geometric patches, not engineered connection specifications.','',
        '## Probe facade gradients','', '| Scene | Budget | Arm | Value | Legal gradient norm |','|---|---|---|---:|---:|']
    for row in r['gradient_record']:lines.append(f'| {row["scene_id"]} | {row["budget"]} | {row["arm"]} | {row["value"]:.8g} | {row["legal_l2"]:.8g} |')
    lines+=['','These are fixed T1 occupancy probes, not model gradients or calibrated weights. A zero gradient can be correct when the capped ratio is below15%. All original probe values/norms reproduce T1.','',
        '## Full per-case facade comparison','', '| Scene | Candidate | Envelope | Budget | Original facade | Endpoint facade |','|---|---|---:|---|---:|---:|']
    grouped=defaultdict(dict)
    for row in r['target_record']:grouped[(row['scene_id'],row['candidate'],row['envelope_radius'],row['budget'])][row['arm']]=row['terms']['facade']
    for key,values in grouped.items():lines.append('| '+' | '.join(map(str,key))+f' | {values["original"]:.8g} | {values["endpoint_allowance"]:.8g} |')
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    records=load();path=REPO/'experiments/reports/A1-facade-comparison.md'
    with path.open('x',encoding='utf-8') as f:f.write(render(records))
    write_once(REPO/'experiments/reports/A1-verification.json',{'run_id':RUN,'artifacts_verified':True,'matched_target_pairs':432,'control_pairs':72,'bound_records_rechecked':144,'independent_gradient_checks':72,'scene_annotations_reconstructed':18})
    print(path)
