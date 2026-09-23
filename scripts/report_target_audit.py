"""Reconstruct T1 findings from verified registered evidence; exclusive publication."""
from pathlib import Path
from collections import defaultdict
import sys,math,itertools
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
from nca.losses import FAMILIES
RUN='20260923T082527Z_845d2aa6aec0'

def load():
    store=RunStore(REPO/'.local-artifacts/runs');problems=store.verify(RUN)
    if problems:raise ValueError(problems)
    directory=store.path(RUN)
    assert read_json(directory/'result.json')['status']=='completed'
    records=defaultdict(list)
    for path in sorted((directory/'events').glob('*.json')):
        event=read_json(path)
        if event['kind']=='artifact' and event['details']['role'] in ('target_record','bound_record','gradient_record','protocol','summary'):
            records[event['details']['role']].append(read_json(directory/event['details']['path']))
    for role,count in [('target_record',432),('bound_record',72),('gradient_record',36),('protocol',1),('summary',1)]:assert len(records[role])==count
    for role,keys in [('target_record',('scene_id','candidate','envelope_radius','budget')),('bound_record',('scene_id','envelope_radius','budget')),('gradient_record',('scene_id','budget'))]:
        assert len({tuple(r[k] for k in keys) for r in records[role]})==len(records[role])
    for r in records['bound_record']:
        with np.load(directory/r['input_fields']['path'],allow_pickle=False) as f:
            guide=f['guide'];env=f['envelope'+str(r['envelope_radius'])];facade=f['facade'];legal=f['permitted']
            contact=int((guide&facade).sum());capacity=int((env&legal).sum())
            denominator=int(f['budget'].sum()) if r['budget']=='site' else capacity
            lower=max(.03*denominator,int(guide.sum()),contact/.15);upper=min(.12*denominator,capacity)
            assert math.isclose(lower,r['joint_minimum_mass'][0],rel_tol=1e-6,abs_tol=1e-6)
            assert math.isclose(upper,r['joint_maximum_mass'][0],rel_tol=1e-6,abs_tol=1e-6)
            nonfacade=int((env&legal&~facade).sum());fraction=contact/(contact+nonfacade) if contact+nonfacade else 0.
            assert ((lower<=upper+1e-8) and fraction<=.15+1e-8)==r['joint_necessary_compatible'][0]
    for r in records['target_record']:
        with np.load(directory/r['input_fields']['path'],allow_pickle=False) as f:
            mask=f[r['candidate']];assert int(mask.sum())==r['mass_voxels']
            assert int((mask&f['facade']).sum())==r['facade_voxels']
            assert math.isclose(r['mass_voxels']*r['voxel_size_m']**3,r['volume_m3'])
    for r in records['gradient_record']:
        with np.load(directory/r['input_fields']['path'],allow_pickle=False) as inp,np.load(directory/r['fields']['path'],allow_pickle=False) as f:
            vectors={}
            for name in FAMILIES:
                grad=f[name].astype(np.float64);legal=grad*inp['permitted'];vectors[name]=legal.flatten()
                assert np.isfinite(grad).all()
                for actual,key in [(np.linalg.norm(grad),'full_l2'),(np.linalg.norm(legal),'legal_l2')]:
                    assert math.isclose(actual,r['gradient_norms'][name][key],rel_tol=1e-9,abs_tol=1e-12)
            for a in FAMILIES:
                for b in FAMILIES:
                    denom=np.linalg.norm(vectors[a])*np.linalg.norm(vectors[b]);value=r['cosines'][a][b]
                    if denom==0:assert value is None
                    else:assert math.isclose(float(np.dot(vectors[a],vectors[b])/denom),value,rel_tol=1e-8,abs_tol=1e-11)
    return records


def render(records):
    targets=records['target_record'];bounds=records['bound_record'];gradients=records['gradient_record']
    lines=['# T1 target compatibility audit','',f'Run `{RUN}`; source `{records["summary"][0]["provenance"]["commit"]}`.',
           '', '432 static target records, 72 necessary-bound records, 36 direct occupancy-gradient cases. All registered hashes verified; bound arithmetic, target mass/contact/physical volume, saved gradient norms and pairwise cosines independently rechecked. No optimizer updates.',
           '', '## Joint compatibility','',
           'Passing bounds is necessary, not sufficient. The sealed scene is excluded from feasible counts and retained in raw records. A failure means these exact losses cannot all be zero, not that architecture in the scene is impossible.',
           '', '| Envelope radius | Budget | Previous bounds + feasible route | Including facade bound + feasible route |',
           '|---:|---|---:|---:|']
    for radius in (3,6):
        for budget in ('site','envelope'):
            selected=[r for r in bounds if r['envelope_radius']==radius and r['budget']==budget and r['route_feasible']]
            lines.append(f'| {radius} | {budget} | {sum(r["budget_compatible"][0] for r in selected)}/17 | {sum(r["joint_necessary_compatible"][0] for r in selected)}/17 |')
    lines+=['','Coverage fixes guide material to one. If C of its cells touch facade, the 15% contact cap requires total mass at least C/0.15. Coverage, zero spill and the upper material budget can contradict that requirement.','',
            '| Scene | Envelope | Mandatory facade cells | Required minimum mass | Allowed maximum mass |',
            '|---|---:|---:|---:|---:|']
    for r in bounds:
        if r['route_feasible'] and r['budget_compatible'][0] and not r['joint_necessary_compatible'][0]:
            lines.append(f'| {r["scene_id"]} | {r["envelope_radius"]} | {r["mandatory_facade_voxels"][0]:.0f} | {r["joint_minimum_mass"][0]:.4f} | {r["joint_maximum_mass"][0]:.4f} |')
    lines+=['','## Candidate outcomes: radius-six envelope budget','',
            'Counts below use all 17 geometrically feasible scenes, including those with conflicting objective bounds. Empty is a negative control. Connected/supported are spatial proxies, not walkability or mechanical safety.', '',
            '| Candidate | Binary connected | All nine near zero | Nonzero thickness | Nonzero sparsity | Nonzero facade |',
            '|---|---:|---:|---:|---:|---:|']
    for name in records['protocol'][0]['candidates']:
        rows=[r for r in targets if r['route_feasible'] and r['candidate']==name and r['envelope_radius']==6 and r['budget']=='envelope']
        lines.append(f'| {name} | {sum(r["metrics"]["connectivity"]["all_connected"] is True for r in rows)}/17 | {sum(r["all_terms_zero"] for r in rows)}/17 | {sum(r["terms"]["thickness"]>1e-7 for r in rows)}/17 | {sum(r["terms"]["sparsity"]>1e-7 for r in rows)}/17 | {sum(r["terms"]["facade"]>1e-7 for r in rows)}/17 |')
    lines+=['','### Zero-loss witnesses across the full matrix','',
            'Near-zero means every term <=1e-7 with nonempty material and a valid context. Passing these proxies is not architectural quality. A one-voxel-wide guide can pass; thickness minimizes bulk, not minimum width, and access measures material rather than circulation void.', '',
            '| Scene | Candidate | Envelope | Budget | Material voxels | Volume (m3) |',
            '|---|---|---:|---|---:|---:|']
    for r in targets:
        if r['all_terms_zero'] and r['context_valid']:
            lines.append(f'| {r["scene_id"]} | {r["candidate"]} | {r["envelope_radius"]} | {r["budget"]} | {r["mass_voxels"]} | {r["volume_m3"]:.3f} |')
    lines+=['','## Direct occupancy gradient scales','',
            'Only the 17 route-feasible scenes appear below; both budget choices remain separate. The synthetic state is high on guide and low elsewhere inside radius six. These are raw, unit-weight direct occupancy derivatives, not model parameter gradients or proposed weights. Legal-coordinate masking removes forbidden coordinates only: it is not the full feasible tangent cone at occupancy 0/1. Max/min ties have implementation-selected subgradients.','',
            '| Budget | Family | Median value | Median legal gradient norm | Range of legal norms |',
            '|---|---|---:|---:|---|']
    for budget in ('site','envelope'):
        rows=[r for r in gradients if r['budget']==budget and r['route_feasible']]
        for name in FAMILIES:
            values=[r['gradient_norms'][name]['value'] for r in rows];norms=[r['gradient_norms'][name]['legal_l2'] for r in rows]
            lines.append(f'| {budget} | {name} | {np.median(values):.6g} | {np.median(norms):.6g} | {min(norms):.6g} - {max(norms):.6g} |')
    lines+=['','### Pairwise opposition on legal coordinates','',
            'Negative cosine means locally opposed direct gradients in this probe. Counts use cosine <-0.01; null values mean a zero norm and are excluded. Alignment is state dependent and is not proof of global incompatibility.','',
            '| Budget | Pair | Opposed / defined | Median cosine |', '|---|---|---:|---:|']
    for budget in ('site','envelope'):
        rows=[r for r in gradients if r['budget']==budget and r['route_feasible']]
        for a,b in itertools.combinations(FAMILIES,2):
            vals=[r['cosines'][a][b] for r in rows if r['cosines'][a][b] is not None]
            count=sum(v<-.01 for v in vals)
            if count:lines.append(f'| {budget} | {a} / {b} | {count}/{len(vals)} | {np.median(vals):.5f} |')
    lines+=['','## All static cases','',
            'All nine raw terms retained below. Near-zero diagnostics do not override invalid scene/context labels. Full fields, binary metrics and all gradient pairs are in the immutable run archive.','',
            '| Scene | Candidate | Envelope | Budget | Valid context | Voxels | '+ ' | '.join(FAMILIES)+' |',
            '|---|---|---:|---|---|---:|'+ '|'.join(['---:']*9)+'|']
    for r in targets:
        lines.append(f'| {r["scene_id"]} | {r["candidate"]} | {r["envelope_radius"]} | {r["budget"]} | {r["context_valid"]} | {r["mass_voxels"]} | '+' | '.join(f'{r["terms"][n]:.6g}' for n in FAMILIES)+' |')
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    records=load();body=render(records)
    path=REPO/'experiments/reports/T1-target-audit.md';path.parent.mkdir(exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(body)
    write_once(REPO/'experiments/reports/T1-verification.json',{'run_id':RUN,'artifacts_verified':True,'bound_records_rechecked':72,'target_records_rechecked':432,'gradient_records_rechecked':36,'report':path.name})
    print(path)
