"""Independent W1 replay and loss verification; exclusive report publication."""
from pathlib import Path
from collections import defaultdict
import sys,math
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
from nca.losses import LossSpec,context_from_scenes,material_envelope,loss_terms
from nca.facade import endpoint_allowance,facade_term
from nca.e0 import evaluate
from scripts.diagnostic_inputs import load_inputs
from scripts.run_target_audit import with_budget
RUN='20260923T084933Z_eb2603cd79f7'


def load():
    torch.set_num_threads(2);store=RunStore(REPO/'.local-artifacts/runs')
    assert not store.verify(RUN);d=store.path(RUN);assert read_json(d/'result.json')['status']=='completed'
    rows=[]
    for path in sorted((d/'events').glob('*.json')):
        event=read_json(path)
        if event['kind']=='artifact' and event['details']['role']=='witness_record':rows.append(read_json(d/event['details']['path']))
    assert len(rows)==18 and len({r['scene_id'] for r in rows})==18
    c1,inputs=load_inputs(REPO);cfg=c1['effective_model_config'];spec=LossSpec()
    for row in rows:
        item=inputs[row['scene_id']]
        with np.load(d/row['fields']['path'],allow_pickle=False) as f:
            guide=f['guide'];material=f['material'];envelope=f['envelope'];allowance=f['allowance']
        assert np.array_equal(guide,item['guide'].numpy())
        reconstructed=guide.copy();charged=None
        ctx=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],material_envelope(item['guide'],item['permitted'],6),item['feasible'])
        allowed,annotation=endpoint_allowance(item['scene'],item['permitted'])
        assert np.array_equal(allowance,allowed.numpy()) and annotation==row['annotation']
        charged=ctx.facade.numpy()&~allowance
        for z,y,x in row['added_cells']:
            assert not reconstructed[0,z,y,x] and envelope[0,z,y,x] and item['permitted'][0,z,y,x] and not charged[0,z,y,x]
            neighbors=[(z+dz,y+dy,x+dx) for dz,dy,dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
            assert any(all(0<=v<32 for v in xyz) and reconstructed[(0,)+xyz] for xyz in neighbors)
            reconstructed[0,z,y,x]=True
        assert np.array_equal(reconstructed,material)
        p=torch.from_numpy(material.copy()).float();terms=with_budget(loss_terms(p,ctx,spec),p,ctx,'envelope',spec)
        terms['facade']=facade_term(p,ctx,allowed,spec)
        assert {k:float(v[0]) for k,v in terms.items()}==row['terms']
        state=item['seed'].clone();state[:,cfg['ch_structure']]=p;assert evaluate(state,cfg,item['scene'])==row['metrics']
        if row['witness']:
            assert row['route_feasible'] and all(abs(float(v[0]))<=1e-7 for v in terms.values())
            assert row['metrics']['connectivity']['all_connected'] and row['metrics']['legality']['illegal_voxels']==0
            assert int(material.sum())==row['target_mass']
        else:assert not row['route_feasible'] and row['status']=='incompatible'
    return rows


def render(rows):
    lines=['# W1 constructive baseline','',f'Run `{RUN}`, source `2965502`.',
        '', 'All18 records verified from registered hashes. Ordered cell additions independently replayed; legality, face adjacency, unchanged guide/envelope/allowance, all nine terms and binary metrics independently recomputed.17 feasible scenes have zero-loss connected witnesses; the sealed reference remains incompatible.',
        '', '| Scene | Status | Original guide cells | Added cells | Final cells | Volume (m3) | Witness |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f'| {r["scene_id"]} | {r["status"]} | {r["guide_voxels"]} | {len(r["added_cells"])} | {r["material_voxels"]} | {r["volume_m3"]:.3f} | {r["witness"]} |')
    lines+=['','The fixed radius6/envelope budget and facade_endpoint_v1 accounting are an experimental contract, not production defaults. The construction starts from a full procedural guide, adds only uncharged legal neighboring cells to meet mass/contact bounds, and refuses new radius2 eroded cores. Lexicographic order is deliberately simple and directionally biased.',
        '', 'This proves numerical consistency with a constructive example for each feasible scene, not architectural quality. Some added material serves only to meet a mass floor or dilute the facade ratio. Simple strands remain zero-loss outcomes. NCA training must be compared against this baseline and demonstrate value beyond matching these losses.',
        '', 'No model training, paid compute, cloud operation or deployment. All18 fields and ordered additions remain in the run archive.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    rows=load();p=REPO/'experiments/reports/W1-witnesses.md'
    with p.open('x',encoding='utf-8') as f:f.write(render(rows))
    write_once(REPO/'experiments/reports/W1-verification.json',{'run_id':RUN,'records_replayed':18,'terms_and_metrics_recomputed':18,'witnesses':17,'sealed_reference_refused':True})
    print(p)
