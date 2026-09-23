"""Verify and publish A2 replay and parameter-gradient evidence without overwrites."""
from collections import defaultdict
from pathlib import Path
import hashlib,json,math,sys,zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_sensitivity import STORE,records
from scripts.diagnostic_inputs import load_inputs
from nca.experiments import read_json,write_once,digest
from nca.access import component_access,component_connectivity
from nca.sensitivity import contexts
from deploy.checkpoints import load_model_c


def verify(run):
    torch.set_num_threads(2);d=STORE.path(run)
    assert not STORE.verify(run) and read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0];replay=records(run,'access_replay');gradient=records(run,'access_gradient')
    assert len(replay)==277 and {r['index'] for r in replay}==set(range(277))
    assert len(gradient)==12 and {r['index'] for r in gradient}==set(range(12))
    events=[read_json(p) for p in (d/'events').glob('*.json')]
    src=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/src) as z:
        for name,expected in protocol['code_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    cfg,_,_=load_model_c();_,inputs=load_inputs(REPO)
    ctxs=contexts(inputs,cfg,[n for n,i in inputs.items() if bool(i['feasible'][0])])
    with torch.no_grad():
        for row in replay:
            source=row['source'];assert source==protocol['sources'][row['index']]
            path=STORE.path(source['source_run'])/source['fields']['path'];assert digest(path)==source['fields']['sha256']
            with np.load(path,allow_pickle=False) as f:p=torch.from_numpy(f['material'].copy())
            ctx,_=ctxs[source['scene']];assert row['scene_hash']==inputs[source['scene']]['scene_hash']
            loss,details=component_access(p,ctx.permitted,ctx.endpoints)
            assert float(loss[0])==row['access_v2'] and details[0]==row['candidate']
            assert component_connectivity(p[0].numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0])==row['binary_v2']
            assert row['access_v1']==source['terms']['access']
            assert row['access_v2']<=row['fixed_worst_unbounded']+1e-6
            assert row['fixed_worst_unbounded']<=row['fixed_worst_64']+1e-6
            assert row['access_v1']<=row['fixed_worst_64']+1e-6
            for recipe,value in source['totals'].items():assert row['totals_v2'][recipe]==value+15*(row['access_v2']-row['access_v1'])
        for row in gradient:
            assert row['case']==protocol['gradient_cases'][row['index']]
            assert row['saved_forward_exact'] and row['frozen_weights_unchanged']
            elements=sum(p['elements'] for p in row['parameter_layout'])
            path=STORE.path(row['case']['source_run'])/row['case']['fields']['path']
            with np.load(d/row['fields']['path'],allow_pickle=False) as f,np.load(path,allow_pickle=False) as original:
                assert np.array_equal(f['material'],original['material']) and np.array_equal(f['raw'],original['raw'])
                for name,summary in row['norms'].items():
                    assert f[name].size==elements and np.isfinite(f[name]).all()
                    for key,arr in [('parameter_l2',f[name]),('last_raw_l2',f['raw_gradient_'+name])]:
                        assert math.isclose(float(np.linalg.norm(arr.astype(float))),summary[key],rel_tol=1e-10,abs_tol=1e-12)
                for a,items in row['cosines'].items():
                    for b,value in items.items():
                        x,y=f[a].astype(float),f[b].astype(float);den=np.linalg.norm(x)*np.linalg.norm(y)
                        if den:assert math.isclose(float(np.dot(x,y)/den),value,rel_tol=1e-10,abs_tol=1e-12)
                        else:assert value is None
    checks={'registered_hashes_verified':True,'source_hashes_verified':len(protocol['code_sha256']),
        'candidate_replays_recomputed':277,'binary_bfs_recomputed':277,
        'gradient_cases_verified':12,'parameter_vectors_verified':72,'last_raw_gradients_verified':72,
        'cosines_verified':432,'saved_forward_fields_match':12,'optimizer_updates':0,
        'scope':'Candidate scores/BFS and vector norms/cosines recomputed. Actual backpropagation is recorded, not independently reimplemented.'}
    return protocol,replay,gradient,checks


def summarize(replay,gradient):
    groups=defaultdict(list)
    for row in replay:groups[row['source']['kind']].append(row)
    summaries=[]
    for kind,rows in groups.items():
        summaries.append({'kind':kind,'cases':len(rows),
            'old_binary_connected':sum(r['source']['metrics']['connectivity']['all_connected'] is True for r in rows),
            'new_binary_connected':sum(r['binary_v2']['all_connected'] for r in rows),
            'old_unscorable':sum(r['old_source_fragmented'] for r in rows),
            'access_lower':sum(r['access_v2']<r['access_v1']-1e-6 for r in rows),
            'access_higher':sum(r['access_v2']>r['access_v1']+1e-6 for r in rows),
            'source_choice_improves':sum(r['access_v2']<r['fixed_worst_unbounded']-1e-6 for r in rows),
            'unbounded_improves':sum(r['fixed_worst_unbounded']<r['fixed_worst_64']-1e-6 for r in rows),
            'worst_destination_increases_loss':sum(r['fixed_worst_64']>r['access_v1']+1e-6 for r in rows),
            'fixed_source_empty':sum(r['fixed_source_material']==0 for r in rows),
            'finite_hop_binary_misses':sum(r['finite_hop_binary_miss'] for r in rows)})
    return {'replay':summaries,'gradients':[{'case':r['case'],'norms':r['norms'],
        'access_v2_coverage_cosine':r['cosines']['access_v2']['coverage'],
        'total_change_cosine':r['cosines']['total_v1']['total_v2']} for r in gradient]}


def render(run,replay,gradient,checks):
    summary=summarize(replay,gradient)
    lines=['# A2: access-contract replay and gradient audit','',f'Run `{run}`; no optimizer updates.',
        '', 'Candidate component_bottleneck_v2 requires one connected material component to touch every entrance region. It also replaces mean-destination scoring by worst-destination strength and removes the spatial hop limit. The attribution columns isolate these changes. Existing results retain their original definitions.',
        '', '## Replay summary','', '| Set | Cases | Old connected | New connected | Old unscorable | Access lower / higher | Source choice improves | Hop removal improves | Worst reduction increases loss |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in summary['replay']:
        lines.append(f'| {r["kind"]} | {r["cases"]} | {r["old_binary_connected"]} | {r["new_binary_connected"]} | {r["old_unscorable"]} | {r["access_lower"]} / {r["access_higher"]} | {r["source_choice_improves"]} | {r["unbounded_improves"]} | {r["worst_destination_increases_loss"]} |')
    lines+=['','Candidate changes are rescoring of saved fields, not learned improvements. W1 controls are included in the sensitivity set. A disconnected fragment cannot pool entrance contact with another component.','',
        '## Actual parameter gradients','', '| Model | Scene | Growth | Access v1 / v2 | Access parameter norm v1 / v2 | Coverage norm | Sparsity norm | Candidate access vs coverage cosine | Total v1 vs v2 cosine |',
        '|---|---|---:|---|---|---:|---:|---|---|']
    for r in gradient:
        c=r['case'];n=r['norms'];fmt=lambda x:'null' if x is None else f'{x:.7g}'
        lines.append(f'| {c["label"]} | {c["scene"]} | {c["steps"]} | {n["access_v1"]["value"]:.7g} / {n["access_v2"]["value"]:.7g} | {n["access_v1"]["parameter_l2"]:.7g} / {n["access_v2"]["parameter_l2"]:.7g} | {n["coverage"]["parameter_l2"]:.7g} | {n["sparsity"]["parameter_l2"]:.7g} | {fmt(r["cosines"]["access_v2"]["coverage"])} | {fmt(r["cosines"]["total_v1"]["total_v2"])} |')
    lines+=['','All forwards match their saved raw/material arrays exactly. Weights remain frozen. Norms are raw derivatives before clipping/Adam; cosines are not optimizer predictions. Zero-vector cosines are null. Topology selection is detached CPU union-find, with a live-tensor critical-voxel derivative and deterministic ties; no GPU efficiency or smoothness claim.','',
        '## Every replay case','', '| Set / model | Scene | Update | Growth | Access old | Fixed worst64 | Fixed worst unbounded | Candidate | Old / new connected |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in replay:
        s=r['source'];lines.append(f'| {s["kind"]} / {s["branch"]} | {s["scene"]} | {s["update"]} | {s["steps"]} | {r["access_v1"]:.7g} | {r["fixed_worst_64"]:.7g} | {r["fixed_worst_unbounded"]:.7g} | {r["access_v2"]:.7g} | {s["metrics"]["connectivity"]["all_connected"]} / {r["binary_v2"]["all_connected"]} |')
    lines+=['','## Verification','', '```json',json.dumps(checks,indent=2),'```','',
        'All source/hash references, critical coordinates, old source ambiguity, fixed-point distances, gradients and per-recipe totals are retained in A2-evidence.json and the immutable run. Candidate totals replace only access coefficient15; other eight families and regularizers remain unchanged. These are development scenes, not unseen validation. No model/recipe promotion, paid compute, Drive operation, deployment or production default change.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    run=sys.argv[1];protocol,replay,gradient,checks=verify(run);out=REPO/'experiments/reports'
    with (out/'A2-access.md').open('x',encoding='utf-8') as f:f.write(render(run,replay,gradient,checks))
    write_once(out/'A2-verification.json',dict(run_id=run,**checks))
    write_once(out/'A2-evidence.json',{'run_id':run,'replay':replay,'gradient':gradient})
    summary=summarize(replay,gradient);write_once(out/'A2-summary.json',dict(run_id=run,**summary))
    print(json.dumps(summary,indent=2))
