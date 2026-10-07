"""Merge original observations and missing slots, with verified source provenance."""
from pathlib import Path
import json, hashlib, itertools
import numpy as np
from continue_r3_contribution import ROOT, OLD, OUT, sha, save, verify_inputs

def main():
    verify_inputs()
    assert json.loads((OUT/'controller.json').read_text())['state']=='completed'
    protocol=json.loads((OLD/'protocol.json').read_text())
    observations=[]; diversity=[]; paired=[]; provenance=[]
    for i,site in enumerate(protocol['sites']):
        fields={}
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                previous=OLD/f'case-{i}'
                oldrows=json.loads((previous/'results.json').read_text()) if (previous/'results.json').exists() else []
                existing=[v for v in oldrows if v['seed']==seed and v['method']==method]
                d=previous if existing else OUT/f'missing-{i}-{seed}-{method}'
                manifest=d/('manifest.json' if (d/'manifest.json').exists() else 'partial-evidence-manifest.json')
                for n,h in json.loads(manifest.read_text()).items(): assert sha(d/n)==h,(d,n)
                row=existing[0] if existing else json.loads((d/'results.json').read_text())[0]
                observations.append(dict(site=site['scene_id'],**row))
                provenance.append(dict(site=site['scene_id'],seed=seed,method=method,directory=str(d),manifest_sha256=sha(manifest)))
                if row['status']!='evaluated': continue
                with np.load(d/f'{method}-{seed}-trajectory.npz') as trajectory:
                    births=trajectory['births']; assert births.sum(0).max()<=1
                    for key in trajectory.files: assert np.isfinite(trajectory[key]).all()
                for step in [64,128]:
                    with np.load(d/f'{method}-{seed}-{step}.npz') as a: f=a['field']
                    assert f.shape==(site['grid_size'],)*3
                    fields[method,seed,step]=f
                assert not (fields[method,seed,64]&~fields[method,seed,128]).any()
        def distance(a,b): return float(1-(a&b).sum()/(a|b).sum())
        for m in protocol['methods']:
            if all((m,s,128) in fields for s in protocol['seeds']):
                diversity.append(dict(site=site['scene_id'],method=m,jaccard_distance=distance(fields[m,2102,128],fields[m,2103,128])))
        for seed in protocol['seeds']:
            for a,b in itertools.combinations(protocol['methods'],2):
                if (a,seed,128) in fields and (b,seed,128) in fields:
                    paired.append(dict(site=site['scene_id'],seed=seed,methods=[a,b],jaccard_distance=distance(fields[a,seed,128],fields[b,seed,128])))
    assert len(observations)==24
    summary={}
    for m in protocol['methods']:
        rows=[v for v in observations if v['method']==m]
        feasible=[v for v in rows if v['site']!='site-blocked-gap-40']
        summary[m]=dict(requests=len(rows),outputs=sum(v['status']=='evaluated' for v in rows),no_certified_output=sum(v['status']=='no_certified_output' for v in rows),accepted=sum(v['accepted'] for v in rows),feasible_accepted=sum(v['accepted'] for v in feasible),feasible_requests=len(feasible),mean_seed_jaccard_feasible=float(np.mean([d['jaccard_distance'] for d in diversity if d['method']==m and d['site']!='site-blocked-gap-40'])))
    parity=[]
    for m in protocol['methods']:
        d=OUT/f'parity-2-2102-{m}'
        for n,h in json.loads((d/'manifest.json').read_text()).items(): assert sha(d/n)==h
        parity.append(json.loads((d/'parity.json').read_text()))
    result=dict(complete=True,summary=summary,observations=observations,seed_diversity=diversity,paired_methods=paired,provenance=provenance,parity=parity,limitations='Four exposed development sites, three feasible. Two correlated firing seeds per site. Neutral removes learned gating and ranking together. Geometry validity and voxel differences do not establish architectural quality or broad generalization.')
    save(OUT/'analysis.json',result)
    save(ROOT/'experiments/reports/R3-contribution-complete-2026-10-08.json',result)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
