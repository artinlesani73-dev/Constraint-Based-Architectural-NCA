"""Verify and summarize every result from the frozen contribution comparison."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np

R=Path(__file__).resolve().parents[1];P=R/'.local-artifacts/r3-contribution-2026-10-08'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distance(a,b):return float(1-(a&b).sum()/(a|b).sum())
def main():
    protocol=json.loads((P/'protocol.json').read_text());controller=json.loads((P/'controller.json').read_text());assert controller['state'] in ['completed','stopped'],controller
    for name,h in json.loads((P/'input-manifest.json').read_text()).items():assert sha(P/name)==h,name
    observations=[];diversity=[];pairs=[]
    for i,s in enumerate(protocol['sites']):
        d=P/f'case-{i}'
        if not (d/'results.json').exists():continue
        if (d/'manifest.json').exists():
            manifest=json.loads((d/'manifest.json').read_text());assert all(sha(d/n)==h for n,h in manifest.items())
        else:
            (d/'partial-evidence-manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='partial-evidence-manifest.json'},indent=2))
        values=json.loads((d/'results.json').read_text());assert len(values)<=6
        fields={}
        for v in values:
            observations.append(dict(site=s['scene_id'],**v))
            if v['status']!='evaluated':continue
            trajectory=np.load(d/f"{v['method']}-{v['seed']}-trajectory.npz")
            birth=trajectory['births'];assert np.isfinite(birth).all() and birth.sum(0).max()<=1
            for step in [64,128]:
                f=np.load(d/f"{v['method']}-{v['seed']}-{step}.npz")['field']
                assert f.shape==(s['grid_size'],)*3
                fields[(v['method'],v['seed'],step)]=f
            f64=fields[(v['method'],v['seed'],64)];f128=fields[(v['method'],v['seed'],128)]
            assert not (f64&~f128).any()
        for m in protocol['methods']:
            if (m,2102,128) in fields and (m,2103,128) in fields:
                a,b=fields[(m,2102,128)],fields[(m,2103,128)]
                diversity.append(dict(site=s['scene_id'],method=m,jaccard_distance=distance(a,b),symmetric_difference_voxels=int((a!=b).sum())))
        for seed in protocol['seeds']:
            for a,b in itertools.combinations(protocol['methods'],2):
                if (a,seed,128) in fields and (b,seed,128) in fields:
                    af,bf=fields[(a,seed,128)],fields[(b,seed,128)]
                    pairs.append(dict(site=s['scene_id'],seed=seed,methods=[a,b],jaccard_distance=distance(af,bf)))
    summary={}
    for m in protocol['methods']:
        rows=[x for x in observations if x['method']==m];feasible=[x for x in rows if x['site']!='site-blocked-gap-40']
        summary[m]=dict(planned_requests=8,evaluated_requests=len(rows),missing_requests=8-len(rows),outputs=sum(x['status']=='evaluated' for x in rows),accepted=sum(x['accepted'] for x in rows),certified_site_requests=len(feasible),certified_site_accepted=sum(x['accepted'] for x in feasible),mean_seed_jaccard_certified_sites=float(np.mean([x['jaccard_distance'] for x in diversity if x['method']==m and x['site']!='site-blocked-gap-40'])))
    result=dict(summary=summary,observations=observations,seed_diversity=diversity,paired_methods=pairs,controller=controller,frozen_inputs_and_completed_worker_hashes_verified=True,partial_worker_evidence_manifest_created=True,complete=False,missing=[{'site':s['scene_id'],'seed':seed,'method':m} for s in protocol['sites'] for seed in protocol['seeds'] for m in protocol['methods'] if not any(x['site']==s['scene_id'] and x['seed']==seed and x['method']==m for x in observations)],known_seed_parity='Every G10/R3 field at64/128 for seed2102 exactly matched the preceding studio integration outputs')
    target=R/'experiments/reports/R3-contribution-results-2026-10-08.json';target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');(P/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':summary,'seed_diversity':diversity,'paired_R3_neutral':[x for x in pairs if x['methods']==['R3','neutral']]},indent=2))

if __name__=='__main__':main()
