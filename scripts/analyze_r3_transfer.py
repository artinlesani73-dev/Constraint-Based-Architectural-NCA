"""Verify all frozen transfer slots and report failures without filtering."""
import itertools,json,sys
import numpy as np
from r3_transfer_study import ROOT,OUT,save,sha,verify

def main():
    verify()
    protocol=json.loads((OUT/'protocol.json').read_text())
    controller=json.loads((OUT/'controller.json').read_text());assert controller['state']=='completed'
    observations=[];diversity=[];pairs=[];provenance=[];certificates=[]
    sys.path.insert(0,str(OUT/'source'))
    from nca.generation_data import seed_inputs
    for i,s in enumerate(protocol['sites']):
        fields={};cert_by_slot=[]
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                d=OUT/f'slot-{i}-{seed}-{method}'
                for n,h in json.loads((d/'manifest.json').read_text()).items():assert sha(d/n)==h,(d,n)
                rows=json.loads((d/'results.json').read_text());assert len(rows)==1
                row=rows[0];assert (row['seed'],row['method'])==(seed,method)
                cert=json.loads((d/'certificate.json').read_text());cert_by_slot.append(cert)
                mem=json.loads((d/'memory.json').read_text());assert mem['peak_working_set_bytes']>0
                observations.append(dict(site=s['scene_id'],certified=cert['certified'],memory=mem,**row))
                provenance.append(dict(site=s['scene_id'],seed=seed,method=method,directory=str(d),manifest_sha256=sha(d/'manifest.json')))
                if row['status']!='evaluated':
                    assert method!='G10' and not cert['certified'] and not row['accepted'];continue
                with np.load(d/'context.npz') as c:initial=seed_inputs(c['condition'])['occupancy'].astype(bool)
                with np.load(d/f'{method}-{seed}-trajectory.npz') as t:
                    for k in t.files:assert np.isfinite(t[k]).all()
                    births=t['births'];assert births.sum(0).max()<=1
                    for step in [64,128]:
                        with np.load(d/f'{method}-{seed}-{step}.npz') as a:f=a['field']
                        assert f.shape==(s['grid_size'],)*3
                        assert np.array_equal(f,initial|births[:step].any(0))
                        fields[method,seed,step]=f
                assert not (fields[method,seed,64]&~fields[method,seed,128]).any()
                expected=all(v['score']['contract_pass'] and v['volume_error']<=.04 for v in row['scores'].values()) and row['growth']<=.05
                assert row['accepted']==expected
        assert all(c==cert_by_slot[0] for c in cert_by_slot)
        certificates.append(dict(site=s['scene_id'],**cert_by_slot[0]))
        def dist(a,b):return float(1-(a&b).sum()/(a|b).sum())
        for m in protocol['methods']:
            if all((m,seed,128) in fields for seed in protocol['seeds']):
                diversity.append(dict(site=s['scene_id'],method=m,certified=cert_by_slot[0]['certified'],jaccard_distance=dist(fields[m,2201,128],fields[m,2202,128])))
        for seed in protocol['seeds']:
            for a,b in itertools.combinations(protocol['methods'],2):
                if (a,seed,128) in fields and (b,seed,128) in fields:pairs.append(dict(site=s['scene_id'],seed=seed,methods=[a,b],jaccard_distance=dist(fields[a,seed,128],fields[b,seed,128])))
    assert len(observations)==24
    def stats(v):return dict(min=min(v),median=float(np.median(v)),max=max(v)) if v else None
    summary={}
    for m in protocol['methods']:
        rows=[v for v in observations if v['method']==m];evaluated=[v for v in rows if v['status']=='evaluated'];certified=[v for v in rows if v['certified']]
        summary[m]=dict(requests=len(rows),outputs=len(evaluated),accepted=sum(v['accepted'] for v in rows),no_certified_output=sum(v['status']=='no_certified_output' for v in rows),certified_requests=len(certified),certified_accepted=sum(v['accepted'] for v in certified),mean_seed_jaccard_certified=float(np.mean([v['jaccard_distance'] for v in diversity if v['method']==m and v['certified']])) if certified else None,
            rollout_seconds=stats([v['rollout_seconds'] for v in evaluated]),peak_worker_working_set_MiB=stats([v['memory']['peak_working_set_bytes']/1048576 for v in evaluated]),peak_worker_commit_MiB=stats([v['memory']['peak_commit_bytes']/1048576 for v in evaluated]))
    result=dict(complete=True,summary=summary,observations=observations,seed_diversity=diversity,paired_methods=pairs,certificates=certificates,provenance=provenance,controller=controller,limitations=protocol['novelty']+' All cases are exposed after this evaluation. Per-method timings pool different grids; consult per-case observations. Capture/setup affects peak process memory. Small authored sample, not a broad generalization claim.')
    save(OUT/'analysis.json',result);save(ROOT/'experiments/reports/R3-transfer-results-2026-10-08.json',result)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
