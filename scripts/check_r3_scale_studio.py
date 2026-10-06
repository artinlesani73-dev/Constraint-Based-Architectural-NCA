"""One bounded HTTP integration comparison on the two frozen scale fixtures."""
from pathlib import Path
import json,time,urllib.request,urllib.error,hashlib,copy
import numpy as np

def main():
    repo=Path(__file__).resolve().parents[1]
    runtime=repo/'.local-artifacts/r3-scale-studio-2026-10-06'
    evidence=runtime/'integration';evidence.mkdir(exist_ok=False)
    index=json.loads((repo/'experiments/recovery-2026-10-04/index.json').read_text())
    baseline=Path(index['artifact_root'])/'G11-R3-Scale-2026-10-04'
    def api(path,body=None):
        request=urllib.request.Request('http://127.0.0.1:8019'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json','X-NCA-Local':'1'})
        try:
            with urllib.request.urlopen(request,timeout=10) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:return error.code,json.load(error)
    scenes=api('/api/scenes')[1];report={'runs':[],'failures':[]}
    try:
        before=api('/api/runs')[1]
        scene=next(s for s in scenes if s['scene_id']=='scale-40')
        for label,grid,site in [('unsupported',64,'scale-40'),('mismatch',40,'scale-32')]:
            bad=copy.deepcopy(scene);bad['grid_size']=grid
            code,result=api('/api/generate',dict(scene=site,seed=2102,volume=.24,custom_scene=bad))
            assert code==400,(label,code,result)
            report[label+'_rejected']=result
        assert len(api('/api/runs')[1])==len(before)
        for n in [32,40]:
            # Exercise the edited-scene path without changing the frozen geometry.
            s=next(s for s in scenes if s['scene_id']==f'scale-{n}')
            code,job=api('/api/generate',dict(scene=s['scene_id'],custom_scene=s,seed=2102,volume=.24,route_mode='original'))
            assert code==202,(code,job)
            rid=job['id'];deadline=time.monotonic()+180
            while True:
                code,run=api('/api/run/'+rid)
                if run['status']['state'] not in ['queued','running']:break
                if time.monotonic()>deadline:raise TimeoutError('Run still active; inspect before retrying '+rid)
                time.sleep(.5)
            assert run['status']['state']=='completed',run['status']
            folder=runtime/'runs'/rid
            for step in [64,128]:
                old=np.load(baseline/f'{n}-{step}.npz')['field']
                new=np.load(folder/f'R3-{step}.npz')['field']
                assert np.array_equal(old,new),(n,step,'R3 parity')
                out=run['result']['outputs'][f'R3-{step}']
                assert out['voxels']==np.flatnonzero(new).tolist()
                assert new.shape==(n,n,n)
            assert json.loads((folder/'config.json').read_text())['grid_size']==n
            # Worker writes the manifest immediately after terminal status.
            for _ in range(40):
                if (folder/'manifest.json').exists():break
                time.sleep(.1)
            manifest=json.loads((folder/'manifest.json').read_text())
            assert all(hashlib.sha256((folder/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
            report['runs'].append({'grid':n,'id':rid,'R3_both_horizons_exact_baseline_parity':True,'status':run['status'],'result':run['result']['stability']})
            print('Verified',n,rid,flush=True)
        draft={'site':'scale-40','base':scene,'values':{'edit0':''},'enabled':True,'seed':'2102','volume':'0.24','route_mode':'original'}
        code,saved=api('/api/drafts',draft);assert code==201
        assert next(d for d in api('/api/drafts')[1] if d['id']==saved['id'])['draft']==draft
        report['draft_40_incomplete_roundtrip']=saved['id'];report['passed']=True
    except Exception as error:
        report['failures'].append(repr(error));report['passed']=False
        raise
    finally:
        (evidence/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Integration passed; no training or held-out evaluation')

if __name__=='__main__':main()
