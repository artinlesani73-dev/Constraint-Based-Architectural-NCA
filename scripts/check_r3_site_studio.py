"""Bounded development integration: reference, two varied sites, blocked site."""
from pathlib import Path
import json,urllib.request,urllib.error,time,hashlib,copy
import numpy as np
def main():
 r=Path(__file__).resolve().parents[1];runtime=r/'.local-artifacts/r3-site-studio-2026-10-08';out=runtime/'integration-v2';out.mkdir(exist_ok=False)
 def api(path,body=None):
  q=urllib.request.Request('http://127.0.0.1:8020'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','X-NCA-Local':'1'})
  try:
   with urllib.request.urlopen(q,timeout=15) as response:return response.status,json.load(response)
  except urllib.error.HTTPError as e:return e.code,json.load(e)
 deadline=time.monotonic()+30
 while True:
  try:
   code,loaded=api('/api/scenes');assert code==200;break
  except urllib.error.URLError:
   if time.monotonic()>deadline:raise
   time.sleep(.5)
 scenes={s['scene_id']:s for s in loaded};report={'runs':[],'rejections':{},'passed':False}
 protocol={'sites':['scale-32','site-staggered-court-40','site-offset-obstacle-40','site-blocked-gap-40'],'seed':2102,'volume':.24,'route_mode':'original','steps':128,'per_job_wait_limit_seconds':180,'interpretation':'Exposed development fixtures; no tuning or retries. Failed quality retained.'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 try:
  base=scenes['site-staggered-court-40'];before=len(api('/api/runs')[1])
  for label in ['too_many','out_of_bounds','extra_face']:
   bad=copy.deepcopy(base)
   if label=='too_many':bad['buildings']+=bad['buildings'][2:]*2
   if label=='out_of_bounds':bad['buildings'][2]['x']=[38,42]
   if label=='extra_face':bad['buildings'][2].update(side='left',gap_facing_x=17)
   code,value=api('/api/generate',dict(scene=base['scene_id'],custom_scene=bad,seed=2102,volume=.24))
   assert code==400,(label,code,value);report['rejections'][label]=value
  assert len(api('/api/runs')[1])==before
  for sid in protocol['sites']:
   code,job=api('/api/generate',dict(scene=sid,custom_scene=scenes[sid],seed=2102,volume=.24,route_mode='original'));assert code==202,(sid,code,job)
   rid=job['id'];deadline=time.monotonic()+180
   while True:
    code,v=api('/api/run/'+rid)
    if v['status']['state'] not in ['running','queued']:break
    if time.monotonic()>deadline:raise TimeoutError('Inspect still-active run '+rid+' before retrying')
    time.sleep(.5)
   assert v['status']['state'] in ['completed','failed_checks'],v['status']
   folder=runtime/'runs'/rid
   for _ in range(40):
    if (folder/'manifest.json').exists():break
    time.sleep(.1)
   hashes=json.loads((folder/'manifest.json').read_text());assert all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
   if sid=='scale-32':
    prior=r/'.local-artifacts/r3-scale-studio-2026-10-06/runs/d71b4acf1c9044f9a4fd3632cc05deb9'
    for m in ['G10','R3']:
     for step in [64,128]:assert np.array_equal(np.load(folder/f'{m}-{step}.npz')['field'],np.load(prior/f'{m}-{step}.npz')['field'])
    report['reference_all_four_fields_exact']=True
   if sid=='site-blocked-gap-40':
    assert not v['result']['certificate']['certified']
    assert all('error' in v['result']['outputs'][f'R3-{step}'] for step in [64,128])
    report['blocked_no_fabricated_R3']=True
   report['runs'].append({'site':sid,'id':rid,'status':v['status']['state'],'certificate':v['result']['certificate']['certified'],'stability':v['result']['stability'],'outputs':{k:val.get('record',val) for k,val in v['result']['outputs'].items()}})
   print(sid,v['status']['state'],rid,flush=True)
  report['passed']=True
 finally:(out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print('Integration complete; quality results include all attempted sites')
if __name__=='__main__':main()
