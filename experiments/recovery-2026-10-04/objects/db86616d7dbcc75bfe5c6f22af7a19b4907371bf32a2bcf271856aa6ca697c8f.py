from pathlib import Path
import json,urllib.request,time,hashlib
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Modes-2026-10-04');E=P/'transfer';E.mkdir(exist_ok=False)
protocol=dict(scenes=['r3-variety-wide-gap','r3-variety-offset-y','r3-variety-partial-obstacle'],modes=['low_y','high_y'],volume=.24,seed=2101,interpretation='six exploratory transfer attempts on three already exposed sites; no tuning; not held-out acceptance',baseline='G11-R3-Variety-2026-10-04 seed2101 for same sites')
(E/'protocol.json').write_text(json.dumps(protocol,indent=2))
def get(path):return json.load(urllib.request.urlopen('http://127.0.0.1:8017'+path))
results=[]
for scene in protocol['scenes']:
 for mode in protocol['modes']:
  req=dict(scene=scene,volume=.24,seed=2101,route_mode=mode)
  answer=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8017/api/generate',json.dumps(req).encode(),{'X-NCA-Local':'1','Content-Type':'application/json'})));rid=answer['id']
  (E/f'{scene}-{mode}-submitted.json').write_text(json.dumps(dict(request=req,id=rid)))
  while True:
   v=get('/api/run/'+rid)
   if v['status']['state'] not in ['running','queued']:break
   time.sleep(1)
  results.append(dict(scene=scene,mode=mode,id=rid,status=v['status'],result=v['result']));(E/'result.json').write_text(json.dumps(results,indent=2));print(scene,mode,v['status']['state'],flush=True)
print('Complete; all failures retained.')
