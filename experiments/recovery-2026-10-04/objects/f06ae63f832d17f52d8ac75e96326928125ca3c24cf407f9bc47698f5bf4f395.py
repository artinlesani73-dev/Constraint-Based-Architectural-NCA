from pathlib import Path
import json,hashlib,urllib.request,urllib.error,copy
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Editor-2026-10-04')
runs=list((P/'runs').iterdir());assert len(runs)==1;p=runs[0]
request=json.loads((p/'request.json').read_text());scene=json.loads((p/'scene.json').read_text());result=json.loads((p/'result.json').read_text())
assert request['custom_scene']==scene==result['scene'];assert scene['buildings'][0]['x'][1]==7 and scene['entrances'][0]['x']==7 and scene['entrances'][0]['z']==9
manifest=json.loads((p/'manifest.json').read_text());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
invalid=copy.deepcopy(request);invalid['custom_scene']['grid_size']=10000
try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8015/api/generate',json.dumps(invalid).encode(),{'X-NCA-Local':'1','Content-Type':'application/json'}));raise AssertionError('invalid grid accepted')
except urllib.error.HTTPError as e:assert e.code==400;error=json.loads(e.read())
assert list((P/'runs').iterdir())==runs
check=dict(run=p.name,request_scene_and_result_exact=True,facade_x_and_connection_z_edits_preserved=True,all_run_hashes_verified=True,oversize_grid_rejected_without_run=error,status=json.loads((p/'status.json').read_text()),family_pass={k:v['record']['score']['family_pass'] for k,v in result['outputs'].items() if 'record' in v})
(P/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print(json.dumps(check))
