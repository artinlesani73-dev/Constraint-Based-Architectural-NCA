from pathlib import Path
import urllib.request,urllib.error,json,importlib.util,uuid,hashlib
import numpy as np
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Studio-2026-10-04');E=P/'verification';E.mkdir(exist_ok=True)
checks={}
for label,payload,origin,wanted in [('invalid_seed',dict(scene='r3-variety-wide-gap',volume=.24,seed=-1),None,400),('foreign_origin',dict(scene='r3-variety-wide-gap',volume=.24,seed=2101),'http://example.com',403)]:
 headers={'Content-Type':'application/json','X-NCA-Local':'1'}
 if origin:headers['Origin']=origin
 try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8014/api/generate',json.dumps(payload).encode(),headers));raise AssertionError('unexpected acceptance')
 except urllib.error.HTTPError as ex:checks[label]=ex.code==wanted;assert checks[label]
run=next(p for p in (P/'runs').iterdir() if json.loads((p/'request.json').read_text())==dict(scene='r3-variety-wide-gap',volume=.24,seed=2101))
assert json.loads((run/'status.json').read_text())['state']=='completed'
ref=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Variety-2026-10-04/cases/r3-variety-wide-gap')
for model in ['G10','R3']:
 for step in [64,128]:
  with np.load(run/f'{model}-{step}.npz') as a,np.load(ref/f'{model}-2101-{step}.npz') as b:assert np.array_equal(a['field'],b['field'])
checks['actual_run_exact_4_field_parity']=True
spec=importlib.util.spec_from_file_location('studio_check',P/'server.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.RUNS=E/'fault-injection';m.RUNS.mkdir(exist_ok=True)
request=dict(scene='r3-variety-wide-gap',volume=.24,seed=2101)
rid=uuid.uuid4().hex;p=m.RUNS/rid;p.mkdir();m.write(p/'test.json',dict(synthetic=True,injection='route_from_context returns None; not evidence of site infeasibility'))
m.route_from_context=lambda c:None;m.lock.acquire();m.generate(rid,request)
result=json.loads((p/'result.json').read_text());assert result['outputs']['R3-128']['error'];assert 'voxels' in result['outputs']['G10-128'];assert json.loads((p/'status.json').read_text())['state']=='failed_checks';checks['no_certificate_preserves_raw_and_explicit_failure']=True
rid=uuid.uuid4().hex;p=m.RUNS/rid;p.mkdir();m.write(p/'test.json',dict(synthetic=True,injection='context construction raises'))
def fail(*a):raise RuntimeError('Synthetic verification fault')
m.target_context=fail;m.lock.acquire();m.generate(rid,request);assert json.loads((p/'status.json').read_text())['state']=='error' and (p/'error.txt').exists() and not m.lock.locked();checks['worker_error_persisted_and_lock_released']=True
(E/'checks.json').write_text(json.dumps(checks,indent=2));print(checks)
