from pathlib import Path
import json,zipfile,hashlib,shutil
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Runtime-Stop-20261003T093130Z');out.mkdir(exist_ok=False)
a=Path('C:/Users/artin/Downloads/20261003T093130Z_0cd5c5706d65.zip');receipt=json.loads(a.with_suffix('.receipt.json').read_bytes())
for p in [a,a.with_suffix('.receipt.json')]:shutil.copy2(p,out/p.name);assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((out/p.name).read_bytes()).digest()
with zipfile.ZipFile(a) as z:
 m=json.loads(z.read('evidence-manifest.json'))
 for n,h in m.items():
  assert hashlib.sha256(z.read(n)).hexdigest()==h
 result=json.loads(z.read('result.json'));identity=json.loads(z.read('worker/identity.json'))
 for name in ['result.json','worker.log']:(out/name).write_bytes(z.read(name))
 (out/'identity.json').write_bytes(z.read('worker/identity.json'))
 assert result['request']['manifest_sha256']=='79a8719b2f35b07e7acb510947723bd9ffc09ff186b3dbcbb0580126cb0fa59e'
 assert result['worker']['completed']==0 and not any('checkpoint-' in n for n in z.namelist())
expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu128','numpy':'2.1.3','python':'3.13.15','cuda_build':'12.8','cudnn':91900}
diff={k:dict(expected=v,actual=identity['runtime'][k]) for k,v in expected.items() if identity['runtime'][k]!=v}
record=dict(run_id='20261003T093130Z_0cd5c5706d65',status='failed',reason='Strict runtime guard mismatch before checkpoint/training',completed_updates=0,verified_payloads=5,receipt=receipt,controlled_seconds=result['wall_seconds'],worker_seconds=result['worker']['wall_seconds'],runtime_differences=diff,artifact_location=str(out),execution_provenance='User supplied execution evidence; no standing retry approval inferred.',quality_evaluation_performed=False)
(r/'experiments/records/20261003T093130Z_0cd5c5706d65.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(out/'verified-record.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
entry='''## D098 — CGR3 stopped on changed Colab runtime — 2026-10-03

Verified user ZIP+receipt20261003T093130Z_0cd5c5706d65:all5payload hashes,
exact unique archive membership and expected CGR3 package manifest. Worker
completed0updates; no checkpoint. Controlled8.742s,worker6.554s,exit1.
T4,Python3.13.15,NumPy2.1.3 unchanged. Expected Torch2.11.0+cu128/CUDA12.8/
cuDNN91900; actual Torch2.11.0+cu130/CUDA13.0/cuDNN92700. Strict stack guard
stopped before training; not a model-quality failure. No final evaluation possible.
Original ZIP,receipt,identity,log,result preserved at:
C:/Users/artin/Documents/Codex/outputs/CGR3-Runtime-Stop-20261003T093130Z.
Tracked record:experiments/records/20261003T093130Z_0cd5c5706d65.json.

Next prepare one bounded compatibility/recovery check for the observed cu130
stack before proposing a fresh CGR3 run. Verify deterministic backward and exact
next-update recovery, including an augmented visit, on that same runtime. Do not
simply bypass the guard, claim cu128/cu130 numerical equivalence, or reuse a prior
runtime-bound checkpoint. Keep prior training package and failed evidence intact;
new package/attempt must have new provenance. No retry approved or launched.
CGR1 stays experimental reference,MG7 live; no TEST,Drive or push.

'''
for name in ['RESUME.md','PLAN.md']:
 p=r/'docs/next-phase'/name;x,y=p.read_text(encoding='utf-8').split('\n',1);p.write_text(x+'\n\n'+entry+y,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 with (r/'docs/next-phase'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry.rstrip()+'\n')
(out/'FINDINGS.md').write_text(entry,encoding='utf-8')
print(json.dumps(diff,indent=2))
