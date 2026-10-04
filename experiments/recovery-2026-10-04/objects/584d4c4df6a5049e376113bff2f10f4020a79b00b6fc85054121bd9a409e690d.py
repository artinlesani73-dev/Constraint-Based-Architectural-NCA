from pathlib import Path
import json,hashlib,zipfile,io
import numpy as np,torch
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum');run=out/'rehearsal/curriculum-runs/20260929T070735Z_0580de3e2476';a=run.with_suffix('.zip')
receipt=json.loads(a.with_suffix('.receipt.json').read_bytes());assert hashlib.sha256(a.read_bytes()).hexdigest()==receipt['sha256']
with zipfile.ZipFile(a) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m)|{'evidence-manifest.json'}
 for name,h in m.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 for i in range(1,9):
  trace=json.loads(z.read(f'worker/update-{i:04d}.json'))
  with np.load(io.BytesIO(z.read(f'worker/training-{i:04d}.npz')),allow_pickle=False) as pack:assert hashlib.sha256(pack['start'].astype('<f4').tobytes(order='C')).hexdigest()==trace['start']['occupancy_sha256']
 p=torch.load(io.BytesIO(z.read('worker/checkpoint-0008.pt')),map_location='cpu',weights_only=True);assert sum(p['start_visits'])==p['completed']==8
result=json.loads((run/'result.json').read_bytes());assert result['status']=='completed' and result['cleanup']['active_processes_after_stop']==0
report=dict(package=json.loads((out/'package-receipt.json').read_bytes()),rehearsal=result,receipt=receipt,verified_payloads=len(m),start_hashes_verified=8,tests={'consolidated_test_passed':True,'seconds':2.848,'covers':['CGR1 first-update parity','augmented next-step exact full-payload recovery','original-input evaluation unchanged','CGR1/CGR2 semantic rejection','visit corruption rejection']},gpu_approved=False,gpu_executed=False)
(r/'experiments/reports/CGR3-readiness.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
entry='''## D097 — CGR3 implemented and packaged; GPU approval pending — 2026-09-29

Separate curriculum_repair session implements D096 with CGR1 model/loss unchanged.
Checkpoint visit counters are checked against full trace; each training update
exports its exact starting occupancy and SHA256. Curriculum only in step(),
original tensors/evaluate inherited unchanged. Same-shape prior restores rejected.
Consolidated CPU test passed2.848s: first original update exactly matches CGR1,
augmented next step restores exact start/loss/state/optimizer/weights, evaluation
uses original input and does not change payload; invalid counters rejected.
Eight-update packaged CPU rehearsal20260929T070735Z_0580de3e2476 completed27.437s;
41payload hashes and8start hashes verified, checkpoint visits sum8,cleanup0active.
Rehearsal first visits are originals; augmented recovery covered by focused test.
Engineering evidence only; no new GPU compatibility/recovery or quality claim.

Ready folder C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum/:
NCA-CGR3-Curriculum.ipynb and NCA-CGR3-Curriculum-Package.zip.
ZIP SHA256498617094d60b12c10a6455daeae3046f9f56acf26667996c3c661b3f036cb97.
Manifest79a8719b2f35b07e7acb510947723bd9ffc09ff186b3dbcbb0580126cb0fa59e.
TRAIN81,heldout0,103payloads; notebook approval gateFalse. See CGR3-readiness.json.
Next request ONE seed1201 T4 job,256updates32steps600controlledseconds; setup,
export,idle extra. Download ZIP+receipt locally,including failures. No retry,
extra seed,Drive,push or live admission. MG7 stays live. Upon returned evidence,
adapt frozen final256 reviewer for new semantics/manifest and verify start visits,
trace and arrays; retain original-input27development evaluation and all prior gates.

'''
for name in ['RESUME.md','PLAN.md']:
 q=r/'docs/next-phase'/name;x,y=q.read_text(encoding='utf-8').split('\n',1);q.write_text(x+'\n\n'+entry+y,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 with (r/'docs/next-phase'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry.rstrip()+'\n')
(r/'docs/next-phase/CGR3_READINESS.md').write_text('# CGR3 readiness\n\n'+entry+'Local copies are same-disk archives, not off-device backup.\n',encoding='utf-8')
(out/'READINESS.md').write_text(entry,encoding='utf-8')
print('Verified',len(m),'payloads and eight starting-state hashes.')
