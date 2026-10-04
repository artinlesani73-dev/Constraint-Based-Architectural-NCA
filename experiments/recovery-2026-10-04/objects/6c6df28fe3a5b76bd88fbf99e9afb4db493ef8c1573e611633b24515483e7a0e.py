from pathlib import Path
import json,hashlib,zipfile,sys
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');sys.path.insert(0,str(r))
from nca.experiments import snapshot_source
out=Path('C:/Users/artin/Documents/Codex/outputs/CGR2-Bulk')
run=out/'rehearsal/bulk-runs/20260928T160237Z_2eabf7d1cf3a';a=run.with_suffix('.zip');receipt=json.loads(a.with_suffix('.receipt.json').read_bytes())
assert hashlib.sha256(a.read_bytes()).hexdigest()==receipt['sha256']
with zipfile.ZipFile(a) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert set(z.namelist())==set(m)|{'evidence-manifest.json'} and len(z.namelist())==len(set(z.namelist()))
 for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
result=json.loads((run/'result.json').read_bytes());assert result['status']=='completed' and result['worker']['completed']==8 and result['cleanup']['active_processes_after_stop']==0
readiness=dict(package=json.loads((out/'package-receipt.json').read_bytes()),rehearsal=result,rehearsal_receipt=receipt,verified_payloads=len(m),tests={'passed':5,'seconds':2.570},gradient_audit=json.loads((out/'train-gradient-audit.json').read_bytes()),gpu_executed=False,approved=False)
(r/'experiments/reports/CGR2-readiness.json').write_text(json.dumps(readiness,indent=2)+'\n',encoding='utf-8')
entry='''## CGR2 objective revision ready; one GPU approval pending — 2026-09-28

D093: separate bulk_repair.py retains CGR1 architecture, seed, data,32steps and
256updates. Add0.5 squared target-full3-cube deficit; intact negative weight3
(previous1.5). Local bulk surrogate, not guaranteed connectivity. Prior code and
checkpoints preserved; semantic restore guard rejects CGR1 weights as resumable
CGR2 state. Five focused tests passed2.570s, including exact CPU recovery.
TRAIN81 gradient audit: finite gradients,54damaged rows nonzero,27intact zero;
no validation/TEST inspected. Eight-update CPU rehearsal completed26.547s,
run20260928T160237Z_2eabf7d1cf3a,41payload hashes verified, no active processes.
These are engineering checks, not trained-model quality or GPU recovery evidence.

Ready files: C:/Users/artin/Documents/Codex/outputs/CGR2-Bulk/
NCA-CGR2-Connected.ipynb and NCA-CGR2-Connected-Package.zip.
Package SHA2564acfe0d207e18f9c58df532a754cebfaebd1769a44689f67f2a402369162316e.
Manifest b3d15f8625637e75ba4b3a59aea8f243facb5d572b0653436acba90afe23b22a.
Upload accepts any filename for exactly one file, while enforcing the checksum.
See BULK_REPAIR_SPEC.md, BULK_REPAIR_PROTOCOL.md, experiments/reports/CGR2-readiness.json.
Next: obtain approval for one Colab T4 seed1201,256updates,32steps,600controlled
seconds (setup/idle/export extra). Gate remains False. Download ZIP+receipt
locally; no Drive operation or off-device backup claimed. No retry/extra seed.
Review final256 accepted occupancy using same27development rows and frozen gates,
report CGR1/NR5/closing3 comparison; no TEST or automatic admission. MG7 stays live.
Adapt review_connected_run.py to CGR2 semantic/module/manifest only after receiving
results; never change the historical CGR1 review or tune gates after evaluation.

'''
for name in ['RESUME.md','PLAN.md']:
 p=r/'docs/next-phase'/name;s=p.read_text(encoding='utf-8');a1,b=s.split('\n',1);p.write_text(a1+'\n\n'+entry+b,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 p=r/'docs/next-phase'/name
 with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry.rstrip()+'\n')
(r/'docs/next-phase/CGR2_READINESS.md').write_text('# CGR2 readiness\n\n'+entry,encoding='utf-8')
snapshot_source(r,out/'local-source-snapshot.zip')
print('Verified',len(m),'rehearsal payloads and recorded readiness.')
