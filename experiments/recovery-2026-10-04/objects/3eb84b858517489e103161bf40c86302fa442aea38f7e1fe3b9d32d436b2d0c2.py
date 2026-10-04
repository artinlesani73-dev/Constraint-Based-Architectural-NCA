from pathlib import Path
import zipfile,json,hashlib
old=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum-cu130');out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(old/'NCA-CGR3-Curriculum-Package.zip') as z:
 payload={n:z.read(n) for n in z.namelist() if n!='manifest.json'};m=json.loads(z.read('manifest.json'))
name='scripts/colab_curriculum_repair.py';s=payload[name].decode()
s=s.replace("'torch':'2.11.0+cu128'","'torch':'2.11.0+cu130'").replace("'cuda_build':'12.8'","'cuda_build':'13.0'").replace("'cudnn':91900","'cudnn':92700")
s=s.replace('These are the stack and hardware type demonstrated by NR2.', 'This stack passed CGR3 recovery run 20261003T094720Z_e18428c17993.').replace('verified NR2 stack','verified CGR3 cu130 stack')
compile(s,name,'exec');payload[name]=s.encode()
for name in ['README.md','docs/next-phase/CURRICULUM_REPAIR_PROTOCOL.md']:
 s=payload[name].decode().replace('2.11.0+cu128','2.11.0+cu130').replace('CUDA12.8','CUDA13.0').replace('cuDNN91900','cuDNN92700')
 s=s.replace("The new model's GPU execution is not yet verified: local CPU evidence only.", 'Two-update deterministic GPU backward and exact augmented-step recovery verified on this stack; full trial quality is untested.')
 s+='\nRuntime update 2026-10-03: compatibility run20261003T094720Z_e18428c17993 passed.\nFresh256-update trial only after approval. Training math,data,seed and curriculum\nunchanged; runtime changed versus previous CGR1/CGR2,so comparisons cannot isolate\ncurriculum effects perfectly. No cross-runtime numerical equivalence is claimed.\nDo not resume a compatibility checkpoint or rerun the old cu128 package.\n'
 payload[name]=s.encode()
m['files']={n:sha(b) for n,b in payload.items()};m['runtime_evidence']='20261003T094720Z_e18428c17993'
a=out/'NCA-CGR3-Curriculum-cu130-Package.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED) as z:
 for n,b in payload.items():z.writestr(n,b)
 z.writestr('manifest.json',json.dumps(m,indent=2))
nb=json.loads((old/'NCA-CGR3-Curriculum.ipynb').read_bytes());previous=json.loads((old/'package-receipt.json').read_bytes())['archive_sha256']
for c in nb['cells']:
 text=''.join(c['source']).replace(previous,sha(a.read_bytes())).replace('NCA-CGR3-Curriculum-Package.zip',a.name)
 if c['cell_type']=='code':compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
assert any('APPROVED_SEED_JOB = False' in ''.join(c['source']) for c in nb['cells'])
(out/'NCA-CGR3-Curriculum-cu130.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
(out/'START-HERE.md').write_bytes(payload['README.md'])
receipt=dict(archive=a.name,archive_sha256=sha(a.read_bytes()),manifest_sha256=sha(json.dumps(m,indent=2).encode()),payload_files=len(payload),notebook_sha256=sha((out/'NCA-CGR3-Curriculum-cu130.ipynb').read_bytes()),gpu_approved=False,gpu_executed=False,runtime_evidence='20261003T094720Z_e18428c17993',repository_sync_pending=True)
(out/'package-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
with zipfile.ZipFile(a) as z:
 assert set(z.namelist())==set(payload)|{'manifest.json'}
 for n,b in payload.items():assert sha(z.read(n))==sha(b)
# Assert every mathematical/model/data file is byte-identical to prior package.
with zipfile.ZipFile(old/'NCA-CGR3-Curriculum-Package.zip') as z:
 changed=[n for n,b in payload.items() if b!=z.read(n)]
assert set(changed)=={'scripts/colab_curriculum_repair.py','README.md','docs/next-phase/CURRICULUM_REPAIR_PROTOCOL.md'}
(out/'verification.json').write_text(json.dumps(dict(changed_files=changed,all_model_data_settings_unchanged=True,all_payload_hashes_verified=True,notebook_cells_compile=True,scope='Runtime guard update only; original CPU integration rehearsal and new GPU recovery evidence retained.'),indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
