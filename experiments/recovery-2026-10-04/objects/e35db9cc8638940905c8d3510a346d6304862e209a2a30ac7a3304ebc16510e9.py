from pathlib import Path
import json,zipfile,hashlib,subprocess
repo=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum-Design-2026-09-29')
a=out.with_suffix('.milestone-verified.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest={}
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():
   n='evidence/'+p.relative_to(out).as_posix();b=p.read_bytes();manifest[n]=sha(b);z.writestr(n,b)
 paths=subprocess.check_output(['git','show','--pretty=format:','--name-only','HEAD'],cwd=repo,text=True).splitlines()
 for name in paths:
  if name.strip():
   b=(repo/name).read_bytes();n='project/'+name;manifest[n]=sha(b);z.writestr(n,b)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(a) as z:
 assert set(z.namelist())==set(manifest)|{'manifest.json'}
 for n,h in manifest.items():assert sha(z.read(n))==h
receipt=dict(archive=str(a),sha256=sha(a.read_bytes()),files=len(manifest),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),off_device_backup=False)
with a.with_suffix('.receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
