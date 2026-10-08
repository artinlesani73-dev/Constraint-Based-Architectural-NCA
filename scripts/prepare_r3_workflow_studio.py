"""Prepare a separate workflow studio, preserving every previous runtime."""
from pathlib import Path
import json,hashlib,subprocess,sys,shutil

def main():
 r=Path(__file__).resolve().parents[1];name='r3-workflow-studio-2026-10-08'
 subprocess.run([sys.executable,str(r/'scripts/prepare_r3_site_studio.py'),'--name',name],check=True)
 out=r/'.local-artifacts'/name
 for p in (r/'experimental/r3-workflow-studio').iterdir():
  if p.suffix in {'.py','.js','.html','.css'}:shutil.copyfile(p,out/p.name)
 identity=json.loads((out/'identity.json').read_text());identity.update(version='R3-workflow-studio-v1',parent_version='R3-site-studio-v1')
 identity['files'].update({n:'' for n in ['workflow.js','export_bundle.py']})
 identity['files']={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in identity['files']}
 (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
