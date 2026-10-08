"""Build a deterministic local release ZIP from verified app and saved history."""
from pathlib import Path
import hashlib,json,platform,shutil,zipfile
import numpy,torch
R=Path(__file__).resolve().parents[1]
BASE=R/'.local-artifacts/r3-workflow-studio-2026-10-08'
OUT=R/'.local-artifacts/r3-local-release-2026-10-08'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 identity=json.loads((BASE/'identity.json').read_text())
 for n,h in identity['files'].items():assert sha(BASE/n)==h,n
 OUT.mkdir(exist_ok=False);app=OUT/'app';app.mkdir()
 for n in identity['files']:
  dest=app/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/n,dest)
 # Packaging-only configurable port. Scientific code is unchanged.
 p=app/'server.py';s=p.read_text(encoding='utf-8')
 s=s.replace('import json,sys,','import os,json,sys,',1)
 s=s.replace("ROOT=Path(__file__).resolve().parent;", "PORT=int(os.environ.get('NCA_STUDIO_PORT','8022')); assert 1024<=PORT<=65535\nROOT=Path(__file__).resolve().parent;",1)
 s=s.replace("'127.0.0.1:8021'", "f'127.0.0.1:{PORT}'").replace("'localhost:8021'", "f'localhost:{PORT}'")
 s=s.replace("'http://127.0.0.1:8021'", "f'http://127.0.0.1:{PORT}'").replace("'http://localhost:8021'", "f'http://localhost:{PORT}'")
 s=s.replace("('127.0.0.1',8021)", "('127.0.0.1',PORT)");assert '8021' not in s;p.write_text(s,encoding='utf-8')
 identity.update(version='R3-local-release-v1',parent_version='R3-workflow-studio-v1')
 identity['files']={n:sha(app/n) for n in identity['files']};save(app/'identity.json',identity)
 history={}
 for runtime in ['r3-scale-studio-2026-10-06','r3-site-studio-2026-10-08','r3-workflow-studio-2026-10-08']:
  base=R/'.local-artifacts'/runtime
  for status in (base/'runs').glob('*/status.json'):
   assert json.loads(status.read_text())['state'] not in ['running','queued'],'Wait for active job: '+str(status)
   manifest=json.loads((status.parent/'manifest.json').read_text())
   for n,h in manifest.items():assert sha(status.parent/n)==h,str(status.parent/n)
  for category in ['runs','drafts']:
   (app/category).mkdir(exist_ok=True)
   for p in (base/category).rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:continue
    name=category+'/'+p.relative_to(base/category).as_posix();dest=app/name;h=sha(p)
    if dest.exists():assert sha(dest)==h,'History collision: '+name
    else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    history.setdefault(name,dict(sha256=h,origins=[]))['origins'].append(runtime)
 save(OUT/'history-index.json',history)
 for p in (R/'experimental/r3-local-release').iterdir():
  if p.is_file():shutil.copyfile(p,OUT/p.name)
 (OUT/'evidence').mkdir()
 for name in ['R3_CONTRIBUTION_COMPLETE.md','R3_TRANSFER_RESULTS.md','R3_WORKFLOW_STUDIO.md']:shutil.copyfile(R/'docs/next-phase'/name,OUT/'evidence'/name)
 for name in ['R3-transfer-results-2026-10-08.json','R3-workflow-check-2026-10-08.json']:shutil.copyfile(R/'experiments/reports'/name,OUT/'evidence'/name)
 save(OUT/'release-manifest.json',dict(version='R3-local-release-v1',runtime=dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__),files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
 archive=OUT.with_suffix('.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():
    info=zipfile.ZipInfo(p.relative_to(OUT).as_posix(),date_time=(2026,10,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes(),compresslevel=9)
 with zipfile.ZipFile(archive) as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==sha(OUT/n),n
 report=dict(version='R3-local-release-v1',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,payloads=len(z.namelist()),saved_runs=len(list((app/'runs').iterdir())),saved_drafts=len(list((app/'drafts').glob('*.json'))),all_payloads_verified=True,source_history_origins=list({o for v in history.values() for o in v['origins']}),scientific_source_model_unchanged=all(identity['files'][n]==h for n,h in json.loads((BASE/'identity.json').read_text())['files'].items() if n.startswith(('source/','model/'))))
 save(R/'experiments/reports/R3-local-release-build-2026-10-08.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
