"""Fresh-folder release smoke check; no generation or dependency installation."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,time,urllib.request,zipfile
R=Path(__file__).resolve().parents[1];PACKAGE=R/'.local-artifacts/r3-local-release-2026-10-08.zip';OUT=R/'.local-artifacts/r3-release-smoke-2026-10-08'
def main():
 OUT.mkdir(exist_ok=False);copy=OUT/'fresh-copy';copy.mkdir();report=dict(passed=False);server=None
 try:
  expected=json.loads((R/'experiments/reports/R3-local-release-build-2026-10-08.json').read_text());assert hashlib.sha256(PACKAGE.read_bytes()).hexdigest()==expected['sha256']
  with zipfile.ZipFile(PACKAGE) as z:
   for name in z.namelist():
    p=(copy/name).resolve();assert p.is_relative_to(copy.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
  check=subprocess.run([sys.executable,str(copy/'launch.py'),'--check'],capture_output=True,text=True,timeout=30)
  (OUT/'preflight.log').write_text(check.stdout+check.stderr,encoding='utf-8');assert check.returncode==0,check.stderr
  def api(path,body=None):
   request=urllib.request.Request('http://127.0.0.1:8022'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','X-NCA-Local':'1','Origin':'http://127.0.0.1:8022'})
   with urllib.request.urlopen(request,timeout=20) as response:return response.read()
  with (OUT/'server.log').open('w',encoding='utf-8') as log:
   server=subprocess.Popen([sys.executable,str(copy/'launch.py'),'--port','8022'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
   deadline=time.monotonic()+35
   while True:
    if server.poll() is not None:raise RuntimeError('Release server exited; inspect log')
    try:runs=json.loads(api('/api/runs'));break
    except OSError:
     if time.monotonic()>deadline:raise
     time.sleep(.5)
   assert len(runs)==expected['saved_runs']
   drafts=json.loads(api('/api/drafts'));assert len(drafts)==expected['saved_drafts']
   rid='8f11090ceaef4f618f853acddb7c4988';v=json.loads(api('/api/run/'+rid));assert v['result']['id']==rid and v['status']['state']=='completed'
   raw=api('/api/export/'+rid);(OUT/'reopened-run.zip').write_bytes(raw)
   with zipfile.ZipFile(io.BytesIO(raw)) as z:
    manifest=json.loads(z.read('export-manifest.json'))
    assert set(z.namelist())==set(manifest)|{'export-manifest.json'}
    for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    assert json.loads(z.read('metadata.json'))['run_id']==rid
   # Port-specific mutation guard accepts the correct local origin; old drafts stay unchanged.
   saved=json.loads(api('/api/drafts',drafts[0]['draft']));assert saved['id'] not in {d['id'] for d in drafts}
   assert len(json.loads(api('/api/drafts')))==len(drafts)+1
   report.update(passed=True,fresh_copy_preflight=True,preserved_runs=len(runs),preserved_drafts=len(drafts),reopened_run=rid,export_hashes_verified=True,custom_port_draft_save=True,new_draft=saved['id'],no_generation=True)
 finally:
  if server is not None and server.poll() is None:server.terminate();server.wait(timeout=15)
  (OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(R/'experiments/reports/R3-local-release-check-2026-10-08.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
