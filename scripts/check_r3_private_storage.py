"""One consolidated storage migration, restart and rejection check; no generation."""
from pathlib import Path
import hashlib, importlib.util, io, json, os, subprocess, sys, time, urllib.request, zipfile
R=Path(__file__).resolve().parents[1]
APP=R/'.local-artifacts/r3-private-storage-v2-2026-10-09'
OUT=R/'.local-artifacts/r3-private-storage-check-2026-10-09'
SOURCE=R/'.local-artifacts/r3-local-release-2026-10-08'

def main():
    OUT.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('private_storage',APP/'storage.py')
    store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
    before=store.inventory(SOURCE/'app')
    app_before={p.relative_to(APP).as_posix():store.digest(p) for p in APP.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    data=OUT/'research-data'
    record=store.import_release(SOURCE,data,APP/'app')
    assert before==store.inventory(data) and record['runs']==15 and record['drafts']==6
    for action in (lambda:store.import_release(SOURCE,data,APP/'app'),
                   lambda:store.open_store(APP/'app',APP/'app')):
        try:action()
        except ValueError:pass
        else:raise AssertionError('Unsafe storage path accepted')
    def api(path,body=None):
        req=urllib.request.Request('http://127.0.0.1:8024'+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:8024','X-NCA-Local':'1'})
        with urllib.request.urlopen(req,timeout=20) as response:return response.read()
    saved=None
    for attempt in range(2):
        with (OUT/f'server-{attempt}.log').open('w',encoding='utf-8') as log:
            server=subprocess.Popen([sys.executable,'-B',str(APP/'launch.py'),'--data-dir',str(data),'--port','8024'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            try:
                deadline=time.monotonic()+35
                while True:
                    if server.poll() is not None:raise RuntimeError('Server exited; inspect log')
                    try:runs=json.loads(api('/api/runs'));break
                    except OSError:
                        if time.monotonic()>deadline:raise
                        time.sleep(.25)
                assert len(runs)==15
                rid='8f11090ceaef4f618f853acddb7c4988'
                assert json.loads(api('/api/run/'+rid))['result']['id']==rid
                drafts=json.loads(api('/api/drafts'))
                if attempt==0:
                    saved=json.loads(api('/api/drafts',drafts[0]['draft']))
                    raw=api('/api/export/'+rid)
                    (OUT/'reopened-run.zip').write_bytes(raw)
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        for name,h in json.loads(z.read('export-manifest.json')).items():
                            assert hashlib.sha256(z.read(name)).hexdigest()==h
                else:assert len(drafts)==7 and saved['id'] in {d['id'] for d in drafts}
            finally:
                if server.poll() is None:server.terminate();server.wait(timeout=15)
    assert store.inventory(SOURCE/'app')==before
    assert all(store.digest(data/name)==h for name,h in before.items())
    assert app_before=={p.relative_to(APP).as_posix():store.digest(p) for p in APP.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    report=dict(passed=True,runs=15,imported_drafts=6,new_draft=saved['id'],
                restart_preserved_draft=True,all_import_hashes_verified=True,
                source_and_app_unchanged=True,export_hashes_verified=True,
                existing_destination_and_app_path_rejected=True,no_generation=True)
    text=json.dumps(report,indent=2)+'\n'
    (OUT/'result.json').write_text(text,encoding='utf-8')
    (R/'experiments/reports/R3-private-storage-check-2026-10-09.json').write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()
