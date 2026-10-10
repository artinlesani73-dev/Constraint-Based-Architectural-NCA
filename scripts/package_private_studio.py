"""Finish the local studio package with a verified, separate research snapshot."""
from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,time,urllib.request,zipfile
R=Path(__file__).resolve().parents[1]
APP=R/'.local-artifacts/r3-research-history-v3-2026-10-10'
DATA=R/'.local-artifacts/r3-research-data-2026-10-09'
OUT=Path('C:/Users/artin/Documents/Codex/outputs/NCA-Private-Studio-2026-10-10')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in ('studio.lock','worker.lock') and '__pycache__' not in p.parts}
def main():
    sys.path.insert(0,str(APP));from jobs import ownership
    manifest=json.loads((APP/'private-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(APP/n)==h,n
    OUT.mkdir(exist_ok=False)
    with ownership(DATA/'studio.lock'),ownership(DATA/'worker.lock'):
        for p in (DATA/'runs').glob('*/status.json'):
            assert json.loads(p.read_text())['state'] not in ('queued','running'),'Active job'
        before=inventory(DATA)
        shutil.copytree(APP,OUT/'Studio',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(DATA,OUT/'ResearchData-Snapshot',ignore=shutil.ignore_patterns('studio.lock','worker.lock','__pycache__'))
        assert before==inventory(OUT/'ResearchData-Snapshot')==inventory(DATA)
    cmd='@echo off\nsetlocal\nif not defined NCA_PYTHON set "NCA_PYTHON='+str(R/'.venv/Scripts/python.exe')+'"\nif not defined NCA_DATA_DIR set "NCA_DATA_DIR='+str(DATA)+'"\n"%NCA_PYTHON%" -B "%~dp0Studio\\launch.py" --data-dir "%NCA_DATA_DIR%" --port 8023 %*\npause\n'
    (OUT/'START-STUDIO.cmd').write_text(cmd,encoding='utf-8')
    readme='''# Private research studio: local release

Double-click START-STUDIO.cmd. Uses existing project Python and the current research folder:
'''+str(DATA)+'''

Open http://127.0.0.1:8023/#skin=graphite . Keep terminal open; Ctrl+C stops the server.
Only one studio/worker may own a research folder. A worker may finish after the server closes; wait before restarting.

Studio/ contains versioned application files. ResearchData-Snapshot/ is a verified delivery snapshot, not the live save location. This avoids splitting new work between copies. Set NCA_DATA_DIR to an absolute compatible store path to select another store; set NCA_PYTHON if the tested environment moved. No dependency installer or standalone executable is included.

All experiments, failures, drafts and annotation revisions live in the research folder. Geometry exports omit annotations; preserve the whole folder for backup. A same-disk snapshot is not an off-device backup.

Upgrade: stop the current server, wait for active worker completion, snapshot the whole data folder, verify its hashes, then start the new app against the same data directory after checking schema compatibility. Never overwrite an old app or data snapshot. Never run a historical app that does not honor ownership locks against an active folder.

Rollback: stop the new app and its work, retain the complete current data folder first. Reuse an older app only if its data schema is compatible. Otherwise restore a verified snapshot into a NEW folder and explicitly point NCA_DATA_DIR there. Newer results remain in the retained folder; do not overwrite them with the snapshot. No automatic merge/import of arbitrary data archives is provided.

Recovery: interrupted jobs retain partial files/status events. Retry as a new run linked to the parent; numerical checkpoint continuation is not implemented. Labels/notes append revisions; stale-tab saves are rejected. Search history opens each run's G10/R3 comparison. Simultaneous cross-run views remain future scope.

Scientific status: occupied voxels are gross building volume, not construction or room plans. R3 is experimental learned growth plus procedural coordination; MG7 remains production. Nine-family checks are geometric acceptance, not architectural/structural certification. Existing source/model unchanged.

Current phase closes with this local package. Next approved direction: context-conditioned growth of architectural mass across multiple sites; first target one small urban block. New representation, training and compute configuration must be defined separately.
'''
    (OUT/'START-HERE.md').write_text(readme,encoding='utf-8')
    snapshot=OUT/'ResearchData-Snapshot';copy=OUT/'Studio'
    preflight=subprocess.run([sys.executable,'-B',str(copy/'launch.py'),'--data-dir',str(snapshot),'--check'],capture_output=True,text=True,timeout=40)
    (OUT/'preflight.log').write_text(preflight.stdout+preflight.stderr,encoding='utf-8');assert preflight.returncode==0
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8028'+path,timeout=20) as response:return response.read()
    with (OUT/'package-check.log').open('w') as log:
        server=subprocess.Popen([sys.executable,'-B',str(copy/'launch.py'),'--data-dir',str(snapshot),'--port','8028'],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+40
            while True:
                if server.poll() is not None:raise RuntimeError('Packaged server exited')
                try:runs=json.loads(get('/api/runs'));break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.25)
            assert len(runs)==len(list((DATA/'runs').iterdir()))
            drafts=json.loads(get('/api/drafts'));assert len(drafts)==len(list((DATA/'drafts').glob('*.json')))
            rid='8f11090ceaef4f618f853acddb7c4988'
            assert json.loads(get('/api/run/'+rid))['result']['id']==rid
            assert 'version' in json.loads(get('/api/annotations/'+rid))
            with zipfile.ZipFile(io.BytesIO(get('/api/export/'+rid))) as z:
                for n,h in json.loads(z.read('export-manifest.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        finally:
            if server.poll() is None:server.terminate();server.wait(timeout=15)
    assert before==inventory(snapshot)==inventory(DATA)
    result=dict(passed=True,runs=len(runs),drafts=len(drafts),annotation_revisions=len(list((DATA/'annotations').glob('*/*.json'))),
        data_files=len(before),data_inventory=before,original_data_unchanged=True,
        packaged_preflight_startup_reopen_export_verified=True,no_generation=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    payloads=inventory(OUT)
    (OUT/'package-manifest.json').write_text(json.dumps(payloads,indent=2)+'\n',encoding='utf-8')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name not in ('studio.lock','worker.lock'):z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        for n,h in payloads.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    receipt=dict(passed=True,folder=str(OUT),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        runs=len(runs),drafts=len(drafts),annotation_revisions=result['annotation_revisions'],payloads=len(payloads)+1)
    (R/'experiments/reports/Private-studio-package-2026-10-10.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
