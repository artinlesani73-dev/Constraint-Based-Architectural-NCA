"""Consolidated worker interruption/ownership fixture plus one real parity run."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,sys,time,urllib.request,uuid
import numpy as np
R=Path(__file__).resolve().parents[1];APP=R/'.local-artifacts/r3-private-jobs-v3-2026-10-10'
OUT=R/'.local-artifacts/r3-private-jobs-check-v4-2026-10-10'
def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(APP));from jobs import ownership,recover,write
    from storage import import_release,inventory
    data=OUT/'data';import_release(R/'.local-artifacts/r3-local-release-2026-10-08',data,APP/'app')
    original=inventory(data)
    fixture=OUT/'fixture';fixture.mkdir()
    (fixture/'server.py').write_text("import threading,time\nlock=threading.Lock()\ndef generate(rid,request):\n from pathlib import Path\n import os\n from jobs import write\n run=Path(os.environ['NCA_STUDIO_DATA'])/'runs'/rid\n write(run/'status.json',dict(id=rid,state='running'))\n (run/'partial.txt').write_text('retained')\n time.sleep(120)\n",encoding='utf-8')
    rid=uuid.uuid4().hex;run=data/'runs'/rid;run.mkdir();write(run/'request.json',{})
    write(run/'status.json',dict(id=rid,state='queued'))
    child=subprocess.Popen([sys.executable,'-B',str(APP/'worker.py'),'--app',str(fixture),'--data',str(data),'--run',rid])
    try:
        deadline=time.monotonic()+10
        while not (run/'partial.txt').exists():
            if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Fixture did not start')
            time.sleep(.1)
        try:
            with ownership(data/'worker.lock'):pass
        except RuntimeError:pass
        else:raise AssertionError('Active worker ownership accepted')
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=10)
    deadline=time.monotonic()+5
    while True:
        try:
            with ownership(data/'worker.lock'):recover(data)
            break
        except RuntimeError:
            if time.monotonic()>deadline:raise
            time.sleep(.1)
    assert (run/'partial.txt').read_text()=='retained'
    assert json.loads((run/'status.json').read_text())['state']=='interrupted'
    assert len(list(run.glob('interruption-*.json')))==1
    recover(data);assert len(list(run.glob('interruption-*.json')))==1
    # Delayed worker must not restart the interrupted record.
    subprocess.run([sys.executable,'-B',str(APP/'worker.py'),'--app',str(fixture),'--data',str(data),'--run',rid],check=True,timeout=10)
    def api(path,body=None):
        req=urllib.request.Request('http://127.0.0.1:8025'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:8025','X-NCA-Local':'1'})
        with urllib.request.urlopen(req,timeout=20) as response:return json.loads(response.read())
    with (OUT/'server.log').open('w') as log:
        server=subprocess.Popen([sys.executable,'-B',str(APP/'launch.py'),'--data-dir',str(data),'--port','8025'],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+40
            while True:
                if server.poll() is not None:raise RuntimeError('Server startup failed')
                try:api('/api/scenes');break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.25)
            duplicate=subprocess.run([sys.executable,'-B',str(APP/'launch.py'),'--data-dir',str(data),'--port','8026'],capture_output=True,text=True,timeout=40)
            (OUT/'duplicate.log').write_text(duplicate.stdout+duplicate.stderr)
            assert duplicate.returncode!=0 and 'busy' in duplicate.stderr
            ref=data/'runs/43182d56' # Resolve the full saved identifier.
            ref=next((data/'runs').glob('43182d56*'))
            request=json.loads((ref/'request.json').read_text());request['parent_run']=ref.name
            new=api('/api/generate',request)['id']
            deadline=time.monotonic()+100
            while True:
                result=api('/api/run/'+new)
                if result['status']['state'] not in ('queued','running'):break
                if time.monotonic()>deadline:raise RuntimeError('Worker timed out')
                time.sleep(.5)
            assert result['status']['state']=='completed',result['status']
            for name in ('G10-64.npz','G10-128.npz','R3-64.npz','R3-128.npz'):
                assert np.array_equal(np.load(ref/name)['field'],np.load(data/'runs'/new/name)['field']),name
            # Watcher must finish manifest finalization before checking all files.
            deadline=time.monotonic()+10
            while True:
                try:
                    with ownership(data/'worker.lock'):pass
                    from storage import digest
                    manifest=json.loads((data/'runs'/new/'manifest.json').read_text())
                    assert 'worker.log' in manifest
                    assert all(digest(data/'runs'/new/n)==h for n,h in manifest.items());break
                except (AssertionError,RuntimeError):
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
        finally:
            if server.poll() is None:server.terminate();server.wait(timeout=10)
    from storage import digest
    assert all(digest(data/n)==h for n,h in original.items())
    report=dict(passed=True,synthetic_interruption_retains_partial=True,recovery_idempotent=True,
        delayed_worker_refused=True,active_worker_and_duplicate_studio_refused=True,
        original_history_unchanged=True,parity_run=new,parent_run=ref.name,
        all_four_fields_exact=True,worker_manifest_verified=True,no_training=True)
    text=json.dumps(report,indent=2)+'\n';(OUT/'result.json').write_text(text)
    (R/'experiments/reports/R3-private-jobs-check-2026-10-10.json').write_text(text);print(text)
if __name__=='__main__':main()
