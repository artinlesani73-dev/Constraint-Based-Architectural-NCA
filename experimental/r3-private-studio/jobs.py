"""Windows local worker ownership and evidence-preserving interruption records."""
from pathlib import Path
import datetime, hashlib, json, msvcrt, os, subprocess, sys, threading, uuid
from contextlib import contextmanager

@contextmanager
def ownership(path):
    # Keep the lock file; deleting it could split ownership across file handles.
    with open(path, 'a+b') as handle:
        handle.seek(0)
        if os.fstat(handle.fileno()).st_size == 0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Research folder is busy in another studio or worker')
        try:yield
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def write(path,value):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def interrupt(run,reason):
    old=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if old['state'] not in ('running','queued'):return
    event=run/('interruption-'+uuid.uuid4().hex+'.json')
    write(event,dict(previous_status=old,reason=reason))
    write(run/'status.json',dict(id=old['id'],state='interrupted',message=reason,
          updated=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write(run/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
          for p in run.iterdir() if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')})

def recover(data):
    for path in (data/'runs').glob('*/status.json'):
        interrupt(path.parent,'Worker stopped before completion. Evidence retained; retry as a new run.')

def start(runid,request,admission,app,data):
    run=data/'runs'/runid
    try:
        with (run/'worker.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-B',str(app.parent/'worker.py'),
                '--app',str(app),'--data',str(data),'--run',runid],stdout=log,stderr=subprocess.STDOUT)
        # Log is kept outside evidence manifest until child has closed it.
        def watch():
            try:
                child.wait()
                with ownership(data/'worker.lock'):
                    interrupt(run,'Worker exited before completion. Evidence retained; retry as a new run.')
                    write(run/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in run.iterdir() if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')})
            finally:admission.release()
        threading.Thread(target=watch,daemon=True).start()
    except Exception:
        interrupt(run,'Worker could not start. Evidence retained; retry as a new run.')
        # The HTTP caller owns admission until start() returns successfully.
        raise
