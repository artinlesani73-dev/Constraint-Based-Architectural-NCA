"""Use the frozen server generation function; never duplicate scientific rollout."""
from pathlib import Path
import argparse, importlib.util, json, os, sys
from jobs import ownership

def main():
    p=argparse.ArgumentParser();p.add_argument('--app',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args()
    if len(a.run)!=32 or any(c not in '0123456789abcdef' for c in a.run):raise ValueError('Invalid run ID')
    app=a.app.resolve();data=a.data.resolve()
    os.environ['NCA_STUDIO_DATA']=str(data)
    sys.path.insert(0,str(app))
    with ownership(data/'worker.lock'):
        run=data/'runs'/a.run
        # A delayed child must not resume a job already classified as interrupted.
        if json.loads((run/'status.json').read_text())['state']!='queued':return
        spec=importlib.util.spec_from_file_location('studio_worker_server',app/'server.py')
        server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        server.lock.acquire()
        server.generate(a.run,json.loads((run/'request.json').read_text(encoding='utf-8')))
if __name__=='__main__':main()
