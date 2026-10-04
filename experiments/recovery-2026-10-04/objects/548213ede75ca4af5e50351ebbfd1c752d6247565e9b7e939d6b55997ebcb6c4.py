"""G2 bounded generation pilot, with embedded exact recovery checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,subprocess,sys,threading,time,traceback,uuid,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import read_json,write_once,digest
from nca.generation_package import verify
from nca.generation_training import SEEDS,SETTINGS
from scripts.colab_repair_preflight import watch_parent,wait_before_deadline,bundle_results


def worker(a):
    if sys.stdin.readline()!='GO\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np,torch
    from nca.generation_training import GenerationSession,equal_tree
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);status='completed';error=None;t=time.monotonic();session=None
    try:
        _,data=verify(ROOT)
        identity={'manifest_sha256':digest(ROOT/'manifest.json'),'settings':SETTINGS}
        def fresh():return GenerationSession(ROOT,data['rows'],identity,device=a.device,seed=a.seed)
        session=fresh();write_once(out/'identity.json',session.identity)
        if a.device=='cuda:0':
            expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
            if any(session.identity['runtime'][k]!=v for k,v in expected.items()):raise ValueError('Runtime changed; stop')
            torch.cuda.reset_peak_memory_stats()
        session.save(out/'checkpoint-0000.pt')
        end=3 if a.cpu_rehearsal else 256
        for update in range(1,end+1):
            trace,state=session.step();session.save(out/f'checkpoint-{update:04d}.pt')
            with (out/f'training-{update:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state)
            write_once(out/f'update-{update:04d}.json',trace)
            if update in (2,3):
                expected_payload=session.payload();clone=fresh();clone.restore(out/f'checkpoint-{update-1:04d}.pt')
                replay,replay_state=clone.step()
                checks={'update':update,'full_payload_equal':equal_tree(expected_payload,clone.payload()),'state_equal':bool(np.array_equal(state,replay_state))}
                write_once(out/f'recovery-{update:04d}.json',checks)
                if not checks['full_payload_equal'] or not checks['state_equal']:raise AssertionError('Exact recovery failed')
                session=clone
            if update==end:
                with (out/'seed-only-boundary.npz').open('xb') as f:np.savez_compressed(f,**session.evaluate(0))
            if a.device=='cuda:0' and torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('Memory cap')
    except BaseException:error=traceback.format_exc();status='failed'
    result={'status':status,'failure':error,'completed':session.completed if session else 0,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda:0' else None,'quality_evaluated':False}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1

def main(a):
    if a.cpu_rehearsal!=(a.device=='cpu'):raise ValueError('CPU is only an compatibility rehearsal')
    if a.device=='cuda:0' and not a.approved_seed_job:raise ValueError('Explicit one-seed compute approval required')
    if not 0<a.seconds<=600:raise ValueError('Maximum600s per job')
    verify(ROOT)
    runs=ROOT/'generation-runs';runs.mkdir(exist_ok=True)
    run=runs/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12]);run.mkdir()
    if a.device=='cuda:0':write_once(ROOT/f'generation-attempt-seed-{a.seed}.json',{'run':str(run),'seed':a.seed,'max_seconds':a.seconds})
    request={'seed':a.seed,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'updates':3 if a.cpu_rehearsal else 256,'replayed_updates':2,
             'max_seconds':a.seconds,'manifest_sha256':digest(ROOT/'manifest.json'),'settings':SETTINGS}
    write_once(run/'request.json',request);print('RUN_DIR='+str(run),flush=True)
    start=time.monotonic();p=None;tree=None;error=None;status='completed';result={};cleanup={}
    try:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--device',a.device,'--seed',str(a.seed),'--output',str(run/'worker')]
        if a.cpu_rehearsal:command+=['--cpu-rehearsal']
        with (run/'worker.log').open('x',encoding='utf-8') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
            if os.name=='nt':
                from deploy.studio_process import WorkerTree
                tree=WorkerTree(p)
            p.stdin.write(b'GO\n');p.stdin.flush()
            wait_before_deadline(p,start+a.seconds)
            result=read_json(run/'worker/result.json')
            if p.returncode or result['status']!='completed':raise RuntimeError('Training worker failed; inspect preserved log')
    except BaseException:status='failed';error=traceback.format_exc()
    finally:
        if tree is not None:
            if tree.active_count():tree.terminate()
            cleanup['active_processes_after_stop']=tree.active_count();tree.close()
        elif p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
        if p is not None:
            p.wait(timeout=10);p.stdin.close();cleanup['exit_code']=p.returncode
    result={'status':status,'failure':error,'request':request,'worker':result,'cleanup':cleanup,'wall_seconds':time.monotonic()-start,
            'interpretation':'G2 generation training evidence; no quality acceptance or automatic deployment.'}
    write_once(run/'result.json',result);archive=bundle_results(run)
    print(json.dumps(result,indent=2));print('DOWNLOAD='+str(archive),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--approved-seed-job',action='store_true');p.add_argument('--cpu-rehearsal',action='store_true');p.add_argument('--seconds',type=float,default=600)
    a=p.parse_args();raise SystemExit(worker(a) if a.worker else main(a))
