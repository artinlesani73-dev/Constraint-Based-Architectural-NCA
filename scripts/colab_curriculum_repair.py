"""One explicitly approved CGR3 training seed, or a bounded eight-update CPU rehearsal."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,subprocess,sys,threading,time,traceback,uuid,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import read_json,write_once,digest
from nca.curriculum_package import verify
from nca.curriculum_repair import SEEDS,SETTINGS
from scripts.colab_repair_preflight import watch_parent,wait_before_deadline,bundle_results


def worker(a):
    if sys.stdin.readline()!='GO\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np
    import torch
    from nca.curriculum_repair import ConnectedSession
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);session=None;status='completed';error=None;t=time.monotonic()
    try:
        _,data=verify(ROOT)
        session=ConnectedSession(ROOT,data['rows'],{'study':SETTINGS,'manifest_sha256':digest(ROOT/'manifest.json')},device=a.device,seed=a.seed)
        write_once(out/'identity.json',session.identity)
        if a.device=='cuda:0':
            rt=session.identity['runtime']
            # These are the stack and hardware type demonstrated by NR2. A changed
            # stack requires review/preflight, not an automatic nondeterministic retry.
            expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu128','numpy':'2.1.3','python':'3.13.15','cuda_build':'12.8','cudnn':91900}
            if any(rt[k]!=v for k,v in expected.items()):raise ValueError('GPU/software differs from verified NR2 stack; stop for review')
            torch.cuda.reset_peak_memory_stats()
        def save_boundary():
            values=session.evaluate(0)
            with (out/f'boundary-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,**values)
        session.save(out/'checkpoint-0000.pt');save_boundary()
        end=8 if a.cpu_rehearsal else 256
        while session.completed<end:
            start=time.monotonic();trace,state=session.step()
            session.save(out/f'checkpoint-{session.completed:04d}.pt')
            with (out/f'training-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state,start=session.last_start)
            write_once(out/f'update-{session.completed:04d}.json',dict(trace,wall_seconds=time.monotonic()-start))
            if session.completed%64==0 or session.completed==end:save_boundary()
            if a.device=='cuda:0' and torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('Reserved GPU memory exceeds80%')
    except BaseException:error=traceback.format_exc();status='failed'
    memory={}
    if session is not None and a.device=='cuda:0':
        memory={'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'total':torch.cuda.get_device_properties(0).total_memory}
    result={'status':status,'failure':error,'seed':a.seed,'completed':session.completed if session else None,
            'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'pid':os.getpid(),'wall_seconds':time.monotonic()-t,'memory':memory}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1


def main(a):
    if a.cpu_rehearsal!=(a.device=='cpu'):raise ValueError('CPU is only an eight-update rehearsal')
    if a.device=='cuda:0' and not a.approved_seed_job:raise ValueError('Explicit one-seed compute approval required')
    if not 0<a.seconds<=600:raise ValueError('Maximum600s per job')
    verify(ROOT)
    runs=ROOT/'curriculum-runs';runs.mkdir(exist_ok=True)
    run=runs/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12]);run.mkdir()
    if a.device=='cuda:0':write_once(ROOT/f'curriculum-attempt-seed-{a.seed}.json',{'run':str(run),'seed':a.seed,'max_seconds':a.seconds})
    request={'seed':a.seed,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'updates':8 if a.cpu_rehearsal else 256,
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
            'interpretation':'Training evidence only. Final quality is evaluated separately on CPU with frozen validation-only rules.'}
    write_once(run/'result.json',result);archive=bundle_results(run)
    print(json.dumps(result,indent=2));print('DOWNLOAD='+str(archive),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--approved-seed-job',action='store_true');p.add_argument('--cpu-rehearsal',action='store_true');p.add_argument('--seconds',type=float,default=600)
    a=p.parse_args();raise SystemExit(worker(a) if a.worker else main(a))
