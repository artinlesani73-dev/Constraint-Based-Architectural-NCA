"""NR2 eight-update process-restart preflight. No long-run training entry point."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from hashlib import sha256

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nca.experiments import read_json,write_once,digest
from nca.colab_package import verify_package


def watch_parent():
    """Detect pipe closure without holding Windows CRT stdin locks during imports."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.PeekNamedPipe.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,
                                      ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
        kernel.PeekNamedPipe.restype=wintypes.BOOL
        handle=msvcrt.get_osfhandle(sys.stdin.fileno())
        while True:
            available=wintypes.DWORD()
            if not kernel.PeekNamedPipe(handle,None,0,None,ctypes.byref(available),None):break
            if available.value:break  # No further input is part of this protocol.
            time.sleep(.1)
    else:
        os.read(sys.stdin.fileno(),1)
    os._exit(71)  # EOF/parent death stops compute; completed checkpoints survive.


def wait_before_deadline(process, deadline):
    # Recheck the monotonic deadline after suspend/resume; a single Windows
    # wait(timeout=...) may exclude low-power time from its timeout accounting.
    while process.poll() is None:
        if time.monotonic()>=deadline:raise TimeoutError('Preflight time cap')
        time.sleep(.05)


def worker(args):
    if sys.stdin.readline()!='GO\n':raise RuntimeError('Owned start handshake required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np
    import torch
    from nca.repair_portable import PortableSession
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    status='completed';failure=None;started=time.perf_counter();session=None
    try:
        _,data=verify_package(ROOT)
        session=PortableSession(ROOT,data['rows'],{'manifest_sha256':digest(ROOT/'manifest.json')},device=args.device)
        if args.device=='cuda:0':torch.cuda.reset_peak_memory_stats()
        write_once(out/'identity.json',session.identity)
        if args.resume:session.restore(args.resume)
        if not 0<=session.completed<=8:raise ValueError('Preflight cursor out of bounds')
        def boundary():
            values=session.evaluate(0)
            with (out/f'boundary-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,**values)
        session.save(out/f'checkpoint-{session.completed:04d}.pt');boundary()
        while session.completed<8:
            tick=time.perf_counter();record,state=session.step()
            session.save(out/f'checkpoint-{session.completed:04d}.pt')
            with (out/f'training-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state)
            write_once(out/f'update-{session.completed:04d}.json',dict(record,wall_seconds=time.perf_counter()-tick))
            if session.completed in (4,8):boundary()
            if args.pause and session.completed==4:
                write_once(out/'ready.json',{'completed':4,'pid':os.getpid()})
                threading.Event().wait()
    except Exception:
        status='failed';failure=traceback.format_exc()
    memory={}
    if session is not None and args.device=='cuda:0':
        memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'total_bytes':torch.cuda.get_device_properties(0).total_memory}
    result={'status':status,'failure':failure,'pid':os.getpid(),'completed':session.completed if session else None,
            'wall_seconds':time.perf_counter()-started,'memory':memory,'device':args.device}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1


def bundle_results(run):
    archive=run.with_suffix('.zip');files={}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file():
                name=p.relative_to(run).as_posix();raw=p.read_bytes();files[name]=sha256(raw).hexdigest();z.writestr(name,raw)
        z.writestr('evidence-manifest.json',json.dumps(files,indent=2))
    with zipfile.ZipFile(archive) as z:
        for name,h in files.items():
            if sha256(z.read(name)).hexdigest()!=h:raise ValueError('Evidence export failed')
    write_once(archive.with_suffix('.receipt.json'),{'archive':archive.name,'sha256':digest(archive),'files':len(files)})
    return archive


def main(args):
    if args.device=='cuda:0' and not args.allow_gpu_preflight:raise ValueError('Explicit approved preflight flag required')
    if not 0<args.seconds<=600:raise ValueError('Preflight capped at 600 seconds')
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    run=root/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12]);run.mkdir()
    if args.device=='cuda:0':
        write_once(ROOT/'gpu-preflight-attempt.json',{'run':str(run),'approved_max_compute_seconds':args.seconds})
    started=time.monotonic();deadline=started+args.seconds
    write_once(run/'request.json',{'device':args.device,'seconds':args.seconds,'updates':8,'seed':1201,
                                 'manifest_sha256':digest(ROOT/'manifest.json')})
    print('RUN_DIR='+str(run),flush=True)
    workers=[];status='completed';failure=None;comparisons=[]
    def launch(name,resume=None,pause=False):
        if time.monotonic()>=deadline:raise TimeoutError('Preflight time cap before worker launch')
        out=run/name;entry={'branch':name};workers.append(entry)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--device',args.device,'--output',str(out)]
        if resume:command+=['--resume',str(resume)]
        if pause:command+=['--pause']
        with (run/(name+'.log')).open('x',encoding='utf-8') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
            entry['launcher_pid']=p.pid;tree=None
            def stop():
                if tree is not None:tree.terminate()
                elif p.poll() is None:os.killpg(p.pid,signal.SIGKILL) if os.name!='nt' else p.kill()
                p.wait(timeout=10)
            try:
                if os.name=='nt':
                    from deploy.studio_process import WorkerTree
                    tree=WorkerTree(p)
                p.stdin.write(b'GO\n');p.stdin.flush()
                if pause:
                    while not (out/'ready.json').exists():
                        if p.poll() is not None:raise RuntimeError('Worker exited before checkpoint4; read its log')
                        if time.monotonic()>=deadline:raise TimeoutError('Preflight time cap')
                        time.sleep(.05)
                    marker=read_json(out/'ready.json')
                    if marker['completed']!=4:raise ValueError('Interruption cursor differs')
                    entry['worker_pid']=marker['pid'];stop();entry['outcome']='deliberately_interrupted'
                else:
                    wait_before_deadline(p,deadline)
                    result=read_json(out/'result.json');entry.update(worker_pid=result['pid'],result=result)
                    if p.returncode or result['status']!='completed':raise RuntimeError('Worker failed; retain logs and stop')
                    entry['outcome']='completed'
            finally:
                if tree is not None:
                    if tree.active_count():tree.terminate()
                    entry['active_processes_after_stop']=tree.active_count();tree.close()
                elif p.poll() is None:stop()
                p.wait(timeout=10);p.stdin.close();entry['exit_code']=p.returncode
        return out
    try:
        verify_package(ROOT)
        base=launch('uninterrupted')
        stopped=launch('interrupted',pause=True)
        resumed=launch('resumed',resume=stopped/'checkpoint-0004.pt')
        import numpy as np
        from nca.repair_portable import read_portable
        from nca.recovery import tree_equal
        identity=read_json(base/'identity.json')
        initial=read_portable(base/'checkpoint-0000.pt',identity);final=read_portable(base/'checkpoint-0008.pt',identity)
        if tree_equal(initial['model'],final['model']) or any(x['pre_clip_gradient_norm']<=0 for x in final['trace']):
            raise ValueError('No actual learning updates')
        for branch,steps in [(stopped,range(5)),(resumed,range(4,9))]:
            for step in steps:
                equal=tree_equal(read_portable(base/f'checkpoint-{step:04d}.pt',identity),read_portable(branch/f'checkpoint-{step:04d}.pt',identity))
                comparisons.append({'branch':branch.name,'step':step,'exact':equal})
                if not equal:raise ValueError('Full checkpoint replay differs; do not relax tolerance')
        for step in range(1,9):
            branch=stopped if step<=4 else resumed
            with np.load(base/f'training-{step:04d}.npz',allow_pickle=False) as a,np.load(branch/f'training-{step:04d}.npz',allow_pickle=False) as b:
                if not np.array_equal(a['state'],b['state']):raise ValueError('Training-state replay differs')
        for step,branch in [(4,stopped),(8,resumed)]:
            with np.load(base/f'boundary-{step:04d}.npz',allow_pickle=False) as a,np.load(branch/f'boundary-{step:04d}.npz',allow_pickle=False) as b:
                if any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Evaluation replay differs')
        if args.device=='cuda:0':
            if any(w['result']['memory']['peak_reserved_bytes']>.8*w['result']['memory']['total_bytes'] for w in workers if 'result' in w):
                raise MemoryError('Preflight reserved more than 80% of GPU memory')
        if time.monotonic()>deadline:raise TimeoutError('Preflight cap exceeded')
    except BaseException:
        status='failed';failure=traceback.format_exc()
    result={'status':status,'failure':failure,'device':args.device,'gpu_recovery_passed':status=='completed' and args.device=='cuda:0',
            'wall_seconds':time.monotonic()-started,'workers':workers,'checkpoint_comparisons':comparisons,
            'interpretation':'Mechanics check only; no quality admission or permission for a longer run.'}
    write_once(run/'result.json',result)
    archive=bundle_results(run)
    print(json.dumps(result,indent=2));print('DOWNLOAD='+str(archive),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--allow-gpu-preflight',action='store_true');p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--output-root',default=str(ROOT/'runs'));p.add_argument('--output');p.add_argument('--resume');p.add_argument('--pause',action='store_true')
    a=p.parse_args();raise SystemExit(worker(a) if a.worker else main(a))
