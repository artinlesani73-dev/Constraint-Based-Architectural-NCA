"""RGR1 cu130 compatibility, recovery and cleanup timing; three optimizer steps."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,subprocess,sys,threading,time,traceback,uuid,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import read_json,write_once,digest
from nca.reversible_package import verify
from nca.reversible_repair import SEEDS,SETTINGS
from scripts.colab_repair_preflight import watch_parent,wait_before_deadline,bundle_results


def worker(a):
    if sys.stdin.readline()!='GO\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np
    import torch,hashlib,gc
    from nca.connected_repair import ConnectedSession as Control
    from nca.reversible_repair import ConnectedSession as Candidate
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);status='completed';error=None;t=time.monotonic();arms={};initial_hashes={};orders={};firing_states={};session=None
    def parameter_hash(model):
        h=hashlib.sha256()
        for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    try:
        _,data=verify(ROOT)
        plan=read_json(ROOT/'paired-plan.json')
        expected_plan={'version':'RGR1_paired_comparison_v1','arms':['CGR1','RGR1'],'updates_per_arm':256,'total_updates':512,'shared_seconds_cap':600,'seed':1201,'steps':32,'automatic_retry':False,'test_evaluation':False}
        if plan!=expected_plan:raise ValueError('Paired plan differs')
        end=2 if a.cpu_rehearsal else 256
        for label,cls in [('CGR1',Control),('RGR1',Candidate)]:
            arm=out/label;arm.mkdir();start=time.monotonic()
            session=cls(ROOT,data['rows'],{'paired_plan':plan,'arm':label,'manifest_sha256':digest(ROOT/'manifest.json')},device=a.device,seed=a.seed)
            write_once(arm/'identity.json',session.identity)
            if a.device=='cuda:0':
                expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
                if any(session.identity['runtime'][k]!=v for k,v in expected.items()):raise ValueError('Runtime changed; stop')
                torch.cuda.reset_peak_memory_stats()
            initial_hashes[label]=parameter_hash(session.model)
            session.save(arm/'checkpoint-0000.pt')
            def boundary():
                with (arm/f'boundary-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,**session.evaluate(0))
            boundary()
            while session.completed<end:
                step_start=time.monotonic();trace,state=session.step()
                session.save(arm/f'checkpoint-{session.completed:04d}.pt')
                with (arm/f'training-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state)
                write_once(arm/f'update-{session.completed:04d}.json',dict(trace,wall_seconds=time.monotonic()-step_start))
                if session.completed%64==0 or session.completed==end:boundary()
                if a.device=='cuda:0' and torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('Memory cap')
            orders[label]=[x['row_index'] for x in session.trace];firing_states[label]=session.firing.get_state().cpu()
            arms[label]={'completed':session.completed,'wall_seconds':time.monotonic()-start,'initial_parameter_sha256':initial_hashes[label],'peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda:0' else None}
            write_once(arm/'result.json',arms[label]);session=None;gc.collect()
            if a.device=='cuda:0':torch.cuda.empty_cache()
        checks={'same_initial_parameters':initial_hashes['CGR1']==initial_hashes['RGR1'],'same_row_order':orders['CGR1']==orders['RGR1'],'same_final_firing_rng':torch.equal(firing_states['CGR1'],firing_states['RGR1'])}
        write_once(out/'pairing-checks.json',checks)
        if not all(checks.values()):raise AssertionError('Pairing mismatch')
    except BaseException:error=traceback.format_exc();status='failed'
    result={'status':status,'failure':error,'arms':arms,'in_progress_completed':session.completed if session else None,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'interpretation':'Training evidence; final quality evaluated later. Partial arms are not a completed comparison.'}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1


def main(a):
    if a.cpu_rehearsal!=(a.device=='cpu'):raise ValueError('CPU is only an compatibility rehearsal')
    if a.device=='cuda:0' and not a.approved_seed_job:raise ValueError('Explicit one-seed compute approval required')
    if not 0<a.seconds<=600:raise ValueError('Maximum600s per job')
    verify(ROOT)
    runs=ROOT/'paired-runs';runs.mkdir(exist_ok=True)
    run=runs/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12]);run.mkdir()
    if a.device=='cuda:0':write_once(ROOT/f'paired-attempt-seed-{a.seed}.json',{'run':str(run),'seed':a.seed,'max_seconds':a.seconds})
    request={'seed':a.seed,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'updates_per_arm':2 if a.cpu_rehearsal else 256,'arms':['CGR1','RGR1'],
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
            'interpretation':'Paired training evidence; no heldout quality evaluation or automatic deployment.'}
    write_once(run/'result.json',result);archive=bundle_results(run)
    print(json.dumps(result,indent=2));print('DOWNLOAD='+str(archive),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--approved-seed-job',action='store_true');p.add_argument('--cpu-rehearsal',action='store_true');p.add_argument('--seconds',type=float,default=600)
    a=p.parse_args();raise SystemExit(worker(a) if a.worker else main(a))
