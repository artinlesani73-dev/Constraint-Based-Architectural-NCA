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
    import torch
    from nca.reversible_repair import ConnectedSession
    from nca.recovery import tree_equal
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);session=None;status='completed';error=None;t=time.monotonic();checks={}
    try:
        _,data=verify(ROOT)
        row=next(r for r in data['rows'] if r['damage']=='cube5')
        identity={'preflight':'RGR1_cu130_recovery_v1','manifest_sha256':digest(ROOT/'manifest.json')}
        session=ConnectedSession(ROOT,[row],identity,device=a.device,seed=a.seed)
        write_once(out/'identity.json',session.identity);write_once(out/'selected-row.json',row)
        if a.device=='cuda:0':
            rt=session.identity['runtime']
            expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
            differences={k:{'expected':v,'actual':rt[k]} for k,v in expected.items() if rt[k]!=v}
            if differences:raise ValueError('Observed runtime changed: '+json.dumps(differences))
            torch.cuda.reset_peak_memory_stats()
        session.save(out/'checkpoint-0000.pt')
        first,_=session.step();session.save(out/'before-replay.pt')
        expected_trace,expected_state=session.step();expected_start=session.tensors(0)[0].detach().cpu().numpy()[0,0].copy();expected_payload=session.payload()
        session.save(out/'after-update.pt')
        with (out/'expected.npz').open('xb') as f:np.savez_compressed(f,start=expected_start,state=expected_state)
        replay=ConnectedSession(ROOT,[row],identity,device=a.device,seed=a.seed)
        replay.restore(out/'before-replay.pt');actual_trace,actual_state=replay.step()
        replay.save(out/'replayed.pt')
        with (out/'replayed.npz').open('xb') as f:np.savez_compressed(f,start=replay.tensors(0)[0].detach().cpu().numpy()[0,0],state=actual_state)
        checks={'trace_exact':expected_trace==actual_trace,'start_exact':np.array_equal(expected_start,replay.tensors(0)[0].detach().cpu().numpy()[0,0]),'state_exact':np.array_equal(expected_state,actual_state),'full_payload_exact':tree_equal(expected_payload,replay.payload())}
        write_once(out/'checks.json',checks)
        if not all(checks.values()):raise AssertionError('Compatibility/recovery checks failed')
        # Direct device-side rule probe: deletion followed by rebirth, no target.
        from nca.reversible_repair import transition,ConnectedRepair as Reversible
        from nca.connected_repair import ConnectedRepair as Control
        anchor=torch.zeros(1,1,5,5,5,dtype=torch.bool,device=a.device);anchor[:,:,2,2,2]=True
        legal=torch.ones_like(anchor);high=torch.ones_like(anchor,dtype=torch.float32)
        grown=transition(anchor,anchor,legal,legal,high)[0]
        deleted=transition(anchor,grown,legal,legal,high*0)[0]
        regrown=transition(anchor,deleted,legal,legal,high)[0]
        if not torch.equal(deleted,anchor) or not torch.equal(grown,regrown):raise AssertionError('Device delete/rebirth failed')
        write_once(out/'transition-check.json',{'delete_rebirth_exact':True})
        # One matched high-birth timing probe per model, plus captured learned output.
        occupancy,features,allowed,_=session.tensors(0);timings=[]
        for label,cls in [('CGR1',Control),('RGR1',Reversible)]:
            torch.manual_seed(1201);model=cls().float().to(a.device)
            with torch.no_grad():
                model.last.bias[0]=2.
                if a.device=='cuda:0':torch.cuda.synchronize()
                start=time.monotonic();values=model.rollout(occupancy,features,allowed,torch.Generator(device=a.device).manual_seed(2101),32)
                if a.device=='cuda:0':torch.cuda.synchronize()
                elapsed=time.monotonic()-start
            timings.append({'model':label,'seconds':elapsed,'occupied':int(values['field'].sum()),'interpretation':'Untrained forced-positive32-step probe; includes cleanup and transfers; single timing.'})
        write_once(out/'timings.json',timings)
        with (out/'learned-boundary.npz').open('xb') as f:np.savez_compressed(f,**replay.evaluate(0))

    except BaseException:error=traceback.format_exc();status='failed'
    memory={}
    if session is not None and a.device=='cuda:0':memory={'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()}
    result={'status':status,'failure':error,'completed':session.completed if session else 0,'optimizer_steps_executed':3 if all(checks.values()) and checks else None,'checks':checks,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'memory':memory}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1


def main(a):
    if a.cpu_rehearsal!=(a.device=='cpu'):raise ValueError('CPU is only an compatibility rehearsal')
    if a.device=='cuda:0' and not a.approved_seed_job:raise ValueError('Explicit one-seed compute approval required')
    if not 0<a.seconds<=120:raise ValueError('Maximum120s per job')
    verify(ROOT)
    runs=ROOT/'compatibility-runs';runs.mkdir(exist_ok=True)
    run=runs/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:12]);run.mkdir()
    if a.device=='cuda:0':write_once(ROOT/f'compatibility-attempt-seed-{a.seed}.json',{'run':str(run),'seed':a.seed,'max_seconds':a.seconds})
    request={'seed':a.seed,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'updates':2,'optimizer_steps_including_replay':3,
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
            'interpretation':'Compatibility and exact recovery check only; not a quality trial or full training run.'}
    write_once(run/'result.json',result);archive=bundle_results(run)
    print(json.dumps(result,indent=2));print('DOWNLOAD='+str(archive),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--approved-seed-job',action='store_true');p.add_argument('--cpu-rehearsal',action='store_true');p.add_argument('--seconds',type=float,default=120)
    a=p.parse_args();raise SystemExit(worker(a) if a.worker else main(a))
