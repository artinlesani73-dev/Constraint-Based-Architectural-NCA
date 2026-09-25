"""NR1: two fixed CPU mechanics pilots with deliberate process interruption/replay."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from nca.experiments import RunStore,read_json,write_once,digest,provenance,snapshot_source
from nca.repair_training import RepairSession,read_payload,latest_verified
from nca.repair_benchmark import load_example,repair_metrics
from nca.massing_targets import evaluate_targets
from nca.recovery import tree_equal
from deploy.studio_process import WorkerTree


def checked_recipe(root):
    recipe=read_json(root/'recipe.json')
    for path,h in recipe['source_sha256'].items():
        if digest(REPO/path)!=h:raise ValueError('Frozen source mismatch: '+path)
    return recipe


def worker(args):
    if sys.stdin.readline()!='GO\n':raise RuntimeError('Owned worker start handshake required')
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();status='completed';failure=None;session=None
    try:
        recipe=checked_recipe(root)
        member=next(x for x in recipe['members'] if x['name']==args.member)
        if not 0<=args.end<=recipe['updates']:raise ValueError('Worker end beyond pilot allowance')
        row=read_json(root/f"inputs/{member['name']}.json")
        inputs,target=load_example(root,row)
        context=read_json(root/f"inputs/{member['name']}-context.json")
        with np.load(root/f"inputs/{member['name']}-context.npz",allow_pickle=False) as p:
            domain=p['domain'];fields={k:p[k] for k in ('permitted','existing','protected','support_boundary')}
        identity={'recipe_sha256':digest(root/'recipe.json'),'dataset':recipe['dataset_run'],
                  'dataset_study_sha256':recipe['dataset_study_sha256'],'member':member,
                  'arrays_sha256':row['arrays_sha256'],'source_sha256':recipe['source_sha256']}
        session=RepairSession(inputs,target,identity,recipe['seed'])
        write_once(out/'identity.json',session.identity)
        if args.resume:session.restore(args.resume)
        if session.completed>args.end:raise ValueError('Resume cursor exceeds requested end')

        def arrays(name,**values):
            with (out/name).open('xb') as f:np.savez_compressed(f,**values)

        def boundary():
            record,state,prob=session.evaluate()
            field=prob>.5
            report,_=evaluate_targets(field,context['scene'],fields,domain)
            record.update(targets=report,**repair_metrics(field,target.astype(bool),inputs['occupancy'].astype(bool),domain,
                                                         float(inputs['context'][6,0,0,0])))
            # Request scalar is float32 in conditioning, but metric must use exact declared request.
            request=context['generation']['spec']['target_fraction']
            record['request_error_cells']=int(field.sum())-int(np.ceil(request*domain.sum()))
            write_once(out/f'boundary-{session.completed:04d}.json',record)
            arrays(f'boundary-{session.completed:04d}.npz',state=state,probability=prob,field=field)

        session.save(out/f'checkpoint-{session.completed:04d}.pt');boundary()
        while session.completed<args.end:
            if time.perf_counter()-start>args.seconds:raise TimeoutError('Worker time cap')
            t=time.perf_counter();record,state=session.step()
            session.save(out/f'checkpoint-{session.completed:04d}.pt')
            arrays(f'training-output-{session.completed:04d}.npz',pre_update_rollout_state=state)
            write_once(out/f'update-{session.completed:04d}.json',dict(record,wall_seconds=time.perf_counter()-t))
            if session.completed in (4,8):boundary()
            if args.pause and session.completed==4:
                write_once(out/'ready-to-interrupt.json',{'pid':os.getpid(),'completed':4})
                sys.stdin.readline()  # Parent owns this process and kills it at this durable boundary.
                raise RuntimeError('Interrupted worker must not be continued in the same process')
        if time.perf_counter()-start>args.seconds:raise TimeoutError('Worker time cap after final save')
    except Exception:
        status='failed';failure=traceback.format_exc()
    result={'status':status,'failure':failure,'completed':session.completed if session else None,
            'wall_seconds':time.perf_counter()-start,'pid':os.getpid(),'resume':args.resume}
    write_once(out/'worker-result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1


def main(args):
    recipe=read_json(REPO/'experiments/configs/NR1-cpu.json')
    store=RunStore(REPO/'.local-artifacts/runs')
    run=store.create('NR1 CPU repair mechanics and process recovery','learned_mechanics',recipe,recipe['seed'],
                     provenance(REPO),args.parent_run or recipe['dataset_run'])
    root=store.path(run);print('RUN_ID='+run,flush=True)
    started=time.perf_counter();deadline=started+recipe['max_seconds']
    status='completed';failure=None;workers=[];comparisons=[]
    def launch(member,name,*,resume=None,pause=False):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('600-second pilot cap')
        out=root/'workers'/name;log=root/(name+'.log')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--root',str(root),'--output',str(out),
             '--member',member,'--end','8','--seconds',str(remaining)]
        if resume:cmd+=['--resume',str(resume)]
        if pause:cmd+=['--pause']
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.PIPE,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            entry={'name':name,'pid':p.pid,'resume':str(resume) if resume else None,'planned_interruption':pause}
            workers.append(entry)
            tree=None
            try:
                tree=WorkerTree(p)
                p.stdin.write(b'GO\n');p.stdin.flush()
                if pause:
                    while not (out/'ready-to-interrupt.json').exists():
                        if p.poll() is not None:raise RuntimeError('Worker exited before interruption marker')
                        if time.perf_counter()>=deadline:raise TimeoutError('Pilot cap before interruption')
                        time.sleep(.05)
                    marker=read_json(out/'ready-to-interrupt.json')
                    if marker['completed']!=4 or type(marker['pid']) is not int:raise ValueError('Interruption marker differs')
                    entry['worker_pid']=marker['pid']
                    tree.terminate();p.wait(timeout=10)
                    entry['outcome']='deliberately_interrupted_after_4'
                else:
                    code=p.wait(timeout=max(.01,deadline-time.perf_counter()))
                    if code or read_json(out/'worker-result.json')['status']!='completed':
                        raise RuntimeError('Worker failed: '+name+'; inspect '+str(log))
                    entry['outcome']='completed'
                    entry['worker_pid']=read_json(out/'worker-result.json')['pid']
                entry['exit_code']=p.returncode
            finally:
                if tree is not None:
                    if tree.active_count():tree.terminate()
                    entry['active_processes_after_stop']=tree.active_count()
                    tree.close()
                elif p.poll() is None:p.kill()
                p.wait(timeout=10)
                entry['exit_code']=p.returncode
                p.stdin.close()
        store.event(run,'worker_finished','Owned worker execution retained',**entry)
        return out
    try:
        snapshot_source(REPO,root/'source.zip')
        write_once(root/'recipe.json',recipe);checked_recipe(root)
        if store.verify(recipe['dataset_run']):raise ValueError('Dataset run corrupt')
        source=store.path(recipe['dataset_run'])
        if digest(source/'study.json')!=recipe['dataset_study_sha256']:raise ValueError('Dataset study changed')
        data=read_json(source/'study.json')
        (root/'inputs').mkdir()
        for member in recipe['members']:
            row=next(x for x in data['examples'] if x['case']==member['case'] and x['damage']==member['damage'])
            if row['split']!='train':raise ValueError('Pilot must never learn from held-out sites')
            target=next(x for x in data['targets'] if x['case']==member['case'])
            original=source/row['arrays'];copied=root/f"inputs/{member['name']}.npz"
            shutil.copyfile(original,copied)
            if digest(copied)!=row['arrays_sha256']:raise ValueError('Copied input mismatch')
            write_once(root/f"inputs/{member['name']}.json",dict(row,arrays=copied.relative_to(root).as_posix()))
            for suffix,key in [('json','json'),('npz','arrays')]:
                shutil.copyfile(source/target[key],root/f"inputs/{member['name']}-context.{suffix}")
            baseline=launch(member['name'],member['name']+'-uninterrupted')
            identity=read_json(baseline/'identity.json')
            first=read_payload(baseline/'checkpoint-0000.pt',identity)
            final=read_payload(baseline/'checkpoint-0008.pt',identity)
            if tree_equal(first['model'],final['model']) or any(x['pre_clip_gradient_norm']<=0 for x in final['trace']):
                raise ValueError('Pilot did not produce actual finite learning updates')
            with np.load(baseline/'boundary-0000.npz',allow_pickle=False) as p,np.load(copied,allow_pickle=False) as q:
                if not np.array_equal(p['field'],q['damaged']):raise ValueError('Initialization changed damaged input')
            interrupted=launch(member['name'],member['name']+'-interrupted',pause=True)
            restart,rejected=latest_verified(interrupted,identity)
            if restart.name!='checkpoint-0004.pt' or rejected:raise ValueError('Unexpected recovery cursor')
            resumed=launch(member['name'],member['name']+'-resumed',resume=restart)
            for branch,steps in [(interrupted,range(5)),(resumed,range(4,9))]:
                for step in steps:
                    a=read_payload(baseline/f'checkpoint-{step:04d}.pt',identity)
                    b=read_payload(branch/f'checkpoint-{step:04d}.pt',identity)
                    equal=tree_equal(a,b)
                    comparisons.append({'member':member['name'],'branch':branch.name,'step':step,'checkpoint_exact':equal})
                    if not equal:raise ValueError('Full checkpoint replay differs')
            for step in (4,8):
                other=interrupted if step==4 else resumed
                if read_json(baseline/f'boundary-{step:04d}.json')!=read_json(other/f'boundary-{step:04d}.json'):
                    raise ValueError('Boundary scores differ')
                with np.load(baseline/f'boundary-{step:04d}.npz',allow_pickle=False) as a,np.load(other/f'boundary-{step:04d}.npz',allow_pickle=False) as b:
                    if any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Boundary arrays differ')
            if time.perf_counter()>deadline:raise TimeoutError('Pilot total cap')
    except Exception:
        status='failed';failure=traceback.format_exc()
    metrics={'admission_gate':status=='completed','workers':len(workers),'exact_checkpoint_comparisons':sum(x['checkpoint_exact'] for x in comparisons),
             'unique_training_updates':16 if status=='completed' else None,'executed_training_updates':32 if status=='completed' else None,
             'wall_seconds':time.perf_counter()-started,'heldout_updates':0,'device':'cpu'}
    result={'run_id':run,'status':status,'failure':failure,'metrics':metrics,'workers':workers,'comparisons':comparisons}
    write_once(root/'study.json',result)
    # Keep every worker file, including partial writes and deliberate interruption evidence.
    for path in sorted(root.rglob('*')):
        if path.is_file() and 'artifacts' not in path.parts and 'events' not in path.parts and path.name!='run.json':
            store.attach(run,path,'pilot_evidence')
    interpretation='Two seen-example CPU mechanics pilots only; no model quality, GPU recovery or production promotion claim.'
    store.finish(run,status,metrics,interpretation)
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'metrics':metrics,
               'interpretation':interpretation,'artifact_location':str(root),'drive_backup':'pending'})
    print(json.dumps(result,indent=2),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--root');parser.add_argument('--output');parser.add_argument('--member')
    parser.add_argument('--end',type=int,default=8);parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--resume');parser.add_argument('--pause',action='store_true')
    args=parser.parse_args();raise SystemExit(worker(args) if args.worker else main(args))
