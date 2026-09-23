"""D1 direct-field recovery, cost pilot, and gated 17-scene control."""
import argparse,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.direct import DirectSession,metadata,CONFIG
from nca.experiments import provenance,snapshot_source,read_json,write_once
from nca.recovery import save_checkpoint,tree_equal
from nca.e0 import evaluate
from nca.objective import weighted_total
from deploy.checkpoints import load_model_c
from scripts.diagnostic_inputs import load_inputs
from scripts.run_sensitivity import STORE,records,record,arrays


def worker(args):
    started=time.perf_counter();d=STORE.path(args.run_id);protocol=read_json(d/'protocol.json');meta=protocol['members'][args.branch]
    session=DirectSession(meta,args.resume);setup=time.perf_counter()-started
    begin=session.completed
    if not begin<args.stop_after<=meta['max_updates']:raise ValueError('Invalid update boundary')
    def state_record(label):
        with torch.no_grad():
            state,values,loss=session.objective()
            row={'branch':args.branch,'scene':meta['scene'],'recipe':meta['recipe'],'completed_updates':session.completed,
                'terms':{k:float(v[0]) for k,v in values['terms'].items()},
                'regularizers':{k:float(v[0]) for k,v in values['regularizers'].items()},
                'mass_ratio':float(values['mass_ratio'][0]),'metrics':evaluate(state,session.config,session.item['scene']),
                'totals':{k:float(weighted_total(values,c['family_weights'],c['regularizer_weights'])) for k,c in protocol['config']['recipes'].items()},
                'fields':arrays(args.run_id,f'{args.branch}-{label}',{'raw':session.model.raw.detach().numpy(),'material':session.model().detach().numpy()})}
        record(args.run_id,f'{args.branch}-{label}',row,'direct_state');return row
    initial=state_record('initial')
    if begin==0:
        _,values,_=session.objective();terms={**values['terms'],**values['regularizers']};grads={};norms={}
        for name,v in terms.items():
            g,=torch.autograd.grad(v.sum(),session.model.raw,retain_graph=True)
            grads[name]=g.detach().numpy();norms[name]=float(g.double().norm())
        record(args.run_id,args.branch+'-probe',{'branch':args.branch,'norms':norms,'fields':arrays(args.run_id,args.branch+'-probe',grads)},'gradient_probe')
        del values,terms,grads
    rows=[]
    for _ in range(begin,args.stop_after):
        tick=time.perf_counter();trace,gradient=session.step();u=session.completed
        path=d/f'{args.branch}-u{u:02d}.pt'
        save_checkpoint(path,session.model,session.optimizer,session.scheduler,session.generator,meta,u)
        ckpt=STORE.attach(args.run_id,path,'direct_checkpoint')
        with torch.no_grad():fields=arrays(args.run_id,f'{args.branch}-u{u:02d}',{'raw':session.model.raw.detach().numpy(),'material':session.model().detach().numpy(),'gradient_before_update':gradient})
        row={'branch':args.branch,'trace':trace,'checkpoint':ckpt,'fields':fields,'seconds':time.perf_counter()-tick}
        record(args.run_id,f'{args.branch}-u{u:02d}',row,'direct_update');rows.append(row)
        if u%8==0 or u==args.stop_after:print(f'{args.branch} update={u} objective_before={trace["loss_before_update"]:.6g}',flush=True)
    final=state_record('final')
    record(args.run_id,args.branch+'-result',{'branch':args.branch,'scene':meta['scene'],'recipe':meta['recipe'],
        'restored_updates':begin,'completed_updates':session.completed,'setup_seconds':setup,'seconds':time.perf_counter()-started,
        'initial':initial,'final':final,'update_seconds':[r['seconds'] for r in rows]},'direct_case')
    return 0


def launch(run,branch,stop,cap,resume=None):
    d=STORE.path(run);cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--run-id',run,'--branch',branch,'--stop-after',str(stop)]
    if resume:cmd+=['--resume',str(resume)]
    log=d/(branch+'.log');tick=time.perf_counter();timed_out=False
    with log.open('x',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:child.kill();child.wait();code=child.returncode;timed_out=True
        except BaseException:child.kill();child.wait();raise
    STORE.attach(run,log,'worker_log')
    record(run,'process-'+branch,{'branch':branch,'seconds':time.perf_counter()-tick,'returncode':code,'timed_out':timed_out,'cap':cap},'process_record')
    print(f'{branch}: exit={code}, seconds={time.perf_counter()-tick:.2f}',flush=True)
    if timed_out:raise TimeoutError('Time cap; completed boundaries and partial artifacts retained')
    if code:raise RuntimeError('Worker failed: '+str(log))


def recovery_check(run):
    assert not STORE.verify(run);d=STORE.path(run);rows=records(run,'direct_update')
    by={b:sorted([r for r in rows if r['branch']==b],key=lambda r:r['trace']['update']) for b in ('whole','prefix','resumed','repeat')}
    load=lambda r:torch.load(d/r['checkpoint']['path'],weights_only=True,map_location='cpu')
    assert [r['trace']['update'] for r in by['whole']]==[1,2,3,4]
    assert [r['trace']['update'] for r in by['prefix']]==[1,2]
    checks={'prefix_checkpoint':tree_equal(load(by['whole'][1]),load(by['prefix'][-1]))}
    for b in ('resumed','repeat'):
        joined=by['prefix']+by[b];assert [r['trace']['update'] for r in joined]==[1,2,3,4]
        checks[b+'_trace']=[r['trace'] for r in by['whole']]==[r['trace'] for r in joined]
        checks[b+'_checkpoint']=tree_equal(load(by['whole'][-1]),load(joined[-1]));checks[b+'_fields']=True
        for a,c in zip(by['whole'],joined):
            with np.load(d/a['fields']['path'],allow_pickle=False) as x,np.load(d/c['fields']['path'],allow_pickle=False) as y:
                checks[b+'_fields'] &= x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
    if not all(checks.values()):raise ValueError('Direct recovery mismatch')
    return checks


def admission(pilot,config):
    assert not STORE.verify(pilot)
    assert read_json(STORE.path(pilot)/'result.json')['status']=='completed'
    cases=records(pilot,'direct_case');assert len(cases)==6
    p90=float(np.percentile([t for r in cases for t in r['update_seconds']],90));setup=max(r['setup_seconds'] for r in cases)
    # Include a conservative startup allowance beyond the session's measured setup.
    startup=max(5.,setup+3.)
    estimate=1.5*(34*startup+34*config['full_updates']*p90)
    return {'pilot_run':pilot,'p90_update_seconds':p90,'max_session_setup_seconds':setup,'startup_allowance_seconds':startup,
        'estimated_full_seconds':estimate,'total_cap_seconds':config['full_total_cap_seconds'],'admitted':estimate<=config['full_total_cap_seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['recovery','pilot','full'])
    p.add_argument('--recovery-run');p.add_argument('--pilot-run');p.add_argument('--parent-run')
    p.add_argument('--worker',action='store_true');p.add_argument('--run-id');p.add_argument('--branch');p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    args=p.parse_args()
    if args.worker:return worker(args)
    if not args.mode:p.error('--mode required')
    torch.set_num_threads(2);config=read_json(REPO/CONFIG);cfg,_,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
    assert not STORE.verify(config['source_comparison_run'])
    assert read_json(STORE.path(config['source_comparison_run'])/'result.json')['status']=='completed'
    members={};gate={};budget=None
    if args.mode=='recovery':
        m=metadata('ref-01-ground-pair','mass_3',cfg,checkpoint,inputs);members={b:m for b in ('whole','prefix','resumed','repeat')}
    else:
        if not args.recovery_run:p.error('--recovery-run required')
        assert read_json(STORE.path(args.recovery_run)/'result.json')['status']=='completed'
        gate=recovery_check(args.recovery_run)
        assert records(args.recovery_run,'protocol')[0]['members']['whole']==metadata('ref-01-ground-pair','mass_3',cfg,checkpoint,inputs)
        scenes=config['pilot_scenes'] if args.mode=='pilot' else config['scenes']
        for i,scene in enumerate(scenes):
            for j,recipe in enumerate(('mapped_30','mass_3')):members[f'c{i:02d}r{j}']=metadata(scene,recipe,cfg,checkpoint,inputs)
        if args.mode=='full':
            if not args.pilot_run:p.error('--pilot-run required')
            old=records(args.pilot_run,'protocol')[0]
            assert old['config']==config and old['members']['c00r0']==metadata(config['pilot_scenes'][0],'mapped_30',cfg,checkpoint,inputs)
            budget=admission(args.pilot_run,config)
            if not budget['admitted']:raise ValueError('Full run exceeds preregistered cost admission: '+str(budget))
    protocol={'protocol':'D1_'+args.mode+'_v1','mode':args.mode,'config':config,'members':members,
        'recovery_run':args.recovery_run,'pilot_run':args.pilot_run,'admission':budget,'paid_compute':False}
    origin=provenance(REPO);run=STORE.create('D1 '+args.mode,'direct_'+args.mode,protocol,0,origin,parent_run=args.parent_run)
    d=STORE.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();status,error='completed',None;checks={}
    try:
        record(run,'protocol',protocol,'protocol');path=d/'source.zip';snapshot_source(REPO,path);STORE.attach(run,path,'source_snapshot')
        if args.mode=='recovery':
            for b,stop in [('whole',4),('prefix',2),('resumed',4),('repeat',4)]:
                resume=None
                if b in ('resumed','repeat'):
                    row=max([r for r in records(run,'direct_update') if r['branch']=='prefix'],key=lambda r:r['trace']['update'])
                    resume=d/row['checkpoint']['path']
                launch(run,b,stop,120,resume)
            checks=recovery_check(run)
        else:
            stop=config['pilot_updates'] if args.mode=='pilot' else config['full_updates']
            for b in members:
                cap=config['pilot_worker_cap_seconds'] if args.mode=='pilot' else min(config['full_case_cap_seconds'],config['full_total_cap_seconds']-(time.perf_counter()-started))
                if cap<=0:raise TimeoutError('Total wall cap reached')
                launch(run,b,stop,cap)
            assert len(records(run,'direct_case'))==len(members)
            checks={'complete_matrix':True,'recovery_gate_passed':True}
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'mode':args.mode,'checks':checks,'cases':len(records(run,'direct_case')),
        'recorded_updates':len(records(run,'direct_update')),'seconds':time.perf_counter()-started,'provenance':origin,
        'artifact_location':f'.local-artifacts/runs/{run}'}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Stopped; complete and partial evidence retained',traceback=error)
    STORE.finish(run,status,checks,'Per-scene optimizer, not a learned generalizing model')
    write_once(REPO/'experiments/records'/(run+'.json'),summary);print(summary,flush=True)
    return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
